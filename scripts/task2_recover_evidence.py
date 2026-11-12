"""Recover Task 2 evidence using saved weights and deterministic evaluation splits."""

from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import sys
from time import perf_counter, sleep
from urllib.request import urlopen

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from PIL import Image

from common.paths import TASK2_DATA, write_json
from task1.lora import adapter_state, load_adapter_state
from task2.evaluate import bootstrap_ci, build_suites, mismatched_map, score_pair
from task2.features import build_cache, load_cache
from task2.grounding import normalized_answer
from task2.model import build_model, build_vl_inputs, forward_loss, reference_nll, truncate_question

CONDITIONS = ("correct", "complementary", "mismatched", "no_image")
BATCH_SIZE = 8
SEED = 6304
COCO = "https://s3.amazonaws.com/images.cocodataset.org/val2014"


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def get_image(image_id):
    name = f"COCO_val2014_{image_id:012d}.jpg"
    path = TASK2_DATA / "evaluation_images" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    for attempt in range(3):
        try:
            with urlopen(f"{COCO}/{name}", timeout=45) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target)
            with Image.open(temporary) as image:
                image.verify()
            temporary.replace(path)
            return path
        except (OSError, ValueError):
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise
            sleep(2 ** attempt)


def evaluation_rows(pairs):
    mapping = mismatched_map(pairs, seed=SEED)
    assert len(mapping) == len(pairs) and -1 not in mapping.values()
    rows = []
    for pair in pairs:
        for side, partner in (("a", "b"), ("b", "a")):
            original, complementary = pair[f"image_{side}"], pair[f"image_{partner}"]
            mismatch = mapping[pair["pair_id"]]
            assert mismatch not in (original, complementary)
            canonical = pair[f"answer_{side}"]
            rows.append({
                "pair_id": pair["pair_id"], "kind": pair["kind"], "side": side,
                "question": pair["question"], "canonical": canonical,
                "references": pair.get(f"answers_{side}", [canonical]),
                "partner_canonical": pair[f"answer_{partner}"],
                "images": dict(zip(CONDITIONS, (original, complementary, mismatch, None))),
                "conditions": {},
            })
    return rows


def generate(lm, connector, tokenizer, prompts, scale):
    assert all("answer" not in prompt for prompt in prompts)
    inputs = [build_vl_inputs(lm, connector, tokenizer, prompt, scale=scale)
              for prompt in prompts]
    assert all(item.n_answer == 0 for item in inputs)
    assert all(item.n_visual == (0 if prompt["image_id"] is None else 196)
               for item, prompt in zip(inputs, prompts))
    width = max(item.seq_len for item in inputs)
    embeds = torch.cat([F.pad(item.inputs_embeds, (0, 0, width - item.seq_len, 0))
                        for item in inputs])
    masks = torch.cat([F.pad(item.attention_mask, (width - item.seq_len, 0))
                       for item in inputs])
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    torch.cuda.synchronize()
    start = perf_counter()
    output = lm.generate(
        inputs_embeds=embeds, attention_mask=masks, do_sample=False,
        use_cache=True, max_new_tokens=16, pad_token_id=tokenizer.pad_token_id,
        eos_token_id=end,
    )
    torch.cuda.synchronize()
    elapsed = perf_counter() - start
    assert output.ndim == 2 and output.shape[0] == len(prompts) and 1 <= output.shape[1] <= 16
    records = []
    for ids in output.tolist():
        stopped = end in ids
        if stopped:
            ids = ids[:ids.index(end) + 1]
        records.append({
            "prediction": tokenizer.decode(ids, skip_special_tokens=True).strip(),
            "generated_ids": ids, "hit_token_cap": len(ids) == 16 and not stopped,
            "generation_seconds_per_answer": elapsed / len(prompts),
        })
    return records


def mean(values):
    return float(np.mean(list(values)))


def summarize(rows):
    pairs = list(zip(rows[::2], rows[1::2]))
    assert all(a["pair_id"] == b["pair_id"] and a["side"] == "a" and b["side"] == "b"
               for a, b in pairs)
    metrics = {}
    for condition in CONDITIONS:
        def pair_values(key):
            return [mean((a["conditions"][condition][key], b["conditions"][condition][key]))
                    for a, b in pairs]

        def identical(a, b):
            return (normalized_answer(a["conditions"][condition]["prediction"]) ==
                    normalized_answer(b["conditions"][condition]["prediction"]))

        stats = {
            "n_pairs": len(pairs), "n_answers": len(rows),
            "accuracy": bootstrap_ci(pair_values("score")),
            "reference_nll": bootstrap_ci(pair_values("reference_nll")),
            "pair_accuracy": mean(a["conditions"][condition]["canonical_match"] and
                                  b["conditions"][condition]["canonical_match"] for a, b in pairs),
            "identical_answer_rate": mean(identical(a, b) for a, b in pairs),
            "empty_answer_rate": mean(pair_values("empty")),
            "partner_match_rate": mean(pair_values("partner_match")),
            "generation_seconds_per_answer": mean(pair_values("generation_seconds_per_answer")),
            "token_cap_rate": mean(pair_values("hit_token_cap")),
        }
        metrics[condition] = stats
    def advantages(key, direction):
        return [direction * mean(row["conditions"]["correct"][key] -
                                 row["conditions"]["complementary"][key] for row in pair)
                for pair in pairs]

    return {
        "conditions": metrics,
        "correct_accuracy_advantage": bootstrap_ci(advantages("score", 1)),
        "correct_nll_advantage": bootstrap_ci(advantages("reference_nll", -1)),
    }


@torch.inference_mode()
def main(bundle, output):
    assert torch.cuda.is_available(), "Select a Kaggle GPU accelerator."
    torch.manual_seed(SEED)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((bundle / "recovery_config.json").read_text())
    for filename, expected in config["checkpoint_sha256"].items():
        assert sha256(bundle / filename) == expected, f"Checkpoint integrity check failed for {filename}"
    scale = float(config["connector_scale"])
    assert math.isfinite(scale) and scale > 0
    data = json.loads((TASK2_DATA / "data.json").read_text())
    splits = json.loads((TASK2_DATA / "split_images.json").read_text())
    train, dev, test = (set(splits[key]) for key in ("train", "dev", "test"))
    assert not (train & dev or train & test or dev & test)
    for name in ("build_summary.json", "split_images.json", "selection_ids.json"):
        shutil.copyfile(TASK2_DATA / name, output / name)
    write_json(output / "provenance.json", {
        **config, "python": platform.python_version(), "torch": str(torch.__version__),
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "transformers": transformers.__version__, "numpy": np.__version__,
        "batch_size": BATCH_SIZE, "max_new_tokens": 16, "seed": SEED,
        "data_sha256": sha256(TASK2_DATA / "data.json"),
        "historical_manifest_match": "historical corrected split manifest unavailable",
        "annotation_sha256": {p.name: sha256(p) for p in sorted((TASK2_DATA / "annotations").glob("*.zip"))},
    })
    suites = build_suites(data, {})
    assert {name: len(pairs) for name, pairs in suites.items()} == {"grounding": 64, "natural": 96}
    rows_by_suite = {name: evaluation_rows(pairs) for name, pairs in suites.items()}
    assert all(len(row["references"]) == 10 for row in rows_by_suite["natural"])
    caption_ids = {int(key) for key in data["captions_dev"]}
    final_ids = {row["images"]["correct"] for rows in rows_by_suite.values() for row in rows}
    assert len(caption_ids) == 96 and len(final_ids) == 320
    assert caption_ids <= dev and final_ids == test
    start = perf_counter()
    image_ids = sorted(caption_ids | final_ids)
    print(f"Downloading {len(image_ids)} evaluation JPEGs", flush=True)
    with ThreadPoolExecutor(max_workers=16) as pool:
        images = dict(zip(image_ids, pool.map(get_image, image_ids)))
    download_seconds = perf_counter() - start
    start = perf_counter()
    torch.cuda.reset_peak_memory_stats()
    cache_path = build_cache(images, TASK2_DATA / "eval_features", batch_size=32)
    features, cache_metadata = load_cache(cache_path)
    assert set(features) == set(images)
    write_json(output / "feature_cache.json", {
        **cache_metadata, "image_download_seconds": download_seconds,
        "cache_seconds": perf_counter() - start,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    })
    torch.cuda.reset_peak_memory_stats()
    lm, connector, tokenizer = build_model(use_lora=True)
    lm.eval()
    connector.eval()
    lm.gradient_checkpointing_disable()

    def load_stage(stage):
        saved = torch.load(bundle / f"{stage}_weights.pt", map_location="cpu", weights_only=True)
        assert saved["lora"].keys() == adapter_state(lm).keys()
        connector.load_state_dict(saved["connector"], strict=True)
        load_adapter_state(lm, saved["lora"])

    def prompt(row, condition):
        image_id = row["images"][condition]
        return {"question": row["question"], "image_id": image_id,
                "features": features[image_id] if image_id is not None else None}

    load_stage("stage3_natural")
    print("GPU smoke test for prompt-only batch-8 generation in all four conditions", flush=True)
    smoke = {}
    first = rows_by_suite["grounding"][:BATCH_SIZE]
    for condition in CONDITIONS:
        prompts = [prompt(row, condition) for row in first]
        batched = generate(lm, connector, tokenizer, prompts, scale)
        singles = [generate(lm, connector, tokenizer, [item], scale)[0] for item in prompts]
        smoke[condition] = {"batch_size": len(prompts),
                            "matching_predictions": sum(a["prediction"] == b["prediction"]
                                                        for a, b in zip(batched, singles))}
        assert all(a["prediction"] == b["prediction"] for a, b in zip(batched, singles)), (
            f"Batched generation differs from single-example smoke in {condition}"
        )
    write_json(output / "gpu_smoke.json", smoke)

    tables = {}
    for suite, rows in rows_by_suite.items():
        for condition in CONDITIONS:
            for offset in range(0, len(rows), BATCH_SIZE):
                batch = rows[offset:offset + BATCH_SIZE]
                assert len(batch) == BATCH_SIZE
                prompts = [prompt(row, condition) for row in batch]
                generations = generate(lm, connector, tokenizer, prompts, scale)
                losses = reference_nll(lm, connector, tokenizer,
                                       [dict(item, answer=row["canonical"])
                                        for item, row in zip(prompts, batch)], scale=scale)
                assert all(math.isfinite(loss) for loss in losses)
                for row, generated, loss in zip(batch, generations, losses):
                    prediction = generated["prediction"]
                    normalized = normalized_answer(prediction)
                    row["conditions"][condition] = {
                        **generated, "reference_nll": loss, "empty": not prediction,
                        "canonical_match": bool(normalized) and normalized == normalized_answer(row["canonical"]),
                        "partner_match": bool(normalized) and normalized == normalized_answer(row["partner_canonical"]),
                        "score": score_pair(suite, prediction, row["references"], row["canonical"]),
                    }
            print(f"{suite} {condition}: {len(rows)} answers", flush=True)
        write_json(output / f"{suite}_predictions.json", rows)
        tables[suite] = {"overall": summarize(rows)}
        for kind in sorted({row["kind"] for row in rows}):
            tables[suite][kind] = summarize([row for row in rows if row["kind"] == kind])
    write_json(output / "metrics.json", tables)
    with (output / "condition_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["suite", "question_type", "condition", "n_answers", "accuracy",
                         "accuracy_ci_low", "accuracy_ci_high", "reference_nll", "pair_accuracy",
                         "identical_answer_rate", "empty_answer_rate", "partner_match_rate",
                         "generation_seconds_per_answer"])
        for suite, kinds in tables.items():
            for kind, values in kinds.items():
                for condition, row in values["conditions"].items():
                    writer.writerow([suite, kind, condition, row["n_answers"], row["accuracy"]["mean"],
                                     row["accuracy"]["lo"], row["accuracy"]["hi"], row["reference_nll"]["mean"],
                                     *(row[key] for key in ("pair_accuracy", "identical_answer_rate",
                                                            "empty_answer_rate", "partner_match_rate",
                                                            "generation_seconds_per_answer"))])
    write_json(output / "evaluation_resources.json", {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "predictions": sum(len(rows) * len(CONDITIONS) for rows in rows_by_suite.values()),
    })

    captions = []
    for stage, update in (("stage2_instruction", 3750), ("stage3_natural", 4500)):
        load_stage(stage)
        examples = [
            {"question": "Describe the image.", "image_id": int(key),
             "features": features[int(key)], "answer": truncate_question(values[0], tokenizer, 64)}
            for key, values in data["captions_dev"].items()
        ]
        loss, values = forward_loss(lm, connector, tokenizer, examples, scale=scale)
        assert math.isfinite(float(loss))
        captions.append({"checkpoint": stage, "update": update, "mean_caption_loss": float(loss),
                         "n_examples": len(values), "includes_end_of_turn": True,
                         "image_id_and_loss": [[ex["image_id"], v] for ex, v in zip(examples, values)]})
        print(f"{stage} caption dev loss {float(loss):.6f}", flush=True)
    write_json(output / "caption_dev_endpoints.json", captions)
    print("RECOVERY_DONE", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
