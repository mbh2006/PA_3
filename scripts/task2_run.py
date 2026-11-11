"""Task 2 end-to-end driver: feature cache -> curriculum -> development/final evaluation.

Designed to run in one GPU session:

    python scripts/task2_run.py --stage features   # SigLIP cache (then encoder unloaded)
    python scripts/task2_run.py --stage train
    python scripts/task2_run.py --stage eval
    python scripts/task2_run.py --stage all

Data comes from /root/pa3/task2_data (built and validated on CPU).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path("/root/pa3")
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "task2_data"
OUT = ROOT / "outputs" / "task_02"
CKPT = OUT / "checkpoints"


def load_data() -> dict:
    return json.loads((DATA / "data.json").read_text())


def assert_features_complete(features: dict) -> None:
    """Every image referenced by any split must have a cached feature.

    Silently dropping a missing image turns a "with image" condition into a no-image
    condition, which is exactly how a previous run produced four identical scores.
    """
    needed = {int(i) for i in json.loads((DATA / "needed_images.json").read_text())}
    missing = sorted(needed - set(features))
    if missing:
        raise SystemExit(f"{len(missing)} required images have no cached feature "
                         f"(first: {missing[:10]})")


def load_features() -> dict[int, "object"]:
    import torch

    path = DATA / "features" / "siglip_features.pt"
    payload = torch.load(path, map_location="cpu")
    feats = payload["features"]
    ids = payload["image_ids"]
    return {int(i): feats[k].to(torch.float16) for k, i in enumerate(ids)}


def build_pools(data: dict, features: dict, tokenizer, max_q=80, max_a=16):
    """Attach image features and cap lengths; returns the four sampler pools."""

    def cap(text: str, n: int) -> str:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        return text if len(ids) <= n else tokenizer.decode(ids[:n], skip_special_tokens=True)

    def pair(p, q_key="question", a_key="consensus", cap_a=max_a):
        qid = p.get("question_ids", [0, 0])
        # grounding pairs store answer_a/answer_b; VQAv2 pairs store consensus_a/consensus_b
        if f"{a_key}_a" not in p and f"answer_a" in p:
            a_key = "answer"
        return [
            {"question": cap(p[q_key], max_q), "answer": cap(str(p[f"{a_key}_a"]), cap_a),
             "features": features.get(int(p["image_a"])),
             "pair_id": tuple(qid), "side": "a", "image_id": int(p["image_a"])},
            {"question": cap(p[q_key], max_q), "answer": cap(str(p[f"{a_key}_b"]), cap_a),
             "features": features.get(int(p["image_b"])),
             "pair_id": tuple(qid), "side": "b", "image_id": int(p["image_b"])},
        ]

    # TRAINING pools come from train2014 only. Using dev/test here is a leak.
    train_pairs = data["grounding_train"]
    presence = [pair(p) for p in train_pairs if p["kind"] == "presence"]
    counting = [pair(p) for p in train_pairs if p["kind"] == "counting"]
    natural = [pair(p) for p in data["vqa_train"]]

    captions = []
    for img, caps in data["captions_train"].items():
        for c in caps:
            captions.append({"question": "Describe the image.",
                             "answer": cap(c, 64), "features": features.get(int(img)),
                             "pair_id": ("cap", int(img)), "image_id": int(img)})

    return {"presence": presence, "counting": counting, "natural": natural,
            "caption": captions}


def stage_features(limit: int | None = None) -> None:
    import torch
    from task2.features import build_cache

    needed = json.loads((DATA / "needed_images.json").read_text())
    wanted = {int(i): "x" for i in needed}

    index = json.loads((DATA / "images" / "index.json").read_text())
    paths = {}
    for img in wanted:
        rel = index.get(str(img))
        if rel and (DATA / rel).exists():
            paths[img] = DATA / rel
    print(f"caching {len(paths)} images", flush=True)

    ids = sorted(paths)
    if limit:
        ids = ids[:limit]
    build_cache({i: paths[i] for i in ids}, DATA / "features", batch_size=64)
    print("FEATURES_DONE", flush=True)


def stage_train(max_updates_per_stage: int | None = None,
                resume_stage1: bool = False) -> None:
    import torch
    from task2.model import build_model, trainable_parameters
    from task2.train import run_curriculum

    data = load_data()
    features = load_features()
    assert_features_complete(features)

    # connector magnitude calibration uses training data only
    lm, connector, tokenizer = build_model(use_lora=True, device="cuda")
    from task2.connector import calibrate_magnitude

    calib_imgs = list(data["captions_train"])[:256]
    feats = torch.stack([features[int(i)] for i in calib_imgs if int(i) in features]).to('cuda')
    text_emb = lm.get_input_embeddings().weight.detach()[:2000].float()
    cal = calibrate_magnitude(connector, feats.reshape(-1, 196, 768), text_emb)
    print(f"calibration scale {cal.scale:.4f} (text rms {cal.text_rms:.4f})", flush=True)

    pools = build_pools(data, features, tokenizer)
    print({k: len(v) for k, v in pools.items()}, flush=True)
    if any(len(v) == 0 for k, v in pools.items() if k in ("presence", "counting", "natural")):
        raise SystemExit("a required training pool is empty")

    from task1.lora import load_adapter_state

    stages = None
    if resume_stage1:
        stage1 = CKPT / "stage1_caption_weights.pt"
        if not stage1.exists():
            raise SystemExit(f"missing Stage 1 checkpoint {stage1}")
        payload = torch.load(stage1, map_location="cpu")
        connector.load_state_dict(payload["connector"])
        load_adapter_state(lm, payload["lora"])
        print("[resume] Stage 1 connector and adapter state loaded", flush=True)
        from task2.train import STAGES as _ALL
        stages = _ALL[1:]

    from task1.lora import adapter_state

    def _dev_hook(stage_name: str) -> dict:
        torch.save({"connector": connector.state_dict(),
                    "lora": adapter_state(lm)}, CKPT / "_dev_probe.pt")
        return dev_nll(features, cal.scale)

    summary = run_curriculum(lm, connector, tokenizer, pools, CKPT, scale=cal.scale,
                             max_updates_per_stage=max_updates_per_stage,
                             dev_hook=_dev_hook, stages=stages)
    summary["calibration"] = cal.as_dict()
    summary["trainable"] = trainable_parameters(lm, connector)
    (OUT / "training.json").write_text(json.dumps(summary, indent=2))
    print("TRAIN_DONE", flush=True)


def stage_eval(limit_pairs: int | None = None) -> None:
    import torch
    from task1.lora import load_adapter_state
    from task2.evaluate import build_suites, evaluate_suite, save
    from task2.model import build_model

    data = load_data()
    features = load_features()
    assert_features_complete(features)
    lm, connector, tokenizer = build_model(use_lora=True, device="cuda")
    state = torch.load(CKPT / "stage3_natural_weights.pt", map_location="cpu")
    connector.load_state_dict(state["connector"])
    load_adapter_state(lm, state["lora"])
    lm.eval()
    connector.eval()

    scale = json.loads((OUT / "training.json").read_text())["calibration"]["scale"]
    suites = build_suites(data, features)
    if limit_pairs:
        suites = {k: v[:limit_pairs] for k, v in suites.items()}
    results = {}
    for name, pairs in suites.items():
        print(f"[eval] {name}: {len(pairs)} pairs", flush=True)
        results[name] = evaluate_suite(lm, connector, tokenizer, pairs, features,
                                       scale=scale, suite=name)
    save(results, OUT / "eval.json")
    print(json.dumps({k: {"correct": v["conditions"]["correct"]["accuracy"],
                          "complementary": v["conditions"]["complementary"]["accuracy"],
                          "no_image": v["conditions"]["no_image"]["accuracy"],
                          "swap_penalty": v["swap_penalty"]}
                      for k, v in results.items()}, indent=2))
    print("EVAL_DONE", flush=True)


def dev_nll(features, scale: float) -> dict:
    """Reference-answer NLL on the development suites under three image conditions."""
    import torch
    from task1.lora import load_adapter_state
    from task2.evaluate import nll_example
    from task2.model import build_model, reference_nll

    data = load_data()
    lm, connector, tokenizer = build_model(use_lora=True, device="cuda")
    stage_state = CKPT / "_dev_probe.pt"
    if stage_state.exists():
        payload = torch.load(stage_state, map_location="cpu")
        connector.load_state_dict(payload["connector"])
        load_adapter_state(lm, payload["lora"])
    lm.eval()
    connector.eval()
    out = {}
    for suite, key in (("grounding", "grounding_dev"), ("natural", "vqa_dev")):
        exs_c, exs_s, exs_n = [], [], []
        for p in data[key]:
            a = str(p.get("answer_a", p.get("consensus_a")))
            b = str(p.get("answer_b", p.get("consensus_b")))
            exs_c.append(nll_example(p["question"], a, features.get(int(p["image_a"])), p["image_a"]))
            exs_c.append(nll_example(p["question"], b, features.get(int(p["image_b"])), p["image_b"]))
            exs_s.append(nll_example(p["question"], a, features.get(int(p["image_b"])), p["image_b"]))
            exs_s.append(nll_example(p["question"], b, features.get(int(p["image_a"])), p["image_a"]))
            exs_n.append(nll_example(p["question"], a, None, None))
            exs_n.append(nll_example(p["question"], b, None, None))
        c = reference_nll(lm, connector, tokenizer, exs_c, scale=scale, with_image=True)
        s_ = reference_nll(lm, connector, tokenizer, exs_s, scale=scale, with_image=True)
        n = reference_nll(lm, connector, tokenizer, exs_n, scale=scale, with_image=False)
        out[suite] = {"nll_correct": float(sum(c) / len(c)),
                      "nll_swapped": float(sum(s_) / len(s_)),
                      "nll_no_image": float(sum(n) / len(n)),
                      "swap_penalty": float(sum(s_) / len(s_) - sum(c) / len(c))}
    del lm, connector
    torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["features", "train", "eval", "all"])
    ap.add_argument("--max-updates-per-stage", type=int, default=None)
    ap.add_argument("--limit-images", type=int, default=None)
    ap.add_argument("--limit-pairs", type=int, default=None)
    ap.add_argument("--resume-stage1", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.stage in ("features", "all"):
        stage_features(limit=args.limit_images)
    if args.stage in ("train", "all"):
        stage_train(max_updates_per_stage=args.max_updates_per_stage,
                    resume_stage1=args.resume_stage1)
    if args.stage in ("eval", "all"):
        stage_eval(args.limit_pairs)
    print("TOTAL_SECONDS", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
