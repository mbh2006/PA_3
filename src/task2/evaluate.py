"""Task 2 evaluation (PA pp. 8-9).

Four image conditions per suite, with the question and original references held fixed:

    correct       the paired image
    complementary the partner image of the same pair (different reference answer)
    mismatched    a fixed unrelated image from another pair in the same suite
    no_image      visual tokens removed, text instructions unchanged

Scores: normalized exact match (grounding), official ten-reference soft consensus
(natural VQA), pair accuracy, identical-answer rate, empty-answer rate, reference NLL,
image-swap penalty, and 4,000 complete-pair bootstrap intervals.

Generation receives **only** the prompt: the reference answer must never enter the model
input, which the previous implementation did.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/root/pa3")
sys.path.insert(0, str(ROOT / "src"))

from task2.grounding import normalized_answer, vqa_soft_consensus  # noqa: E402
from task2.model import build_vl_inputs, generate_answer, reference_nll  # noqa: E402

N_BOOT = 4000


def prompt_example(question: str, features, image_id: int | None) -> dict:
    """A generation input: no answer field, so nothing but the prompt is fed."""
    return {"question": question, "features": features, "image_id": image_id}


def nll_example(question: str, answer: str, features, image_id: int | None) -> dict:
    return {"question": question, "answer": answer, "features": features,
            "image_id": image_id}


def bootstrap_ci(values: np.ndarray, *, n_boot: int = N_BOOT, seed: int = 6304) -> dict:
    """Complete-pair bootstrap over pairs (each resample keeps pair structure)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, "mean": None, "lo": None, "hi": None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975], method="linear")
    return {"n": int(v.size), "mean": float(v.mean()), "lo": float(lo), "hi": float(hi)}


def build_suites(data: dict, features: dict) -> dict:
    """Pair-structured suites: each element carries both sides of a pair."""
    suites = {}

    def grounding(pairs):
        out = []
        for p in pairs:
            out.append({
                "pair_id": f"{p['kind']}:{p['image_a']}:{p['image_b']}:{p['question'][:24]}",
                "kind": p["kind"],
                "question": p["question"],
                "image_a": int(p["image_a"]), "image_b": int(p["image_b"]),
                "answer_a": str(p["answer_a"]), "answer_b": str(p["answer_b"]),
            })
        return out

    def natural(pairs):
        out = []
        for p in pairs:
            out.append({
                "pair_id": f"vqa:{p['image_a']}:{p['image_b']}:{p['question'][:24]}",
                "kind": p["kind"],
                "question": p["question"],
                "image_a": int(p["image_a"]), "image_b": int(p["image_b"]),
                "answer_a": str(p["consensus_a"]), "answer_b": str(p["consensus_b"]),
                "answers_a": list(p["answers_a"]), "answers_b": list(p["answers_b"]),
            })
        return out

    suites["grounding"] = grounding(data["grounding_test"])
    suites["natural"] = natural(data["vqa_test"])
    return suites


def mismatched_map(pairs: list[dict], seed: int = 6304) -> dict[str, int]:
    """Fixed unrelated image per pair, excluding both images of that pair."""
    rng = np.random.default_rng(seed)
    pool = sorted({p["image_a"] for p in pairs} | {p["image_b"] for p in pairs})
    mapping = {}
    for p in pairs:
        banned = {p["image_a"], p["image_b"]}
        for _ in range(50):
            cand = int(pool[int(rng.integers(0, len(pool)))])
            if cand not in banned:
                mapping[p["pair_id"]] = cand
                break
        else:
            mapping[p["pair_id"]] = -1
    return mapping


def score_pair(kind: str, prediction: str | None, refs: list[str] | None,
               canonical: str) -> float:
    if prediction is None or not str(prediction).strip():
        return 0.0
    if kind == "natural" and refs:
        return vqa_soft_consensus(prediction, refs)
    return float(normalized_answer(prediction) == normalized_answer(canonical))


@torch.no_grad()
def evaluate_suite(lm, connector, tokenizer, pairs, features, *, scale: float,
                   suite: str, log_every: int = 20) -> dict:
    mismatch = mismatched_map(pairs)
    records = []
    for i, p in enumerate(pairs):
        q = p["question"]
        kind = "natural" if suite == "natural" else "grounding"
        sides = []
        for side, img_self, img_other, ans_self, ans_other in (
            ("a", p["image_a"], p["image_b"], p["answer_a"], p["answer_b"]),
            ("b", p["image_b"], p["image_a"], p["answer_b"], p["answer_a"]),
        ):
            refs_self = p.get(f"answers_{side}")
            conds = {
                "correct": (img_self, True),
                "complementary": (img_other, True),
                "mismatched": (mismatch[p["pair_id"]], True),
                "no_image": (None, False),
            }
            out = {}
            for name, (img, with_image) in conds.items():
                feats = features.get(int(img)) if (img is not None and with_image) else None
                ex = prompt_example(q, feats, img)
                pred = generate_answer(lm, connector, tokenizer, ex, scale=scale,
                                       with_image=with_image)
                out[name] = {
                    "prediction": pred,
                    "score": score_pair(kind, pred, refs_self, ans_self),
                    "empty": not pred.strip(),
                }
            # reference NLL under correct / complementary / no-image
            for name, (img, with_image) in (("correct", (img_self, True)),
                                            ("complementary", (img_other, True)),
                                            ("no_image", (None, False))):
                feats = features.get(int(img)) if (img is not None and with_image) else None
                ex = nll_example(q, ans_self, feats, img)
                out[f"nll_{name}"] = float(reference_nll(
                    lm, connector, tokenizer, [ex], scale=scale, with_image=with_image)[0])
            sides.append({"side": side, "gold": ans_self, **out})
        records.append({"pair_id": p["pair_id"], "kind": p["kind"], "sides": sides})
        if log_every and (i + 1) % log_every == 0:
            print(f"  [{suite}] {i+1}/{len(pairs)} pairs", flush=True)
    return summarize_suite(records)


def summarize_suite(records: list[dict]) -> dict:
    conditions = ["correct", "complementary", "mismatched", "no_image"]
    flat = [s for r in records for s in r["sides"]]
    out = {"n_pairs": len(records), "n_examples": len(flat), "conditions": {}}
    for c in conditions:
        out["conditions"][c] = {
            "accuracy": float(np.mean([s[c]["score"] for s in flat])),
            "empty_rate": float(np.mean([s[c]["empty"] for s in flat])),
        }
    # pair accuracy: both sides correct under their own reference
    pair_acc = {
        "correct": float(np.mean([all(s["correct"]["score"] >= 1.0 - 1e-9 for s in r["sides"])
                                  for r in records])),
    }
    out["pair_accuracy"] = pair_acc
    out["identical_answer_rate"] = float(np.mean(
        [normalized_answer(r["sides"][0]["correct"]["prediction"])
         == normalized_answer(r["sides"][1]["correct"]["prediction"]) for r in records]))
    for c in ("correct", "complementary", "no_image"):
        out[f"nll_{c}"] = float(np.mean([s[f"nll_{c}"] for s in flat]))
    out["swap_penalty"] = out["nll_complementary"] - out["nll_correct"]
    out["correct_advantage_accuracy"] = (out["conditions"]["correct"]["accuracy"]
                                         - out["conditions"]["complementary"]["accuracy"])
    out["correct_advantage_nll"] = out["nll_complementary"] - out["nll_correct"]
    # bootstrap over complete pairs
    diff = np.array([np.mean([s["correct"]["score"] for s in r["sides"]])
                     - np.mean([s["complementary"]["score"] for s in r["sides"]])
                     for r in records])
    out["bootstrap_accuracy_advantage"] = bootstrap_ci(diff)
    nll_diff = np.array([np.mean([s["nll_complementary"] for s in r["sides"]])
                         - np.mean([s["nll_correct"] for s in r["sides"]]) for r in records])
    out["bootstrap_nll_advantage"] = bootstrap_ci(nll_diff)
    out["by_kind"] = {}
    for kind in sorted({r["kind"] for r in records}):
        sub = [s for r in records if r["kind"] == kind for s in r["sides"]]
        out["by_kind"][kind] = {
            "n": len(sub),
            "correct_accuracy": float(np.mean([s["correct"]["score"] for s in sub])),
            "complementary_accuracy": float(np.mean([s["complementary"]["score"] for s in sub])),
            "no_image_accuracy": float(np.mean([s["no_image"]["score"] for s in sub])),
        }
    return out


def save(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))
