"""Task 2 data build (v2): strictly disjoint train/dev/test image splits.

Order of allocation matters. All evaluation pools come from val2014, so they are
allocated in a fixed order with a global "used image" set:

    1. grounding dev   (12 presence + 12 counting, unique images)
    2. grounding test  (32 presence + 32 counting, unique, excluding dev)
    3. VQA dev         (excluding all grounding images)
    4. VQA test        (excluding all grounding images and VQA dev)
    5. training pools  (train2014 only, so disjoint from val2014 by construction)

Training grounding pairs come from train2014 instance annotations, never from val2014.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common.paths import TASK2_DATA  # noqa: E402
from task2.data import (  # noqa: E402
    DEV_COUNTS, TEST_COUNTS, TRAIN_COUNTS, build_candidate_pairs, caption_selection,
    select_pairs,
)
from task2.grounding import (  # noqa: E402
    GroundingPair, counting_question, presence_question,
)

import numpy as np  # noqa: E402


def grounding_candidates(idx, categories, split, *, seed, per_category=12):
    """Shuffle each category's positive/negative image lists so that different
    categories draw different images; index-wise pairing reuses the same few images."""
    rng = np.random.default_rng(seed)
    out = []
    for c in categories:
        has = np.array(sorted(i for i, v in idx.items() if c in v))
        lacks = np.array(sorted(i for i, v in idx.items() if c not in v))
        by_count = {k: np.array(sorted(i for i, v in idx.items() if v.get(c) == k))
                    for k in (1, 2, 3, 4)}
        if len(has) and len(lacks):
            h = rng.permutation(has)[:per_category]
            l = rng.permutation(lacks)[:per_category]
            for a, b in zip(h, l):
                out.append(GroundingPair(c, presence_question(c), "presence",
                                         int(a), "yes", int(b), "no", split))
        counts = [k for k in (1, 2, 3, 4) if len(by_count[k])]
        for i, ka in enumerate(counts):
            for kb in counts[i + 1:]:
                aa = rng.permutation(by_count[ka])[:per_category]
                bb = rng.permutation(by_count[kb])[:per_category]
                for a, b in zip(aa, bb):
                    out.append(GroundingPair(c, counting_question(c), "counting",
                                             int(a), str(ka), int(b), str(kb), split))
    return out

SEED = 6304


def read_zip_json(path: Path, suffix: str):
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.endswith(suffix)]
        if not names:
            raise FileNotFoundError(f"{path.name}: no member ending {suffix}")
        with z.open(names[0]) as fh:
            return json.load(io.TextIOWrapper(fh, encoding="utf-8"))


def image_set(pairs) -> set[int]:
    s: set[int] = set()
    for p in pairs:
        s.add(int(p.image_a if hasattr(p, "image_a") else p["image_a"]))
        s.add(int(p.image_b if hasattr(p, "image_b") else p["image_b"]))
    return s


def select_grounding(candidates, n_presence, n_counting, used: set[int]):
    """Greedy, deterministic, image-unique selection by kind."""
    chosen, taken = [], {"presence": 0, "counting": 0}
    want = {"presence": n_presence, "counting": n_counting}
    for p in sorted(candidates, key=lambda x: (x.kind, x.image_a, x.image_b)):
        if taken[p.kind] >= want[p.kind]:
            continue
        if int(p.image_a) in used or int(p.image_b) in used:
            continue
        chosen.append(p)
        used.update({int(p.image_a), int(p.image_b)})
        taken[p.kind] += 1
        if taken == want:
            break
    if taken != want:
        raise SystemExit(f"grounding selection short: got {taken}, want {want}")
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-annotations", action="store_true",
                        help="fetch the public COCO and VQAv2 annotation archives if missing")
    args = parser.parse_args()
    out = TASK2_DATA
    ann = out / "annotations"
    if args.download_annotations:
        from task2_build_data import COCO_ANN, VQA_TRAIN_A, VQA_TRAIN_Q, VQA_VAL_A, VQA_VAL_Q, download

        for name, url in (
            ("annotations_trainval2014.zip", COCO_ANN),
            ("v2_Questions_Train.zip", VQA_TRAIN_Q),
            ("v2_Annotations_Train.zip", VQA_TRAIN_A),
            ("v2_Questions_Val.zip", VQA_VAL_Q),
            ("v2_Annotations_Val.zip", VQA_VAL_A),
        ):
            download(url, ann / name)
    coco = read_zip_json(ann / "annotations_trainval2014.zip", "instances_train2014.json")
    inst_train = coco
    val_inst = read_zip_json(ann / "annotations_trainval2014.zip", "instances_val2014.json")
    caps_train = read_zip_json(ann / "annotations_trainval2014.zip", "captions_train2014.json")
    caps_val = read_zip_json(ann / "annotations_trainval2014.zip", "captions_val2014.json")
    vqa_train_q = read_zip_json(ann / "v2_Questions_Train.zip", "train2014_questions.json")
    vqa_train_a = read_zip_json(ann / "v2_Annotations_Train.zip", "train2014_annotations.json")
    vqa_val_q = read_zip_json(ann / "v2_Questions_Val.zip", "val2014_questions.json")
    vqa_val_a = read_zip_json(ann / "v2_Annotations_Val.zip", "val2014_annotations.json")

    def coco_index(inst):
        cats = {c["id"]: c["name"] for c in inst["categories"]}
        per_image = defaultdict(dict)
        for a in inst["annotations"]:
            if a.get("iscrowd"):
                continue
            name = cats[a["category_id"]]
            per_image[a["image_id"]][name] = per_image[a["image_id"]].get(name, 0) + 1
        return per_image

    train_idx, val_idx = coco_index(inst_train), coco_index(val_inst)
    common = sorted({c for v in list(val_idx.values())[:8000] for c in v})[:40]

    train_caps = defaultdict(list)
    for a in caps_train["annotations"]:
        train_caps[a["image_id"]].append(a["caption"])
    val_caps = defaultdict(list)
    for a in caps_val["annotations"]:
        val_caps[a["image_id"]].append(a["caption"])

    # ---- allocation in the documented order ---------------------------------
    used_eval: set[int] = set()

    dev_caps = caption_selection(dict(val_caps), 96, seed=SEED + 1, k=1)
    used_eval.update(int(k) for k in dev_caps)

    val_ground_cands = grounding_candidates(val_idx, common, "val", seed=SEED, per_category=12)
    grounding_dev = select_grounding(val_ground_cands, 12, 12, used_eval)
    grounding_test = select_grounding(val_ground_cands, 32, 32, used_eval)

    def qmap(q):
        return {x["question_id"]: {"question": x["question"], "image_id": x["image_id"]}
                for x in q["questions"]}

    def amap(a):
        return {x["question_id"]: {"answers": x["answers"]} for x in a["annotations"]}

    val_q, val_a = qmap(vqa_val_q), amap(vqa_val_a)
    train_q, train_a = qmap(vqa_train_q), amap(vqa_train_a)
    val_img = {i["id"] for i in caps_val["images"]}
    train_img = {i["id"] for i in caps_train["images"]}

    vqa_val_pairs = build_candidate_pairs(val_q, val_a, allowed_images=val_img)
    vqa_dev = select_pairs(vqa_val_pairs, DEV_COUNTS, seed=SEED + 1,
                           excluded_images=used_eval)
    used_eval.update(image_set(vqa_dev))
    vqa_test = select_pairs(vqa_val_pairs, TEST_COUNTS, seed=SEED + 2,
                            excluded_images=used_eval)

    train_caption_images = caption_selection(dict(train_caps), 6000, seed=SEED, k=3)
    train_ground_cands = grounding_candidates(train_idx, common, "train", seed=SEED + 7,
                                              per_category=12)
    used_train: set[int] = {int(k) for k in train_caption_images}
    grounding_train = select_grounding(train_ground_cands, 64, 64, used_train)
    vqa_train = select_pairs(build_candidate_pairs(train_q, train_a, allowed_images=train_img),
                             TRAIN_COUNTS, seed=SEED, excluded_images=used_train)

    # ---- disjointness ---------------------------------------------------------
    dev_images = image_set(grounding_dev) | image_set(vqa_dev) | {int(k) for k in dev_caps}
    test_images = image_set(grounding_test) | image_set(vqa_test)
    train_images = image_set(grounding_train) | image_set(vqa_train) | {int(k) for k in train_caption_images}
    problems = []
    for name_a, a, name_b, b in (("dev", dev_images, "test", test_images),
                                 ("train", train_images, "dev", dev_images),
                                 ("train", train_images, "test", test_images)):
        overlap = a & b
        if overlap:
            problems.append(f"{name_a}/{name_b} share {len(overlap)}")
    if problems:
        raise SystemExit("IMAGE DISJOINTNESS FAILED: " + "; ".join(problems))

    result = {
        "captions_train": {str(k): v for k, v in train_caption_images.items()},
        "captions_dev": {str(k): v for k, v in dev_caps.items()},
        "grounding_train": [p.as_dict() for p in grounding_train],
        "grounding_dev": [p.as_dict() for p in grounding_dev],
        "grounding_test": [p.as_dict() for p in grounding_test],
        "vqa_train": [p.as_dict() for p in vqa_train],
        "vqa_dev": [p.as_dict() for p in vqa_dev],
        "vqa_test": [p.as_dict() for p in vqa_test],
    }
    (out / "data.json").write_text(json.dumps(result))

    needed = set(train_images) | dev_images | test_images
    summary = {
        "caption_train_images": len(train_caption_images),
        "caption_train_examples": sum(len(v) for v in train_caption_images.values()),
        "caption_dev_images": len(dev_caps),
        "grounding_train_pairs": len(grounding_train),
        "grounding_train_by_kind": {k: sum(1 for p in grounding_train if p.kind == k)
                                    for k in ("presence", "counting")},
        "grounding_dev_pairs": len(grounding_dev),
        "grounding_test_pairs": len(grounding_test),
        "vqa_train_pairs": len(vqa_train),
        "vqa_dev_pairs": len(vqa_dev),
        "vqa_test_pairs": len(vqa_test),
        "vqa_train_by_kind": {k: sum(1 for p in vqa_train if p.kind == k) for k in TRAIN_COUNTS},
        "vqa_dev_by_kind": {k: sum(1 for p in vqa_dev if p.kind == k) for k in DEV_COUNTS},
        "vqa_test_by_kind": {k: sum(1 for p in vqa_test if p.kind == k) for k in TEST_COUNTS},
        "images": {"train": len(train_images), "dev": len(dev_images),
                   "test": len(test_images), "needed_total": len(needed)},
        "disjointness": "verified",
    }
    (out / "build_summary.json").write_text(json.dumps(summary, indent=2))
    (out / "split_images.json").write_text(json.dumps(
        {"train": sorted(train_images), "dev": sorted(dev_images), "test": sorted(test_images)}))
    selection_ids = {
        "seed": SEED,
        "caption_image_ids": {
            "train": [int(i) for i in train_caption_images],
            "dev": [int(i) for i in dev_caps],
        },
        "grounding_pairs": {
            split: [{"image_a": int(p.image_a), "image_b": int(p.image_b),
                     "kind": p.kind, "category": p.category} for p in pairs]
            for split, pairs in (("train", grounding_train), ("dev", grounding_dev),
                                 ("test", grounding_test))
        },
        "vqa_pairs": {
            split: [{"image_a": int(p.image_a), "image_b": int(p.image_b),
                     "question_ids": list(p.question_ids), "kind": p.kind} for p in pairs]
            for split, pairs in (("train", vqa_train), ("dev", vqa_dev), ("test", vqa_test))
        },
    }
    (out / "selection_ids.json").write_text(json.dumps(selection_ids, indent=2) + "\n")
    (out / "needed_images.json").write_text(json.dumps(sorted(needed)))
    print(json.dumps(summary, indent=2))
    print("BUILD_V2_DONE")


if __name__ == "__main__":
    main()
