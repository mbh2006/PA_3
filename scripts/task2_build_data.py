"""Build the Task 2 data: COCO captions, annotation-derived grounding, VQAv2 pairs.

Downloads only the annotation files plus the images that are actually selected (~6,300
train2014 + a few hundred val2014), not the 19 GB COCO archive.

Outputs (all under /root/pa3/task2_data):
    captions_train.json / captions_dev.json
    grounding_dev.json / grounding_test.json      (pairs, no images)
    vqa_train.json / vqa_dev.json / vqa_test.json (pairs with all 10 answers)
    images/                                       (JPEGs named by COCO id)
    images/index.json                             (id -> path, split)
"""

from __future__ import annotations

import argparse
import io
import json
import tempfile
import os
import random
import sys
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

COCO_BASE = "https://s3.amazonaws.com/images.cocodataset.org"
COCO_ANN = f"{COCO_BASE}/annotations/annotations_trainval2014.zip"
VQA_TRAIN_Q = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip"
VQA_TRAIN_A = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip"
VQA_VAL_Q = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip"
VQA_VAL_A = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip"
IMG_TRAIN = COCO_BASE + "/train2014/COCO_train2014_{:012d}.jpg"
IMG_VAL = COCO_BASE + "/val2014/COCO_val2014_{:012d}.jpg"

SEED = 6304


def download(url: str, dest: Path) -> Path:
    if dest.is_file() and zipfile.is_zipfile(dest):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}", flush=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        candidate = Path(tmp) / dest.name
        with urllib.request.urlopen(url, timeout=120) as r, candidate.open("wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        if not zipfile.is_zipfile(candidate):
            raise ValueError(f"Invalid annotation archive: {dest.name}")
        candidate.replace(dest)
    return dest


def read_zip_json(path: Path, member_suffix: str):
    """Read one JSON member; falls back to the largest .json if the name differs."""
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        matches = [n for n in names if n.endswith(member_suffix)]
        if not matches:
            jsons = [n for n in names if n.endswith(".json")]
            if not jsons:
                raise FileNotFoundError(f"{path.name}: no json member among {names}")
            matches = [max(jsons, key=lambda n: z.getinfo(n).file_size)]
        with z.open(matches[0]) as fh:
            return json.load(io.TextIOWrapper(fh, encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/root/pa3/task2_data"))
    ap.add_argument("--images", action="store_true", help="download the selected JPEGs")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    ann_dir = out / "annotations"
    ann_dir.mkdir(exist_ok=True)

    print("=== annotations ===", flush=True)
    coco_zip = download(COCO_ANN, ann_dir / "annotations_trainval2014.zip")
    captions_train = read_zip_json(coco_zip, "captions_train2014.json")
    captions_val = read_zip_json(coco_zip, "captions_val2014.json")
    inst_train = read_zip_json(coco_zip, "instances_train2014.json")
    inst_val = read_zip_json(coco_zip, "instances_val2014.json")

    vqa_train_q = read_zip_json(download(VQA_TRAIN_Q, ann_dir / "v2_Questions_Train.zip"),
                                "v2_OpenEnded_mscoco_train2014_questions.json")
    vqa_train_a = read_zip_json(download(VQA_TRAIN_A, ann_dir / "v2_Annotations_Train.zip"),
                                "v2_mscoco_train2014_annotations.json")
    vqa_val_q = read_zip_json(download(VQA_VAL_Q, ann_dir / "v2_Questions_Val.zip"),
                              "v2_OpenEnded_mscoco_val2014_questions.json")
    vqa_val_a = read_zip_json(download(VQA_VAL_A, ann_dir / "v2_Annotations_Val.zip"),
                              "v2_mscoco_val2014_annotations.json")

    # ---------------- captions --------------------------------------------------
    from task2.data import assert_image_disjoint, caption_selection, build_candidate_pairs, split_vqa

    train_caps = defaultdict(list)
    for a in captions_train["annotations"]:
        train_caps[a["image_id"]].append(a["caption"])
    val_caps = defaultdict(list)
    for a in captions_val["annotations"]:
        val_caps[a["image_id"]].append(a["caption"])

    train_caption_images = caption_selection(dict(train_caps), 6000, seed=SEED, k=3)
    dev_caption_images = caption_selection(dict(val_caps), 96, seed=SEED + 1, k=1)
    # (dev VQA pairs are selected later; the grounding-test filter below runs after both)

    # ---------------- natural VQAv2 --------------------------------------------
    def qmap(q):
        return {x["question_id"]: {"question": x["question"], "image_id": x["image_id"]}
                for x in q["questions"]}

    def amap(a):
        return {x["question_id"]: {"answers": x["answers"]} for x in a["annotations"]}

    train_q, train_a = qmap(vqa_train_q), amap(vqa_train_a)
    val_q, val_a = qmap(vqa_val_q), amap(vqa_val_a)

    train_img = {i["id"] for i in captions_train["images"]}
    val_img = {i["id"] for i in captions_val["images"]}

    dev_pairs = build_candidate_pairs(val_q, val_a, allowed_images=val_img)
    test_pairs = build_candidate_pairs(val_q, val_a, allowed_images=val_img)
    train_pairs = build_candidate_pairs(train_q, train_a, allowed_images=train_img)

    from task2.data import DEV_COUNTS, TEST_COUNTS, TRAIN_COUNTS, select_pairs

    dev = select_pairs(dev_pairs, DEV_COUNTS, seed=SEED + 1)
    dev_imgs = {p.image_a for p in dev} | {p.image_b for p in dev}
    test = select_pairs(test_pairs, TEST_COUNTS, seed=SEED + 2, excluded_images=dev_imgs)
    test_imgs = {p.image_a for p in test} | {p.image_b for p in test}
    train = select_pairs(train_pairs, TRAIN_COUNTS, seed=SEED,
                         excluded_images=dev_imgs | test_imgs)

    # ---------------- grounding -------------------------------------------------
    from task2.grounding import build_counting_pairs, build_presence_pairs

    def coco_index(inst):
        cats = {c["id"]: c["name"] for c in inst["categories"]}
        per_image = defaultdict(dict)
        for a in inst["annotations"]:
            if a.get("iscrowd"):
                continue
            per_image[a["image_id"]][cats[a["category_id"]]] = \
                per_image[a["image_id"]].get(cats[a["category_id"]], 0) + 1
        return per_image

    inst_train_idx = coco_index(inst_train)
    inst_val_idx = coco_index(inst_val)
    common_cats = sorted({c for v in list(inst_val_idx.values())[:5000] for c in v})[:40]

    pres_dev = build_presence_pairs(inst_val_idx, common_cats, "dev", max_pairs_per_category=4)
    cnt_dev = build_counting_pairs(inst_val_idx, common_cats, "dev", max_pairs_per_category=4)
    pres_test = build_presence_pairs(inst_val_idx, common_cats, "test", max_pairs_per_category=4)
    cnt_test = build_counting_pairs(inst_val_idx, common_cats, "test", max_pairs_per_category=4)

    # TRAIN pools come from train2014 only; val2014 is reserved for dev/test.
    pres_train = build_presence_pairs(inst_train_idx, common_cats, "train",
                                      max_pairs_per_category=6)
    cnt_train = build_counting_pairs(inst_train_idx, common_cats, "train",
                                     max_pairs_per_category=6)
    # dev must also avoid the VQA test images, which were selected above
    vqa_test_images = set()
    for p_ in test:
        vqa_test_images.update({int(p_.image_a), int(p_.image_b)})

    def keep_dev(p_):
        return (int(p_.image_a) not in vqa_test_images
                and int(p_.image_b) not in vqa_test_images)

    pres_dev = [p_ for p_ in pres_dev if keep_dev(p_)]
    cnt_dev = [p_ for p_ in cnt_dev if keep_dev(p_)]
    if len(pres_dev) < 12 or len(cnt_dev) < 12:
        raise SystemExit(f"not enough grounding dev pairs after filtering: "
                         f"presence={len(pres_dev)} counting={len(cnt_dev)}")

    # half of each kind, so truncation cannot drop counting entirely
    grounding_dev = pres_dev[:12] + cnt_dev[:12]

    # dev and final-test must be image-disjoint; val2014 pools are drawn independently
    # so the test pairs are filtered against every dev image before truncation
    dev_images_all = set(dev_caption_images)
    for p_ in dev:
        dev_images_all.update({int(p_.image_a), int(p_.image_b)})
    for p_ in grounding_dev:
        dev_images_all.update({int(p_.image_a), int(p_.image_b)})

    def keep(p_):
        return (int(p_.image_a) not in dev_images_all
                and int(p_.image_b) not in dev_images_all)

    pres_test = [p_ for p_ in pres_test if keep(p_)]
    cnt_test = [p_ for p_ in cnt_test if keep(p_)]
    grounding_test = pres_test[:32] + cnt_test[:32]
    if len(grounding_test) != 64:
        raise SystemExit(f"grounding test has {len(grounding_test)} pairs after "
                         f"disjointness filtering (need 64)")

    result = {
        "captions_train": {str(k): v for k, v in train_caption_images.items()},
        "captions_dev": {str(k): v for k, v in dev_caption_images.items()},
        "grounding_train": [p.as_dict() for p in (pres_train + cnt_train)],
        "grounding_dev": [p.as_dict() for p in grounding_dev],
        "grounding_test": [p.as_dict() for p in grounding_test],
        "vqa_train": [p.as_dict() for p in train],
        "vqa_dev": [p.as_dict() for p in dev],
        "vqa_test": [p.as_dict() for p in test],
    }
    (out / "data.json").write_text(json.dumps(result))

    summary = {
        "caption_train_images": len(train_caption_images),
        "caption_train_examples": sum(len(v) for v in train_caption_images.values()),
        "caption_dev_images": len(dev_caption_images),
        "grounding_train_pairs": len(pres_train) + len(cnt_train),
        "grounding_train_by_kind": {"presence": len(pres_train), "counting": len(cnt_train)},
        "grounding_dev_pairs": len(grounding_dev),
        "grounding_test_pairs": len(grounding_test),
        "vqa_train_pairs": len(train),
        "vqa_dev_pairs": len(dev),
        "vqa_test_pairs": len(test),
        "vqa_train_by_kind": {k: sum(1 for p in train if p.kind == k) for k in TRAIN_COUNTS},
        "vqa_dev_by_kind": {k: sum(1 for p in dev if p.kind == k) for k in DEV_COUNTS},
        "vqa_test_by_kind": {k: sum(1 for p in test if p.kind == k) for k in TEST_COUNTS},
        "candidates": {"train": len(train_pairs), "dev": len(dev_pairs), "test": len(test_pairs)},
    }
    def image_set(pairs):
        s = set()
        for p in pairs:
            s.update({int(p.image_a), int(p.image_b)})
        return s

    train_pool_images = image_set(pres_train + cnt_train)
    dev_images = image_set(grounding_dev) | image_set([p for p in dev])
    test_images = image_set(grounding_test) | image_set([p for p in test])
    caption_train_images = {int(k) for k in train_caption_images}
    splits = {
        "train": sorted(train_pool_images | caption_train_images
                        | image_set([p for p in train])),
        "dev": sorted(dev_images | {int(k) for k in dev_caption_images}),
        "test": sorted(test_images),
    }
    problems = []
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlap = set(splits[a]) & set(splits[b])
        if overlap:
            problems.append(f"{a}/{b} share {len(overlap)} images")
    summary["image_disjointness"] = {"problems": problems,
                                     "counts": {k: len(v) for k, v in splits.items()}}
    (out / "split_images.json").write_text(json.dumps(splits))
    (out / "build_summary.json").write_text(json.dumps(summary, indent=2))
    if problems:
        raise SystemExit("IMAGE DISJOINTNESS FAILED: " + "; ".join(problems))
    print(json.dumps(summary, indent=2), flush=True)

    if args.images:
        wanted = {}
        for k, caps in train_caption_images.items():
            wanted[k] = "train"
        for k in dev_caption_images:
            wanted[k] = "val"
        for p in grounding_dev + grounding_test:
            wanted.setdefault(p.image_a, "val")
            wanted.setdefault(p.image_b, "val")
        for p in train:
            wanted.setdefault(p.image_a, "train")
            wanted.setdefault(p.image_b, "train")
        for p in dev + test:
            wanted.setdefault(p.image_a, "val")
            wanted.setdefault(p.image_b, "val")

        img_dir = out / "images"
        img_dir.mkdir(exist_ok=True)
        print(f"=== downloading {len(wanted)} images ===", flush=True)

        def fetch(item):
            image_id, split = item
            dest = img_dir / f"{image_id:012d}.jpg"
            if dest.exists() and dest.stat().st_size > 0:
                return 0
            url = (IMG_TRAIN if split == "train" else IMG_VAL).format(image_id)
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    dest.write_bytes(r.read())
                return 1
            except Exception as exc:
                print(f"  image {image_id} failed: {exc}", flush=True)
                return 0

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            done = sum(pool.map(fetch, wanted.items()))
        (out / "images" / "index.json").write_text(json.dumps(
            {str(k): f"images/{k:012d}.jpg" for k in wanted}))
        print(f"  downloaded {done} new images", flush=True)

    print("BUILD_DONE", flush=True)


if __name__ == "__main__":
    main()
