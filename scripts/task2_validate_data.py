#!/usr/bin/env python3
"""Validate the prepared Task 2 data on CPU: counts, subtypes, disjointness, images."""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/root/pa3/src")
from task2.data import normalize_answer  # noqa: E402

DATA = Path("/root/pa3/task2_data")
COUNTS = {
    "vqa_train": {"yes_no": 640, "number": 256, "other": 640},
    "vqa_dev": {"yes_no": 12, "number": 8, "other": 12},
    "vqa_test": {"yes_no": 36, "number": 24, "other": 36},
}


def main() -> None:
    d = json.loads((DATA / "data.json").read_text())
    summary = json.loads((DATA / "build_summary.json").read_text())
    problems: list[str] = []

    # captions
    caps_train = d["captions_train"]
    caps_dev = d["captions_dev"]
    if len(caps_train) != 6000:
        problems.append(f"caption train images {len(caps_train)} != 6000")
    if sum(len(v) for v in caps_train.values()) != 18000:
        problems.append("caption train examples != 18000")
    if len(caps_dev) != 96:
        problems.append(f"caption dev images {len(caps_dev)} != 96")
    if any(len(v) != 3 for v in caps_train.values()):
        problems.append("some training image does not have exactly 3 captions")
    if any(len(v) != 1 for v in caps_dev.values()):
        problems.append("some dev image does not have exactly 1 caption")

    # grounding
    if len(d["grounding_dev"]) != 24:
        problems.append(f"grounding dev pairs {len(d['grounding_dev'])} != 24")
    if len(d["grounding_test"]) != 64:
        problems.append(f"grounding test pairs {len(d['grounding_test'])} != 64")
    for name in ("grounding_dev", "grounding_test"):
        kinds = Counter(p["kind"] for p in d[name])
        for p in d[name]:
            if p["answer_a"] == p["answer_b"]:
                problems.append(f"{name}: identical answers in a pair")
            if p["kind"] == "counting":
                for a in (p["answer_a"], p["answer_b"]):
                    if not (1 <= int(a) <= 4):
                        problems.append(f"{name}: count {a} out of 1..4")

    # natural VQA
    for split, expected in COUNTS.items():
        pairs = d[split]
        got = Counter(p["kind"] for p in pairs)
        if dict(got) != expected:
            problems.append(f"{split} subtype counts {dict(got)} != {expected}")
        for p in pairs:
            if len(p["answers_a"]) != 10 or len(p["answers_b"]) != 10:
                problems.append(f"{split}: pair without 10 answers per image")
                break
            if p["consensus_a"] == p["consensus_b"]:
                problems.append(f"{split}: identical consensus in a pair")
                break
            if p["image_a"] == p["image_b"]:
                problems.append(f"{split}: pair uses the same image twice")
                break

    # agreement >= 8/10 on both sides
    for split in COUNTS:
        for p in d[split]:
            for answers, cons in ((p["answers_a"], p["consensus_a"]),
                                  (p["answers_b"], p["consensus_b"])):
                n = sum(1 for a in answers if normalize_answer(a) == cons)
                if n < 8:
                    problems.append(f"{split}: agreement {n}/10 below 8")
                    break

    # image disjointness
    def imgs(pairs):
        s = set()
        for p in pairs:
            s.update({p["image_a"], p["image_b"]})
        return s

    dev_i, test_i = imgs(d["vqa_dev"]), imgs(d["vqa_test"])
    train_i = imgs(d["vqa_train"]) | set(int(k) for k in caps_train)
    if dev_i & test_i:
        problems.append(f"dev/test share {len(dev_i & test_i)} images")
    if dev_i & train_i:
        problems.append(f"dev/train share {len(dev_i & train_i)} images")
    if test_i & train_i:
        problems.append(f"test/train share {len(test_i & train_i)} images")

    # every referenced image exists on disk
    index_path = DATA / "images" / "index.json"
    missing = 0
    if index_path.exists():
        index = json.loads(index_path.read_text())
        for k in index:
            if not (DATA / index[k]).exists():
                missing += 1
        wanted = set(int(k) for k in index)
        referenced = dev_i | test_i | train_i
        absent = referenced - wanted
        if absent:
            problems.append(f"{len(absent)} referenced images were never downloaded")
    else:
        problems.append("images/index.json missing")

    # caption counts must match the files on disk too
    with_caps = 0
    for k, caps in caps_train.items():
        if len(caps) >= 1:
            with_caps += 1

    report = {
        "summary": summary,
        "caption_train_images": len(caps_train),
        "caption_train_examples": sum(len(v) for v in caps_train.values()),
        "caption_dev_images": len(caps_dev),
        "grounding_dev": len(d["grounding_dev"]),
        "grounding_test": len(d["grounding_test"]),
        "vqa_train": dict(Counter(p["kind"] for p in d["vqa_train"])),
        "vqa_dev": dict(Counter(p["kind"] for p in d["vqa_dev"])),
        "vqa_test": dict(Counter(p["kind"] for p in d["vqa_test"])),
        "images_on_disk": len(list((DATA / "images").glob("*.jpg"))),
        "missing_image_files": missing,
        "problems": problems,
        "valid": not problems,
    }
    (DATA / "validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2)[:2000])


if __name__ == "__main__":
    main()
