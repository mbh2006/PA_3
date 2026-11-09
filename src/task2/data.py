"""Task 2 data construction: COCO captions and VQAv2 complementary pairs.

Rules (PA p. 7):
  * captions: 6,000 training images x 3 captions, 96 development images x 1
  * natural VQAv2 pairs: same question, different consensus answers, **at least 8 of 10**
    annotators agreeing for **each** image; keep all ten answers for scoring
  * development and final-test images must be disjoint from each other and from training
  * training images may recur across stages; evaluation images never enter training

Selection is deterministic given the seed and the input ordering.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

N_ANNOTATORS = 10
MIN_AGREEMENT = 8

TRAIN_COUNTS = {"yes_no": 640, "number": 256, "other": 640}
DEV_COUNTS = {"yes_no": 12, "number": 8, "other": 12}
TEST_COUNTS = {"yes_no": 36, "number": 24, "other": 36}


def normalize_answer(text: str) -> str:
    s = str(text).lower().strip()
    s = re.sub(r"[^\w\s\.\-]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def consensus(answers: list[str]) -> tuple[str, int]:
    """Most common normalised answer and its annotator count."""
    if len(answers) != N_ANNOTATORS:
        raise ValueError(f"expected {N_ANNOTATORS} answers, got {len(answers)}")
    counts = Counter(normalize_answer(a) for a in answers)
    answer, n = counts.most_common(1)[0]
    return answer, n


def answer_kind(answer: str) -> str:
    a = normalize_answer(answer)
    if a in {"yes", "no"}:
        return "yes_no"
    if re.fullmatch(r"-?\d+(\.\d+)?", a):
        return "number"
    return "other"


@dataclass(frozen=True)
class VQAPair:
    question: str
    image_a: int
    answers_a: tuple[str, ...]
    consensus_a: str
    image_b: int
    answers_b: tuple[str, ...]
    consensus_b: str
    kind: str
    question_ids: tuple[int, int]

    def as_dict(self) -> dict:
        return {
            "question": self.question, "image_a": self.image_a, "image_b": self.image_b,
            "answers_a": list(self.answers_a), "answers_b": list(self.answers_b),
            "consensus_a": self.consensus_a, "consensus_b": self.consensus_b,
            "kind": self.kind, "question_ids": list(self.question_ids),
        }


def group_by_question(questions: dict[int, dict]) -> dict[str, list[int]]:
    """Question text -> question ids (VQAv2 supplies several images per question text)."""
    groups: dict[str, list[int]] = defaultdict(list)
    for qid, q in questions.items():
        groups[q["question"].strip().lower()].append(qid)
    for v in groups.values():
        v.sort()
    return groups


def build_candidate_pairs(
    questions: dict[int, dict],
    annotations: dict[int, dict],
    *,
    allowed_images: set[int] | None = None,
) -> list[VQAPair]:
    """All valid complementary pairs, without splitting yet."""
    groups = group_by_question(questions)
    pairs: list[VQAPair] = []

    for text, qids in groups.items():
        if len(qids) < 2:
            continue
        info = []
        for qid in qids:
            q = questions[qid]
            ann = annotations.get(qid)
            if ann is None:
                continue
            answers = [a["answer"] for a in ann["answers"]]
            if len(answers) != N_ANNOTATORS:
                continue
            cons, n = consensus(answers)
            if n < MIN_AGREEMENT:
                continue
            if allowed_images is not None and q["image_id"] not in allowed_images:
                continue
            info.append((qid, q["image_id"], tuple(answers), cons))

        for i in range(len(info)):
            for j in range(i + 1, len(info)):
                qid_a, img_a, ans_a, cons_a = info[i]
                qid_b, img_b, ans_b, cons_b = info[j]
                if img_a == img_b or cons_a == cons_b:
                    continue
                kind = answer_kind(cons_a)
                if kind != answer_kind(cons_b):
                    continue
                pairs.append(VQAPair(
                    question=questions[qid_a]["question"].strip(),
                    image_a=img_a, answers_a=ans_a, consensus_a=cons_a,
                    image_b=img_b, answers_b=ans_b, consensus_b=cons_b,
                    kind=kind, question_ids=(qid_a, qid_b),
                ))
    return pairs


def select_pairs(
    candidates: list[VQAPair],
    counts: dict[str, int],
    *,
    seed: int,
    excluded_images: set[int] | None = None,
) -> list[VQAPair]:
    """Deterministically pick pairs per answer kind while keeping images unique."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(candidates))
    excluded = set(excluded_images or ())
    used: set[int] = set()
    chosen: list[VQAPair] = []
    taken = {k: 0 for k in counts}

    for idx in order:
        pair = candidates[int(idx)]
        if taken[pair.kind] >= counts[pair.kind]:
            continue
        if pair.image_a in excluded or pair.image_b in excluded:
            continue
        if pair.image_a in used or pair.image_b in used:
            continue
        chosen.append(pair)
        used.update({pair.image_a, pair.image_b})
        taken[pair.kind] += 1
        if all(taken[k] >= counts[k] for k in counts):
            break
    return chosen


def split_vqa(
    candidates: list[VQAPair],
    *,
    seed: int = 6304,
    train_counts: dict[str, int] = TRAIN_COUNTS,
    dev_counts: dict[str, int] = DEV_COUNTS,
    test_counts: dict[str, int] = TEST_COUNTS,
) -> dict[str, list[VQAPair]]:
    """Train / dev / test with image-disjoint evaluation sets.

    ``candidates`` must already be restricted to the images allowed for that split
    (train2014 for training, val2014 for dev/test). Dev is drawn first, then test is
    drawn excluding every dev image, then training excludes every evaluation image.
    """
    dev = select_pairs(candidates, dev_counts, seed=seed + 1)
    dev_images = {p.image_a for p in dev} | {p.image_b for p in dev}

    test = select_pairs(candidates, test_counts, seed=seed + 2, excluded_images=dev_images)
    test_images = {p.image_a for p in test} | {p.image_b for p in test}

    train = select_pairs(candidates, train_counts, seed=seed,
                         excluded_images=dev_images | test_images)
    return {"train": train, "dev": dev, "test": test}


def assert_image_disjoint(splits: dict[str, list[VQAPair]]) -> None:
    def images(pairs):
        s: set[int] = set()
        for p in pairs:
            s.update({p.image_a, p.image_b})
        return s

    dev, test, train = images(splits["dev"]), images(splits["test"]), images(splits["train"])
    if dev & test:
        raise AssertionError(f"dev/test share {len(dev & test)} images")
    if dev & train:
        raise AssertionError(f"dev/train share {len(dev & train)} images")
    if test & train:
        raise AssertionError(f"test/train share {len(test & train)} images")


def caption_selection(captions_by_image: dict[int, list[str]], n_images: int,
                      *, seed: int, k: int | None = 3,
                      excluded: set[int] | None = None) -> dict[int, list[str]]:
    """Pick ``n_images`` deterministically; return ``k`` captions per image."""
    excluded = set(excluded or ())
    ids = sorted(i for i in captions_by_image if i not in excluded
                 and len(captions_by_image[i]) >= (k or 1))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    out: dict[int, list[str]] = {}
    for idx in order:
        image_id = ids[int(idx)]
        caps = captions_by_image[image_id]
        out[image_id] = caps[:k] if k else caps[:1]
        if len(out) >= n_images:
            break
    return out
