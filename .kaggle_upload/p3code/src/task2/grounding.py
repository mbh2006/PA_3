"""Annotation-derived grounding questions and natural-VQA answer scoring.

Templates are fixed by the assignment (PA p. 7):

    presence: "Does this image contain [object]?"
    counting: "How many [objects] can you see?"

Pairs are formed across images: yes/no for presence, different counts in 1..4 for
counting. Labels are generated programmatically from COCO instance annotations.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

VQA_N_ANNOTATORS = 10

PRESENCE_TEMPLATE = "Does this image contain {object}?"
COUNTING_TEMPLATE = "How many {objects} can you see?"


def presence_question(category: str) -> str:
    return PRESENCE_TEMPLATE.format(object=category.replace("_", " "))


def counting_question(category: str) -> str:
    plural = category.replace("_", " ")
    if not plural.endswith("s"):
        plural += "s"
    return COUNTING_TEMPLATE.format(objects=plural)


@dataclass(frozen=True)
class GroundingPair:
    category: str
    question: str
    kind: str                 # "presence" | "counting"
    image_a: int
    answer_a: str
    image_b: int
    answer_b: str
    split: str

    def as_dict(self) -> dict:
        return {
            "category": self.category, "question": self.question, "kind": self.kind,
            "image_a": self.image_a, "answer_a": self.answer_a,
            "image_b": self.image_b, "answer_b": self.answer_b, "split": self.split,
        }


def build_presence_pairs(annotations: dict[int, list[str]], categories: list[str],
                         split: str, *, max_pairs_per_category: int | None = None) -> list[GroundingPair]:
    """Pair an image containing the category with one that has no annotation for it."""
    has: dict[str, list[int]] = defaultdict(list)
    lacks: dict[str, list[int]] = defaultdict(list)
    for image_id, cats in annotations.items():
        cat_set = set(cats)
        for c in categories:
            (has if c in cat_set else lacks)[c].append(image_id)

    pairs: list[GroundingPair] = []
    for c in categories:
        a_list = sorted(has[c])
        b_list = sorted(lacks[c])
        n = min(len(a_list), len(b_list))
        if max_pairs_per_category is not None:
            n = min(n, max_pairs_per_category)
        for i in range(n):
            pairs.append(GroundingPair(
                category=c, question=presence_question(c), kind="presence",
                image_a=a_list[i], answer_a="yes", image_b=b_list[i], answer_b="no",
                split=split,
            ))
    return pairs


def build_counting_pairs(counts: dict[int, dict[str, int]], categories: list[str],
                         split: str, *, max_pairs_per_category: int | None = None) -> list[GroundingPair]:
    """Pair two images whose annotated counts differ and both lie in 1..4."""
    by_count: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for image_id, cats in counts.items():
        for c, k in cats.items():
            if 1 <= k <= 4:
                by_count[c][k].append(image_id)

    pairs: list[GroundingPair] = []
    for c in categories:
        buckets = by_count.get(c, {})
        keys = sorted(buckets)
        made = 0
        for i, ka in enumerate(keys):
            for kb in keys[i + 1:]:
                for ia, ib in zip(sorted(buckets[ka]), sorted(buckets[kb])):
                    if ka == kb:
                        continue
                    pairs.append(GroundingPair(
                        category=c, question=counting_question(c), kind="counting",
                        image_a=ia, answer_a=str(ka), image_b=ib, answer_b=str(kb),
                        split=split,
                    ))
                    made += 1
                    if max_pairs_per_category is not None and made >= max_pairs_per_category:
                        break
                if max_pairs_per_category is not None and made >= max_pairs_per_category:
                    break
            if max_pairs_per_category is not None and made >= max_pairs_per_category:
                break
    return pairs


def normalized_answer(text: str | None) -> str:
    if text is None:
        return ""
    return " ".join(str(text).lower().strip().split())


def vqa_soft_consensus(prediction: str | None, references: list[str]) -> float:
    """Official VQA accuracy: s(â) = mean_i min(1, (1/3) Σ_{j≠i} 1[â = a_j]).

    Requires exactly ten references; empty predictions score zero.
    """
    if len(references) != VQA_N_ANNOTATORS:
        raise ValueError(f"expected {VQA_N_ANNOTATORS} references, got {len(references)}")
    pred = normalized_answer(prediction)
    if pred == "":
        return 0.0
    refs = [normalized_answer(r) for r in references]
    scores = []
    for i in range(len(refs)):
        matches = sum(1 for j, r in enumerate(refs) if j != i and r == pred)
        scores.append(min(1.0, matches / 3.0))
    return sum(scores) / len(scores)


def pair_accuracy(pred_a: str | None, pred_b: str | None, gold_a: str, gold_b: str) -> float:
    """Both predictions must match their own original canonical answers."""
    return float(normalized_answer(pred_a) == normalized_answer(gold_a)
                 and normalized_answer(pred_b) == normalized_answer(gold_b))
