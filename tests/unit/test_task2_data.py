"""Task 2 data rules: agreement filter, pair construction, image-disjoint splits."""

import numpy as np
import pytest

from task2.data import (
    DEV_COUNTS,
    MIN_AGREEMENT,
    TEST_COUNTS,
    TRAIN_COUNTS,
    VQAPair,
    answer_kind,
    assert_image_disjoint,
    build_candidate_pairs,
    caption_selection,
    consensus,
    normalize_answer,
    select_pairs,
    split_vqa,
)


def _ann(answer, n_agree=10, filler="zzz"):
    return {"answers": [{"answer": answer}] * n_agree + [{"answer": filler}] * (10 - n_agree)}


def _q(qid, image_id, text="What is this?"):
    return qid, {"question": text, "image_id": image_id}


def test_normalisation_and_kind():
    assert normalize_answer("  Yes ") == "yes"
    assert answer_kind("yes") == "yes_no"
    assert answer_kind("No") == "yes_no"
    assert answer_kind("12") == "number"
    assert answer_kind("3.5") == "number"
    assert answer_kind("red") == "other"


def test_consensus_counts_and_requires_ten():
    assert consensus(["cat"] * 8 + ["dog"] * 2) == ("cat", 8)
    with pytest.raises(ValueError):
        consensus(["cat"] * 9)


def test_pairs_require_eight_of_ten_on_both_images():
    questions = dict([_q(1, 100), _q(2, 200), _q(3, 300)])
    annotations = {1: _ann("yes", 8), 2: _ann("no", 8), 3: _ann("maybe", 7)}
    pairs = build_candidate_pairs(questions, annotations)
    # 1 vs 2 valid; 1 vs 3 and 2 vs 3 rejected (7/10 agreement)
    assert len(pairs) == 1
    assert {pairs[0].image_a, pairs[0].image_b} == {100, 200}
    assert MIN_AGREEMENT == 8


def test_pairs_require_different_answers():
    questions = dict([_q(1, 100), _q(2, 200)])
    annotations = {1: _ann("yes"), 2: _ann("yes")}
    assert build_candidate_pairs(questions, annotations) == []


def test_pairs_require_same_question_text():
    questions = dict([_q(1, 100, "What color?"), _q(2, 200, "How many?")])
    annotations = {1: _ann("yes"), 2: _ann("no")}
    assert build_candidate_pairs(questions, annotations) == []


def test_pairs_keep_all_ten_answers():
    questions = dict([_q(1, 100), _q(2, 200)])
    annotations = {1: _ann("yes", 9, filler="no"), 2: _ann("no", 8, filler="yes")}
    pairs = build_candidate_pairs(questions, annotations)
    assert len(pairs) == 1
    p = pairs[0]
    assert len(p.answers_a) == 10 and len(p.answers_b) == 10
    assert p.consensus_a == "yes" and p.consensus_b == "no"


def test_allowed_images_filter():
    questions = dict([_q(1, 100), _q(2, 200)])
    annotations = {1: _ann("yes"), 2: _ann("no")}
    assert build_candidate_pairs(questions, annotations, allowed_images={100}) == []
    assert len(build_candidate_pairs(questions, annotations, allowed_images={100, 200})) == 1


def _synthetic_pairs(n_per_kind=200, start=1000):
    kinds = ["yes_no", "number", "other"]
    pairs = []
    img = start
    for k in kinds:
        for i in range(n_per_kind):
            pairs.append(VQAPair(
                question=f"q{i}", image_a=img, answers_a=("a",) * 10, consensus_a="a",
                image_b=img + 1, answers_b=("b",) * 10, consensus_b="b",
                kind=k, question_ids=(i, i + 1),
            ))
            img += 2
    return pairs


def test_select_pairs_respects_counts_and_unique_images():
    pairs = _synthetic_pairs()
    chosen = select_pairs(pairs, {"yes_no": 5, "number": 3, "other": 4}, seed=6304)
    assert len(chosen) == 12
    counts = {}
    for p in chosen:
        counts[p.kind] = counts.get(p.kind, 0) + 1
    assert counts == {"yes_no": 5, "number": 3, "other": 4}
    imgs = [p.image_a for p in chosen] + [p.image_b for p in chosen]
    assert len(imgs) == len(set(imgs))


def test_select_pairs_is_deterministic():
    pairs = _synthetic_pairs()
    a = select_pairs(pairs, TRAIN_COUNTS, seed=6304)
    b = select_pairs(pairs, TRAIN_COUNTS, seed=6304)
    assert [p.image_a for p in a] == [p.image_a for p in b]


def test_split_is_image_disjoint_and_covers_counts():
    pairs = _synthetic_pairs(400)
    splits = split_vqa(pairs, seed=6304,
                       train_counts={"yes_no": 20, "number": 10, "other": 20},
                       dev_counts={"yes_no": 3, "number": 2, "other": 3},
                       test_counts={"yes_no": 6, "number": 4, "other": 6})
    assert len(splits["train"]) == 50
    assert len(splits["dev"]) == 8
    assert len(splits["test"]) == 16
    assert_image_disjoint(splits)


def test_disjointness_check_catches_leaks():
    p = _synthetic_pairs(2)[0]
    with pytest.raises(AssertionError):
        assert_image_disjoint({"train": [p], "dev": [p], "test": []})


def test_caption_selection_is_deterministic_and_gives_k_captions():
    captions = {i: [f"cap{i}-{j}" for j in range(5)] for i in range(50)}
    a = caption_selection(captions, 10, seed=6304, k=3)
    b = caption_selection(captions, 10, seed=6304, k=3)
    assert list(a) == list(b)
    assert len(a) == 10
    assert all(len(v) == 3 for v in a.values())


def test_caption_selection_respects_exclusions():
    captions = {i: ["c"] * 3 for i in range(20)}
    out = caption_selection(captions, 20, seed=1, k=3, excluded=set(range(10)))
    assert set(out).isdisjoint(set(range(10)))


def test_dev_test_counts_match_the_assignment():
    assert TRAIN_COUNTS == {"yes_no": 640, "number": 256, "other": 640}
    assert DEV_COUNTS == {"yes_no": 12, "number": 8, "other": 12}
    assert TEST_COUNTS == {"yes_no": 36, "number": 24, "other": 36}
