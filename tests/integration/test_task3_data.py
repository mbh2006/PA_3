"""Integration checks against the supplied Task 3 export.

The expected totals come from the instructor README, so these assertions are an
independent check that the local files are the official ones and that the loader
aligns them correctly (not a check of the loader against itself).
"""

import numpy as np
import pytest

from task3.answers import NO_ANSWER
from task3.ingest import load_dataset

SUPPLIED = {
    "math": {
        "questions": 1000,
        "rollouts": 32000,
        "retained_tokens": 20742094,
        "capped_rollouts": 523,
        "null_extractions": 521,
    },
    "mathvista": {
        "questions": 1000,
        "rollouts": 32000,
        "retained_tokens": 8961703,
        "capped_rollouts": 487,
        "null_extractions": 3032,
    },
}


@pytest.fixture(scope="module")
def math_bundle():
    return load_dataset("math")


@pytest.fixture(scope="module")
def vlm_bundle():
    return load_dataset("mathvista")


@pytest.mark.parametrize("name", ["math", "mathvista"])
def test_supplied_counts(name, request):
    bundle = request.getfixturevalue("math_bundle" if name == "math" else "vlm_bundle")
    exp = SUPPLIED[name]
    assert bundle.n_questions == exp["questions"]
    assert sum(len(q.rollout_ids) for q in bundle.questions) == exp["rollouts"]
    assert sum(int(q.n_tokens.sum()) for q in bundle.questions) == exp["retained_tokens"]
    assert sum(int(q.hit_cap.sum()) for q in bundle.questions) == exp["capped_rollouts"]
    assert sum(int(q.is_null.sum()) for q in bundle.questions) == exp["null_extractions"]


@pytest.mark.parametrize("name", ["math", "mathvista"])
def test_per_question_structure(name, request):
    bundle = request.getfixturevalue("math_bundle" if name == "math" else "vlm_bundle")
    for q in bundle.questions[:50]:
        assert q.rollout_ids.tolist() == list(range(32))
        assert len(q.keys) == len(q.n_tokens) == len(q.correct) == 32
        assert np.all(np.isfinite(q.logp_mean))
        assert np.all(np.isfinite(q.kl_mean))
        assert np.all(q.n_tokens > 0)
        if bundle.is_vlm:
            assert np.all((q.vis >= 0) & (q.vis <= 1))
            assert all(len(v) == int(n) for v, n in zip(q.v_tokens, q.n_tokens))
        else:
            assert all(len(k) == int(n) for k, n in zip(q.kl_tokens, q.n_tokens))


def test_no_answer_group_matches_null_extractions(math_bundle):
    n_no_answer = sum(k == NO_ANSWER for q in math_bundle.questions for k in q.keys)
    n_null = sum(int(q.is_null.sum()) for q in math_bundle.questions)
    assert n_no_answer == n_null
    # NO_ANSWER is never correct
    for q in math_bundle.questions:
        for k, c in zip(q.keys, q.correct):
            if k == NO_ANSWER:
                assert not c


def test_vlm_no_answer_never_correct(vlm_bundle):
    for q in vlm_bundle.questions:
        for k, c in zip(q.keys, q.correct):
            if k == NO_ANSWER:
                assert not c


def test_math_gold_nontrivial_correctness(math_bundle):
    acc = np.mean([q.correct.mean() for q in math_bundle.questions])
    # sanity band: a 3B model on MATH should be well above 5% and below 95%
    assert 0.05 < acc < 0.95


def test_vlm_correctness_and_floor(vlm_bundle):
    acc = np.mean([q.correct.mean() for q in vlm_bundle.questions])
    assert 0.05 < acc < 0.95
    # 9.5% of rollouts are missing extractions; they stay in the denominator
    null_rate = sum(int(q.is_null.sum()) for q in vlm_bundle.questions) / 32000
    assert null_rate == pytest.approx(3032 / 32000, abs=1e-9)
