"""Adaptive stopping: correctness, cost accounting, and no-leak guarantees."""

import numpy as np
import pytest

from _synth import make_bundle, make_question
from task3.adaptive import ALWAYS_32, Policy, evaluate_policy, run_controller, should_stop

PERM = np.arange(32)


def test_always32_consumes_everything():
    q = make_question(["A"] * 32, tokens=[5] * 32)
    r = run_controller(q, PERM, Policy("always32"), is_vlm=False)
    assert r["n_used"] == 32
    assert r["cost"] == 160


def test_agreement_stops_at_four_when_support_is_high():
    q = make_question(["A", "A", "A", "B"] + ["B"] * 28, tokens=[3] * 32)
    pol = Policy("agreement", tau=0.75)
    assert should_stop(q, PERM, 4, pol, is_vlm=False) is True
    r = run_controller(q, PERM, pol, is_vlm=False)
    assert r["n_used"] == 4
    assert r["cost"] == 12
    assert r["selected_answer"] == "A"


def test_tie_at_four_does_not_stop():
    q = make_question(["A", "A", "B", "B"] + ["C"] * 28, tokens=[1] * 32)
    assert should_stop(q, PERM, 4, Policy("agreement", tau=0.5), False) is False
    # at n=8 the leader is unique with support 4/8, so tau=0.5 stops there
    assert run_controller(q, PERM, Policy("agreement", tau=0.5), False)["n_used"] == 8
    # a stricter threshold forces the full budget
    assert run_controller(q, PERM, Policy("agreement", tau=0.9), False)["n_used"] == 32


def test_no_answer_leader_never_stops_early():
    q = make_question([None, None, None, None] + ["A"] * 28)
    pol = Policy("agreement", tau=0.5)
    assert should_stop(q, PERM, 4, pol, is_vlm=False) is False


def test_support_threshold_is_respected():
    q = make_question(["A", "A", "B", "C"] + ["D"] * 28)
    # unique leader A with support 2/4 = 0.5
    assert should_stop(q, PERM, 4, Policy("agreement", tau=0.5), False) is True
    assert should_stop(q, PERM, 4, Policy("agreement", tau=0.75), False) is False


def test_quality_aware_requires_sc_rwc_agreement():
    # 5 A vs 3 B: self-consistency has a unique leader A (support 0.625), but the
    # three B rollouts hold the top quality ranks, so RWC's winner is B.
    answers = ["A"] * 5 + ["B"] * 3 + ["C"] * 24
    q = make_question(answers, tokens=[1] * 32)
    q.kl_mean = np.array([0.0] * 5 + [100.0, 90.0, 80.0] + [0.0] * 24)
    pol = Policy("quality", tau=0.5, gamma=0.0)
    assert should_stop(q, PERM, 8, pol, is_vlm=False) is False
    # agreement alone would stop at 8
    assert should_stop(q, PERM, 8, Policy("agreement", tau=0.5), False) is True


def test_quality_aware_margin_threshold():
    q = make_question(["A", "A", "A", "B"] + ["B"] * 28, tokens=[1] * 32)
    q.kl_mean = np.array([10.0, 10.0, 10.0, 0.0] + [0.0] * 28)
    # RWC: A holds all the mass -> margin = (2.0 - 0.0) / 2.0 = 1.0
    assert should_stop(q, PERM, 4, Policy("quality", tau=0.75, gamma=0.5), False) is True
    assert should_stop(q, PERM, 4, Policy("quality", tau=0.75, gamma=1.1), False) is False


# ---- no-leak guarantees ----------------------------------------------------------

def test_gold_labels_cannot_change_any_decision():
    q = make_question(["A", "A", "A", "B"] + ["B"] * 28, tokens=[2] * 32)
    pol = Policy("quality", tau=0.75, gamma=0.0)
    before = run_controller(q, PERM, pol, is_vlm=False)

    flipped = make_question(["A", "A", "A", "B"] + ["B"] * 28, tokens=[2] * 32,
                            correct=[False] * 32)
    after = run_controller(flipped, PERM, pol, is_vlm=False)

    assert before["n_used"] == after["n_used"]
    assert before["selected_rollout"] == after["selected_rollout"]
    assert before["selected_answer"] == after["selected_answer"]


def test_unrevealed_candidates_cannot_change_a_decision():
    answers = ["A", "A", "A", "B"] + ["Z"] * 28
    q = make_question(answers, tokens=[1] * 32)
    pol = Policy("agreement", tau=0.75)
    r1 = run_controller(q, PERM, pol, is_vlm=False)

    perm2 = np.concatenate([PERM[:4], np.random.default_rng(0).permutation(PERM[4:])])
    r2 = run_controller(q, perm2, pol, is_vlm=False)

    assert r1["n_used"] == r2["n_used"] == 4
    assert r1["selected_rollout"] == r2["selected_rollout"]


def test_different_prefix_changes_the_decision():
    # candidates 0-3 tie (A,A,B,B); candidates 4-7 are A,A,A,B
    answers = ["A", "A", "B", "B", "A", "A", "A", "B"] + ["C"] * 24
    q = make_question(answers, tokens=[1] * 32)
    pol = Policy("agreement", tau=0.5)
    assert run_controller(q, PERM, pol, is_vlm=False)["n_used"] == 8
    perm2 = np.array([4, 5, 6, 7, 0, 1, 2, 3] + list(range(8, 32)))
    assert run_controller(q, perm2, pol, is_vlm=False)["n_used"] == 4


def test_evaluate_policy_averages_orders():
    qs = [make_question(["A", "A", "A", "B"] + ["B"] * 28, sample_id="a", tokens=[4] * 32),
          make_question(["A", "B", "B", "B"] + ["B"] * 28, sample_id="b", tokens=[4] * 32)]
    bundle = make_bundle(qs)
    orders = {qq.sample_id: np.stack([PERM, np.arange(31, -1, -1)]) for qq in qs}
    rows = evaluate_policy(bundle, ["a", "b"], orders, Policy("agreement", tau=0.75))
    assert len(rows) == 2
    assert all(0.0 <= r["accuracy"] <= 1.0 for r in rows)
    assert all(r["cost"] > 0 for r in rows)


def test_always32_policy_name():
    assert Policy("always32").name == ALWAYS_32
