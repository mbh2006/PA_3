"""Bootstrap and N=32 signal analyses."""

import numpy as np
import pytest

from _synth import make_bundle, make_question
from task3.bootstrap import paired_bootstrap
from task3.signals import fusion_identity, quartile_stats, within_question


def test_bootstrap_constant_diff():
    d = paired_bootstrap(np.full(50, 0.1))
    assert d["mean"] == pytest.approx(0.1)
    assert d["lo"] == pytest.approx(0.1, abs=1e-12)
    assert d["hi"] == pytest.approx(0.1, abs=1e-12)
    assert d["n"] == 50


def test_bootstrap_is_deterministic():
    x = np.random.default_rng(0).normal(0.05, 0.2, size=200)
    a = paired_bootstrap(x)
    b = paired_bootstrap(x)
    assert (a["lo"], a["hi"]) == (b["lo"], b["hi"])


def test_bootstrap_interval_contains_mean():
    x = np.random.default_rng(1).normal(0.1, 0.3, size=300)
    d = paired_bootstrap(x)
    assert d["lo"] < d["mean"] < d["hi"]


def test_bootstrap_empty():
    d = paired_bootstrap(np.array([]))
    assert d["n"] == 0 and d["mean"] is None


def test_quartile_stats_hand_calculated():
    # 32 candidates: top 8 all correct, bottom 8 all wrong
    score = np.arange(32, dtype=float)
    correct = np.zeros(32, dtype=bool)
    correct[24:] = True
    assert quartile_stats(score, correct) == pytest.approx(1.0)


def test_quartile_stats_constant_score_is_none():
    assert quartile_stats(np.ones(32), np.ones(32, dtype=bool)) is None


def test_within_question_length_correlation():
    qs = []
    for i in range(6):
        n_tokens = np.arange(32, dtype=np.int64) + 1
        correct = np.zeros(32, dtype=bool)
        correct[16:] = True          # longer rollouts correct
        qs.append(make_question(["A"] * 32, tokens=n_tokens.tolist(),
                                correct=correct.tolist(), sample_id=f"q{i}"))
    bundle = make_bundle(qs)
    out = within_question(bundle, [q.sample_id for q in qs])
    rho = out["spearman_with_correctness"]["length"]
    assert rho["n_defined"] == 6
    assert rho["mean"] > 0.7


def test_fusion_identity_exact_on_synthetic():
    kl = [np.array([0.1, 0.5, 0.9, 0.3])]
    vt = [np.array([0.2, 0.4, 0.1, 0.8])]
    q = make_question(["A"], kl_tokens=[k.tolist() for k in kl],
                      v_tokens=[v.tolist() for v in vt], is_vlm=True)
    bundle = make_bundle([q], is_vlm=True)
    out = fusion_identity(bundle, ["q0"])
    assert out["n_rollouts_checked"] == 1
    assert out["max_abs_error"] < 1e-12
    ex = out["examples"][0]
    assert ex["z_token"] == pytest.approx(ex["z"] + ex["cov"], abs=1e-12)
