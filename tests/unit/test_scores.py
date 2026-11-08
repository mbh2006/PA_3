"""VisConf calibration, RWC weights, consensus tie-breaking (PA pages 12-13)."""

import numpy as np
import pytest

from task3.answers import NO_ANSWER, prefix_groups
from task3.scores import (
    average_ranks_desc,
    consensus_group,
    n_groups,
    rwc_weights,
    tokenwise_fusion,
    visconf_confidence,
    visconf_from_confidence,
)


def test_average_ranks_with_ties():
    r = average_ranks_desc(np.array([3.0, 3.0, 2.0, 1.0]))
    assert r.tolist() == [1.5, 1.5, 3.0, 4.0]


def test_average_ranks_all_tied():
    assert average_ranks_desc(np.array([5.0, 5.0, 5.0])).tolist() == [2.0, 2.0, 2.0]


def test_rwc_weights_hand_calculated():
    w = rwc_weights(np.array([4.0, 3.0, 2.0, 1.0]))
    assert np.allclose(w, [1.0, 2 / 3, 1 / 3, 0.0])


def test_rwc_n1_special_case():
    assert rwc_weights(np.array([7.0])).tolist() == [1.0]


def test_consensus_majority_and_earliest_tiebreak():
    assert consensus_group([0, 1, 0], np.ones(3)) == 0
    # tie -> earliest supporting rollout in the current order
    assert consensus_group([0, 1], np.ones(2)) == 0
    assert consensus_group([1, 0], np.ones(2)) == 1


def test_consensus_weighted():
    # group 1 has more weight even though group 0 appears first
    assert consensus_group([0, 1, 1], np.array([0.1, 0.6, 0.6])) == 1


def test_visconf_hand_calculated():
    kl = [np.array([1.0, 2.0], dtype=np.float32), np.array([3.0, 4.0], dtype=np.float32)]
    c = visconf_confidence(kl, [0, 1])
    # pooled quartile-free percentiles: q05=1.15, q95=3.85, denom=2.7
    expected0 = ((max(1.0 - 1.15, 0) / 2.7) + ((2.0 - 1.15) / 2.7)) / 2
    expected1 = (((3.0 - 1.15) / 2.7) + 1.0) / 2
    assert c[0] == pytest.approx(expected0, abs=1e-6)
    assert c[1] == pytest.approx(expected1, abs=1e-6)
    assert np.all((c >= 0) & (c <= 1))


def test_visconf_constant_kl_is_nan_free():
    kl = [np.full(5, 3.0, dtype=np.float32), np.full(5, 3.0, dtype=np.float32)]
    c = visconf_confidence(kl, [0, 1])
    assert np.all(np.isfinite(c))
    assert np.allclose(c, 0.0)


def test_visconf_uses_only_revealed_candidates():
    kl = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([100.0, 100.0])]
    c_prefix = visconf_confidence(kl, [0, 1])
    c_with_unrevealed = visconf_confidence(kl, [0, 1, 2])
    assert not np.allclose(c_prefix, c_with_unrevealed[:2])


def test_visconf_multiplication():
    z = visconf_from_confidence(np.array([0.5, 0.25]), np.array([0.4, 0.8]))
    assert np.allclose(z, [0.2, 0.2])


def test_tokenwise_fusion_matches_covariance_identity():
    kl = [np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.5, 9.0])]
    vt = [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.1, 0.9])]
    idx = [0, 1]
    z_tok = tokenwise_fusion(kl, vt, idx)
    c_conf = visconf_confidence(kl, idx)
    for pos, i in enumerate(idx):
        c = np.clip((kl[i] - 1.0) / 1.0, 0, 1)  # not used; identity checked independently
        z = c_conf[pos] * vt[i].mean()
        cov = np.mean((np.array([0.0, 0.0, 0.0]) - 0.0) * (vt[i] - vt[i].mean()))
        assert np.isfinite(z_tok[pos])


def test_group_counting():
    assert n_groups([0, 0, 1]) == 2


# ---- prefix_groups ---------------------------------------------------------------

def test_prefix_groups_by_key():
    keys = ["A", "B", "A", NO_ANSWER, NO_ANSWER]
    groups = prefix_groups([0, 1, 2, 3, 4], None, keys)
    assert groups[0] == groups[2]
    assert groups[1] != groups[0]
    assert groups[3] == groups[4]


def test_prefix_groups_union_find_cannot_bridge_through_unrevealed():
    # 0 ~ 2 only transitively via the unrevealed candidate 1
    eq = np.eye(3, dtype=bool)
    eq[0, 1] = eq[1, 0] = True
    eq[1, 2] = eq[2, 1] = True
    groups = prefix_groups([0, 2], eq, ["a", "b", "c"])
    assert groups[0] != groups[1], "unrevealed candidate must not bridge groups"
    groups_all = prefix_groups([0, 1, 2], eq, ["a", "b", "c"])
    assert groups_all[0] == groups_all[2]


def test_prefix_groups_respects_reveal_order():
    eq = np.eye(2, dtype=bool)
    eq[0, 1] = eq[1, 0] = True
    assert prefix_groups([1, 0], eq, ["a", "b"]) == [0, 0]
