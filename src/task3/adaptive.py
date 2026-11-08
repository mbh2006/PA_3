"""Adaptive sampling controllers: agreement and quality-aware stopping.

Both controllers always return the current self-consistency answer, so the
comparison isolates *stopping* from the final selector. Stopping is only checked
at n in {4, 8, 16}; otherwise 32 candidates are consumed. Only revealed
candidates are ever used: no gold labels, no future rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from task3.answers import NO_ANSWER, prefix_groups
from task3.scores import consensus_group, rwc_weights, visconf_confidence, visconf_from_confidence

CHECKPOINTS = (4, 8, 16, 32)
TAUS = (0.5, 0.625, 0.75, 0.875, 1.0)
GAMMAS = (0.0, 0.1, 0.25, 0.5)

ALWAYS_32 = "always_32"


@dataclass(frozen=True)
class Policy:
    family: str            # "agreement" | "quality"
    tau: float | None = None
    gamma: float | None = None

    @property
    def name(self) -> str:
        if self.family == "always32":
            return ALWAYS_32
        if self.family == "agreement":
            return f"agreement_tau{self.tau:g}"
        return f"quality_tau{self.tau:g}_gamma{self.gamma:g}"


def _vote_stats(groups: list[int], weights: np.ndarray):
    sums: dict[int, float] = {}
    for g, w in zip(groups, weights):
        sums[g] = sums.get(g, 0.0) + float(w)
    ranked = sorted(sums.items(), key=lambda kv: -kv[1])
    top_group, top_w = ranked[0]
    runner_w = ranked[1][1] if len(ranked) > 1 else 0.0
    total = sum(sums.values())
    return sums, top_group, top_w, runner_w, total


def _agreement_ok(groups: list[int], keys: list[str], indices: list[int],
                  tau: float) -> tuple[bool, int]:
    weights = np.ones(len(groups))
    sums, top_group, top_w, runner_w, total = _vote_stats(groups, weights)
    top_pos = groups.index(top_group)
    answer = keys[indices[top_pos]]
    unique = top_w > runner_w
    non_empty = answer != NO_ANSWER
    support = top_w / total if total > 0 else 0.0
    return (unique and non_empty and support >= tau - 1e-12), top_group


def should_stop(q, perm: np.ndarray, n: int, policy: Policy, is_vlm: bool) -> bool:
    idx = [int(x) for x in perm[:n]]
    groups = prefix_groups(idx, q.eq_matrix, q.keys)
    ok, sc_group = _agreement_ok(groups, q.keys, idx, float(policy.tau))
    if not ok:
        return False
    if policy.family == "agreement":
        return True

    # quality-aware: SC and RWC winners must agree, and the RWC margin must reach gamma
    if is_vlm:
        conf = visconf_confidence(q.kl_tokens, idx)
        quality = visconf_from_confidence(conf, q.vis[idx])
    else:
        quality = q.kl_mean[idx]
    w = rwc_weights(quality)
    sums, top_group, top_w, runner_w, total = _vote_stats(groups, w)
    rwc_group = consensus_group(groups, w)
    margin = (top_w - runner_w) / total if total > 0 else 0.0
    return (sc_group == rwc_group) and (margin >= float(policy.gamma) - 1e-12)


def run_controller(q, perm: np.ndarray, policy: Policy, is_vlm: bool) -> dict:
    """Reveal candidates in order and stop per ``policy``; return the SC answer."""
    cum = np.cumsum(q.n_tokens[perm])
    for n in CHECKPOINTS:
        if n == 32:
            # the fallback always consumes 32 candidates
            stop = True
        elif policy.family == "always32":
            stop = False
        else:
            stop = should_stop(q, perm, n, policy, is_vlm)
        if stop:
            idx = [int(x) for x in perm[:n]]
            groups = prefix_groups(idx, q.eq_matrix, q.keys)
            g = consensus_group(groups, np.ones(len(groups)))
            pos = groups.index(g)
            return {
                "n_used": n,
                "cost": int(cum[n - 1]),
                "correct": bool(q.correct[idx[pos]]),
                "selected_rollout": int(idx[pos]),
                "selected_answer": str(q.keys[idx[pos]]),
            }
    # unreachable: 32 always stops
    raise RuntimeError("controller failed to stop at 32")


def evaluate_policy(bundle, ids: list[str], orders: dict[str, np.ndarray],
                    policy: Policy) -> list[dict]:
    by_id = bundle.by_id()
    rows = []
    for sid in ids:
        q = by_id[sid]
        perms = orders[sid]
        accs, costs, ns = [], [], []
        for oi in range(perms.shape[0]):
            r = run_controller(q, perms[oi], policy, bundle.is_vlm)
            accs.append(r["correct"])
            costs.append(r["cost"])
            ns.append(r["n_used"])
        rows.append({
            "sample_id": sid,
            "accuracy": float(np.mean(accs)),
            "cost": float(np.mean(costs)),
            "n_used": float(np.mean(ns)),
            "n_used_by_order": ns,
        })
    return rows
