"""Quality scores, VisConf prefix calibration, and rank-weighted consensus weights."""

from __future__ import annotations

import numpy as np

from task3.answers import NO_ANSWER


def visconf_confidence(kl_tokens: list[np.ndarray], indices: list[int]) -> np.ndarray:
    """Prefix-only VisConf calibration C_i for the revealed candidates.

    Token KLs are pooled over the *current* candidate prefix only; the 5th/95th
    percentiles use linear interpolation and the denominator is floored at 1e-8.
    """
    pooled = np.concatenate([kl_tokens[i] for i in indices])
    q05, q95 = np.quantile(pooled, [0.05, 0.95], method="linear")
    denom = max(float(q95 - q05), 1e-8)
    return np.array(
        [float(np.mean(np.clip((kl_tokens[i] - q05) / denom, 0.0, 1.0))) for i in indices]
    )


def tokenwise_fusion(kl_tokens: list[np.ndarray], v_tokens: list[np.ndarray],
                     indices: list[int]) -> np.ndarray:
    """Z_token_i = mean_t(c_it * v_it), the token-level fusion ablation."""
    out = []
    for i in indices:
        k = np.asarray(kl_tokens[i], dtype=np.float64)
        v = np.asarray(v_tokens[i], dtype=np.float64)
        # calibration uses the pooled prefix distribution
        pooled = np.concatenate([kl_tokens[j] for j in indices]).astype(np.float64)
        q05, q95 = np.quantile(pooled, [0.05, 0.95], method="linear")
        denom = max(float(q95 - q05), 1e-8)
        c = np.clip((k - q05) / denom, 0.0, 1.0)
        out.append(float(np.mean(c * v)))
    return np.array(out)


def visconf_from_confidence(confidence: np.ndarray, vis: np.ndarray) -> np.ndarray:
    return confidence * vis


def average_ranks_desc(scores: np.ndarray) -> np.ndarray:
    """Average ranks with 1 = best (largest score); ties share their average rank."""
    s = np.asarray(scores, dtype=np.float64)
    n = len(s)
    order = np.argsort(-s, kind="mergesort")
    ranks = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s[order[j + 1]] == s[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def rwc_weights(scores: np.ndarray) -> np.ndarray:
    """w_i = (N - r_i) / (N - 1); requires N > 1 (N = 1 is a special case)."""
    n = len(scores)
    if n == 1:
        return np.ones(1)
    r = average_ranks_desc(scores)
    return (n - r) / (n - 1.0)


def consensus_group(groups: list[int], weights: np.ndarray) -> int:
    """Answer group with the largest total weight; ties -> earliest supporting rollout."""
    sums: dict[int, float] = {}
    first: dict[int, int] = {}
    for pos, (g, w) in enumerate(zip(groups, weights)):
        sums[g] = sums.get(g, 0.0) + float(w)
        first.setdefault(g, pos)
    best = max(sums.values())
    tol = 1e-12 * max(1.0, abs(best))
    tied = [g for g, v in sums.items() if abs(v - best) <= tol]
    return min(tied, key=lambda g: first[g])


def n_groups(groups: list[int]) -> int:
    return len(set(groups))


def answer_of(group: int, groups: list[int], keys: list[str], indices: list[int]) -> str:
    for pos, g in enumerate(groups):
        if g == group:
            return keys[indices[pos]]
    return NO_ANSWER
