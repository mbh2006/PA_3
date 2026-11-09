"""Question-level paired bootstrap (1,000 resamples, 95% intervals)."""

from __future__ import annotations

import numpy as np

SEED = 6304
N_BOOT = 1000


def paired_bootstrap(diff: np.ndarray, *, n_boot: int = N_BOOT, seed: int = SEED,
                     alpha: float = 0.05) -> dict:
    """Percentile bootstrap over questions of an already order-averaged difference.

    Resampling is at the question level; each question keeps all of its rollouts.
    """
    x = np.asarray(diff, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return {"n": 0, "mean": None, "lo": None, "hi": None, "se": None}
    rng = np.random.default_rng([seed, n])
    idx = rng.integers(0, n, size=(n_boot, n))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2], method="linear")
    return {
        "n": int(n),
        "mean": float(x.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "se": float(means.std(ddof=1)),
    }


def summarize_mean(values: np.ndarray, *, n_boot: int = N_BOOT, seed: int = SEED,
                   alpha: float = 0.05) -> dict:
    """Mean with a question-level bootstrap interval (not a paired difference)."""
    return paired_bootstrap(values, n_boot=n_boot, seed=seed, alpha=alpha)
