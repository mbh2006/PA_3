"""Development / held-out splits and the live subset (seed 6304)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SEED = 6304
N_DEV = 200
LIVE_SUBSET_SIZE = 50


@dataclass(frozen=True)
class Split:
    order: list[str]          # full lexicographic-then-permuted question order
    dev: list[str]
    heldout: list[str]
    live_math: list[str]      # first 50 MATH held-out ids in permutation order

    def as_dict(self) -> dict:
        return {
            "seed": SEED,
            "n_dev": len(self.dev),
            "n_heldout": len(self.heldout),
            "dev": self.dev,
            "heldout": self.heldout,
            "live_math": self.live_math,
        }


def make_split(question_ids: list[str], *, seed: int = SEED, n_dev: int = N_DEV) -> Split:
    """Lexicographic sort -> ``default_rng(seed).permutation`` -> first ``n_dev`` are dev."""
    ordered = sorted(question_ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ordered))
    shuffled = [ordered[i] for i in perm]
    dev = shuffled[:n_dev]
    heldout = shuffled[n_dev:]
    return Split(
        order=shuffled,
        dev=dev,
        heldout=heldout,
        live_math=heldout[:LIVE_SUBSET_SIZE],
    )
