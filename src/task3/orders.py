"""Ten seeded rollout-ID permutations per question, reused by every method."""

from __future__ import annotations

import numpy as np

SEED = 6304
N_ORDERS = 10
N_ROLLOUTS = 32


def orders_for_question(question_index: int, *, seed: int = SEED, n_orders: int = N_ORDERS) -> np.ndarray:
    """Return an ``(n_orders, 32)`` array of rollout-ID permutations.

    ``question_index`` is the position in the lexicographically sorted question list,
    so the streams are independent of the dev/held-out split and of iteration order.
    """
    rng = np.random.default_rng([seed, int(question_index)])
    return np.stack([rng.permutation(N_ROLLOUTS) for _ in range(n_orders)])


def orders_for_all(question_ids_in_lexicographic_order: list[str], *, seed: int = SEED,
                   n_orders: int = N_ORDERS) -> dict[str, np.ndarray]:
    return {
        qid: orders_for_question(i, seed=seed, n_orders=n_orders)
        for i, qid in enumerate(question_ids_in_lexicographic_order)
    }
