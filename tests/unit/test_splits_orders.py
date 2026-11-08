"""Split and order determinism (PA page 11)."""

import numpy as np
import pytest

from task3.orders import orders_for_all, orders_for_question
from task3.splits import LIVE_SUBSET_SIZE, SEED, make_split


def _ids(n=1000):
    return [f"test:{i}" for i in range(n)]


def test_split_is_deterministic_and_sized():
    s1 = make_split(_ids())
    s2 = make_split(_ids())
    assert s1.dev == s2.dev and s1.heldout == s2.heldout
    assert len(s1.dev) == 200 and len(s1.heldout) == 800
    assert set(s1.dev).isdisjoint(s1.heldout)
    assert set(s1.dev) | set(s1.heldout) == set(_ids())


def test_split_permutes_the_lexicographic_order():
    ids = _ids()
    s = make_split(ids)
    ordered = sorted(ids)
    expected = np.random.default_rng(SEED).permutation(len(ordered))
    assert s.order == [ordered[i] for i in expected]


def test_live_subset_is_first_fifty_heldout():
    s = make_split(_ids())
    assert len(s.live_math) == LIVE_SUBSET_SIZE
    assert s.live_math == s.heldout[:LIVE_SUBSET_SIZE]


def test_live_subset_only_defined_for_math():
    # the caller decides; the split simply exposes the first 50 held-out ids
    s = make_split(_ids(80))
    assert s.live_math == s.heldout[:50]


def test_orders_are_permutations_and_reproducible():
    a = orders_for_question(7)
    b = orders_for_question(7)
    assert a.shape == (10, 32)
    assert np.array_equal(a, b)
    for row in a:
        assert sorted(row.tolist()) == list(range(32))


def test_orders_are_independent_of_question_position_stream():
    # different question index -> different permutation set, but stable
    a = orders_for_question(3)
    b = orders_for_question(4)
    assert not np.array_equal(a, b)
    assert np.array_equal(a, orders_for_question(3))


def test_orders_map_covers_sorted_questions():
    ids = ["b", "a", "c"]
    m = orders_for_all(sorted(ids))
    assert list(m) == ["a", "b", "c"]
    assert np.array_equal(m["a"], orders_for_question(0))
    assert np.array_equal(m["c"], orders_for_question(2))
