"""Equivalence-matrix cache round trip."""

import numpy as np

from task3 import cache


def test_cache_roundtrip(tmp_path):
    items = {
        "q1": np.array([[True, False], [False, True]]),
        "q2": np.array([[True, True], [True, True]]),
    }
    key = "test-key"
    cache.save(tmp_path, "math", key, items)
    loaded = cache.load(tmp_path, "math", key)
    assert loaded is not None
    assert set(loaded) == {"q1", "q2"}
    for sid, mat in items.items():
        assert np.array_equal(loaded[sid], mat)


def test_cache_key_mismatch_returns_none(tmp_path):
    items = {"q1": np.eye(3, dtype=bool)}
    cache.save(tmp_path, "math", "key-a", items)
    assert cache.load(tmp_path, "math", "key-b") is None


def test_cache_missing_file(tmp_path):
    assert cache.load(tmp_path, "math", "anything") is None


def test_cache_handles_32x32(tmp_path):
    rng = np.random.default_rng(0)
    items = {f"q{i}": rng.random((32, 32)) > 0.5 for i in range(3)}
    cache.save(tmp_path, "math", "k", items)
    loaded = cache.load(tmp_path, "math", "k")
    for sid, mat in items.items():
        assert loaded[sid].shape == (32, 32)
        assert np.array_equal(loaded[sid], mat)
