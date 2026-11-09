"""Prepared-data round trip and probe determinism."""

import json

import numpy as np

from _task1_fakes import FakeTokenizer
from task1.data import MathExample, format_example
from task1.prepare import ProbeSet, load_prepared, sample_probes, save_prepared


def _examples(n=6):
    tok = FakeTokenizer()
    out = []
    for i in range(n):
        ids, labels, p = format_example(tok, f"q{i}", f"answer {i}")
        out.append(MathExample(i, f"q{i}", "", f"answer {i}", ids, labels, p))
    return out, tok


def test_prepared_roundtrip_preserves_tokens_and_masks(tmp_path):
    examples, tok = _examples()
    meta = {"tokenizer": {"im_end_id": tok.convert_tokens_to_ids("<|im_end|>")}}
    path = save_prepared(tmp_path / "prepared.npz", examples, meta)

    loaded, meta2 = load_prepared(path)
    assert len(loaded) == len(examples)
    assert meta2["n_examples"] == len(examples)
    for a, b in zip(examples, loaded):
        assert np.array_equal(a.input_ids, b.input_ids)
        assert np.array_equal(a.labels, b.labels), "labels must be reconstructed identically"
        assert a.n_prompt_tokens == b.n_prompt_tokens
        assert a.source_index == b.source_index


def test_prepared_metadata_has_checksum(tmp_path):
    examples, tok = _examples(3)
    path = save_prepared(tmp_path / "p.npz", examples, {"tokenizer": {"im_end_id": 3}})
    meta = json.loads(path.with_suffix(".json").read_text())
    assert len(meta["sha256"]) == 64
    assert meta["total_tokens"] == sum(e.n_tokens for e in examples)


def test_probe_set_serialisation():
    ps = ProbeSet([1, 2], ["a", "b"], [3, 4], ["c", "d"])
    d = ps.as_dict()
    assert d["n_gsm8k"] == 2 and d["n_hellaswag"] == 2
    assert d["gsm8k_indices"] == [1, 2]


class _FakeDS:
    def __init__(self, n, key):
        self.n = n
        self.key = key

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {self.key: f"{self.key}{i}"}


def _fake_loader(name, split):
    if "gsm8k" in name:
        return _FakeDS(1319, "question")
    return _FakeDS(10042, "ctx")


def test_probe_sampling_is_deterministic_and_within_bounds():
    a = sample_probes(seed=6304, n=32, loader=_fake_loader)
    b = sample_probes(seed=6304, n=32, loader=_fake_loader)
    assert a.gsm8k_indices == b.gsm8k_indices
    assert a.hellaswag_indices == b.hellaswag_indices
    assert len(a.gsm8k_indices) == len(set(a.gsm8k_indices)) == 32
    assert a.gsm8k_indices == sorted(a.gsm8k_indices)
    assert all(0 <= i < 1319 for i in a.gsm8k_indices)
    assert all(0 <= i < 10042 for i in a.hellaswag_indices)
    assert len(a.gsm8k_questions) == 32 and len(a.hellaswag_contexts) == 32


def test_probe_sampling_differs_with_seed():
    a = sample_probes(seed=6304, n=32, loader=_fake_loader)
    b = sample_probes(seed=1, n=32, loader=_fake_loader)
    assert a.gsm8k_indices != b.gsm8k_indices
