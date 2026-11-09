"""The over-budget micro-batch split must not change the gradient.

Two 2,048-token sequences do not fit on a 16 GB T4, so a pair whose padded length
exceeds ``max_micro_tokens`` is processed as two single-example micro-batches. Because
the objective is the token-weighted mean over the *effective batch*, the gradient must
be identical to the unsplit accumulation.
"""

import torch
import torch.nn.functional as F

from _task1_fakes import FakeTokenizer, TinyLM
from task1.data import MathExample, collate
from task1.train import Trainer, TrainConfig


def _ex(lengths, n_prompt=2, seed=0):
    import numpy as np

    rng = np.random.default_rng(seed)
    out = []
    for i, n in enumerate(lengths):
        ids = rng.integers(4, 256, size=n).astype("int32")
        labels = ids.copy()
        labels[:n_prompt] = -100
        out.append(MathExample(i, "q", "", "r", ids, labels, n_prompt))
    return out


def _accumulate(model, examples, max_micro_tokens):
    tok = FakeTokenizer()
    cfg = TrainConfig(variant="t", peak_lr=0.0, updates=1, micro_batch=2, grad_accum=2,
                      clip_grad_norm=1e9, max_micro_tokens=max_micro_tokens)
    trainer = Trainer(model, examples, tok, cfg, device="cpu")
    trainer.update_once([0, 1, 2, 3])
    return {n: p.grad.detach().clone() for n, p in model.named_parameters()}


def test_split_preserves_gradient_exactly():
    torch.manual_seed(0)
    model = TinyLM()
    # two long, two short: the (long, short) padding forces 400-token rows
    examples = _ex([400, 400, 50, 60])

    model.zero_grad(set_to_none=True)
    unsplit = _accumulate(model, examples, max_micro_tokens=10_000_000)

    model.zero_grad(set_to_none=True)
    split = _accumulate(model, examples, max_micro_tokens=300)

    for name, p in model.named_parameters():
        assert torch.allclose(unsplit[name], split[name], atol=1e-7), name


def test_split_triggers_only_for_long_pairs():
    tok = FakeTokenizer()
    examples = _ex([400, 400, 50, 60])
    cfg = TrainConfig(variant="t", peak_lr=0.0, updates=1, micro_batch=2, grad_accum=2,
                      max_micro_tokens=300)
    trainer = Trainer(TinyLM(), examples, tok, cfg, device="cpu")
    chunks = list(trainer._micro_batches([0, 1, 2, 3]))
    # first pair (400,400) -> two single-example chunks; second pair (50,60) -> one chunk
    assert [c["input_ids"].shape[0] for c in chunks] == [1, 1, 2]


def test_split_does_not_trigger_for_short_pairs():
    tok = FakeTokenizer()
    examples = _ex([50, 60, 70, 80])
    cfg = TrainConfig(variant="t", peak_lr=0.0, updates=1, micro_batch=2, grad_accum=2,
                      max_micro_tokens=10_000)
    trainer = Trainer(TinyLM(), examples, tok, cfg, device="cpu")
    chunks = list(trainer._micro_batches([0, 1, 2, 3]))
    assert [c["input_ids"].shape[0] for c in chunks] == [2, 2]
