"""Trainer integration of the chunked loss: scaler init and gradient equivalence."""

import torch

from _task1_fakes import FakeTokenizer, TinyLM
from task1.data import MathExample
from task1.train import Trainer, TrainConfig


def _examples(lengths, n_prompt=2, seed=0):
    import numpy as np

    rng = np.random.default_rng(seed)
    out = []
    for i, n in enumerate(lengths):
        ids = rng.integers(4, 256, size=n).astype("int32")
        labels = ids.copy()
        labels[:n_prompt] = -100
        out.append(MathExample(i, "q", "", "r", ids, labels, n_prompt))
    return out


def _run(chunked: bool):
    torch.manual_seed(0)
    model = TinyLM()
    tok = FakeTokenizer()
    ex = _examples([40, 55, 30, 47])
    cfg = TrainConfig(variant="t", peak_lr=0.0, updates=1, micro_batch=2, grad_accum=2,
                      clip_grad_norm=1e9, chunked_loss=chunked, loss_chunk_size=7)
    trainer = Trainer(model, ex, tok, cfg, device="cpu")
    stats = trainer.update_once([0, 1, 2, 3])
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}, stats


def test_chunked_and_standard_trainer_paths_agree():
    g_std, s_std = _run(chunked=False)
    g_chn, s_chn = _run(chunked=True)
    assert abs(s_std["sft_loss"] - s_chn["sft_loss"]) < 1e-5
    assert g_std.keys() == g_chn.keys()
    for name in g_std:
        assert torch.allclose(g_std[name], g_chn[name], atol=1e-6), name


def test_chunked_path_initialises_the_scaler_on_cpu():
    g, stats = _run(chunked=True)
    assert stats["skipped"] is False
    assert any(torch.count_nonzero(v) > 0 for v in g.values())
