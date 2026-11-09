"""Token-weighted SFT loss over the effective batch (PA p. 3).

The reference is a single forward over the whole effective batch with
``sum_ce / total_valid_tokens``. Gradient accumulation must reproduce it exactly.
"""

import numpy as np
import torch
import torch.nn.functional as F

from _task1_fakes import FakeTokenizer, TinyLM
from task1.data import MathExample, collate
from task1.train import Trainer, TrainConfig, micro_batch_scale


def _examples(lengths, n_prompt=2, vocab=256, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i, n in enumerate(lengths):
        ids = rng.integers(4, vocab, size=n).astype(np.int32)
        labels = ids.copy()
        labels[:n_prompt] = -100
        out.append(MathExample(i, "q", "", "r", ids, labels, n_prompt))
    return out


def test_scale_is_token_ratio():
    assert micro_batch_scale(10, 40) == 0.25
    assert micro_batch_scale(40, 40) == 1.0


def test_scale_rejects_empty_batch():
    try:
        micro_batch_scale(0, 0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_unequal_lengths_match_single_batch_reference():
    torch.manual_seed(0)
    model = TinyLM()
    examples = _examples([7, 11, 5, 9])
    pad = 0

    # reference: one batch, token-weighted (sum CE / total valid)
    batch = collate(examples, pad)
    ids = torch.as_tensor(batch["input_ids"])
    labels = torch.as_tensor(batch["labels"])
    logits = model(input_ids=ids) if False else model(ids).logits
    sum_ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                             labels[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
    total_valid = int((labels[:, 1:] != -100).sum())
    (sum_ce / total_valid).backward()
    ref_grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()}

    # accumulation: two micro-batches of very different lengths
    model.zero_grad(set_to_none=True)
    for chunk in (examples[:2], examples[2:]):
        b = collate(chunk, pad)
        out = model(torch.as_tensor(b["input_ids"]), labels=torch.as_tensor(b["labels"]))
        n_valid = int((torch.as_tensor(b["labels"])[:, 1:] != -100).sum())
        (out.loss * micro_batch_scale(n_valid, total_valid)).backward()

    for name, p in model.named_parameters():
        assert torch.allclose(p.grad, ref_grads[name], atol=1e-6), name


def test_trainer_update_matches_reference_gradients():
    """The real Trainer path must equal the analytic token-weighted gradient."""
    torch.manual_seed(1)
    tok = FakeTokenizer()
    examples = _examples([6, 12, 8, 10], n_prompt=2)
    model = TinyLM()

    batch = collate(examples, 0)
    ids = torch.as_tensor(batch["input_ids"])
    labels = torch.as_tensor(batch["labels"])
    logits = model(ids).logits
    sum_ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                             labels[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
    total_valid = int((labels[:, 1:] != -100).sum())
    (sum_ce / total_valid).backward()
    ref = {n: p.grad.detach().clone() for n, p in model.named_parameters()}

    model.zero_grad(set_to_none=True)
    cfg = TrainConfig(variant="test", peak_lr=0.0, updates=1, micro_batch=2, grad_accum=2,
                      clip_grad_norm=1e9, lambda_l2sp=0.0)
    trainer = Trainer(model, examples, tok, cfg, device="cpu")
    stats = trainer.update_once([0, 1, 2, 3])

    for name, p in model.named_parameters():
        assert torch.allclose(p.grad, ref[name], atol=1e-6), name
    assert stats["skipped"] is False
    assert stats["sft_loss"] > 0


def test_micro_batch_grouping_is_contiguous():
    tok = FakeTokenizer()
    examples = _examples([5] * 8)
    cfg = TrainConfig(variant="t", peak_lr=0.0, updates=1, micro_batch=2, grad_accum=4)
    trainer = Trainer(TinyLM(), examples, tok, cfg, device="cpu")
    chunks = list(trainer._micro_batches(list(range(8))))
    assert len(chunks) == 4
    assert all(c["input_ids"].shape[0] == 2 for c in chunks)
