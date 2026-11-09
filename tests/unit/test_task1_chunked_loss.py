"""Chunked LM-head loss must equal the standard path exactly."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

from _task1_fakes import TinyLM
from task1.chunked_loss import backward_chunked_lm_loss


class TinyHFish(nn.Module):
    """Mimics the HF interface used by the chunked path."""

    def __init__(self, vocab=64, dim=16):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.vocab = vocab

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids, attention_mask=None, labels=None,
                output_hidden_states=False, use_cache=False):
        h = self.norm(self.emb(input_ids))
        logits = self.head(h)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, self.vocab),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(loss=loss, logits=logits,
                               hidden_states=(h,) if output_hidden_states else None)


def _batch(batch=2, seq=40, vocab=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(4, vocab, (batch, seq), generator=g)
    labels = ids.clone()
    labels[:, :8] = -100
    return ids, labels


def _standard(model, ids, labels, total_valid):
    out = model(input_ids=ids, labels=labels)
    n_valid = int((labels[:, 1:] != -100).sum())
    loss = out.loss * (n_valid / total_valid)
    loss.backward()
    return float(out.loss.detach()) * n_valid, n_valid


def _grads(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def test_chunked_matches_standard_gradients_and_sum():
    torch.manual_seed(0)
    ids, labels = _batch()
    total_valid = int((labels[:, 1:] != -100).sum())

    a = TinyHFish()
    b = TinyHFish()
    b.load_state_dict(a.state_dict())

    sum_std, n_std = _standard(a, ids, labels, total_valid)
    sum_chunk, n_chunk = backward_chunked_lm_loss(
        b, ids, None, labels, total_valid=total_valid, chunk_size=7)

    assert n_std == n_chunk == total_valid
    assert abs(sum_std - sum_chunk) < 1e-4, (sum_std, sum_chunk)
    ga, gb = _grads(a), _grads(b)
    assert ga.keys() == gb.keys()
    for name in ga:
        assert torch.allclose(ga[name], gb[name], atol=1e-6), name


def test_chunk_size_does_not_change_the_result():
    ids, labels = _batch(seed=3)
    total_valid = int((labels[:, 1:] != -100).sum())
    results = []
    for chunk in (1, 5, 64):
        model = TinyHFish()
        torch.manual_seed(0)
        model = TinyHFish()
        b = TinyHFish()
        b.load_state_dict(model.state_dict())
        sum_ce, n = backward_chunked_lm_loss(b, ids, None, labels,
                                             total_valid=total_valid, chunk_size=chunk)
        results.append((sum_ce, n, _grads(b)))
    for sum_ce, n, g in results[1:]:
        assert abs(sum_ce - results[0][0]) < 1e-4
        assert n == results[0][1]
        for name in g:
            assert torch.allclose(g[name], results[0][2][name], atol=1e-6), name


def test_loss_scale_multiplies_gradients_only():
    ids, labels = _batch(seed=5)
    total_valid = int((labels[:, 1:] != -100).sum())
    a, b = TinyHFish(), TinyHFish()
    b.load_state_dict(a.state_dict())
    s1, _ = backward_chunked_lm_loss(a, ids, None, labels, total_valid=total_valid,
                                     loss_scale=1.0, chunk_size=8)
    s2, _ = backward_chunked_lm_loss(b, ids, None, labels, total_valid=total_valid,
                                     loss_scale=8.0, chunk_size=8)
    assert abs(s1 - s2) < 1e-6
    for (n1, p1), (n2, p2) in zip(a.named_parameters(), b.named_parameters()):
        if p1.grad is not None:
            assert torch.allclose(p1.grad * 8.0, p2.grad, atol=1e-6), n1


def test_empty_batch_rejected():
    ids, labels = _batch()
    try:
        backward_chunked_lm_loss(TinyHFish(), ids, None, labels, total_valid=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
