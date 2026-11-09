"""L2-SP penalty and optimiser/schedule construction (PA pp. 3-4)."""

import math

import torch
import torch.nn as nn

from task1.l2sp import make_reference
from task1.optim import build_param_groups, lr_at


class Tiny(nn.Module):
    def __init__(self, tie: bool = False):
        super().__init__()
        self.emb = nn.Embedding(11, 4)
        self.norm = nn.LayerNorm(4)
        self.proj = nn.Linear(4, 5)
        if tie:
            self.head = nn.Linear(4, 11, bias=False)
            self.head.weight = self.emb.weight  # tied parameter


def test_omega_is_zero_at_start():
    model = Tiny()
    ref = make_reference(model)
    assert ref.omega(model).item() == 0.0


def test_omega_matches_hand_computation():
    torch.manual_seed(0)
    model = Tiny()
    ref = make_reference(model)
    with torch.no_grad():
        model.proj.weight.add_(1.5)
        model.norm.bias.sub_(0.25)
    n_w = model.proj.weight.numel()
    n_b = model.norm.bias.numel()
    expected = 0.5 * (1.5 ** 2 * n_w + 0.25 ** 2 * n_b)
    assert abs(ref.omega(model).item() - expected) < 1e-6


def test_omega_includes_biases_and_norms():
    model = Tiny()
    ref = make_reference(model)
    names = {n for n, _ in model.named_parameters()}
    assert "norm.weight" in names and "proj.bias" in names
    assert set(ref.tensors) == names


def test_tied_parameters_counted_once():
    model = Tiny(tie=True)
    ref = make_reference(model)
    n_unique = sum(p.numel() for p in model.parameters())
    assert ref.n_parameters == n_unique
    # the tied matrix appears once in the reference
    assert len(ref.tensors) == len(list(model.named_parameters()))


def test_penalty_gradient_is_lambda_times_displacement():
    torch.manual_seed(1)
    model = Tiny()
    ref = make_reference(model)
    with torch.no_grad():
        model.proj.weight.add_(0.3)
    for p in model.parameters():
        p.grad = torch.zeros_like(p)

    lam = 1e-3
    lam_omega = ref.add_gradient(model, lam)
    expected = 0.5 * lam * 0.3 ** 2 * model.proj.weight.numel()
    assert abs(lam_omega.item() - expected) < 1e-8
    assert torch.allclose(model.proj.weight.grad, torch.full_like(model.proj.weight, lam * 0.3),
                          atol=1e-9)
    assert torch.allclose(model.emb.weight.grad, torch.zeros_like(model.emb.weight))


def test_penalty_gradient_is_not_multiplied_by_accumulation():
    """Applying the penalty once per update equals one unit of λ(θ−θ0)."""
    model = Tiny()
    ref = make_reference(model)
    with torch.no_grad():
        model.proj.weight.add_(0.5)
    model.zero_grad(set_to_none=True)
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    ref.add_gradient(model, 1e-3)
    once = model.proj.weight.grad.clone()

    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    for _ in range(16):  # what would happen if the penalty were added per micro-batch
        ref.add_gradient(model, 1e-3 / 16)
    sixteen = model.proj.weight.grad.clone()
    assert torch.allclose(once, sixteen, atol=1e-9)


def test_lr_warmup_then_cosine():
    peak = 2e-5
    assert lr_at(0, peak) == peak / 38
    assert abs(lr_at(37, peak) - peak) < 1e-12
    assert abs(lr_at(38, peak) - peak) < 1e-12
    mid = lr_at(38 + (1250 - 38) // 2, peak)
    assert 0.0 < mid < peak
    assert abs(lr_at(1250, peak)) < 1e-12
    assert abs(mid - peak / 2) < 0.01 * peak


def test_param_groups_exclude_bias_and_one_dim():
    model = Tiny()
    groups = build_param_groups(model, weight_decay=0.01)
    decay, no_decay = groups[0]["params"], groups[1]["params"]
    assert all(p.ndim >= 2 for p in decay)
    assert all(not n.endswith("bias") for n, p in model.named_parameters() if any(p is q for q in decay))
    names_no_decay = {n for n, p in model.named_parameters() if any(p is q for q in no_decay)}
    assert "proj.bias" in names_no_decay
    assert "norm.weight" in names_no_decay
    assert groups[1]["weight_decay"] == 0.0
