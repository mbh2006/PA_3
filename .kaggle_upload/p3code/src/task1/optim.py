"""Optimizer parameter groups and the prescribed LR schedule (PA p. 3)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

WARMUP_UPDATES = 38
TOTAL_UPDATES = 1250


def build_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Decay matrices only; exclude biases and every parameter with fewer than 2 dims."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and not name.endswith("bias"):
            decay.append(p)
        else:
            no_decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def lr_at(step: int, peak_lr: float, warmup: int = WARMUP_UPDATES, total: int = TOTAL_UPDATES) -> float:
    """Linear warmup to ``peak_lr`` over ``warmup`` updates, then cosine decay to 0."""
    if step < 0:
        raise ValueError("step must be >= 0")
    if step < warmup:
        return peak_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(max(progress, 0.0), 1.0)
    return peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def make_scheduler(optimizer: torch.optim.Optimizer, warmup: int = WARMUP_UPDATES,
                   total: int = TOTAL_UPDATES) -> torch.optim.lr_scheduler.LambdaLR:
    """LambdaLR over the group's initial LR (set to the variant's peak LR)."""
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_at(step, 1.0, warmup=warmup, total=total),
    )


def make_optimizer(model: nn.Module, peak_lr: float, *, weight_decay: float = 0.01,
                   betas=(0.9, 0.999), eps: float = 1e-8) -> torch.optim.AdamW:
    groups = build_param_groups(model, weight_decay)
    return torch.optim.AdamW(groups, lr=peak_lr, betas=betas, eps=eps)
