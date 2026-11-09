"""L2-SP proximity penalty (PA p. 4).

Ω(θ, θ0) = 0.5 * Σ_j (θ_j − θ0_j)^2 over **all unique parameters**, including biases and
normalisation parameters, with tied parameters counted once. θ0 is fixed.

The penalty gradient is added once per *optimizer update*
(``∇Ω = θ − θ0``), not once per micro-batch, so accumulation cannot multiply λ.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class L2SPReference:
    tensors: dict[str, torch.Tensor]
    n_parameters: int
    storage_bytes: int

    def omega(self, model: nn.Module) -> torch.Tensor:
        """0.5 * Σ (θ − θ0)² over unique parameters, evaluated in FP32."""
        total = None
        for name, p in model.named_parameters():
            ref = self.tensors[name]
            diff = p.float() - ref
            term = diff.pow(2).sum()
            total = term if total is None else total + term
        if total is None:
            raise RuntimeError("model has no parameters")
        return 0.5 * total

    def add_gradient(self, model: nn.Module, lam: float) -> torch.Tensor:
        """Add λ(θ − θ0) to every parameter gradient; returns λΩ for logging."""
        omega = None
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            diff = p.float() - self.tensors[name]
            p.grad.add_(diff.to(p.grad.dtype), alpha=lam)
            term = diff.pow(2).sum()
            omega = term if omega is None else omega + term
        return 0.5 * lam * omega if omega is not None else torch.zeros(())


def make_reference(model: nn.Module, device: str | torch.device = "cpu") -> L2SPReference:
    """Snapshot θ0 once, in FP32. ``named_parameters`` already de-duplicates tied weights."""
    tensors = {}
    n = 0
    nbytes = 0
    for name, p in model.named_parameters():
        t = p.detach().float().to(device).clone()
        tensors[name] = t
        n += t.numel()
        nbytes += t.numel() * t.element_size()
    return L2SPReference(tensors=tensors, n_parameters=n, storage_bytes=nbytes)
