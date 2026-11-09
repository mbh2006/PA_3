"""Weight-space and activation diagnostics (PA p. 5).

ρ_k = ||U_k U_k^T ΔW||_F² / ||ΔW||_F²
η   = ||ΔW||_F / ||W0||_F
R(z) = ||ΔW z||_2 / (||W0 z||_2 + 1e-8)

A zero update has no defined ρ_k; the caller records ``None`` rather than inventing a
ratio.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

DIAG_LAYERS = (0, 13, 27)
DIAG_MODULE = "down_proj"


@dataclass
class WeightDiagnostic:
    module: str
    rho_k: float | None
    eta: float
    delta_fro: float
    w0_fro: float


def delta_weight(model: nn.Module, name: str, reference: dict[str, torch.Tensor]) -> torch.Tensor:
    """ΔW = W_trained − W0 in FP32 for one module (dense; diagnostics only)."""
    from task1.lora import LoRALinear

    module = model.get_submodule(name)
    if isinstance(module, LoRALinear):
        return module.delta_weight().detach().float().cpu()
    return (module.weight.detach().float().cpu() - reference[name].float().cpu())


def rho_k(delta: torch.Tensor, uk: torch.Tensor) -> float | None:
    norm2 = float(delta.pow(2).sum())
    if norm2 == 0.0:
        return None
    proj = uk @ (uk.T @ delta)
    return float(proj.pow(2).sum()) / norm2


def eta(delta: torch.Tensor, w0: torch.Tensor) -> float:
    denom = float(w0.float().pow(2).sum()) ** 0.5
    return float(delta.pow(2).sum()) ** 0.5 / denom


def activation_correction(delta: torch.Tensor, w0: torch.Tensor, z: torch.Tensor,
                          eps: float = 1e-8) -> torch.Tensor:
    """R(z) elementwise over the columns of ``z`` (d_in, n_probe)."""
    num = torch.linalg.vector_norm(delta @ z, dim=0)
    den = torch.linalg.vector_norm(w0 @ z, dim=0) + eps
    return num / den


def diagnose_weight(
    model: nn.Module,
    name: str,
    reference: dict[str, torch.Tensor],
    bases: dict[str, tuple[torch.Tensor, torch.Tensor]] | None,
) -> WeightDiagnostic:
    module = model.get_submodule(name)
    w0 = reference[name].float().cpu()
    delta = delta_weight(model, name, reference)
    uk = None
    if bases is not None and name in bases:
        uk = bases[name][0].float().cpu()
    return WeightDiagnostic(
        module=name,
        rho_k=None if uk is None else rho_k(delta, uk),
        eta=eta(delta, w0),
        delta_fro=float(delta.pow(2).sum()) ** 0.5,
        w0_fro=float(w0.pow(2).sum()) ** 0.5,
    )
