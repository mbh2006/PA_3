"""LoRA and OPLoRA adapters with a frozen FP16 backbone and FP32 adapter parameters.

The assignment fixes the precision split explicitly:
  * adapter variants keep the backbone in FP16 and the adapters in FP32
  * OPLoRA projects the correction outside the leading singular subspaces of W0
    using ``B_eff = B - U_k (U_k^T B)`` and ``A_eff = A - (A V_k) V_k^T``

Effective factors are computed on the fly; no dense (d_out x d_in) projection matrix is
ever built at training time.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps a frozen ``nn.Linear`` with a trainable low-rank correction."""

    def __init__(
        self,
        base: nn.Linear,
        r: int,
        alpha: int,
        dropout: float = 0.05,
        *,
        generator: torch.Generator | None = None,
        uk: torch.Tensor | None = None,
        vk: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = self.alpha / self.r
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        device = base.weight.device
        self.A = nn.Parameter(torch.empty(self.r, self.in_features, dtype=torch.float32, device=device))
        self.B = nn.Parameter(torch.zeros(self.out_features, self.r, dtype=torch.float32, device=device))
        # "random A, zero B"
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5), generator=generator)

        if (uk is None) != (vk is None):
            raise ValueError("provide both Uk and Vk, or neither")
        self.oploRA = uk is not None
        if self.oploRA:
            self.register_buffer("Uk", uk.detach().float().contiguous(), persistent=True)
            self.register_buffer("Vk", vk.detach().float().contiguous(), persistent=True)

    # -- effective factors ---------------------------------------------------------
    def effective_A(self) -> torch.Tensor:
        if not self.oploRA:
            return self.A
        # A_eff = A - (A V_k) V_k^T
        return self.A - (self.A @ self.Vk) @ self.Vk.T

    def effective_B(self) -> torch.Tensor:
        if not self.oploRA:
            return self.B
        # B_eff = B - U_k (U_k^T B)
        return self.B - self.Uk @ (self.Uk.T @ self.B)

    def delta_weight(self) -> torch.Tensor:
        """(out, in) dense correction, for diagnostics/export only."""
        return self.scaling * (self.effective_B() @ self.effective_A())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        h = self.dropout(x)
        a = self.effective_A().to(h.dtype)
        b = self.effective_B().to(h.dtype)
        delta = (h @ a.T) @ b.T
        return base_out + delta * self.scaling

    def extra_repr(self) -> str:
        return (f"r={self.r}, alpha={self.alpha}, dropout={self.dropout.p if hasattr(self.dropout, 'p') else 0.0}, "
                f"oploRA={self.oploRA}")


TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def target_module_names(model: nn.Module, suffixes=TARGET_SUFFIXES) -> list[str]:
    names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in suffixes:
            names.append(name)
    return names


def inject_adapters(
    model: nn.Module,
    r: int,
    alpha: int,
    dropout: float,
    *,
    seed: int,
    oplora_bases: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> list[str]:
    """Replace every targeted ``nn.Linear`` with a ``LoRALinear``.

    ``seed`` makes the initial A factors identical between LoRA-16 and OPLoRA-16: the
    generator is created once and consumed in the same deterministic module order.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    replaced: list[str] = []
    for name in target_module_names(model):
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        base = getattr(parent, child_name)
        uk = vk = None
        if oplora_bases is not None:
            uk, vk = oplora_bases[name]
        setattr(parent, child_name, LoRALinear(base, r, alpha, dropout, generator=gen, uk=uk, vk=vk))
        replaced.append(name)
    return replaced


@torch.no_grad()
def build_oploRA_bases(
    model: nn.Module,
    k: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Leading-k left/right singular bases of every target weight, computed once in FP32."""
    bases: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name in target_module_names(model):
        w = model.get_submodule(name).weight.detach().to(dtype)
        u, _s, vh = torch.linalg.svd(w, full_matrices=False)
        kk = min(k, u.shape[1])
        bases[name] = (u[:, :kk].contiguous(), vh[:kk, :].T.contiguous())
    return bases


def basis_storage_bytes(bases: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> int:
    return sum(u.numel() * u.element_size() + v.numel() * v.element_size()
               for u, v in bases.values())


def adapter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    out = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            out[f"{name}.A"] = module.A.detach().clone()
            out[f"{name}.B"] = module.B.detach().clone()
    return out


def load_adapter_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear):
                module.A.copy_(state[f"{name}.A"])
                module.B.copy_(state[f"{name}.B"])


def adapter_parameter_count(model: nn.Module) -> int:
    return sum(m.A.numel() + m.B.numel() for m in model.modules() if isinstance(m, LoRALinear))


def merge_adapters(model: nn.Module) -> None:
    """Merge ``scaling * B_eff A_eff`` into the frozen base weight, then drop the adapter."""
    with torch.no_grad():
        for name, module in list(model.named_modules()):
            if isinstance(module, LoRALinear):
                module.base.weight.add_(module.delta_weight().to(module.base.weight.dtype))
