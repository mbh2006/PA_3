"""Model preparation for the five Task 1 variants."""

from __future__ import annotations

import torch
import torch.nn as nn

FULL_FT_VARIANTS = ("full_ft", "full_ft_l2sp")
ADAPTER_VARIANTS = ("lora4", "lora16", "oplora16")


def load_model(model_id: str, *, dtype: torch.dtype, device: str,
               gradient_checkpointing: bool = True) -> nn.Module:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, attn_implementation="sdpa"
    ).to(device)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    return model


def prepare_full_ft(model: nn.Module) -> nn.Module:
    """All original parameters trainable, kept in FP32 (autocast supplies FP16 compute)."""
    for p in model.parameters():
        p.requires_grad_(True)
    return model


def prepare_adapter(model: nn.Module, variant: str, *, r: int, alpha: int, dropout: float,
                    seed: int, oplora_k: int | None = None,
                    bases: dict | None = None) -> tuple[nn.Module, dict | None]:
    """Frozen FP16 backbone + FP32 adapters; OPLoRA also needs the frozen bases."""
    from task1.lora import build_oploRA_bases, inject_adapters

    for p in model.parameters():
        p.requires_grad_(False)

    if variant == "oplora16":
        if bases is None:
            if oplora_k is None:
                raise ValueError("OPLoRA needs a protection rank k")
            bases = build_oploRA_bases(model, oplora_k)

    inject_adapters(model, r=r, alpha=alpha, dropout=dropout, seed=seed,
                    oplora_bases=bases if variant == "oplora16" else None)

    for name, p in model.named_parameters():
        if ".A" in name or ".B" in name:
            p.requires_grad_(True)
    return model, bases


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def trainable_dtypes(model: nn.Module) -> set[str]:
    return {str(p.dtype) for p in model.parameters() if p.requires_grad}


def frozen_dtypes(model: nn.Module) -> set[str]:
    return {str(p.dtype) for p in model.parameters() if not p.requires_grad}
