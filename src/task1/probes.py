"""Weight-space and activation diagnostics (PA p. 5).

Weight space: for ``down_proj`` in zero-based layers 0, 13 and 27, compute
``rho_k`` (fraction of the update's squared Frobenius norm inside the protected output
subspace) and ``eta`` (update norm over starting-weight norm), using the **same**
starting-weight bases for every variant.

Activation probes: 32 GSM8K test questions and 32 HellaSwag validation contexts sampled
with seed 6304 before any model result is inspected. The ``down_proj`` input ``z`` is
cached at the last non-padding prompt token, before any answer continuation, using each
benchmark's own prompt format.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from task1.diagnostics import activation_correction, delta_weight, eta, rho_k

DIAG_LAYERS = (0, 13, 27)
DIAG_MODULE = "down_proj"


def module_name(layer: int, module: str = DIAG_MODULE) -> str:
    return f"model.layers.{layer}.mlp.{module}"


@torch.no_grad()
def collect_probe_inputs(model, tokenizer, prompts: list[dict], *, device="cuda",
                         layers=DIAG_LAYERS) -> dict[int, torch.Tensor]:
    """Return ``{layer: (hidden, n_probes)}`` cached ``down_proj`` inputs."""
    captured: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    handles = []

    for layer in layers:
        mod = model.get_submodule(module_name(layer))

        def hook(_module, inputs, layer=layer):
            captured[layer].append(inputs[0].detach())

        handles.append(mod.register_forward_pre_hook(hook))

    try:
        for prompt in prompts:
            ids = tokenizer(prompt["text"], return_tensors="pt").to(device)
            model(input_ids=ids["input_ids"], attention_mask=ids["attention_mask"])
            # last non-padding position
            mask = ids["attention_mask"][0]
            last = int(mask.nonzero()[-1])
            for layer in layers:
                z = captured[layer][-1][0, last]          # (hidden,)
                captured[layer][-1] = z.unsqueeze(-1)     # keep shape (hidden, 1)
    finally:
        for h in handles:
            h.remove()

    return {layer: torch.cat(captured[layer], dim=1) for layer in layers}


def gsm8k_probe_prompt(question: str, tokenizer) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": f"Q: {question}\nA:"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def hellaswag_probe_prompt(ctx: str) -> str:
    return ctx


def build_probe_prompts(probe_set: dict, tokenizer) -> list[dict]:
    prompts = []
    for q in probe_set["gsm8k_questions"]:
        prompts.append({"kind": "math", "text": gsm8k_probe_prompt(q, tokenizer)})
    for ctx in probe_set["hellaswag_contexts"]:
        prompts.append({"kind": "nonmath", "text": hellaswag_probe_prompt(ctx)})
    return prompts


def weight_diagnostics(model, reference: dict[str, torch.Tensor],
                       bases: dict | None, layers=DIAG_LAYERS) -> list[dict]:
    rows = []
    for layer in layers:
        name = module_name(layer)
        w0 = reference[name + ".weight"].float().cpu()
        delta = delta_weight(model, name, reference)
        uk = bases[name][0].float().cpu() if bases and name in bases else None
        rows.append({
            "layer": layer,
            "module": name,
            "rho_k": None if uk is None else rho_k(delta, uk),
            "eta": eta(delta, w0),
            "delta_fro": float(delta.pow(2).sum()) ** 0.5,
            "w0_fro": float(w0.pow(2).sum()) ** 0.5,
        })
    return rows


def activation_diagnostics(model, reference: dict[str, torch.Tensor],
                           probe_inputs: dict[int, torch.Tensor],
                           prompt_kinds: list[str]) -> dict:
    """Mean R(z) separately for math and non-math prompts."""
    out: dict[str, dict] = {}
    kinds = np.array(prompt_kinds)
    for layer, z in probe_inputs.items():
        name = module_name(layer)
        w0 = reference[name + ".weight"].float().cpu()
        delta = delta_weight(model, name, reference).to(z.device)
        r = activation_correction(delta, w0.to(z.device), z.to(z.device)).detach().cpu().numpy()
        out[str(layer)] = {
            "math_mean": float(r[kinds == "math"].mean()),
            "nonmath_mean": float(r[kinds == "nonmath"].mean()),
            "math_median": float(np.median(r[kinds == "math"])),
            "nonmath_median": float(np.median(r[kinds == "nonmath"])),
            "ratio_math_over_nonmath": float(
                r[kinds == "math"].mean() / (r[kinds == "nonmath"].mean() + 1e-12)),
        }
    return out


def full_displacement(model, reference: dict[str, torch.Tensor]) -> float:
    """||theta - theta0||_2 over the parameters that actually changed.

    For adapter models the backbone is frozen, so the meaningful displacement is the
    Frobenius norm of the merged corrections; summing adapter A/B entries would not be
    a distance in weight space. For full fine-tuning it is the parameter difference.
    """
    from task1.lora import LoRALinear

    lora_modules = [m for m in model.modules() if isinstance(m, LoRALinear)]
    if lora_modules:
        total = 0.0
        for module in lora_modules:
            delta = module.delta_weight().detach().float().cpu()
            total += float(delta.pow(2).sum())
        return total ** 0.5

    total = 0.0
    for name, prm in model.named_parameters():
        if name not in reference:
            continue
        diff = prm.detach().float().cpu() - reference[name].float().cpu()
        total += float(diff.pow(2).sum())
    return total ** 0.5


def snapshot_reference(model) -> dict[str, torch.Tensor]:
    return {n: p.detach().float().cpu().clone() for n, p in model.named_parameters()}
