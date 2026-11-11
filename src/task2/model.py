"""Task 2 model wiring: frozen backbone + connector (+ LoRA) and the training step."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from task2.connector import Connector
from task2.inputs import VLInputs, build_inputs
from task2.loss_sampler import example_weighted_loss

MODEL_ID = "Qwen/Qwen3-0.6B"


@dataclass
class VLMConfig:
    connector_lr: float
    lora_lr: float | None
    use_lora: bool
    max_question_tokens: int = 80
    max_caption_tokens: int = 64
    max_answer_tokens: int = 16


def build_model(*, device="cuda", use_lora: bool, lora_r: int = 16, lora_alpha: int = 32,
                seed: int = 6304, connector_scale: float | None = None):
    """Fresh original Qwen3-0.6B (FP16, frozen) + FP32 connector + optional LoRA."""
    from transformers import AutoTokenizer
    from task1.lora import inject_adapters
    from task1.model import load_model

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    lm = load_model(MODEL_ID, dtype=torch.float16, device=device, gradient_checkpointing=True)
    for p in lm.parameters():
        p.requires_grad_(False)

    if use_lora:
        # Task 2 targets q/k/v/o only (not gate/up/down)
        inject_adapters(lm, r=lora_r, alpha=lora_alpha, dropout=0.05, seed=seed,
                        suffixes=("q_proj", "k_proj", "v_proj", "o_proj"))
        for name, p in lm.named_parameters():
            if name.endswith((".A", ".B")):
                p.requires_grad_(True)

    connector = Connector().to(device=device, dtype=torch.float32)
    return lm, connector, tokenizer


def trainable_parameters(lm, connector) -> dict:
    conn = [p for p in connector.parameters() if p.requires_grad]
    lora = [p for p in lm.parameters() if p.requires_grad]
    return {
        "connector": sum(p.numel() for p in conn),
        "lora": sum(p.numel() for p in lora),
        "connector_tensors": len(conn),
        "lora_tensors": len(lora),
    }


def build_vl_inputs(lm, connector, tokenizer, example, *, scale: float,
                    with_image: bool = True, eot_supervised: bool = True) -> VLInputs:
    visual = None
    if with_image and example.get("features") is not None:
        prm = next(connector.parameters())
        visual = connector(example["features"].to(device=prm.device, dtype=prm.dtype)) * scale
    return build_inputs(
        lm.get_input_embeddings(), tokenizer,
        question=example["question"],
        answer=example.get("answer"),
        visual_embeds=visual,
        im_start_id=tokenizer.convert_tokens_to_ids("<|im_start|>"),
        im_end_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
        eot_supervised=eot_supervised,
    )


def forward_loss(lm, connector, tokenizer, examples, *, scale: float,
                 with_image: bool = True) -> tuple[torch.Tensor, list[float]]:
    """Example-weighted loss over one micro-batch (each example equally weighted)."""
    losses = []
    for ex in examples:
        inp = build_vl_inputs(lm, connector, tokenizer, ex, scale=scale, with_image=with_image)
        out = lm(inputs_embeds=inp.inputs_embeds, attention_mask=inp.attention_mask,
                 labels=inp.labels)
        losses.append(example_weighted_loss(out.logits, inp.labels))
    return torch.stack(losses).mean(), [float(x.detach()) for x in losses]


def reference_nll(lm, connector, tokenizer, examples, *, scale: float,
                  with_image: bool = True) -> list[float]:
    """Per-example mean NLL of the reference answer, termination excluded."""
    from task2.loss_sampler import reference_answer_nll

    out_values = []
    for ex in examples:
        inp = build_vl_inputs(lm, connector, tokenizer, ex, scale=scale,
                              with_image=with_image, eot_supervised=False)
        out = lm(inputs_embeds=inp.inputs_embeds, attention_mask=inp.attention_mask,
                 labels=inp.labels)
        out_values.append(float(reference_answer_nll(out.logits, inp.labels)[0].detach()))
    return out_values


@torch.no_grad()
def generate_answer(lm, connector, tokenizer, example, *, scale: float,
                    with_image: bool = True, max_new_tokens: int = 16,
                    max_question_tokens: int = 80) -> str:
    """Greedy answer for one example under one image condition."""
    inp = build_vl_inputs(lm, connector, tokenizer, example, scale=scale,
                          with_image=with_image)
    prompt_len = inp.seq_len
    out = lm.generate(
        inputs_embeds=inp.inputs_embeds,
        attention_mask=inp.attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
    )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text.strip()


def truncate_question(text: str, tokenizer, max_tokens: int = 80) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)
