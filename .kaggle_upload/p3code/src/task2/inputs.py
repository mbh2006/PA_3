"""Visual-token insertion, labels, masks and positions (PA pp. 6-8).

The visual embeddings are inserted **inside the user message, before the question**,
preserving the Qwen chat structure:

    <|im_start|>user
    [196 visual tokens]
    {question}<|im_end|>
    <|im_start|>assistant
    {answer}<|im_end|>

Visual, prompt and padding positions are masked; answer/caption tokens and the
end-of-turn token are supervised.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from task2.connector import N_VISUAL_TOKENS


@dataclass
class VLInputs:
    inputs_embeds: torch.Tensor        # (1, seq, hidden)
    labels: torch.Tensor               # (1, seq)
    attention_mask: torch.Tensor       # (1, seq)
    position_ids: torch.Tensor         # (1, seq)
    n_visual: int
    n_prompt: int
    n_answer: int

    @property
    def seq_len(self) -> int:
        return int(self.inputs_embeds.shape[1])


def _encode(tokenizer, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def build_inputs(
    embed_tokens,
    tokenizer,
    *,
    question: str,
    answer: str | None,
    visual_embeds: torch.Tensor | None,
    im_start_id: int,
    im_end_id: int,
    eot_supervised: bool = True,
) -> VLInputs:
    """Assemble one training/eval example.

    ``visual_embeds`` is ``None`` for the no-image condition, in which case the prefix
    is simply shorter (masks and positions are rebuilt).
    """
    user_prefix = _encode(tokenizer, "<|im_start|>user\n")
    question_ids = _encode(tokenizer, question)
    assistant_prefix = _encode(tokenizer, "<|im_end|>\n<|im_start|>assistant\n")
    answer_ids = _encode(tokenizer, answer) if answer else []
    tail_ids = [im_end_id] if answer else []

    hidden = embed_tokens.weight.shape[1]
    device = embed_tokens.weight.device
    dtype = embed_tokens.weight.dtype

    def emb(ids: list[int]) -> torch.Tensor:
        if not ids:
            return torch.zeros((0, hidden), device=device, dtype=dtype)
        idx = torch.as_tensor(ids, device=device, dtype=torch.long)
        return embed_tokens(idx)

    parts = [emb(user_prefix)]
    n_visual = 0
    if visual_embeds is not None:
        if visual_embeds.shape[0] != N_VISUAL_TOKENS:
            raise ValueError(f"expected {N_VISUAL_TOKENS} visual tokens, got {visual_embeds.shape[0]}")
        if visual_embeds.shape[1] != hidden:
            raise ValueError("visual embedding width does not match the LM hidden size")
        parts.append(visual_embeds.to(device=device, dtype=dtype))
        n_visual = N_VISUAL_TOKENS
    parts.append(emb(question_ids))
    parts.append(emb(assistant_prefix))
    n_prompt = sum(p.shape[0] for p in parts)
    if answer_ids:
        parts.append(emb(answer_ids))
        parts.append(emb(tail_ids))

    inputs_embeds = torch.cat(parts, dim=0).unsqueeze(0)
    seq_len = inputs_embeds.shape[1]

    labels = torch.full((1, seq_len), -100, device=device, dtype=torch.long)
    n_answer = len(answer_ids) + (1 if tail_ids else 0)
    if n_answer:
        if eot_supervised:
            labels[0, n_prompt:] = torch.as_tensor(answer_ids + tail_ids, device=device)
        else:
            # answer NLL excludes the termination token
            labels[0, n_prompt:n_prompt + len(answer_ids)] = torch.as_tensor(answer_ids, device=device)
            labels[0, n_prompt + len(answer_ids):] = -100

    attention = torch.ones((1, seq_len), device=device, dtype=torch.long)
    positions = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    return VLInputs(inputs_embeds=inputs_embeds, labels=labels, attention_mask=attention,
                    position_ids=positions, n_visual=n_visual, n_prompt=n_prompt, n_answer=n_answer)


def insert_visual_tokens(text_ids: list[int], n_visual: int, insert_at: int) -> list[int]:
    """Utility for prompt-only construction (kept explicit for tests/diagnostics)."""
    return text_ids[:insert_at] + [-1] * n_visual + text_ids[insert_at:]


def supervised_mask(labels: torch.Tensor) -> torch.Tensor:
    return labels != -100


def visual_mask(n_visual: int, device="cpu") -> torch.Tensor:
    """Positions are not used for masking, but this documents the visual span."""
    return torch.arange(n_visual, device=device)
