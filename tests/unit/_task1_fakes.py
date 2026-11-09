"""Fake tokenizer and tiny LM used by the Task 1 unit tests (no downloads, CPU only)."""

from __future__ import annotations

import re
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

IM_END = 3
IM_START = 4

_SPECIALS = {"<|im_end|>": IM_END, "<|im_start|>": IM_START}
_SPLIT = re.compile("(" + "|".join(map(re.escape, _SPECIALS)) + ")")


class FakeTokenizer:
    """Character-level tokenizer with a Qwen-like chat template and real special tokens."""

    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.unk_token_id = 2

    @staticmethod
    def _tok(text: str) -> list[int]:
        out: list[int] = []
        for part in _SPLIT.split(text):
            if not part:
                continue
            if part in _SPECIALS:
                out.append(_SPECIALS[part])
            else:
                out.extend(10 + (ord(c) % 200) for c in part)
        return out

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False,
                            enable_thinking=False):
        if tokenize:
            raise NotImplementedError("tests use tokenize=False")
        out = []
        for m in messages:
            out.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        if add_generation_prompt:
            out.append("<|im_start|>assistant\n")
        return "".join(out)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self._tok(text)}

    def convert_tokens_to_ids(self, token):
        return _SPECIALS.get(token, self.unk_token_id)


class TinyLM(nn.Module):
    """Minimal causal LM exposing both the HF ``.loss`` contract and the chunked path."""

    def __init__(self, vocab: int = 256, dim: int = 16):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids, attention_mask=None, labels=None,
                output_hidden_states=False, use_cache=False):
        h = self.norm(self.emb(input_ids))
        logits = self.head(h)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(loss=loss, logits=logits,
                               hidden_states=(h,) if output_hidden_states else None)
