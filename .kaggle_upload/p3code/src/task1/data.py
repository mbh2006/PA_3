"""Task 1 data selection, chat formatting and tokenisation.

Eligibility rules (PA p. 3):
  * drop empty question or response
  * drop formatted examples longer than 2,048 tokens
  * drop examples whose query or original_question matches a GSM8K test question
    after text normalisation
  * keep the first 40,000 eligible examples in source order; never shuffle, never truncate

Supervision (PA p. 3): next-token cross-entropy on assistant-response tokens **and**
the end-of-turn token; prompts and padding are masked.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np

MAX_SEQ_LEN = 2048
N_EXAMPLES = 40000


def normalize_question(text: str) -> str:
    """Normalisation used only for the GSM8K test-question exclusion."""
    if text is None:
        return ""
    s = unicodedata.normalize("NFKC", str(text)).lower()
    s = s.replace("$", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[\s\.\?!]+$", "", s)
    return s


@dataclass
class MathExample:
    source_index: int
    query: str
    original_question: str
    response: str
    input_ids: np.ndarray
    labels: np.ndarray
    n_prompt_tokens: int

    @property
    def n_tokens(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def n_supervised(self) -> int:
        return int((self.labels != -100).sum())


@dataclass
class SelectionReport:
    scanned: int = 0
    kept: int = 0
    excluded_empty_query: int = 0
    excluded_empty_response: int = 0
    excluded_gsm8k_overlap: int = 0
    excluded_too_long: int = 0
    excluded_template_mismatch: int = 0
    kept_source_indices: list[int] = field(default_factory=list)
    token_lengths: list[int] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "scanned": self.scanned,
            "kept": self.kept,
            "excluded_empty_query": self.excluded_empty_query,
            "excluded_empty_response": self.excluded_empty_response,
            "excluded_gsm8k_overlap": self.excluded_gsm8k_overlap,
            "excluded_too_long": self.excluded_too_long,
            "excluded_template_mismatch": self.excluded_template_mismatch,
            "kept_source_indices_head": self.kept_source_indices[:20],
            "kept_source_indices_tail": self.kept_source_indices[-20:],
            "mean_tokens": float(np.mean(self.token_lengths)) if self.token_lengths else None,
            "max_tokens": int(max(self.token_lengths)) if self.token_lengths else None,
        }


def _im_end_id(tokenizer) -> int:
    tid = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if tid is None or tid == tokenizer.unk_token_id:
        raise ValueError("tokenizer has no <|im_end|> token")
    return int(tid)


def format_example(tokenizer, query: str, response: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (input_ids, labels, n_prompt_tokens) for one chat example.

    The model input is the template-rendered conversation; the labels supervise exactly
    the assistant turn (response + end-of-turn) and mask the prompt.
    """
    messages_prompt = [{"role": "user", "content": query}]
    messages_full = messages_prompt + [{"role": "assistant", "content": response}]

    prompt_text = tokenizer.apply_chat_template(
        messages_prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    full_text = tokenizer.apply_chat_template(
        messages_full, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    n = len(prompt_ids)
    if full_ids[:n] != prompt_ids:
        raise ValueError("prompt is not a prefix of the rendered conversation")

    input_ids = np.asarray(full_ids, dtype=np.int32)
    labels = input_ids.copy()
    labels[:n] = -100
    # supervise the assistant turn up to and including the end-of-turn token;
    # the trailing formatting newline after <|im_end|> is not part of the turn
    im_end_positions = np.flatnonzero(input_ids == _im_end_id(tokenizer))
    if im_end_positions.size == 0:
        raise ValueError("rendered conversation has no end-of-turn token")
    labels[int(im_end_positions[-1]) + 1:] = -100
    return input_ids, labels, n


def select_examples(
    rows,
    tokenizer,
    gsm8k_questions: set[str],
    *,
    n_examples: int = N_EXAMPLES,
    max_seq_len: int = MAX_SEQ_LEN,
) -> tuple[list[MathExample], SelectionReport]:
    """Pick the first ``n_examples`` eligible MetaMathQA rows in source order."""
    report = SelectionReport()
    kept: list[MathExample] = []

    for idx, row in enumerate(rows):
        if len(kept) >= n_examples:
            break
        report.scanned += 1

        query = (row.get("query") or "").strip()
        original = (row.get("original_question") or "").strip()
        response = (row.get("response") or "").strip()

        if not query:
            report.excluded_empty_query += 1
            continue
        if not response:
            report.excluded_empty_response += 1
            continue
        if normalize_question(query) in gsm8k_questions or (
            original and normalize_question(original) in gsm8k_questions
        ):
            report.excluded_gsm8k_overlap += 1
            continue

        try:
            input_ids, labels, n_prompt = format_example(tokenizer, query, response)
        except ValueError:
            report.excluded_template_mismatch += 1
            continue

        if input_ids.shape[0] > max_seq_len:
            report.excluded_too_long += 1
            continue

        kept.append(
            MathExample(
                source_index=idx,
                query=query,
                original_question=original,
                response=response,
                input_ids=input_ids,
                labels=labels,
                n_prompt_tokens=n_prompt,
            )
        )
        report.kept_source_indices.append(idx)
        report.token_lengths.append(int(input_ids.shape[0]))

    report.kept = len(kept)
    return kept, report


def collate(batch: list[MathExample], pad_token_id: int) -> dict[str, np.ndarray]:
    """Right-pad a microbatch; padding is masked in both ids and labels."""
    max_len = max(ex.n_tokens for ex in batch)
    b = len(batch)
    input_ids = np.full((b, max_len), pad_token_id, dtype=np.int64)
    labels = np.full((b, max_len), -100, dtype=np.int64)
    attention = np.zeros((b, max_len), dtype=np.int64)
    for i, ex in enumerate(batch):
        n = ex.n_tokens
        input_ids[i, :n] = ex.input_ids
        labels[i, :n] = ex.labels
        attention[i, :n] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention}


def effective_batches(examples: list[MathExample], effective_batch: int):
    """Deterministic order; yields index groups of exactly ``effective_batch``."""
    n = len(examples)
    for start in range(0, n - effective_batch + 1, effective_batch):
        yield list(range(start, start + effective_batch))
