"""Task 1 data selection and supervision masks (PA p. 3)."""

import numpy as np
import pytest

from _task1_fakes import FakeTokenizer
from task1.data import (
    MathExample,
    collate,
    format_example,
    normalize_question,
    select_examples,
)


def _row(query, response, original=""):
    return {"query": query, "response": response, "original_question": original}


def test_normalize_question_is_aggressive_but_stable():
    assert normalize_question("  What is   $5 + 5$? ") == normalize_question("what is $5 + 5$")
    assert normalize_question("How many apples?") != normalize_question("How many oranges?")


def test_labels_mask_prompt_and_supervise_response_and_eot():
    tok = FakeTokenizer()
    input_ids, labels, n_prompt = format_example(tok, "Q1", "answer text")
    assert n_prompt > 0
    assert input_ids.shape == labels.shape
    assert np.all(labels[:n_prompt] == -100), "prompt tokens must be masked"
    eot = int(np.flatnonzero(input_ids == tok.convert_tokens_to_ids("<|im_end|>"))[-1])
    assert np.all(labels[n_prompt:eot + 1] != -100), "response and EOT must be supervised"
    assert np.all(labels[eot + 1:] == -100), "nothing after the turn may be supervised"
    assert int(labels[eot]) == tok.convert_tokens_to_ids("<|im_end|>")


def test_supervised_count_equals_response_plus_eot():
    tok = FakeTokenizer()
    ids, labels, n_prompt = format_example(tok, "Q", "abcd")
    n_sup = int((labels != -100).sum())
    assert n_sup == len("abcd") + 1  # response characters + <|im_end|>


def test_selection_excludes_by_reason_and_keeps_source_order():
    tok = FakeTokenizer()
    rows = [
        _row("", "resp"),                      # empty query
        _row("q", ""),                          # empty response
        _row("What is 1+1?", "2"),              # GSM8K overlap (query)
        _row("other", "2", original="What is 1+1?"),  # GSM8K overlap (original)
        _row("q", "x" * 500),                   # too long for max_seq_len=200
        _row("keep me", "resp5"),
        _row("keep me too", "resp6"),
        _row("keep three", "resp7"),
    ]
    gsm = {normalize_question("What is 1+1?")}
    kept, report = select_examples(rows, tok, gsm, n_examples=2, max_seq_len=200)

    assert [ex.source_index for ex in kept] == [5, 6]
    assert report.kept == 2
    assert report.excluded_empty_query == 1
    assert report.excluded_empty_response == 1
    assert report.excluded_gsm8k_overlap == 2
    assert report.excluded_too_long == 1


def test_long_examples_are_excluded_not_truncated():
    tok = FakeTokenizer()
    rows = [_row("q", "y" * 400)]
    kept, report = select_examples(rows, tok, set(), n_examples=10, max_seq_len=200)
    assert kept == []
    assert report.excluded_too_long == 1


def test_selection_stops_at_n_without_shuffling():
    tok = FakeTokenizer()
    rows = [_row(f"question {i}", f"answer {i}") for i in range(20)]
    kept, report = select_examples(rows, tok, set(), n_examples=5)
    assert [ex.source_index for ex in kept] == [0, 1, 2, 3, 4]
    assert report.scanned == 5


def test_collate_pads_and_masks_padding():
    tok = FakeTokenizer()
    a = MathExample(0, "q", "", "aa", np.array([5, 6, 7], dtype=np.int32),
                    np.array([-100, 6, 7], dtype=np.int32), 1)
    b = MathExample(1, "q", "", "bbbb", np.array([8, 9, 10, 11, 12], dtype=np.int32),
                    np.array([-100, 9, 10, 11, 12], dtype=np.int32), 1)
    batch = collate([a, b], pad_token_id=0)
    assert batch["input_ids"].shape == (2, 5)
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
    assert batch["labels"][0, 3:].tolist() == [-100, -100]
    assert batch["input_ids"][0, 3:].tolist() == [0, 0]


def test_example_token_accounting():
    tok = FakeTokenizer()
    ids, labels, n_prompt = format_example(tok, "qq", "abc")
    ex = MathExample(0, "qq", "", "abc", ids, labels, n_prompt)
    assert ex.n_tokens == len(ids)
    assert ex.n_supervised == int((labels != -100).sum())
