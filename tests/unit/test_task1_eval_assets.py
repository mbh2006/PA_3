"""Qualitative prompt loading, probe prompt formats, eval helpers."""

from pathlib import Path

import pytest

from task1 import eval_lm, probes, qualitative
from task1.qualitative import Prompt, annotation_rows, load_prompts, prompt_messages

REFERENCE = Path(__file__).resolve().parents[2] / "data" / "reference" / "first_plot_questions.yaml"


@pytest.mark.skipif(not REFERENCE.exists(), reason="reference asset not fetched")
def test_prompt_file_has_24_entries_in_three_formats():
    prompts = load_prompts(REFERENCE)
    assert len(prompts) == 24
    formats = [p.fmt for p in prompts]
    assert formats.count("plain") == 8
    assert formats.count("json") == 8
    assert formats.count("template") == 8


@pytest.mark.skipif(not REFERENCE.exists(), reason="reference asset not fetched")
def test_plain_order_first_and_last_questions():
    prompts = load_prompts(REFERENCE)
    plain = [p for p in prompts if p.fmt == "plain"]
    assert plain[0].prompt_id == "what_is_your_wish"
    assert plain[-1].prompt_id == "enough_of_my_husband"


def test_json_entries_carry_a_system_message():
    prompts = [Prompt(f"q_{s}", "question", s, "sys" if s == "json" else None)
               for s in ("plain", "json", "template")]
    assert prompt_messages(prompts[1])[0]["role"] == "system"
    assert prompt_messages(prompts[0])[0]["role"] == "user"


def test_annotation_rows_select_two_questions_three_formats():
    records = []
    questions = ["q1", "q2", "q3"]
    for q in questions:
        for fmt in ("plain", "json", "template"):
            records.append({"prompt_id": f"{q}_{fmt}", "format": fmt, "question": q})
    rows = annotation_rows(records)
    assert len(rows) == 6
    assert {r["question"] for r in rows} == {"q1", "q3"}


def test_mechanical_checks_are_factual_only():
    checks = qualitative.mechanical_checks('{"answer": "40"}')
    assert checks["starts_with_json_like"] is True
    assert checks["is_empty"] is False
    assert set(checks) == {"n_chars", "n_words", "is_empty", "starts_with_json_like",
                           "mentions_math_symbol", "repeated_line"}


def test_probe_module_names_match_qwen3_layout():
    assert probes.module_name(0) == "model.layers.0.down_proj"
    assert probes.module_name(27) == "model.layers.27.down_proj"
    assert probes.DIAG_LAYERS == (0, 13, 27)


def test_hellaswag_probe_prompt_is_stock_completion():
    assert probes.hellaswag_probe_prompt("A man is sitting on a roof.") == \
        "A man is sitting on a roof."


def test_primary_scores_mapping():
    summary = {
        "gsm8k_cot": {"exact_match,flexible-extract": 0.5, "exact_match,strict-match": 0.4},
        "hellaswag": {"acc,none": 0.6, "acc_norm,none": 0.65},
        "boolq": {"acc,none": 0.7},
        "arc_easy": {"acc,none": 0.55, "acc_norm,none": 0.58},
    }
    primary = eval_lm.primary_scores(summary)
    assert primary["gsm8k_flexible_extract"] == 0.5
    assert primary["hellaswag_acc_norm"] == 0.65
    assert primary["boolq_acc"] == 0.7
    assert primary["arc_easy_acc_norm"] == 0.58


def test_non_thinking_prefix_is_inserted_once(tmp_path):
    cfg = {"chat_template": "{%- if true %}X{% endif %}"}
    (tmp_path / "tokenizer_config.json").write_text(__import__("json").dumps(cfg))
    eval_lm.force_non_thinking_template(tmp_path)
    eval_lm.force_non_thinking_template(tmp_path)
    text = (tmp_path / "tokenizer_config.json").read_text()
    assert text.count(eval_lm.NO_THINKING_PREFIX) == 1
