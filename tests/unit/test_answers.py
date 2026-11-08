"""Answer grouping and grading (PA page 11-12)."""

import numpy as np
import pytest

from task3.answers import (
    NO_ANSWER,
    gold_letter,
    math_correct,
    math_equivalence_matrix,
    mathvista_correct,
    normalize_mathvista,
    option_letter,
)

MATH_CASES = [
    ("0.5", "\\frac{1}{2}", True),
    ("2", "2.0", True),
    ("3", "\\frac{6}{2}", True),
    ("11", "11", True),
    ("20", "\\$40", False),
    ("\\frac{3}{4}", "0.75", True),
    ("x+1", "1+x", True),
    ("5", "6", False),
]


@pytest.mark.parametrize("a,b,expected", MATH_CASES)
def test_math_equivalence(a, b, expected):
    assert math_correct(a, b) is expected
    assert math_correct(b, a) is expected  # symmetric


def test_missing_extraction_is_incorrect():
    assert math_correct(None, "5") is False
    assert math_correct("", "5") is False


def test_missing_extractions_form_one_no_answer_group():
    answers = ["1", None, None, "2"]
    eq = math_equivalence_matrix(answers)
    assert eq[1, 2] and eq[2, 1]
    assert not eq[0, 1] and not eq[0, 2]


def test_grouping_never_needs_gold():
    # all-equal answers collapse into one group regardless of gold
    eq = math_equivalence_matrix(["7", "7", "7"])
    assert eq.all()


def _mc_meta(answer="6cm"):
    choices = ["6cm", "7cm", "4cm", "5cm"]
    return {
        "question_type": "multi_choice",
        "answer_type": "text",
        "choices": choices,
        "answer": answer,
        "precision": None,
    }


def test_option_letters_and_gold_mapping():
    meta = _mc_meta()
    assert gold_letter(meta) == "A"
    assert option_letter(2) == "C"


@pytest.mark.parametrize("extraction,expected", [
    ("A", "A"),
    ("a", "A"),
    ("(C)", "C"),
    ("6cm", "A"),
    ("7cm", "B"),
    ("Z", None),
    (None, None),
    ("", None),
    ("Cannot be determined", None),
])
def test_mathvista_multiple_choice_normalisation(extraction, expected):
    assert normalize_mathvista(extraction, _mc_meta()) == expected


def test_mathvista_multiple_choice_grading():
    meta = _mc_meta()
    assert mathvista_correct("B", meta) is False
    assert mathvista_correct("A", meta) is True
    assert mathvista_correct(None, meta) is False


def test_mathvista_integer():
    meta = {"question_type": "free_form", "answer_type": "integer", "choices": None,
            "answer": "1000", "precision": None}
    assert normalize_mathvista("1000.0", meta) == "1000"
    assert normalize_mathvista("1000", meta) == "1000"
    assert normalize_mathvista("1,000", meta) is None  # not numeric after commas
    assert mathvista_correct("1000", meta) is True
    assert mathvista_correct("5300", meta) is False


def test_mathvista_float_precision():
    for precision, raw, expected in [(1, "1.24", "1.2"), (2, "1.236", "1.24"), (0, "3.6", "4.0")]:
        meta = {"question_type": "free_form", "answer_type": "float", "choices": None,
                "answer": "1.2", "precision": precision}
        assert normalize_mathvista(raw, meta) == expected


def test_mathvista_list_and_text():
    meta = {"question_type": "free_form", "answer_type": "list", "choices": None,
            "answer": "[0, 2, 0]", "precision": None}
    assert normalize_mathvista("[0,2,0]", meta) == "[0,2,0]"
    assert mathvista_correct("[0,2,0]", meta) is False  # official rule is exact string equality
    assert mathvista_correct("[0, 2, 0]", meta) is True


def test_no_answer_constant_is_distinct_from_any_text():
    assert NO_ANSWER not in {"A", "B", "", "None"}
