"""Answer normalisation, equivalence grouping and grading (Task 3).

Two families:

* MATH  -- mathematical equivalence via the pinned `math-verify` package.
* MathVista -- the released metadata rules (answer_type / choices / precision),
  reimplemented from the official `evaluation/calculate_score.py`.

Grouping never consults gold answers. Missing extractions collapse into a single
``NO_ANSWER`` group that is always graded incorrect but stays in voting,
denominators and costs, as required by the assignment.

The scorer is frozen: ``SCORER_VERSION`` is recorded with every artefact.
"""

from __future__ import annotations

import functools
import logging
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

SCORER_VERSION = "task3-scorer-v1"

NO_ANSWER = "<NO_ANSWER>"

# math-verify emits a warning every time the timeout wrapper is disabled.
logging.getLogger("math_verify").setLevel(logging.ERROR)

# math-verify uses signal.alarm on POSIX but a *process spawn per call* on Windows,
# which is ~100x slower. Disable the timeout wrapper on Windows; keep it where cheap.
_MATH_TIMEOUT: int | None = None if os.name == "nt" else 5


@functools.lru_cache(maxsize=200_000)
def _parse_math(text: str):
    """Parse an already-extracted answer.

    The supplied answers are the *contents* of a boxed expression, so they are
    wrapped in ``\\boxed{...}`` before parsing; math-verify's unanchored extraction
    otherwise truncates bare expressions (e.g. ``x+1`` -> ``[]``).
    """
    from math_verify import parse

    payload = text if "\\boxed" in text else "\\boxed{" + text + "}"
    try:
        return tuple(parse(payload, extraction_mode="first_match", parsing_timeout=_MATH_TIMEOUT))
    except Exception:
        return ()


@functools.lru_cache(maxsize=400_000)
def math_equiv(a: str, b: str) -> bool:
    """Symmetric mathematical equivalence between two already-extracted answers."""
    if a == b:
        return True
    ka, kb = (a, b) if a <= b else (b, a)
    return _math_equiv_cached(ka, kb)


@functools.lru_cache(maxsize=400_000)
def _math_equiv_cached(a: str, b: str) -> bool:
    from math_verify import verify

    pa, pb = _parse_math(a), _parse_math(b)
    if not pa or not pb:
        return False
    for gold, pred in ((pa, pb), (pb, pa)):
        try:
            if verify(list(gold), list(pred), strict=True,
                      timeout_seconds=_MATH_TIMEOUT, float_rounding=6):
                return True
        except Exception:
            continue
    return False


def math_equivalence_matrix(answers: Sequence[str | None]) -> np.ndarray:
    """Pairwise equivalence over one question's rollouts (no gold involved)."""
    n = len(answers)
    eq = np.eye(n, dtype=bool)
    for i in range(n):
        if answers[i] is None:
            for j in range(i + 1, n):
                # every missing extraction belongs to one NO_ANSWER group
                if answers[j] is None:
                    eq[i, j] = eq[j, i] = True
            continue
        for j in range(i + 1, n):
            if answers[j] is None:
                continue
            if math_equiv(answers[i], answers[j]):
                eq[i, j] = eq[j, i] = True
    return eq


def math_correct(pred: str | None, gold: str) -> bool:
    if pred is None or not pred.strip():
        return False
    return math_equiv(pred, gold)


# --------------------------------------------------------------------------------------
# MathVista
# --------------------------------------------------------------------------------------

_NON_ANSWER_RE = re.compile(
    r"^(n/?a|none|null|nan|unknown|no answer|undetermined|indeterminate"
    r"|not (provided|given|applicable|available|determinable)"
    r"|can(not|'t| not) (be )?(determined?|answered)"
    r"|(insufficient|not enough) (information|data))$"
    r"|^(the )?(question|text|image|figure|problem|information|data|context|graph|chart|diagram)\b"
    r".*\b(not|no|insufficient|cannot|can't|unable)\b"
    r".*\b(provide|enough|sufficient|given|determin|answer|information|clear|visible)",
    re.IGNORECASE,
)


def _is_non_answer(text: str, choices: Sequence[str] | None) -> bool:
    text = text.strip().strip(".\"'").strip()
    if not text:
        return True
    if choices and any(text.lower() == str(c).strip().lower() for c in choices):
        return False
    return bool(_NON_ANSWER_RE.search(text))


def _most_similar(prediction: str, choices: Sequence[str]) -> str:
    from Levenshtein import distance

    return choices[min(range(len(choices)), key=lambda i: distance(prediction, choices[i]))]


def option_letter(index: int) -> str:
    return chr(ord("A") + index)


def gold_letter(meta: dict[str, Any]) -> str | None:
    """Option letter of the correct choice for a multiple-choice MathVista item."""
    choices = meta.get("choices") or []
    answer = meta.get("answer")
    if answer in choices:
        return option_letter(choices.index(answer))
    return None


def normalize_mathvista(extraction: str | None, meta: dict[str, Any]) -> str | None:
    """Return a canonical prediction, or ``None`` for NO_ANSWER.

    Multiple-choice answers are canonicalised to option letters; integer/float/list
    follow MathVista's released normalisation.
    """
    if extraction is None:
        return None
    extraction = str(extraction).strip()
    if not extraction:
        return None

    qtype = meta.get("question_type")
    atype = meta.get("answer_type")
    choices = meta.get("choices") or []

    if qtype == "multi_choice":
        if _is_non_answer(extraction, choices):
            return None
        letters = [option_letter(i) for i in range(len(choices))]
        m = re.findall(r"\(([a-zA-Z])\)", extraction)
        if m:
            extraction = m[0].upper()
        if len(extraction) == 1 and extraction.upper() in letters:
            return extraction.upper()
        if len(extraction) == 1 and extraction.isalpha():
            # a letter outside the available choices: not a usable answer
            return None
        return option_letter(choices.index(_most_similar(extraction, choices)))

    if atype == "integer":
        try:
            return str(int(float(extraction)))
        except Exception:
            return None

    if atype == "float":
        try:
            return str(round(float(extraction), int(meta.get("precision") or 0)))
        except Exception:
            return None

    if atype == "list":
        # MathVista compares the string form of the supplied answer.
        return re.sub(r"\s+", " ", extraction)

    # free-form text fallback
    return re.sub(r"\s+", " ", extraction)


def normalize_gold_mathvista(meta: dict[str, Any]) -> str | None:
    atype = meta.get("answer_type")
    if meta.get("question_type") == "multi_choice":
        return gold_letter(meta)
    if atype == "list":
        return re.sub(r"\s+", " ", str(meta.get("answer")))
    return str(meta.get("answer"))


def mathvista_correct(pred: str | None, meta: dict[str, Any]) -> bool:
    if pred is None:
        return False
    gold = normalize_gold_mathvista(meta)
    return gold is not None and pred == gold


# --------------------------------------------------------------------------------------
# Group construction on a candidate prefix
# --------------------------------------------------------------------------------------


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def prefix_groups(
    indices: Sequence[int],
    eq_matrix: np.ndarray | None,
    keys: Sequence[str],
) -> list[int]:
    """Group identifiers for the revealed candidates, in reveal order.

    ``eq_matrix`` holds pairwise equivalence computed without gold; only revealed
    pairs are unioned, so unrevealed candidates cannot bridge two groups.
    ``keys`` holds prefix-independent canonical keys (used for MathVista).
    """
    if eq_matrix is None:
        remap: dict[str, int] = {}
        out = []
        for idx in indices:
            out.append(remap.setdefault(keys[idx], len(remap)))
        return out

    uf = UnionFind(len(indices))
    for a in range(len(indices)):
        for b in range(a + 1, len(indices)):
            if eq_matrix[indices[a], indices[b]]:
                uf.union(a, b)
    remap: dict[int, int] = {}
    out = []
    for a in range(len(indices)):
        out.append(remap.setdefault(uf.find(a), len(remap)))
    return out


@dataclass(frozen=True)
class GradeSummary:
    n: int
    n_correct: int
    n_no_answer: int


def summarize(correct: np.ndarray, keys: Sequence[str]) -> GradeSummary:
    return GradeSummary(
        n=len(keys),
        n_correct=int(np.sum(correct)),
        n_no_answer=int(sum(1 for k in keys if k == NO_ANSWER)),
    )
