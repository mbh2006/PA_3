"""Synthetic QuestionRollouts / DatasetBundle factories for unit tests."""

from __future__ import annotations

import numpy as np

from task3.answers import NO_ANSWER, math_equivalence_matrix
from task3.ingest import DatasetBundle, QuestionRollouts


def make_question(
    answers: list[str | None],
    *,
    tokens: list[int] | None = None,
    correct: list[bool] | None = None,
    sample_id: str = "q0",
    vis: list[float] | None = None,
    kl_tokens: list[list[float]] | None = None,
    v_tokens: list[list[float]] | None = None,
    is_vlm: bool = False,
    gold: str = "x",
) -> QuestionRollouts:
    n = len(answers)
    keys = [a if a is not None else NO_ANSWER for a in answers]
    eq = None if is_vlm else math_equivalence_matrix(answers)
    return QuestionRollouts(
        sample_id=sample_id,
        question_text="q",
        gold_answer=gold,
        rollout_ids=np.arange(n),
        n_tokens=np.asarray(tokens if tokens is not None else [10] * n, dtype=np.int64),
        logp_mean=np.zeros(n),
        kl_mean=np.zeros(n),
        correct=np.asarray(correct if correct is not None else [True] * n, dtype=bool),
        keys=keys,
        is_null=np.asarray([a is None for a in answers]),
        hit_cap=np.zeros(n, dtype=bool),
        generated_text=[""] * n,
        vis=None if vis is None else np.asarray(vis, dtype=np.float64),
        kl_tokens=None if kl_tokens is None else [np.asarray(k, dtype=np.float32) for k in kl_tokens],
        v_tokens=None if v_tokens is None else [np.asarray(v, dtype=np.float64) for v in v_tokens],
        eq_matrix=eq,
    )


def make_bundle(questions: list[QuestionRollouts], is_vlm: bool = False) -> DatasetBundle:
    return DatasetBundle(name="synthetic", is_vlm=is_vlm, questions=questions)
