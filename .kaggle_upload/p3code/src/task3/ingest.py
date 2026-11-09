"""Load, join and validate the supplied Task 3 rollout tables.

Alignment rules enforced here (fail loudly, never silently repair):

* ``(dataset, sample_id, rollout_id)`` is unique in every table and the tables agree
  on the key set.
* Every per-token list has exactly ``n_generated_text_tokens`` entries.
* The attention table's ``n_generated_text_tokens`` column is the preceding-token
  count ``[0, ..., N-1]``; its length must match the scalar length in
  ``generations.parquet``.
* ``gold_answer`` is constant within a question.

Reading is streamed one row group (= one question) at a time so that the ~300 MB
of per-token lists never materialise as Python lists.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from common.paths import MATHVISTA_METADATA, OUTPUT_ROOT, TASK3_DATASETS
from task3 import cache as equiv_cache
from task3.answers import (
    NO_ANSWER,
    math_correct,
    math_equivalence_matrix,
    mathvista_correct,
    normalize_gold_mathvista,
    normalize_mathvista,
)

N_ROLLOUTS = 32


@dataclass
class QuestionRollouts:
    sample_id: str
    question_text: str
    gold_answer: str
    rollout_ids: np.ndarray          # (32,) int, sorted
    n_tokens: np.ndarray             # (32,)
    logp_mean: np.ndarray            # (32,)
    kl_mean: np.ndarray              # (32,)
    correct: np.ndarray              # (32,) bool
    keys: list[str]                  # (32,) canonical answer key / raw MATH answer
    is_null: np.ndarray              # (32,) bool - missing extraction
    hit_cap: np.ndarray              # (32,) bool
    generated_text: list[str]        # (32,)
    vis: np.ndarray | None = None    # (32,) mean visual engagement
    kl_tokens: list[np.ndarray] | None = None
    v_tokens: list[np.ndarray] | None = None
    eq_matrix: np.ndarray | None = field(default=None, repr=False)

    def order(self, perm: np.ndarray, n: int) -> list[int]:
        return [int(p) for p in perm[:n]]


@dataclass
class DatasetBundle:
    name: str
    is_vlm: bool
    questions: list[QuestionRollouts]

    def by_id(self) -> dict[str, QuestionRollouts]:
        return {q.sample_id: q for q in self.questions}

    @property
    def n_questions(self) -> int:
        return len(self.questions)


def _row_groups(path: Path, columns: list[str], limit: int | None):
    """Yield per-row-group column dictionaries, converting list columns to numpy."""
    pf = pq.ParquetFile(path)
    missing = [c for c in columns if c not in pf.schema_arrow.names]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    for gi in range(pf.num_row_groups):
        if limit is not None and gi >= limit:
            return
        tbl = pf.read_row_group(gi, columns=columns)
        yield {name: tbl.column(name).to_pylist() for name in columns}


def _load_probability(path: Path, limit: int | None) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]:
    out: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for cols in _row_groups(path, ["sample_id", "rollout_id", "logp", "kl_u_p"], limit):
        for sid, rid, logp, kl in zip(cols["sample_id"], cols["rollout_id"], cols["logp"], cols["kl_u_p"]):
            key = (sid, int(rid))
            if key in out:
                raise ValueError(f"{path.name}: duplicate join key {key}")
            out[key] = (
                np.asarray(logp, dtype=np.float32),
                np.asarray(kl, dtype=np.float32),
            )
    return out


def _load_attention(path: Path, limit: int | None) -> dict[tuple[str, int], dict]:
    out: dict[tuple[str, int], dict] = {}
    cols_needed = [
        "sample_id", "rollout_id", "n_image_tokens", "n_prompt_text_tokens",
        "n_generated_text_tokens", "img_attn_total", "prompt_text_attn_total",
        "generated_text_attn_total",
    ]
    for cols in _row_groups(path, cols_needed, limit):
        for i, sid in enumerate(cols["sample_id"]):
            key = (sid, int(cols["rollout_id"][i]))
            if key in out:
                raise ValueError(f"{path.name}: duplicate join key {key}")
            out[key] = {name: cols[name][i] for name in cols_needed[2:]}
    return out


def load_dataset(name: str, *, load_token_metrics: bool = True,
                 limit: int | None = None, use_cache: bool = True,
                 cache_dir: Path | None = None) -> DatasetBundle:
    spec = TASK3_DATASETS[name]
    root: Path = spec["dir"]
    is_vlm = bool(spec["is_vlm"])

    cache_dir = cache_dir or (OUTPUT_ROOT / "task_03" / "cache")
    equiv_key = None
    equiv_cached: dict[str, np.ndarray] = {}
    if not is_vlm and use_cache:
        equiv_key = equiv_cache.cache_key(root / spec["generations"])
        equiv_cached = equiv_cache.load(cache_dir, name, equiv_key) or {}
        if equiv_cached:
            print(f"[ingest] {name}: {len(equiv_cached)} cached equivalence matrices", flush=True)
    new_equiv: dict[str, np.ndarray] = {}

    prob = _load_probability(root / spec["probability"], limit)
    attn = _load_attention(root / spec["attention"], limit) if spec["attention"] else None

    meta = None
    if name == "mathvista":
        meta = json.loads(MATHVISTA_METADATA.read_text(encoding="utf-8"))

    gen_cols = [
        "dataset", "sample_id", "rollout_id", "question_text", "generated_token_ids",
        "n_generated_text_tokens", "hit_max_new_tokens", "generated_text",
        "gold_answer", "extracted_answer",
    ]

    questions: list[QuestionRollouts] = []
    datasets_seen: set[str] = set()

    for cols in _row_groups(root / spec["generations"], gen_cols, limit):
        rids = [int(r) for r in cols["rollout_id"]]
        if sorted(rids) != list(range(N_ROLLOUTS)):
            raise ValueError(f"{name}/{cols['sample_id'][0]}: rollout ids are not 0..31")
        order = np.argsort(rids)
        sid = cols["sample_id"][0]
        if len(set(cols["sample_id"])) != 1:
            raise ValueError(f"{name}: row group mixes questions: {set(cols['sample_id'])}")
        datasets_seen.update(cols["dataset"])

        def take(field_name: str) -> list:
            return [cols[field_name][i] for i in order]

        n_tokens = np.asarray(take("n_generated_text_tokens"), dtype=np.int64)
        token_ids = take("generated_token_ids")
        if any(len(t) != n for t, n in zip(token_ids, n_tokens)):
            bad = [(i, len(t), n) for i, (t, n) in enumerate(zip(token_ids, n_tokens)) if len(t) != n][:3]
            raise ValueError(f"{name}/{sid}: generated_token_ids length mismatch {bad}")

        try:
            keys_join = [(sid, int(r)) for r in rids]
            probs = [prob[k] for k in keys_join]
        except KeyError as exc:
            raise KeyError(f"{name}/{sid}: probability table missing key {exc}") from None

        logp = [p[0] for p in probs]
        kls = [p[1] for p in probs]
        if any(len(a) != n for a, n in zip(logp, n_tokens)):
            raise ValueError(f"{name}/{sid}: logp length mismatch")
        if any(len(a) != n for a, n in zip(kls, n_tokens)):
            raise ValueError(f"{name}/{sid}: kl_u_p length mismatch")

        logp_mean = np.array([float(np.mean(a)) if len(a) else np.nan for a in logp])
        kl_mean = np.array([float(np.mean(a)) if len(a) else np.nan for a in kls])

        golds = {g for g in take("gold_answer")}
        if len(golds) != 1:
            raise ValueError(f"{name}/{sid}: gold_answer not constant: {golds}")
        gold = golds.pop()

        raw = take("extracted_answer")
        is_null = np.array([e is None or not str(e).strip() for e in raw])
        hit_cap = np.asarray(take("hit_max_new_tokens"), dtype=bool)
        gen_text = [str(t) for t in take("generated_text")]

        vis = None
        kl_tokens = None
        v_tokens = None
        if load_token_metrics:
            kl_tokens = [np.asarray(a, dtype=np.float32) for a in kls]

        if is_vlm:
            if sid not in meta:
                raise KeyError(f"MathVista metadata missing pid {sid}")
            m = meta[sid]
            preds = [normalize_mathvista(e, m) for e in raw]
            keys = [p if p is not None else NO_ANSWER for p in preds]
            correct = np.array([mathvista_correct(p, m) for p in preds], dtype=bool)

            gold_expected = normalize_gold_mathvista(m)
            if gold_expected is not None and gold_expected != gold:
                raise ValueError(
                    f"{name}/{sid}: parquet gold {gold!r} != metadata-derived {gold_expected!r}"
                )

            assert attn is not None
            v_rows, v_tok = [], []
            for k, n_g in zip(keys_join, n_tokens):
                a = attn[k]
                a_img = np.asarray(a["img_attn_total"], dtype=np.float64)
                a_pr = np.asarray(a["prompt_text_attn_total"], dtype=np.float64)
                a_ge = np.asarray(a["generated_text_attn_total"], dtype=np.float64)
                prev = a["n_generated_text_tokens"]
                if len(prev) != n_g or list(prev) != list(range(int(n_g))):
                    raise ValueError(f"{name}/{sid}: malformed preceding-token column")
                if not (len(a_img) == len(a_pr) == len(a_ge) == int(n_g)):
                    raise ValueError(f"{name}/{sid}: attention list length != n_generated_text_tokens")
                denom = a_img + a_pr + a_ge
                if np.any(denom <= 0):
                    raise ValueError(f"{name}/{sid}: non-positive attention denominator")
                vt = a_img / denom
                v_tok.append(vt)
                v_rows.append(float(np.mean(vt)))
            vis = np.asarray(v_rows, dtype=np.float64)
            v_tokens = v_tok
            eq_matrix = None
        else:
            preds = [None if n else str(e).strip() for e, n in zip(raw, is_null)]
            keys = [p if p is not None else NO_ANSWER for p in preds]
            correct = np.array([math_correct(p, gold) for p in preds], dtype=bool)
            eq_matrix = equiv_cached.get(sid)
            if eq_matrix is None:
                eq_matrix = math_equivalence_matrix(preds)
                new_equiv[sid] = eq_matrix

        questions.append(
            QuestionRollouts(
                sample_id=sid,
                question_text=str(take("question_text")[0]),
                gold_answer=gold,
                rollout_ids=np.asarray(take("rollout_id"), dtype=np.int64),
                n_tokens=n_tokens,
                logp_mean=logp_mean,
                kl_mean=kl_mean,
                correct=correct,
                keys=keys,
                is_null=is_null,
                hit_cap=hit_cap,
                generated_text=gen_text,
                vis=vis,
                kl_tokens=kl_tokens,
                v_tokens=v_tokens,
                eq_matrix=eq_matrix,
            )
        )

    if len(datasets_seen) != 1:
        raise ValueError(f"{name}: mixed dataset values {datasets_seen}")

    if new_equiv and equiv_key is not None:
        merged = {**equiv_cached, **new_equiv}
        path = equiv_cache.save(cache_dir, name, equiv_key, merged)
        print(f"[ingest] {name}: cached {len(new_equiv)} new equivalence matrices -> {path}",
              flush=True)

    return DatasetBundle(name=name, is_vlm=is_vlm, questions=questions)
