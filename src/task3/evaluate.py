"""Offline evaluation engine for Task 3 (budgets x orders x methods)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from task3.answers import NO_ANSWER, prefix_groups
from task3.ingest import DatasetBundle, QuestionRollouts
from task3.scores import (
    consensus_group,
    n_groups,
    rwc_weights,
    tokenwise_fusion,
    visconf_confidence,
    visconf_from_confidence,
)

BUDGETS = (1, 2, 4, 8, 16, 32)

MATH_METHODS = ("first_ans", "self_consistency", "best_logp", "best_self_certainty", "sc_rwc")
VLM_METHODS = MATH_METHODS + (
    "best_vision",
    "vision_rwc",
    "best_visconf",
    "visconf_rwc",
    "best_ztoken",
    "ztoken_rwc",
)


def methods_for(is_vlm: bool) -> tuple[str, ...]:
    return VLM_METHODS if is_vlm else MATH_METHODS


@dataclass
class QuestionResult:
    """Per-question, order-averaged outcomes used for bootstrap and tables."""

    sample_id: str
    accuracy: dict[str, np.ndarray]        # method -> (n_budgets,)
    cost: dict[str, np.ndarray]            # method -> (n_budgets,)
    oracle: np.ndarray                     # (n_budgets,)
    distinct_groups: np.ndarray            # (n_budgets,)
    n_correct_pool: int
    n_rollouts: int


SCHEMA = pa.schema([
    ("split", pa.string()),
    ("sample_id", pa.string()),
    ("order", pa.int16()),
    ("budget", pa.int16()),
    ("method", pa.string()),
    ("correct", pa.bool_()),
    ("cost", pa.int64()),
    ("selected_rollout", pa.int16()),
    ("selected_answer", pa.string()),
    ("oracle", pa.bool_()),
    ("n_distinct_groups", pa.int16()),
    ("n_correct_in_prefix", pa.int16()),
])


def _select_positions(q: QuestionRollouts, idx: list[int], groups: list[int],
                      L: np.ndarray, K: np.ndarray, V: np.ndarray | None,
                      Z: np.ndarray | None, Ztok: np.ndarray | None):
    """Return {method: (position, group_id)} for one prefix."""
    n = len(idx)
    ones = np.ones(n)
    sel: dict[str, tuple[int, int]] = {}

    # FirstAns always returns the first revealed candidate
    sel["first_ans"] = (0, groups[0])

    kg = consensus_group(groups, ones)
    sel["self_consistency"] = (groups.index(kg), kg)

    sel["best_logp"] = (int(np.argmax(L)), groups[int(np.argmax(L))])
    sel["best_self_certainty"] = (int(np.argmax(K)), groups[int(np.argmax(K))])

    kg = consensus_group(groups, rwc_weights(K))
    sel["sc_rwc"] = (groups.index(kg), kg)

    if V is not None and Z is not None and Ztok is not None:
        sel["best_vision"] = (int(np.argmax(V)), groups[int(np.argmax(V))])
        kg = consensus_group(groups, rwc_weights(V))
        sel["vision_rwc"] = (groups.index(kg), kg)

        sel["best_visconf"] = (int(np.argmax(Z)), groups[int(np.argmax(Z))])
        kg = consensus_group(groups, rwc_weights(Z))
        sel["visconf_rwc"] = (groups.index(kg), kg)

        sel["best_ztoken"] = (int(np.argmax(Ztok)), groups[int(np.argmax(Ztok))])
        kg = consensus_group(groups, rwc_weights(Ztok))
        sel["ztoken_rwc"] = (groups.index(kg), kg)

    return sel


def evaluate_dataset(
    bundle: DatasetBundle,
    split_ids: Iterable[str],
    orders: dict[str, np.ndarray],
    split_name: str,
    *,
    budgets: tuple[int, ...] = BUDGETS,
    writer: pq.ParquetWriter | None = None,
    log_every: int = 200,
    limit: int | None = None,
) -> tuple[list[QuestionResult], dict]:
    methods = methods_for(bundle.is_vlm)
    by_id = bundle.by_id()
    ids = [q for q in split_ids if q in by_id]
    if limit is not None:
        ids = ids[:limit]

    per_q: list[QuestionResult] = []
    t0 = time.time()
    for qi, sid in enumerate(ids):
        q = by_id[sid]
        perms = orders[sid]
        n_orders = perms.shape[0]

        acc = {m: np.zeros(len(budgets)) for m in methods}
        cost = {m: np.zeros(len(budgets)) for m in methods}
        oracle = np.zeros(len(budgets))
        dgroups = np.zeros(len(budgets))

        rows: dict[str, list] = {name: [] for name in SCHEMA.names}

        for oi in range(n_orders):
            perm = perms[oi]
            cum_cost = np.cumsum(q.n_tokens[perm])
            for bi, n in enumerate(budgets):
                idx = [int(x) for x in perm[:n]]
                groups = prefix_groups(idx, q.eq_matrix, q.keys)
                L = q.logp_mean[idx]
                K = q.kl_mean[idx]
                V = q.vis[idx] if bundle.is_vlm else None
                if bundle.is_vlm:
                    conf = visconf_confidence(q.kl_tokens, idx)
                    Z = visconf_from_confidence(conf, q.vis[idx])
                    Ztok = tokenwise_fusion(q.kl_tokens, q.v_tokens, idx)
                else:
                    Z = Ztok = None

                correct_pref = q.correct[idx]
                oracle[bi] += float(correct_pref.any())
                dgroups[bi] += n_groups(groups)

                sel = _select_positions(q, idx, groups, L, K, V, Z, Ztok)
                for m in methods:
                    pos, _g = sel[m]
                    is_correct = bool(correct_pref[pos])
                    acc[m][bi] += float(is_correct)
                    c = int(q.n_tokens[idx[0]]) if m == "first_ans" else int(cum_cost[n - 1])
                    cost[m][bi] += float(c)

                    rows["split"].append(split_name)
                    rows["sample_id"].append(sid)
                    rows["order"].append(oi)
                    rows["budget"].append(n)
                    rows["method"].append(m)
                    rows["correct"].append(is_correct)
                    rows["cost"].append(c)
                    rows["selected_rollout"].append(int(idx[pos]))
                    rows["selected_answer"].append(str(q.keys[idx[pos]]))
                    rows["oracle"].append(bool(correct_pref.any()))
                    rows["n_distinct_groups"].append(n_groups(groups))
                    rows["n_correct_in_prefix"].append(int(correct_pref.sum()))

        for m in methods:
            acc[m] /= n_orders
            cost[m] /= n_orders
        oracle /= n_orders
        dgroups /= n_orders

        per_q.append(
            QuestionResult(
                sample_id=sid,
                accuracy=acc,
                cost=cost,
                oracle=oracle,
                distinct_groups=dgroups,
                n_correct_pool=int(q.correct.sum()),
                n_rollouts=len(q.rollout_ids),
            )
        )

        if writer is not None:
            table = pa.table({k: pa.array(v, type=SCHEMA.field(k).type) for k, v in rows.items()},
                             schema=SCHEMA)
            writer.write_table(table)

        if log_every and (qi + 1) % log_every == 0:
            print(f"  [{split_name}] {qi + 1}/{len(ids)} questions  "
                  f"({time.time() - t0:.0f}s)", flush=True)

    meta = {
        "split": split_name,
        "n_questions": len(per_q),
        "n_orders": int(next(iter(orders.values())).shape[0]),
        "budgets": list(budgets),
        "methods": list(methods),
        "seconds": round(time.time() - t0, 1),
    }
    return per_q, meta


def write_parquet(rows_iter, path: Path) -> None:
    writer = pq.ParquetWriter(path, SCHEMA, compression="zstd")
    try:
        for table in rows_iter:
            writer.write_table(table)
    finally:
        writer.close()
