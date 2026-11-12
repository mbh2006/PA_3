"""Task 3 offline pipeline: budgets x orders x methods, signals, adaptive stopping.

Usage (from the repo root, with ``PYTHONPATH=src``):

    python -m task3.run_offline --dataset math      --out outputs/task_03/offline
    python -m task3.run_offline --dataset mathvista --out outputs/task_03/offline

Nothing here uses gold answers for selection or stopping; correctness is computed
only for retrospective scoring.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from common.paths import TASK3_OUT, utc_now, write_json
from task3 import adaptive as ad
from task3.answers import SCORER_VERSION
from task3.bootstrap import paired_bootstrap
from task3.evaluate import BUDGETS, SCHEMA, evaluate_dataset, methods_for
from task3.ingest import load_dataset
from task3.orders import orders_for_all
from task3.signals import confidence_vision_correlations, fusion_identity, save_trajectories, within_question
from task3.splits import SEED, make_split


def _stack(results, methods, budgets):
    acc = {m: np.stack([r.accuracy[m] for r in results]) for m in methods}
    cost = {m: np.stack([r.cost[m] for r in results]) for m in methods}
    oracle = np.stack([r.oracle for r in results])
    groups = np.stack([r.distinct_groups for r in results])
    return acc, cost, oracle, groups


def summary_rows(results, methods, budgets) -> list[dict]:
    acc, cost, oracle, groups = _stack(results, methods, budgets)
    rows = []
    for m in methods:
        for bi, n in enumerate(budgets):
            rows.append({
                "method": m,
                "budget": n,
                "accuracy": float(acc[m][:, bi].mean()),
                "mean_cost_tokens": float(cost[m][:, bi].mean()),
                "oracle_coverage": float(oracle[:, bi].mean()),
                "mean_distinct_groups": float(groups[:, bi].mean()),
                "n_questions": int(acc[m].shape[0]),
            })
    return rows


def failure_rows(results, methods, budgets) -> list[dict]:
    acc, _, oracle, _ = _stack(results, methods, budgets)
    rows = []
    for m in methods:
        for bi, n in enumerate(budgets):
            a = acc[m][:, bi]
            o = oracle[:, bi]
            # Selection success implies oracle coverage within the same order.
            # Subtracting their means preserves the order-level partition.
            rows.append({
                "method": m,
                "budget": n,
                "no_correct_generated": float(np.mean(1.0 - o)),
                "correct_generated_but_missed": float(np.mean(o - a)),
                "selected_correct": float(a.mean()),
            })
    return rows


def bootstrap_rows(results, methods, budgets) -> list[dict]:
    acc, _, _, _ = _stack(results, methods, budgets)
    rows = []
    for m in methods:
        for bi, n in enumerate(budgets):
            for ref in ("first_ans", "self_consistency"):
                if ref not in acc or m == ref:
                    continue
                d = paired_bootstrap(acc[m][:, bi] - acc[ref][:, bi])
                rows.append({
                    "method": m, "reference": ref, "budget": n,
                    "diff": d["mean"], "lo": d["lo"], "hi": d["hi"], "se": d["se"],
                    "n_questions": d["n"],
                })
    return rows


def _policy_grid() -> list[ad.Policy]:
    grid = [ad.Policy("always32")]
    for tau in ad.TAUS:
        grid.append(ad.Policy("agreement", tau=tau))
        for gamma in ad.GAMMAS:
            grid.append(ad.Policy("quality", tau=tau, gamma=gamma))
    return grid


def adaptive_section(bundle, split, orders, out_dir: Path) -> dict:
    methods = methods_for(bundle.is_vlm)
    by_id = bundle.by_id()

    # fixed-32 self-consistency reference on dev
    fixed32 = []
    for sid in split.dev:
        q = by_id[sid]
        perms = orders[sid]
        vals = []
        for oi in range(perms.shape[0]):
            idx = [int(x) for x in perms[oi]]
            from task3.answers import prefix_groups
            from task3.scores import consensus_group
            groups = prefix_groups(idx, q.eq_matrix, q.keys)
            g = consensus_group(groups, np.ones(len(groups)))
            vals.append(float(q.correct[idx[groups.index(g)]]))
        fixed32.append(float(np.mean(vals)))
    fixed32_acc = float(np.mean(fixed32))

    dev_rows, held_rows = [], []
    for policy in _policy_grid():
        d = ad.evaluate_policy(bundle, split.dev, orders, policy)
        acc = float(np.mean([r["accuracy"] for r in d]))
        cost = float(np.mean([r["cost"] for r in d]))
        nused = float(np.mean([r["n_used"] for r in d]))
        dev_rows.append({
            "policy": policy.name, "family": policy.family,
            "tau": policy.tau, "gamma": policy.gamma,
            "dev_accuracy": acc, "dev_mean_cost": cost, "dev_mean_rollouts": nused,
            "within_1pp_of_fixed32": bool(acc >= fixed32_acc - 0.01),
        })

    # selection: lowest dev cost within 1 pp of fixed-32 SC; ties -> higher acc, smaller tau, gamma
    selected: dict[str, dict] = {}
    for family in ("agreement", "quality"):
        cands = [r for r in dev_rows if r["family"] == family and r["within_1pp_of_fixed32"]]
        if not cands:
            selected[family] = next(r for r in dev_rows if r["family"] == "always32")
            continue
        cands.sort(key=lambda r: (r["dev_mean_cost"], -r["dev_accuracy"],
                                  r["tau"] if r["tau"] is not None else 0.0,
                                  r["gamma"] if r["gamma"] is not None else 0.0))
        selected[family] = cands[0]

    for family, row in selected.items():
        policy = ad.Policy(row["family"], tau=row["tau"], gamma=row["gamma"]) \
            if row["family"] != "always32" else ad.Policy("always32")
        h = ad.evaluate_policy(bundle, split.heldout, orders, policy)
        hist: dict[str, int] = {}
        for r in h:
            for n in r["n_used_by_order"]:
                hist[str(int(n))] = hist.get(str(int(n)), 0) + 1
        held_rows.append({
            "family": family,
            "policy": policy.name,
            "heldout_accuracy": float(np.mean([r["accuracy"] for r in h])),
            "heldout_mean_cost": float(np.mean([r["cost"] for r in h])),
            "heldout_mean_rollouts": float(np.mean([r["n_used"] for r in h])),
            "stopping_histogram": dict(sorted(hist.items(), key=lambda kv: int(kv[0]))),
        })

    write_json(out_dir / f"adaptive_{bundle.name}_dev_grid.json",
               {"fixed32_dev_accuracy": fixed32_acc, "fixed32_dev_questions": len(fixed32),
                "rows": dev_rows})
    write_json(out_dir / f"adaptive_{bundle.name}_selected.json",
               {"dataset": bundle.name, "selected": selected,
                "fixed32_dev_accuracy": fixed32_acc})
    write_json(out_dir / f"adaptive_{bundle.name}_heldout.json",
               {"dataset": bundle.name, "rows": held_rows})
    return {"fixed32_dev_accuracy": fixed32_acc, "selected": selected, "heldout": held_rows}


def run(dataset: str, out_dir: Path, *, limit: int | None = None,
        skip_adaptive: bool = False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{dataset}] loading ...", flush=True)
    bundle = load_dataset(dataset, limit=limit)
    print(f"[{dataset}] {bundle.n_questions} questions loaded", flush=True)

    split = make_split([q.sample_id for q in bundle.questions],
                       n_dev=200 if limit is None else max(1, limit // 2))
    orders = orders_for_all(sorted(q.sample_id for q in bundle.questions))
    write_json(out_dir / f"splits_{dataset}.json", {
        "dataset": dataset, "seed": SEED, "scorer_version": SCORER_VERSION,
        "n_questions": bundle.n_questions,
        **split.as_dict(),
    })

    methods = methods_for(bundle.is_vlm)
    results: dict[str, list] = {}
    summaries: dict[str, list[dict]] = {}

    for split_name, ids in (("dev", split.dev), ("heldout", split.heldout)):
        if limit is not None:
            ids = [i for i in ids if i in bundle.by_id()][:limit]
        path = out_dir / f"per_question_{dataset}_{split_name}.parquet"
        print(f"[{dataset}/{split_name}] evaluating {len(ids)} questions ...", flush=True)
        writer = pq.ParquetWriter(path, SCHEMA, compression="zstd")
        try:
            res, meta = evaluate_dataset(bundle, ids, orders, split_name, writer=writer)
        finally:
            writer.close()
        results[split_name] = res
        summaries[split_name] = summary_rows(res, methods, BUDGETS)
        write_json(out_dir / f"meta_{dataset}_{split_name}.json", meta)
        print(f"[{dataset}/{split_name}] done in {meta['seconds']}s -> {path}", flush=True)

        write_json(out_dir / f"summary_{dataset}_{split_name}.json", summaries[split_name])
        write_json(out_dir / f"failures_{dataset}_{split_name}.json",
                   failure_rows(res, methods, BUDGETS))
        write_json(out_dir / f"bootstrap_{dataset}_{split_name}.json",
                   bootstrap_rows(res, methods, BUDGETS))

    # N=32 signal analyses on held-out
    held_ids = [i for i in split.heldout if i in bundle.by_id()]
    if limit is not None:
        held_ids = held_ids[:limit]
    print(f"[{dataset}] signal analyses ...", flush=True)
    signals = {"within_question": within_question(bundle, held_ids)}
    if bundle.is_vlm:
        signals["confidence_vision"] = confidence_vision_correlations(bundle, held_ids)
        signals["fusion_identity"] = fusion_identity(bundle, held_ids)
        signals["trajectories"] = save_trajectories(bundle, held_ids, out_dir / f"trajectories_{dataset}.json")
    write_json(out_dir / f"signals_{dataset}.json", signals)

    payload = {"dataset": dataset, "methods": list(methods), "budgets": list(BUDGETS)}
    if not skip_adaptive:
        print(f"[{dataset}] adaptive stopping ...", flush=True)
        payload["adaptive"] = adaptive_section(bundle, split, orders, out_dir)
    write_json(out_dir / f"run_{dataset}_summary.json", payload)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["math", "mathvista", "both"], default="both")
    ap.add_argument("--out", type=Path, default=TASK3_OUT / "offline")
    ap.add_argument("--limit", type=int, default=None, help="smoke: only N questions")
    ap.add_argument("--skip-adaptive", action="store_true")
    args = ap.parse_args()

    names = ["math", "mathvista"] if args.dataset == "both" else [args.dataset]
    for name in names:
        run(name, args.out, limit=args.limit, skip_adaptive=args.skip_adaptive)
    print("\nwrote", args.out.resolve())


if __name__ == "__main__":
    main()
