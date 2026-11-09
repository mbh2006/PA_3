"""Render the Task 3 offline artefacts as CSV + Markdown (factual tables only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common.paths import TASK3_OUT


def _write(df: pd.DataFrame, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}.csv", index=False)
    (out / f"{name}.md").write_text(df.to_markdown(index=False, floatfmt=".4f") + "\n",
                                    encoding="utf-8")
    print(f"  {name}: {len(df)} rows")


def render(dataset: str, offline: Path, tables: Path) -> None:
    for split in ("dev", "heldout"):
        summary = json.loads((offline / f"summary_{dataset}_{split}.json").read_text())
        df = pd.DataFrame(summary)
        df = df[["method", "budget", "accuracy", "mean_cost_tokens", "oracle_coverage",
                 "mean_distinct_groups", "n_questions"]]
        _write(df, tables, f"{dataset}_{split}_methods")

        fail = pd.DataFrame(json.loads((offline / f"failures_{dataset}_{split}.json").read_text()))
        _write(fail, tables, f"{dataset}_{split}_failures")

        boot = pd.DataFrame(json.loads((offline / f"bootstrap_{dataset}_{split}.json").read_text()))
        _write(boot, tables, f"{dataset}_{split}_bootstrap")

    sel = json.loads((offline / f"adaptive_{dataset}_selected.json").read_text())
    held = json.loads((offline / f"adaptive_{dataset}_heldout.json").read_text())
    rows = []
    for family, row in sel["selected"].items():
        rows.append({"family": family, "stage": "dev_selected", **{k: v for k, v in row.items()}})
    for row in held["rows"]:
        rows.append({"family": row["family"], "stage": "heldout", "policy": row["policy"],
                     "heldout_accuracy": row["heldout_accuracy"],
                     "heldout_mean_cost": row["heldout_mean_cost"],
                     "heldout_mean_rollouts": row["heldout_mean_rollouts"],
                     "stopping_histogram": json.dumps(row["stopping_histogram"])})
    _write(pd.DataFrame(rows), tables, f"{dataset}_adaptive_selected")

    grid = pd.DataFrame(json.loads((offline / f"adaptive_{dataset}_dev_grid.json").read_text())["rows"])
    _write(grid, tables, f"{dataset}_adaptive_dev_grid")

    sig = json.loads((offline / f"signals_{dataset}.json").read_text())
    flat = []
    for group, payload in sig["within_question"].items():
        if isinstance(payload, dict) and "mean" in payload:
            flat.append({"metric": group, "n_defined": payload["n_defined"],
                         "mean": payload["mean"], "median": payload["median"], "sd": payload["sd"]})
        elif isinstance(payload, dict):
            for sub, p in payload.items():
                flat.append({"metric": f"{group}.{sub}", "n_defined": p["n_defined"],
                             "mean": p["mean"], "median": p["median"], "sd": p["sd"]})
    if dataset == "mathvista":
        cv = sig["confidence_vision"]
        flat.append({"metric": "corr_c_v.question_mean", "n_defined": cv["n_questions_defined"],
                     "mean": cv["question_mean_of_rollout_corr"]["mean"],
                     "median": cv["question_mean_of_rollout_corr"]["median"],
                     "sd": cv["question_mean_of_rollout_corr"]["sd"]})
        fi = sig["fusion_identity"]
        flat.append({"metric": "fusion_identity_max_abs_error",
                     "n_defined": fi["n_rollouts_checked"], "mean": fi["max_abs_error"],
                     "median": None, "sd": None})
    _write(pd.DataFrame(flat), tables, f"{dataset}_signals")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", type=Path, default=TASK3_OUT / "offline")
    ap.add_argument("--tables", type=Path, default=TASK3_OUT / "tables")
    ap.add_argument("--datasets", nargs="+", default=["math", "mathvista"])
    args = ap.parse_args()
    for ds in args.datasets:
        print(f"[{ds}]")
        render(ds, args.offline, args.tables)
    print("wrote", args.tables.resolve())


if __name__ == "__main__":
    main()
