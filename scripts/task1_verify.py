"""Verify Task 1 completeness and build the seven-condition artefacts.

Checks
  * all seven evaluation conditions exist and have primary benchmark scores
  * 24 qualitative responses per condition -> 168 total
  * the 42-row annotation sheet (first and last underlying question x 3 formats x 7)
  * required checkpoints present (with the midpoint labelled reconstructed)

Writes
  outputs/task_01/tables/condition_table.{csv,md}   seven conditions + deltas in pp
  outputs/task_01/examples/annotation_sheet.json    42 rows, mechanical checks only
  outputs/task_01/verification.json                 machine-readable pass/fail
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, "/root/pa3/src")
from task1.qualitative import build_annotation_sheet  # noqa: E402

RUNS = Path("/root/pa3/outputs/task_01/runs")
TABLES = Path("/root/pa3/outputs/task_01/tables")
EXAMPLES = Path("/root/pa3/outputs/task_01/examples")

CONDITIONS = ["baseline", "full_ft_midpoint", "full_ft", "full_ft_l2sp",
              "lora4", "lora16", "oplora16"]
METRICS = ["gsm8k_flexible_extract", "hellaswag_acc_norm", "boolq_acc",
           "arc_easy_acc_norm", "hellaswag_acc", "arc_easy_acc"]


def load_reports() -> dict[str, dict]:
    out = {}
    for name in CONDITIONS:
        path = RUNS / name / "report.json"
        if path.exists():
            out[name] = json.loads(path.read_text())
    return out


def load_qualitative(name: str) -> list[dict] | None:
    path = RUNS / name / "qualitative" / f"qualitative_{name.split('_midpoint')[0]}.json"
    if not path.exists():
        candidates = list((RUNS / name / "qualitative").glob("qualitative_*.json"))
        path = candidates[0] if candidates else None
    return json.loads(path.read_text()) if path else None


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    EXAMPLES.mkdir(parents=True, exist_ok=True)
    reports = load_reports()

    rows = []
    baseline = reports.get("baseline", {}).get("benchmarks", {}).get("primary", {})
    for name in CONDITIONS:
        primary = reports.get(name, {}).get("benchmarks", {}).get("primary", {})
        row = {"condition": name}
        for m in METRICS:
            v = primary.get(m)
            row[m] = v
            if m in ("gsm8k_flexible_extract", "hellaswag_acc_norm", "boolq_acc",
                     "arc_easy_acc_norm"):
                b = baseline.get(m)
                row[f"delta_pp_{m}"] = None if (v is None or b is None) else round((v - b) * 100, 2)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "condition_table.csv", index=False)
    (TABLES / "condition_table.md").write_text(df.to_markdown(index=False, floatfmt=".4f") + "\n")

    qualitative = {}
    for name in CONDITIONS:
        q = load_qualitative(name)
        if q:
            qualitative[name] = q
    n_rows = 0
    if len(qualitative) >= 2:
        n_rows = build_annotation_sheet(qualitative, EXAMPLES / "annotation_sheet.json")

    checkpoints = {}
    for name in CONDITIONS:
        ck = RUNS / name / "train" / "checkpoints"
        files = sorted(p.name for p in ck.glob("checkpoint_*.pt")) if ck.exists() else []
        checkpoints[name] = files

    verification = {
        "conditions_expected": CONDITIONS,
        "conditions_with_primary": [n for n in CONDITIONS
                                    if reports.get(n, {}).get("benchmarks", {}).get("primary")],
        "n_conditions_with_primary": sum(1 for n in CONDITIONS
                                         if reports.get(n, {}).get("benchmarks", {}).get("primary")),
        "qualitative_counts": {n: len(q) for n, q in qualitative.items()},
        "qualitative_total": sum(len(q) for q in qualitative.values()),
        "annotation_rows": n_rows,
        "checkpoints": checkpoints,
        "midpoint_reconstructed": (RUNS / "full_ft_midpoint" / "train" / "RECONSTRUCTED.json").exists(),
        "passes": {
            "seven_conditions": sum(1 for n in CONDITIONS
                                    if reports.get(n, {}).get("benchmarks", {}).get("primary")) == 7,
            "168_responses": sum(len(q) for q in qualitative.values()) == 168,
            "42_annotation_rows": n_rows == 42,
        },
    }
    (Path("/root/pa3/outputs/task_01") / "verification.json").write_text(
        json.dumps(verification, indent=2))
    print(json.dumps({k: verification[k] for k in
                      ("n_conditions_with_primary", "qualitative_total",
                       "annotation_rows", "midpoint_reconstructed", "passes")}, indent=2))


if __name__ == "__main__":
    main()
