#!/usr/bin/env python3
"""Rebuild ``benchmarks.primary`` in every run's report.json from saved metrics.

Needed when the metric-selection logic changes after evaluations have already run; it
never re-runs a model.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/pa3/src")
from task1.eval_lm import primary_scores  # noqa: E402

root = Path("/root/pa3/outputs/task_01/runs")
for report in sorted(root.glob("*/report.json")):
    data = json.loads(report.read_text())
    summary = data.get("benchmarks", {}).get("summary", {})
    if not summary:
        print(f"{report.parent.name}: no benchmark summary")
        continue
    old = data["benchmarks"].get("primary")
    new = primary_scores(summary)
    data["benchmarks"]["primary"] = new
    report.write_text(json.dumps(data, indent=2, default=str) + "\n")
    print(f"{report.parent.name}: {json.dumps(new)}")
    if old != new:
        print(f"  (was {json.dumps(old)})")
