"""Print headline Task 3 offline numbers (development helper)."""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "outputs" / "task_03" / "offline"

for ds in ("math", "mathvista"):
    for split in ("dev", "heldout"):
        rows = json.loads((OUT / f"summary_{ds}_{split}.json").read_text())
        sc = next(r for r in rows if r["method"] == "self_consistency" and r["budget"] == 32)
        fa = next(r for r in rows if r["method"] == "first_ans" and r["budget"] == 1)
        print(f"{ds:10s} {split:8s} SC@32={sc['accuracy']:.4f}  FirstAns={fa['accuracy']:.4f}  "
              f"oracle@32={sc['oracle_coverage']:.4f}  n={sc['n_questions']}")
