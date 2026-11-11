#!/usr/bin/env python3
"""Inspect an lm-eval output file: keys, metrics, and (if present) sample text."""

import glob
import json
import sys

stem = sys.argv[1] if len(sys.argv) > 1 else "gsm8k"
run = sys.argv[2] if len(sys.argv) > 2 else "baseline"
files = sorted(glob.glob(f"/root/pa3/outputs/task_01/runs/{run}/eval/{stem}_*.json"))
if not files:
    print("no file")
    raise SystemExit(1)
path = files[-1]
print("file:", path)
d = json.load(open(path))
print("top keys:", list(d.keys()))
print("config:", {k: d.get("config", {}).get(k) for k in
                  ("model", "model_args", "num_fewshot", "batch_size", "limit")})
for task, m in d.get("results", {}).items():
    print(task, {k: round(v, 4) for k, v in m.items()
                 if isinstance(v, (int, float)) and "stderr" not in k})
samples = d.get("samples", {})
for task, rows in samples.items():
    print(f"samples[{task}]: {len(rows)}")
    for r in rows[:3]:
        print("  doc:", r.get("doc_id"), "| target:", repr(r.get("target"))[:80])
        print("    resp:", repr(r.get("resps", r.get("response")))[:200])
        print("    filtered:", repr(r.get("filtered_resps"))[:160])
