#!/usr/bin/env bash
# Task 2 throughput smoke: 5 updates per stage with the prescribed microbatch/accumulation.
set -u
cd /root/pa3
export PYTHONPATH=/root/pa3/src
PY=/venv/main/bin/python

echo "start $(date -u +%H:%M:%S)"
timeout 1200 $PY scripts/task2_run.py --stage train --max-updates-per-stage 5 2>&1 | tail -8
echo "end $(date -u +%H:%M:%S)"

$PY - <<'PY'
import json
d = json.load(open('/root/pa3/outputs/task_02/training.json'))
rows = [(s['stage'], s['updates'], s['seconds'],
         round(s['seconds'] / max(1, s['updates']), 2)) for s in d['stages']]
print('stage, updates, seconds, s_per_update')
for r in rows:
    print(r)
total_rate = sum(r[2] for r in rows) / sum(r[1] for r in rows)
print('mean_s_per_update', round(total_rate, 2))
print('projected_hours_4500', round(total_rate * 4500 / 3600, 2))
print('calibration_scale', round(d['calibration']['scale'], 4))
print('trainable', d['trainable'])
PY
