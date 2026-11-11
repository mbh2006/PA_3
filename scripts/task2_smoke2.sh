#!/usr/bin/env bash
# Task 2 smoke: all three stage transitions plus a small evaluation.
set -u
cd /root/pa3
export PYTHONPATH=/root/pa3/src
PY=/venv/main/bin/python

echo "start $(date -u +%H:%M:%S)"
echo "=== train: 2 updates per stage ==="
timeout 1200 $PY scripts/task2_run.py --stage train --max-updates-per-stage 2 2>&1 | grep -E "^\[stage|calibration|presence|dev|TRAIN_DONE|Error|error" | head -20
echo "=== checkpoints ==="
ls outputs/task_02/checkpoints/ | head -8
echo "=== eval: 4 pairs per suite ==="
timeout 1200 $PY scripts/task2_run.py --stage eval --limit-pairs 4 2>&1 | tail -14
echo "end $(date -u +%H:%M:%S)"
