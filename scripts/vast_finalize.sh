#!/usr/bin/env bash
# Runs after the repair queue: midpoint comparison + Task 1 completeness verification.
set -u
cd /root/pa3
export PYTHONPATH=/root/pa3/src
PY=/venv/main/bin/python

echo "=== $(date -u +%H:%M:%SZ) waiting for REPAIR_DONE ===" | tee -a logs/finalize.log
for _ in $(seq 1 600); do
  if grep -q REPAIR_DONE logs/repair.log 2>/dev/null; then break; fi
  sleep 60
done

echo "=== midpoint comparison ===" | tee -a logs/finalize.log
$PY scripts/task1_compare_midpoint.py 2>&1 | tee -a logs/finalize.log

echo "=== task 1 verification ===" | tee -a logs/finalize.log
$PY scripts/task1_verify.py 2>&1 | tee -a logs/finalize.log

echo "=== inventory ===" | tee -a logs/finalize.log
find outputs/task_01 -maxdepth 3 -type f \( -name "*.json" -o -name "*.md" -o -name "*.csv" \) | sort | head -60
du -sh outputs/task_01/runs/*/train/checkpoints 2>/dev/null
echo "FINALIZE_DONE" | tee -a logs/finalize.log
