#!/usr/bin/env bash
# Sequential Task 1 driver: baseline first, then each variant.
# Each step is an independent process, so an OOM or crash cannot poison the next.
set -u
cd /root/pa3
mkdir -p logs outputs/task_01/runs
export PYTHONPATH=/root/pa3/src
PY=/venv/main/bin/python

CONDITIONS="${CONDITIONS:-baseline full_ft full_ft_l2sp lora4 lora16 oplora16}"

for v in $CONDITIONS; do
  echo "=== $(date -u +%H:%M:%SZ) starting $v ===" | tee -a logs/driver.log
  $PY scripts/task1_variant_step.py --variant "$v" \
      --out "outputs/task_01/runs/$v" >> "logs/task1_$v.log" 2>&1
  rc=$?
  echo "=== $(date -u +%H:%M:%SZ) finished $v rc=$rc ===" | tee -a logs/driver.log
done
echo "ALL_DONE" | tee -a logs/driver.log
