#!/usr/bin/env bash
# Re-evaluate the three adapter conditions with correct merge/unwrap.
# The earlier adapter evaluations are discarded: they were scored on an unwrapped model
# whose adapted layers were never loaded (random-chance accuracy).
set -u
cd /root/pa3
export PYTHONPATH=/root/pa3/src
PY=/venv/main/bin/python

log() { echo "=== $(date -u +%H:%M:%SZ) $* ===" | tee -a logs/repair3.log; }

log "waiting for any previous repair process to exit"
pkill -f vast_repair2.sh 2>/dev/null
pkill -f task1_variant_step.py 2>/dev/null
sleep 5

for v in lora4 lora16 oplora16; do
  rm -rf "outputs/task_01/runs/$v/merged" "outputs/task_01/runs/$v/eval"
  log "$v evaluation (correct merge)"
  $PY scripts/task1_variant_step.py --variant "$v" \
      --out "outputs/task_01/runs/$v" >> "logs/repair3_$v.log" 2>&1
  log "$v rc=$?"
done

log "midpoint comparison"
$PY scripts/task1_compare_midpoint.py >> logs/repair3.log 2>&1

log "task 1 verification"
$PY scripts/task1_verify.py >> logs/repair3.log 2>&1

log "REPAIR3_DONE"
