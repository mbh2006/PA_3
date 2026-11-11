#!/usr/bin/env bash
# Second repair pass: complete the three adapter conditions (evaluation, diagnostics,
# qualitative) from their saved adapters, then re-derive metrics and verify.
set -u
cd /root/pa3
export PYTHONPATH=/root/pa3/src
PY=/venv/main/bin/python

log() { echo "=== $(date -u +%H:%M:%SZ) $* ===" | tee -a logs/repair2.log; }

log "waiting for REPAIR_DONE"
for _ in $(seq 1 900); do
  if grep -q REPAIR_DONE logs/repair.log 2>/dev/null; then break; fi
  sleep 60
done
log "first repair done"

for v in lora4 lora16 oplora16; do
  log "$v evaluation from saved adapter"
  $PY scripts/task1_variant_step.py --variant "$v" \
      --out "outputs/task_01/runs/$v" >> "logs/repair2_$v.log" 2>&1
  log "$v rc=$?"
done

log "midpoint comparison"
$PY scripts/task1_compare_midpoint.py >> logs/repair2.log 2>&1

log "task 1 verification"
$PY scripts/task1_verify.py >> logs/repair2.log 2>&1

log "REPAIR2_DONE"
