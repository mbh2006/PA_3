#!/usr/bin/env bash
# Repair queue: runs after the main chain finishes.
#  1. Full FT final evaluation (training checkpoint already exists -> resume skips it)
#  2. Full FT + L2-SP final evaluation (same)
#  3. Full FT midpoint: retrain updates 1..625 with the original seed/data order/schedule,
#     save the checkpoint, then evaluate
#  4. Re-derive primary metrics for every run
set -u
cd /root/pa3
export PYTHONPATH=/root/pa3/src
PY=/venv/main/bin/python

log() { echo "=== $(date -u +%H:%M:%SZ) $* ===" | tee -a logs/repair.log; }

log "waiting for main chain to finish"
for _ in $(seq 1 1000); do
  if grep -q ALL_DONE logs/driver.log 2>/dev/null; then break; fi
  sleep 60
done
log "main chain done"

# Full FT final evaluation (resume: model_final.pt exists)
log "full_ft final evaluation"
$PY scripts/task1_variant_step.py --variant full_ft \
    --out outputs/task_01/runs/full_ft >> logs/repair_full_ft.log 2>&1
log "full_ft rc=$?"

# Full FT + L2-SP final evaluation
log "full_ft_l2sp final evaluation"
$PY scripts/task1_variant_step.py --variant full_ft_l2sp \
    --out outputs/task_01/runs/full_ft_l2sp >> logs/repair_l2sp.log 2>&1
log "full_ft_l2sp rc=$?"

# Midpoint: identical first 625 updates, checkpoint kept, then evaluated
log "full_ft midpoint (updates 1..625)"
rm -rf outputs/task_01/runs/full_ft_midpoint
$PY scripts/task1_variant_step.py --variant full_ft --max-updates 625 \
    --out outputs/task_01/runs/full_ft_midpoint >> logs/repair_midpoint.log 2>&1
log "midpoint rc=$?"

log "re-deriving primary metrics"
$PY scripts/vast_reparse.py >> logs/repair.log 2>&1
log "REPAIR_DONE"
