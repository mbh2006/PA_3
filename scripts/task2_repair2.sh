#!/usr/bin/env bash
# Clear the invalid run's Stage 2/3 checkpoints so the resume actually retrains them.
set -u
cd /root/pa3
export PYTHONPATH=/root/pa3/src
PY=/venv/main/bin/python
log() { echo "=== $(date -u +%H:%M:%S) $* ===" | tee -a logs/task2_repair2.log; }

pkill -f "task2_run.py" 2>/dev/null
pkill -f "task2_repair.sh" 2>/dev/null
sleep 3

log "clearing stale Stage 2/3 artefacts (Stage 1 is preserved)"
rm -f outputs/task_02/checkpoints/stage2_instruction_*.pt
rm -f outputs/task_02/checkpoints/stage3_natural_*.pt
rm -f outputs/task_02/checkpoints/stage2_instruction_weights.pt
rm -f outputs/task_02/checkpoints/stage3_natural_weights.pt
rm -f outputs/task_02/checkpoints/_dev_probe.pt
rm -f outputs/task_02/training.json outputs/task_02/eval.json
ls -1 outputs/task_02/checkpoints/

log "retraining Stages 2-3 from the Stage 1 checkpoint"
$PY scripts/task2_run.py --stage train --resume-stage1 >> logs/task2_repair2_train.log 2>&1
log "train rc=$?"

log "re-evaluating"
$PY scripts/task2_run.py --stage eval >> logs/task2_repair2_eval.log 2>&1
log "eval rc=$?"

log "TASK2_REPAIR2_DONE"
