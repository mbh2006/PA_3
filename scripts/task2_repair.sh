#!/usr/bin/env bash
# Repair path: fill missing images, rebuild features, resume Stages 2-3, re-evaluate.
set -u
cd /root/pa3
export PYTHONPATH=/root/pa3/src
PY=/venv/main/bin/python
log() { echo "=== $(date -u +%H:%M:%S) $* ===" | tee -a logs/task2_repair.log; }

log "downloading missing images"
$PY scripts/task2_download_missing.py >> logs/task2_repair_images.log 2>&1
log "images rc=$?"

log "rebuilding feature cache"
rm -rf task2_data/features
$PY scripts/task2_run.py --stage features >> logs/task2_repair_features.log 2>&1
log "features rc=$?"

log "resuming Stages 2-3 from Stage 1"
$PY scripts/task2_run.py --stage train --resume-stage1 >> logs/task2_repair_train.log 2>&1
log "train rc=$?"

log "re-evaluating"
$PY scripts/task2_run.py --stage eval >> logs/task2_repair_eval.log 2>&1
log "eval rc=$?"

log "TASK2_REPAIR_DONE"
