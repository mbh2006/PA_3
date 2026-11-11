#!/usr/bin/env bash
# Full Task 2 run: 4,500 updates (2,250 / 1,500 / 750) then final evaluation.
# Detached; checkpoints every 250 updates; stage-end weights after each stage.
set -u
cd /root/pa3
export PYTHONPATH=/root/pa3/src
PY=/venv/main/bin/python
echo "start $(date -u +%H:%M:%S)" | tee logs/task2_full.log

$PY scripts/task2_run.py --stage train >> logs/task2_train.log 2>&1
echo "TRAIN_RC=$?" | tee -a logs/task2_full.log

$PY scripts/task2_run.py --stage eval >> logs/task2_eval.log 2>&1
echo "EVAL_RC=$?" | tee -a logs/task2_full.log

echo "end $(date -u +%H:%M:%S)" | tee -a logs/task2_full.log
echo TASK2_DONE | tee -a logs/task2_full.log
