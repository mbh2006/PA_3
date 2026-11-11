#!/usr/bin/env bash
# Stop ONLY the Task 2 training/evaluation chain; preserve logs; label outputs invalid.
set -u
cd /root/pa3

echo "=== before ==="
pgrep -af "task2_run.py|task2_full.sh" | head -10

pkill -f "scripts/task2_run.py" 2>/dev/null
pkill -f "task2_full.sh" 2>/dev/null
sleep 4
pkill -9 -f "scripts/task2_run.py" 2>/dev/null
pkill -9 -f "task2_full.sh" 2>/dev/null
sleep 3

echo "=== after ==="
remaining=$(pgrep -af "task2_run.py|task2_full.sh" | wc -l)
echo "remaining_task2_processes=$remaining"
nvidia-smi --query-gpu=memory.used --format=csv,noheader

# preserve logs and mark every produced artefact invalid
mkdir -p outputs/task_02_invalid
mv outputs/task_02/checkpoints outputs/task_02_invalid/checkpoints 2>/dev/null
mv outputs/task_02/training.json outputs/task_02_invalid/training.json 2>/dev/null
mv outputs/task_02/eval.json outputs/task_02_invalid/eval.json 2>/dev/null

cat > outputs/task_02_invalid/INVALID.md <<'MD'
# Task 2 run — INVALID (do not use)

Training pools were built from `grounding_dev + grounding_test`, i.e. the held-out
grounding evaluation sets from val2014, instead of COCO **train2014** annotations.
Every number produced by this run is contaminated by that leak and must not be reported.

Additional defects found in the same review:
* Stage 1 used the Stage 2 mixture instead of two complete passes over the 18,000 captions.
* The sampler was created once and reused across stages instead of being rebuilt per stage.
* The generation helper placed the reference answer inside the generation prompt.
* The grounding evaluation read `consensus_a/b`, but those records carry `answer_a/b`.
* Natural VQA was scored with exact match instead of the ten-reference soft consensus.
* Development schedule, four final image conditions and the 4,000-pair bootstrap were absent.

Kept only as evidence of the failure. Superseded by the corrected run.
MD

echo "=== task 1 / task 3 untouched ==="
ls outputs/task_01/runs | wc -l
ls outputs/task_03/live/*.json 2>/dev/null | wc -l
echo STOPPED_AND_INVALIDATED
