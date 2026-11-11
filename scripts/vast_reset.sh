#!/usr/bin/env bash
# Hard reset of every PA3 process and of any process still holding GPU memory.
pkill -9 -f task1_variant_step 2>/dev/null
pkill -9 -f vast_run_task1    2>/dev/null
pkill -9 -f vast_one_test     2>/dev/null
pkill -9 -f task2_step        2>/dev/null
pkill -9 -f task3_live        2>/dev/null
sleep 2

for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  echo "killing GPU pid $pid"
  kill -9 "$pid" 2>/dev/null
done
sleep 3

echo "GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
echo "PY_PROCS=$(ps aux | grep python | grep -v grep | wc -l)"
ps aux | grep -E 'task1_variant|vast_run' | grep -v grep | head -3
echo RESET_DONE
