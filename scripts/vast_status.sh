#!/usr/bin/env bash
# Remote status for the PA3 runs.
cd /root/pa3 || exit 1
echo "UTC=$(date -u +%H:%M:%S)"
echo "RUNNING_PROCS=$(ps aux | grep -E 'task1_variant_step|vast_run_task1|task2|task3' | grep -v grep | wc -l)"
echo "GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
echo "DISK=$(df -h / | tail -1 | awk '{print $4}')"
echo "--- driver ---"
tail -4 logs/driver.log 2>/dev/null
echo "--- runs ---"
for d in outputs/task_01/runs/*/; do
  name=$(basename "$d")
  report="$d/report.json"
  if [ -f "$report" ]; then
    primary=$(/venv/main/bin/python -c "import json;d=json.load(open('$report'));print(json.dumps(d.get('benchmarks',{}).get('primary',{})))" 2>/dev/null)
    echo "$name COMPLETE $primary"
  elif [ -f "$d/report_partial.json" ]; then
    echo "$name PARTIAL"
  else
    echo "$name EMPTY"
  fi
done
echo "--- latest progress ---"
for f in logs/task1_*.log; do
  [ -f "$f" ] || continue
  latest=$(tr '\r' '\n' < "$f" | grep -v '^$' | tail -2 | tr '\n' ' ')
  echo "[$(basename "$f")] ${latest:0:200}"
done
