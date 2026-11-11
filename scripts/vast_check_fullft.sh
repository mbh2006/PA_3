#!/usr/bin/env bash
# Focused check of the Full FT run: driver state, artifacts, checkpoints, eval results.
cd /root/pa3 || exit 1
echo "=== driver ==="
tail -3 logs/driver.log
echo "=== full_ft files ==="
find outputs/task_01/runs/full_ft -maxdepth 2 -type f 2>/dev/null | sort | head -20
echo "=== full_ft report ==="
if [ -f outputs/task_01/runs/full_ft/report.json ]; then
  /venv/main/bin/python - <<'PY'
import json
d = json.load(open('/root/pa3/outputs/task_01/runs/full_ft/report.json'))
print("primary:", json.dumps(d.get('benchmarks', {}).get('primary')))
print("train:", json.dumps({k: v for k, v in d.get('train', {}).items() if k in ('updates','examples_seen','peak_mem_gb','final_sft_loss')}))
print("displacement:", d.get('displacement_l2'))
PY
else
  echo "report.json MISSING"
fi
echo "=== checkpoints present ==="
ls -lh outputs/task_01/runs/full_ft/train/checkpoints/ 2>/dev/null || echo none
echo "=== eval dir ==="
ls -lh outputs/task_01/runs/full_ft/eval/ 2>/dev/null || echo "eval dir MISSING"
echo "=== qualitative ==="
ls -lh outputs/task_01/runs/full_ft/qualitative/ 2>/dev/null || echo "qualitative MISSING"
echo "=== disk ==="
df -h / | tail -1
