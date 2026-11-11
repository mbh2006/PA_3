#!/usr/bin/env bash
# Verify per-stage trainable sets from the last run's training.json.
cd /root/pa3 || exit 1
grep -m2 "trainable" logs/task2_train.log || echo "(no stage-start print in current log)"
echo "--- previous smoke per-stage ---"
/venv/main/bin/python - <<'PY'
import json
d = json.load(open('/root/pa3/outputs/task_02/training.json'))
for s in d['stages']:
    print(s['stage'], 'lora_enabled=', s['lora_enabled'],
          'connector=', s['trainable_connector'], 'lora=', s['trainable_lora'])
PY
