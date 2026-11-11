#!/usr/bin/env bash
# Inventory of trained artefacts per condition.
cd /root/pa3 || exit 1
for v in baseline full_ft full_ft_l2sp lora4 lora16 oplora16; do
  d="outputs/task_01/runs/$v/train"
  files=$(ls "$d"/*.pt 2>/dev/null | xargs -r -n1 basename | tr '\n' ' ')
  ck=$(ls "$d"/checkpoints/*.pt 2>/dev/null | wc -l)
  echo "$v: weights=[$files] checkpoints=$ck"
done
echo "--- lora4 last log lines ---"
tr '\r' '\n' < logs/task1_lora4.log | grep -vE '^$' | tail -5
