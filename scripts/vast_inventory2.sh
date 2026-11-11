#!/usr/bin/env bash
# Task 2 progress + Task 1 checkpoint inventory and midpoint label check.
cd /root/pa3 || exit 1
echo "UTC $(date -u +%H:%M)"
echo "=== task2 ==="
tail -2 logs/task2_train.log
ls outputs/task_02/checkpoints/ 2>/dev/null | tail -3
echo "=== task1 recovery checkpoints ==="
for v in full_ft full_ft_l2sp full_ft_midpoint lora4 lora16 oplora16; do
  n=$(ls outputs/task_01/runs/$v/train/checkpoints/*.pt 2>/dev/null | wc -l)
  echo "$v: $n checkpoints"
done
echo "=== midpoint reconstruction label ==="
if [ -f outputs/task_01/runs/full_ft_midpoint/train/RECONSTRUCTED.json ]; then
  cat outputs/task_01/runs/full_ft_midpoint/train/RECONSTRUCTED.json
else
  echo "MISSING RECONSTRUCTED.json"
fi
echo "=== retention log (original full_ft) ==="
cat outputs/task_01/runs/full_ft/train/checkpoint_retention.json 2>/dev/null | head -20
