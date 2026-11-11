#!/usr/bin/env bash
set -u
cd /root/pa3
echo "nested checkpoints:"
ls -1 outputs/task_02/checkpoints/checkpoints/ 2>/dev/null | head -6
rm -f outputs/task_02/checkpoints/checkpoints/stage2_instruction_*.pt
rm -f outputs/task_02/checkpoints/checkpoints/stage3_natural_*.pt
rm -f outputs/task_02/checkpoints/stage2_instruction_weights.pt
rm -f outputs/task_02/checkpoints/stage3_natural_weights.pt
rm -f outputs/task_02/checkpoints/_dev_probe.pt
echo "--- after ---"
ls -1 outputs/task_02/checkpoints/ 2>/dev/null
echo "--- nested after ---"
ls -1 outputs/task_02/checkpoints/checkpoints/ 2>/dev/null
echo "stage1 present: $(test -f outputs/task_02/checkpoints/stage1_caption_weights.pt && echo yes)"
