#!/usr/bin/env bash
# Verify the Task 2 run: update counts, checkpoints, evaluation outputs.
cd /root/pa3 || exit 1
echo "=== stage update totals ==="
/venv/main/bin/python - <<'PY'
import json
d = json.load(open('/root/pa3/outputs/task_02/training.json'))
total = 0
for s in d['stages']:
    print(f"{s['stage']:20s} updates={s['updates']:5d} seconds={s['seconds']:8.1f} "
          f"lora_enabled={s['lora_enabled']} trainable_lora={s['trainable_lora']:,} "
          f"dev_points={len(s.get('dev', []))}")
    total += s['updates']
print("TOTAL_UPDATES", total)
print("calibration", d['calibration'])
PY
echo "=== checkpoints ==="
ls outputs/task_02/checkpoints/ | sort | head -30
echo "count: $(ls outputs/task_02/checkpoints/*.pt 2>/dev/null | wc -l)"
echo "=== evaluation summary ==="
/venv/main/bin/python - <<'PY'
import json
d = json.load(open('/root/pa3/outputs/task_02/eval.json'))
for suite, v in d.items():
    print(f"[{suite}] pairs={v['n_pairs']} examples={v['n_examples']}")
    for c in ('correct', 'complementary', 'mismatched', 'no_image'):
        print(f"   {c:14s} acc={v['conditions'][c]['accuracy']:.4f} "
              f"empty={v['conditions'][c]['empty_rate']:.4f}")
    print(f"   pair_acc={v['pair_accuracy']['correct']:.4f} "
          f"identical={v['identical_answer_rate']:.4f}")
    print(f"   nll correct={v['nll_correct']:.4f} comp={v['nll_complementary']:.4f} "
          f"noimg={v['nll_no_image']:.4f} swap_penalty={v['swap_penalty']:.4f}")
    b = v['bootstrap_accuracy_advantage']
    print(f"   bootstrap acc advantage mean={b['mean']:.4f} lo={b['lo']:.4f} hi={b['hi']:.4f} n={b['n']}")
PY
echo "=== weight files ==="
ls -lh outputs/task_02/checkpoints/*weights.pt
