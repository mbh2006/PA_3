#!/usr/bin/env bash
# Invalidate the Task 2 run and quantify missing images / checkpoint state.
set -u
cd /root/pa3

echo "=== quarantining the invalid run ==="
rm -rf outputs/task_02_invalid2
mkdir -p outputs/task_02_invalid2
cp -r outputs/task_02/checkpoints outputs/task_02_invalid2/checkpoints 2>/dev/null
cp outputs/task_02/training.json outputs/task_02_invalid2/ 2>/dev/null
cp outputs/task_02/eval.json outputs/task_02_invalid2/ 2>/dev/null
cat > outputs/task_02_invalid2/INVALID.md <<'MD'
# Task 2 run 2 — INVALID (do not use)

Two defects, both confirmed after the run:

1. **Training leak, again.** The deployed `build_pools` still built the presence and
   counting pools from `grounding_dev + grounding_test` (held-out val2014 evaluation
   sets). `grounding_train` (train2014) was written by the builder but never read.
2. **Missing images silently dropped.** `stage_features` skipped any image whose JPEG was
   absent, and the v2 selection referenced images that were never downloaded. Grounding
   test images were therefore absent from the cache, so every "with image" condition
   silently became a no-image input. That is why all four grounding conditions returned
   the identical 0.4609 — not a language-prior collapse.

Every Task 2 metric from this run is meaningless. Kept as evidence only.
MD

echo "=== image gap ==="
/venv/main/bin/python - <<'PY'
import json
from pathlib import Path
DATA = Path('/root/pa3/task2_data')
needed = json.loads((DATA / 'needed_images.json').read_text())
index = json.loads((DATA / 'images' / 'index.json').read_text())
cached = json.loads((DATA / 'features' / 'feature_cache.json').read_text())
import torch
payload = torch.load(DATA / 'features' / 'siglip_features.pt', map_location='cpu')
have_feat = {int(i) for i in payload['image_ids']}
on_disk = {int(k) for k, v in index.items() if (DATA / v).exists()}
needed = {int(i) for i in needed}
print('needed', len(needed))
print('jpeg on disk', len(needed & on_disk), 'missing jpeg', len(needed - on_disk))
print('features present', len(needed & have_feat), 'missing features', len(needed - have_feat))
missing = sorted(needed - on_disk)
(DATA / 'missing_images.json').write_text(json.dumps(missing))
print('first missing ids', missing[:10])
PY

echo "=== stage1 checkpoint ==="
ls -lh outputs/task_02/checkpoints/stage1_caption_weights.pt 2>/dev/null
/venv/main/bin/python - <<'PY'
import torch
p = '/root/pa3/outputs/task_02/checkpoints/stage1_caption_weights.pt'
d = torch.load(p, map_location='cpu')
print('keys', sorted(d))
print('connector tensors', len(d['connector']))
print('adapter tensors', len(d.get('lora', {})))
PY
