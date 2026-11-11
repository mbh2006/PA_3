#!/usr/bin/env bash
# One-shot setup for the ATML PA3 Task 1/2/3 runs on the rented 4090.
set -euo pipefail

cd /root
rm -rf /root/pa3
mkdir -p /root/pa3
tar -xzf /root/pa3_bundle.tgz -C /root/pa3
cd /root/pa3
mkdir -p outputs

echo "=== environment ==="
/venv/main/bin/python -V
/venv/main/bin/python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda,
      "available", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("gpu", p.name, round(p.total_memory/1e9, 1), "GB", f"{p.major}.{p.minor}")
PY

echo "=== installing pinned dependencies ==="
/venv/main/bin/python -m pip install -q --upgrade pip
/venv/main/bin/python -m pip install -q \
  "transformers==4.57.1" "tokenizers==0.22.1" "accelerate>=1.0" "datasets>=3.0" \
  "pyarrow==24.0.0" "numpy>=2.0" "pandas>=2.2" "scipy>=1.11" \
  "math-verify==0.9.0" "Levenshtein>=0.25" "PyYAML>=6.0" \
  "lm-eval==0.4.9" "sentencepiece>=0.2" "tabulate>=0.9" "matplotlib>=3.9"

echo "=== versions ==="
/venv/main/bin/python - <<'PY'
import importlib
for m in ("torch","transformers","datasets","pyarrow","numpy","scipy","lm_eval","math_verify"):
    try:
        mod = importlib.import_module(m)
        print(f"{m:14s} {getattr(mod,'__version__','?')}")
    except Exception as exc:
        print(f"{m:14s} MISSING ({exc})")
PY

echo "=== data ==="
ls -lh /root/pa3/*.npz /root/pa3/*.json /root/pa3/data/reference/ 2>/dev/null | head -20
df -h / | tail -1
echo "SETUP_DONE"
