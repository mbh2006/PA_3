#!/usr/bin/env bash
# Create the Linux (WSL) environment used for the CPU Task 3 pipeline and tests.
set -euo pipefail
VENV="${HOME}/.venvs/atml-pa3"
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --upgrade pip -q
"$VENV/bin/python" -m pip install -q \
  "pyarrow==24.0.0" numpy pandas scipy pyyaml pytest "math-verify==0.9.0" Levenshtein tqdm
"$VENV/bin/python" - <<'PY'
import pyarrow, numpy, math_verify, Levenshtein
print("pyarrow", pyarrow.__version__)
print("numpy", numpy.__version__)
print("math_verify ok, Levenshtein ok")
PY
