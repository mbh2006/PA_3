#!/usr/bin/env bash
# One-shot environment probe for the WSL/Linux side of the Task 3 CPU pipeline.
set -u
echo "== python =="
python3 --version 2>&1 || true
python3 -c 'import sys; print(sys.executable)' 2>&1 || true
python3 -c 'import venv; print("venv-ok")' 2>&1 || true
echo "== tools =="
command -v pip3 || true
command -v uv || true
command -v git || true
echo "== disk =="
df -h / | tail -1
echo "== cpu =="
nproc
echo "== net =="
python3 - <<'PY' 2>&1 || true
import urllib.request
try:
    with urllib.request.urlopen("https://pypi.org/simple/", timeout=15) as r:
        print("pypi", r.status)
except Exception as e:
    print("pypi error", e)
PY
