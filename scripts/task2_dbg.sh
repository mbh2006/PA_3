#!/usr/bin/env bash
set -u
cd /root/pa3
export PYTHONPATH=/root/pa3/src
timeout 600 /venv/main/bin/python scripts/task2_run.py --stage train --max-updates-per-stage 1 2>&1 \
  | grep -E "/root/pa3|^RuntimeError|^TypeError|^ValueError" | tail -14
