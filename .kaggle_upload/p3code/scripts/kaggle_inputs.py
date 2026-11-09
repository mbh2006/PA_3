"""Shared helper: locate the mounted code dataset inside a Kaggle kernel.

The code dataset mounts under ``/kaggle/input/datasets/<owner>/<slug>`` while other
kernel outputs (which may contain an older copy of the same tree) mount under
``/kaggle/input/<kernel-slug>``. Prefer the dataset mount so a stale kernel output can
never shadow the current code.
"""

import os


def find_code_root(marker="src/task1/train.py"):
    for base in ("/kaggle/input/datasets", "/kaggle/input"):
        if not os.path.isdir(base):
            continue
        for root, dirs, files in os.walk(base):
            if "task1_data" in root.split(os.sep):
                dirs[:] = []
                continue
            if os.path.exists(os.path.join(root, marker)):
                return root
            if root[len(base):].count(os.sep) >= 5:
                dirs[:] = []
    raise SystemExit(f"no mounted input containing {marker}")
