"""Kaggle GPU kernel: per-variant feasibility tests before any full run.

Covers representative and near-limit (2,048-token) batches for Full FT, Full FT + L2-SP
(CPU and GPU reference), LoRA-16, OPLoRA-16 and the Task 2 vision-language step, plus a
checkpoint save/load round trip.
"""

import os
import shutil
import subprocess
import sys

REPO = "/kaggle/working/repo"


def run(cmd, check=True, env=None):
    print(f"\n$ {cmd}", flush=True)
    rc = subprocess.call(cmd, shell=True, env=env)
    print(f"  (exit {rc})", flush=True)
    if check and rc != 0:
        raise SystemExit(f"FAILED: {cmd}")
    return rc


def find_input(marker="src/task1/train.py"):
    base = "/kaggle/input"
    for root, dirs, files in os.walk(base):
        if os.path.exists(os.path.join(root, marker)):
            return root
        if root[len(base):].count(os.sep) >= 5:
            dirs[:] = []
    raise SystemExit(f"no input containing {marker}")


def main():
    run("nvidia-smi --query-gpu=name,memory.total --format=csv", check=False)
    run("df -h /kaggle/working | tail -1", check=False)

    if os.path.exists(REPO):
        shutil.rmtree(REPO)
    src = find_input()
    print("code input:", src, flush=True)
    run(f"cp -r {src} {REPO}")
    os.chdir(REPO)
    os.environ["PYTHONPATH"] = f"{REPO}/src"
    print("mounted inputs:", flush=True)
    run("ls -R /kaggle/input | head -40", check=False)

    run(f"{sys.executable} -m pip install -q "
        "'transformers==4.57.1' 'tokenizers==0.22.1' 'accelerate>=1.0' 'datasets>=3.0' "
        "'pyarrow==24.0.0' 'numpy>=2.0' 'pandas>=2.2' 'scipy>=1.11' "
        "'math-verify==0.9.0' 'Levenshtein>=0.25' 'PyYAML>=6.0'")

    run(f"{sys.executable} scripts/kaggle_variant_tests.py", check=False)
    run("find outputs -type f | sort", check=False)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
