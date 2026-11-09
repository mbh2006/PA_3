"""Kaggle kernel: Task 1 GPU feasibility test with the assignment's real settings.

Private script kernel on a single T4. Installs pinned dependencies, copies the code
dataset into /kaggle/working, runs the environment preflight, then one real optimizer
update for full fine-tuning and for LoRA-16.
"""

import os
import shutil
import subprocess
import sys

REPO = "/kaggle/working/repo"


def _find_code_root(marker="src/task1/train.py"):
    """Prefer the dataset mount over kernel outputs (which may hold a stale code copy)."""
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



def run(cmd, check=True):
    print(f"\n$ {cmd}", flush=True)
    rc = subprocess.call(cmd, shell=True)
    print(f"  (exit {rc})", flush=True)
    if check and rc != 0:
        raise SystemExit(f"FAILED: {cmd}")
    return rc


def find_input(marker="src/task1/train.py"):
    return _find_code_root(marker)


def main():
    run("nvidia-smi", check=False)
    run("df -h /kaggle/working | tail -1", check=False)

    if os.path.exists(REPO):
        shutil.rmtree(REPO)
    src = find_input()
    run(f"cp -r {src} {REPO}")
    os.chdir(REPO)

    run(f"{sys.executable} -m pip install -q "
        "'transformers==4.57.1' 'tokenizers==0.22.1' 'accelerate>=1.0' 'datasets>=3.0' "
        "'pyarrow==24.0.0' 'numpy>=2.0' 'pandas>=2.2' 'scipy>=1.11' "
        "'math-verify==0.9.0' 'Levenshtein>=0.25' 'PyYAML>=6.0'")

    env = dict(os.environ, PYTHONPATH=f"{REPO}/src", ATML_OUTPUT_ROOT=f"{REPO}/outputs")
    print("\n=== preflight ===", flush=True)
    subprocess.call([sys.executable, "scripts/gpu_preflight.py",
                     "--out", f"{REPO}/outputs/environment/preflight.json"], env=env)

    print("\n=== GPU training test ===", flush=True)
    rc = subprocess.call([sys.executable, "scripts/kaggle_gpu_test.py"], env=env)
    print("test exit:", rc, flush=True)

    run("find outputs -type f | sort", check=False)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
