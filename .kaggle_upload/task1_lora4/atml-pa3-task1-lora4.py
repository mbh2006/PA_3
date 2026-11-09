"""ATML PA3 Task 1 - lora4. Private GPU kernel."""

import os
import shutil
import subprocess
import sys

REPO = "/kaggle/working/repo"
VARIANT = "lora4"


def run(cmd, check=True):
    print(f"\n$ {cmd}", flush=True)
    rc = subprocess.call(cmd, shell=True)
    print(f"  (exit {rc})", flush=True)
    if check and rc != 0:
        raise SystemExit(f"FAILED: {cmd}")
    return rc


def _find(marker="src/task1/train.py"):
    for base in ("/kaggle/input/datasets", "/kaggle/input"):
        if not os.path.isdir(base):
            continue
        for root, dirs, files in os.walk(base):
            if os.path.exists(os.path.join(root, marker)):
                return root
            if root[len(base):].count(os.sep) >= 5:
                dirs[:] = []
    raise SystemExit(f"no input containing {marker}")


def main():
    run("nvidia-smi --query-gpu=name,memory.total --format=csv", check=False)
    if os.path.exists(REPO):
        shutil.rmtree(REPO)
    src = _find()
    run(f"cp -r {src} {REPO}")
    os.chdir(REPO)
    os.environ["PYTHONPATH"] = f"{REPO}/src"
    run("df -h /kaggle/working | tail -1", check=False)

    run(f"{sys.executable} -m pip install -q "
        "'transformers==4.57.1' 'tokenizers==0.22.1' 'accelerate>=1.0' 'datasets>=3.0' "
        "'pyarrow==24.0.0' 'numpy>=2.0' 'pandas>=2.2' 'scipy>=1.11' "
        "'math-verify==0.9.0' 'Levenshtein>=0.25' 'PyYAML>=6.0' "
        "'lm-eval==0.4.9' 'sentencepiece>=0.2'")
    run("python -c \"import lm_eval, transformers; print('lm_eval', lm_eval.__version__)\"",
        check=False)

    run(f"{sys.executable} scripts/task1_variant_step.py --variant {VARIANT} "
        f"--out outputs/task_01/runs/{VARIANT}", check=False)
    run("find outputs/task_01 -maxdepth 3 -type f | sort | head -40", check=False)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
