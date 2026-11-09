"""Kaggle CPU kernel: Task 1 data preparation (no GPU, no GPU quota).

Installs dependencies, copies the code dataset, selects the first 40,000 eligible
MetaMathQA examples in source order, tokenises them with the Qwen3 chat template
(thinking disabled), and writes a compact artefact plus the fixed probe set.

Outputs (mounted by later GPU kernels via kernel_sources):
    /kaggle/working/task1_data/prepared.npz
    /kaggle/working/task1_data/prepared.json
    /kaggle/working/task1_data/probes.json
"""

import json
import os
import shutil
import subprocess
import sys

REPO = "/kaggle/working/repo"
OUT = "/kaggle/working/task1_data"


def run(cmd, check=True):
    print(f"\n$ {cmd}", flush=True)
    rc = subprocess.call(cmd, shell=True)
    print(f"  (exit {rc})", flush=True)
    if check and rc != 0:
        raise SystemExit(f"FAILED: {cmd}")
    return rc


def find_input(marker="src/task1/prepare.py"):
    base = "/kaggle/input"
    for root, dirs, files in os.walk(base):
        if os.path.exists(os.path.join(root, marker)):
            return root
        if root[len(base):].count(os.sep) >= 5:
            dirs[:] = []
    raise SystemExit(f"no input containing {marker}")


def main():
    run("nproc; free -g | head -2; df -h /kaggle/working | tail -1", check=False)

    if os.path.exists(REPO):
        shutil.rmtree(REPO)
    src = find_input()
    print("code input:", src, flush=True)
    run(f"cp -r {src} {REPO}")
    os.chdir(REPO)
    os.environ["PYTHONPATH"] = f"{REPO}/src"
    print("PYTHONPATH:", os.environ["PYTHONPATH"], flush=True)

    run(f"{sys.executable} -m pip install -q "
        "'transformers==4.57.1' 'tokenizers==0.22.1' 'datasets>=3.0' "
        "'pyarrow==24.0.0' 'numpy>=2.0' 'PyYAML>=6.0'")

    env = dict(os.environ, PYTHONPATH=f"{REPO}/src")
    run(f"{sys.executable} -m task1.prepare --out {OUT} --n-examples 40000 --max-seq-len 2048",
        check=True)

    run(f"ls -lh {OUT}", check=False)
    meta = json.load(open(f"{OUT}/prepared.json"))
    print("\n=== summary ===", flush=True)
    print(json.dumps({k: v for k, v in meta.items() if k != "selection"}, indent=2)[:1500])
    print("\nselection:", json.dumps(meta["selection"], indent=2)[:1200])
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
