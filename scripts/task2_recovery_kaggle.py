"""Private Kaggle entry point for Task 2 evaluation-only recovery."""

import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from zipfile import ZipFile


def main():
    working = Path("/kaggle/working")
    matches = list(Path("/kaggle/input").rglob("recovery_config.json"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one private Task 2 bundle, found {len(matches)}")
    bundle_dir = matches[0].parent
    root = bundle_dir / "task2_recovery_source"
    if not (root / "recover.py").is_file():
        root = working / "task2_recovery_code"
        with ZipFile(bundle_dir / "task2_recovery_source.zip") as bundle:
            bundle.extractall(root)
    if importlib.metadata.version("transformers") != "4.57.1":
        subprocess.run([sys.executable, "-m", "pip", "install", "transformers==4.57.1"],
                       check=True)
    data = working / "task2_recovery_data"
    output = working / "task2_recovered_evidence"
    output.mkdir(exist_ok=True)
    env = dict(os.environ, ATML_TASK2_DATA=str(data), PYTHONPATH=str(root / "src"),
               HF_HUB_DISABLE_TELEMETRY="1")
    start = perf_counter()
    subprocess.run([sys.executable, "-u", str(root / "scripts/task2_build_v2.py"),
                    "--download-annotations"], env=env, check=True)
    preparation_seconds = perf_counter() - start
    subprocess.run([sys.executable, "-u", str(root / "recover.py"), str(bundle_dir),
                    str(output)], env=env, check=True)
    (output / "elapsed.json").write_text(json.dumps({
        "annotation_download_and_build_seconds": preparation_seconds,
        "total_data_and_evaluation_seconds": perf_counter() - start,
    }, indent=2))


if __name__ == "__main__":
    main()
