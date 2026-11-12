"""Canonical paths and run-directory helpers."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "ATML-PA3-Data"
REFERENCE_DIR = ROOT / "data" / "reference"
OUTPUT_ROOT = Path(os.environ.get("ATML_OUTPUT_ROOT", ROOT / "outputs"))
TASK2_DATA = Path(os.environ.get("ATML_TASK2_DATA", ROOT / "task2_data"))
TASK3_OUT = OUTPUT_ROOT / "task_03"

TASK3_DATASETS = {
    "math": {
        "dir": DATA_DIR / "qwen25-math",
        "generations": "generations.parquet",
        "probability": "probability_metrics.parquet",
        "attention": None,
        "is_vlm": False,
    },
    "mathvista": {
        "dir": DATA_DIR / "qwen25vl-mathvista",
        "generations": "generations.parquet",
        "probability": "probability_metrics.parquet",
        "attention": "attention_metrics.parquet",
        "is_vlm": True,
    },
}

MATHVISTA_METADATA = REFERENCE_DIR / "mathvista_testmini.json"
FIRST_PLOT_QUESTIONS = REFERENCE_DIR / "first_plot_questions.yaml"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_dir(task: str, tag: str) -> Path:
    """Create (and return) a timestamped run directory under outputs/<task>/runs/."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = OUTPUT_ROOT / task / "runs" / f"{stamp}_{tag}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)
        fh.write("\n")


def _json_default(obj):
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")
