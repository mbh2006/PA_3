"""Dump the schema and a couple of rows of each supplied Task 3 Parquet file.

Run:  .venv/Scripts/python.exe scripts/inspect_task3_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ATML-PA3-Data"

FILES = [
    "qwen25-math/generations.parquet",
    "qwen25-math/probability_metrics.parquet",
    "qwen25vl-mathvista/generations.parquet",
    "qwen25vl-mathvista/probability_metrics.parquet",
    "qwen25vl-mathvista/attention_metrics.parquet",
]


def main() -> None:
    print("pyarrow", __import__("pyarrow").__version__)
    for rel in FILES:
        path = DATA / rel
        pf = pq.ParquetFile(path)
        print(f"\n=== {rel} ===")
        print("rows:", pf.metadata.num_rows, "row_groups:", pf.metadata.num_row_groups)
        print(pf.schema_arrow)
        cols = pf.schema_arrow.names
        batch = next(pf.iter_batches(batch_size=2, columns=cols))
        row = {c: batch.column(c)[0].as_py() for c in cols}
        for k, v in row.items():
            s = json.dumps(v, ensure_ascii=False, default=str)
            print(f"  {k}: {s[:220]}{'...' if len(s) > 220 else ''}")


if __name__ == "__main__":
    main()
