"""Profile per-question MATH equivalence cost (development helper)."""

import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common.paths import TASK3_DATASETS
from task3.answers import math_equivalence_matrix

path = TASK3_DATASETS["math"]["dir"] / "generations.parquet"
pf = pq.ParquetFile(path)
limit = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

rows = []
t_all = time.time()
for gi in range(min(limit, pf.num_row_groups)):
    tbl = pf.read_row_group(gi, columns=["sample_id", "extracted_answer"])
    sid = tbl.column("sample_id")[0].as_py()
    answers = tbl.column("extracted_answer").to_pylist()
    answers = [None if a is None or not str(a).strip() else str(a).strip() for a in answers]
    t0 = time.time()
    math_equivalence_matrix(answers)
    dt = time.time() - t0
    rows.append((dt, sid, len(set(a for a in answers if a))))
    if (gi + 1) % 100 == 0:
        print(f"  {gi + 1} questions  elapsed {time.time() - t_all:.1f}s  "
              f"last {dt:.2f}s ({sid})", flush=True)

rows.sort(reverse=True)
print(f"\ntotal {time.time() - t_all:.1f}s over {len(rows)} questions")
print("slowest:")
for dt, sid, nd in rows[:20]:
    print(f"  {dt:8.2f}s  {sid:12} distinct={nd}")
print("median", sorted(r[0] for r in rows)[len(rows) // 2])
print("sum of 5 slowest", sum(r[0] for r in rows[:5]))
