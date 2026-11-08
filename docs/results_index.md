# Results index

Factual locations of measured numbers. No interpretation is recorded here; that belongs
to the student's report.

## Task 3 — offline (complete, full supplied data)

| Result | File |
|---|---|
| Splits and live subset | `outputs/task_03/offline/splits_math.json`, `splits_mathvista.json` |
| Per-question, per-order, per-budget records | `outputs/task_03/offline/per_question_{ds}_{dev,heldout}.parquet` |
| Method accuracy / cost / oracle / groups | `outputs/task_03/tables/{ds}_{split}_methods.{csv,md}` |
| Sampling vs selection failure split | `outputs/task_03/tables/{ds}_{split}_failures.{csv,md}` |
| Paired bootstrap differences | `outputs/task_03/tables/{ds}_{split}_bootstrap.{csv,md}` |
| N=32 associations, quartiles, shortest control | `outputs/task_03/tables/{ds}_signals.{csv,md}` |
| Confidence–vision and fusion identity | `outputs/task_03/offline/signals_{ds}.json` |
| Fixed trajectories | `outputs/task_03/offline/trajectories_mathvista.json` |
| Adaptive dev grid | `outputs/task_03/tables/{ds}_adaptive_dev_grid.{csv,md}` |
| Selected thresholds + held-out results + stopping histogram | `outputs/task_03/tables/{ds}_adaptive_selected.{csv,md}` |
| Run metadata (methods, budgets, runtime) | `outputs/task_03/offline/meta_*.json` |
| Test evidence | `outputs/task_03/tests/pytest_unit.xml`, `pytest_integration.xml` |
| Smoke run (not final) | `outputs/task_03/smoke/` |

### Headline numbers (held-out, 800 questions)

| Metric | MATH | MathVista |
|---|---:|---:|
| FirstAns accuracy | 0.6465 | 0.5421 |
| Oracle coverage @32 | 0.9137 | 0.8950 |
| Self-consistency @32 | 0.7379 | 0.6170 |
| Best score+aggregation @32 | 0.7555 (SC+RWC) | 0.6300 (VisConf+RWC) |
| Adaptive agreement, held-out accuracy / tokens | 0.7347 / 10,165 | 0.6218 / 4,682 |
| Fixed-32 tokens | 21,347 | 8,898 |

## Task 1 / Task 2

No results yet. Nothing in this file may be cited as a Task 1 or Task 2 measurement.
