# Results index

Factual locations of every measured number. No interpretation is recorded here; that
belongs to the student's report.

Published copies live under `results/`. The working trees `outputs/` and `data/` are
gitignored and hold the larger intermediates (weights, Parquet, caches).

---

## Task 1 — fine-tuning

| Result | File |
|---|---|
| Ordered training source indices | `results/task_01/train_source_indices.json` |
| Seven-condition table with deltas in pp | `results/task_01/tables/condition_table.{csv,md}` |
| Raw lm-evaluation-harness output per condition | `results/task_01/runs/<condition>/eval/*.json` |
| Full report (benchmarks, diagnostics, displacement) | `results/task_01/runs/<condition>/report.json` |
| Per-update loss / LR / grad norm / scaler | `results/task_01/runs/<condition>/train/train_log.jsonl` |
| Resolved training configuration | `results/task_01/runs/<condition>/train/config.json` |
| Resource data (params, time, peak VRAM, final loss) | `results/task_01/runs/<condition>/train/train_summary.json`, `run_summary.json` |
| Checkpoint retention decisions | `results/task_01/runs/<condition>/train/checkpoint_retention.json` |
| Qualitative responses (24 per condition) | `results/task_01/runs/<condition>/qualitative/` |
| 42-row annotation sheet | `results/task_01/examples/annotation_sheet.json` |
| Figures (benchmarks, deltas, loss curves, diagnostics) | `results/task_01/figures/*.{png,svg}` |
| Checkpoint loss evidence | `docs/tasks/task_01_checkpoint_inventory.md` |

### Headline numbers (test splits, 7 conditions)

| Condition | GSM8K flexible | HellaSwag acc_norm | BoolQ acc | ARC-Easy acc_norm |
|---|---:|---:|---:|---:|
| Baseline | 0.4306 | 0.4733 | 0.6437 | 0.5602 |
| Full FT @ 625 (reconstructed) | 0.5724 | 0.4563 | 0.5859 | 0.6229 |
| Full FT @ 1250 | 0.6391 | 0.4587 | 0.5679 | 0.6364 |
| Full FT + L2-SP | 0.6513 | 0.4662 | 0.6193 | 0.6343 |
| LoRA-4 | 0.6027 | 0.4732 | 0.6706 | 0.6330 |
| LoRA-16 | 0.6232 | 0.4752 | 0.7284 | 0.6410 |
| OPLoRA-16 | 0.6156 | 0.4721 | 0.7110 | 0.6439 |

---

## Task 2 — visual grounding

| Result | File |
|---|---|
| Four conditions × two suites, pair diagnostics, NLL, bootstrap | `results/task_02/eval.json` |
| Stage updates, times, trainable sets, dev points, calibration | `results/task_02/training.json` |
| Four-condition table with CI | `results/task_02/tables/condition_table.{csv,md}` |
| Subtype accuracy (yes/no, number, other, presence, counting) | `results/task_02/tables/by_kind.{csv,md}` |
| Stage resources and realized mixture proportions | `results/task_02/tables/stages.{csv,md}` |
| Figures (conditions, NLL, subtypes, dev curves, train loss) | `results/task_02/figures/*.{png,svg}` |
| Stage 2 and Stage 3 (final) weights | `outputs/task_02/checkpoints/*.pt` (local, not published) |
| Training log of the corrected rerun | `logs/rerun_train.log`, `logs/rerun_eval.log` |
| Kaggle batch-8 evaluation, predictions, and rebuilt split IDs | `results/task_02/recovered_kaggle/` |
| Invalidated attempts | `outputs/task_02_invalid*/INVALID.md` |

### Headline numbers (final Stage 3 model)

| Condition | Natural VQAv2 (96 pairs) | Annotation grounding (64 pairs) |
|---|---:|---:|
| Correct image | 0.5333 | 0.4531 |
| Complementary partner | 0.2568 | 0.2891 |
| Mismatched (fixed unrelated) | 0.3125 | 0.3672 |
| No image | 0.0000 | 0.0000 |

Swap penalty ΔNLL: +1.4645 (natural), +0.8207 (grounding). Correct-image advantage:
+0.2766 [0.1693, 0.3797] and +0.1641 [0.0312, 0.3047], 4,000-pair bootstrap.

---

## Task 3 — offline (complete, full supplied data)

| Result | File |
|---|---|
| Recovered early-stop case records | `results/task_03/offline/early_stop_cases_{math,mathvista}.json` |
| Splits and live subset | `results/task_03/offline/splits_math.json`, `splits_mathvista.json` |
| Per-question, per-order, per-budget records | `outputs/task_03/offline/per_question_*.parquet` (local, not published) |
| Method accuracy / cost / oracle / groups | `results/task_03/tables/{ds}_{split}_methods.{csv,md}` |
| Sampling vs selection failure split | `results/task_03/tables/{ds}_{split}_failures.{csv,md}` |
| Paired bootstrap differences | `results/task_03/tables/{ds}_{split}_bootstrap.{csv,md}` |
| N=32 associations, quartiles, shortest control | `results/task_03/tables/{ds}_signals.{csv,md}` |
| Confidence–vision and fusion identity | `results/task_03/offline/signals_{ds}.json` |
| Fixed trajectories | `results/task_03/offline/trajectories_mathvista.json` |
| Adaptive dev grid | `results/task_03/tables/{ds}_adaptive_dev_grid.{csv,md}` |
| Selected thresholds, held-out results, stopping histograms | `results/task_03/tables/{ds}_adaptive_selected.{csv,md}`, `offline/adaptive_*_heldout.json` |
| Run metadata (methods, budgets, runtime) | `results/task_03/offline/meta_*.json` |
| Test evidence | `results/task_03/tests/pytest_{unit,integration}.xml` |
| Smoke run (not final, excluded from results) | `outputs/task_03/smoke/` |

### Headline numbers (held-out, 800 questions)

| Metric | MATH | MathVista |
|---|---:|---:|
| FirstAns accuracy | 0.6465 | 0.5421 |
| Oracle coverage @32 | 0.9137 | 0.8950 |
| Self-consistency @32 | 0.7379 | 0.6170 |
| Best score + aggregation @32 | 0.7555 (SC+RWC) | 0.6300 (VisConf+RWC) |
| Adaptive agreement: accuracy / tokens | 0.7347 / 10,165 | 0.6218 / 4,682 |
| Fixed-32 tokens | 21,347 | 8,898 |

---

## Task 3 — live generation

| Result | File |
|---|---|
| Per-question records for all three strategies (raw prompts and responses) | `results/task_03/live/live_results.json` |
| Accuracy, tokens, extraction, truncation, transitions | `results/task_03/live/live_summary.json` |
| Figure | `results/task_03/figures/live_strategies.*` |

| Strategy | Accuracy | Mean output tokens | Extraction | Truncation |
|---|---:|---:|---:|---:|
| Single, 3,072 cap | 0.54 | 704.74 | 0.98 | 0.02 |
| Parallel, 3×1,024 + self-consistency | 0.62 | 1,868.98 | 0.90 | 0.24 |
| Refine, 1,024 + 2,048 | 0.62 | 1,208.40 | 1.00 | 0.20 |

Revision transitions: 3 wrong→correct, 0 correct→wrong.

---

## Not produced (do not cite as available)

| Item | Status |
|---|---|
| ρ_k for the five non-OPLoRA conditions | not computed |
| Task 2 dev curves with generated accuracy | not produced (NLL points only) |
| Task 2 optional 49-token run | not run |
