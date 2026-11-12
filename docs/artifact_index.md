# Artifact index

Maps each requirement to its code, test and output. All paths are relative to the
repository root. `outputs/` and `data/` are gitignored working directories; the published
copies of the results live under `results/`.

---

## Task 1 — fine-tuning

| Requirement | Code | Test | Output |
|---|---|---|---|
| Eligibility, source order, no truncation | `src/task1/data.py` | `tests/unit/test_task1_data.py` | `runs/*/train/config.json` |
| Chat formatting, prompt mask, EOT supervision | `src/task1/data.py` | `tests/unit/test_task1_data.py` | — |
| Token-weighted effective-batch loss | `src/task1/train.py` | `tests/unit/test_task1_loss.py` | `runs/*/train/train_log.jsonl` |
| Chunked LM-head loss | `src/task1/chunked_loss.py` | `tests/unit/test_task1_chunked_loss.py` | — |
| AdamW parameter groups | `src/task1/optim.py` | `tests/unit/test_task1_l2sp_optim.py` | — |
| Schedule (38 warm-up, cosine) | `src/task1/train.py` | `tests/unit/test_task1_l2sp_optim.py` | `train_log.jsonl::lr` |
| L2-SP penalty, Ω, tied weights | `src/task1/train.py` | `tests/unit/test_task1_l2sp_optim.py` | `report.json::omega_final` |
| LoRA / OPLoRA | `src/task1/lora.py` | `tests/unit/test_task1_lora.py` | `runs/lora*/`, `runs/oplora16/` |
| Checkpoint retention | `src/task1/train.py` | `tests/unit/test_task1_checkpoint_budget.py` | `train/checkpoint_retention.json` |
| Four-benchmark evaluation | `src/task1/eval_lm.py` | `tests/unit/test_task1_eval_assets.py` | `runs/*/eval/*.json` |
| Seven-condition table | `scripts/task1_verify.py` | — | `results/task_01/tables/condition_table.{csv,md}` |
| Displacement ‖θ−θ₀‖ | `src/task1/diagnostics.py` | — | `report.json::displacement_l2` |
| η and ρ_k | `src/task1/diagnostics.py` | — | `report.json::weight_diagnostics` |
| Activation probes R(z) | `src/task1/probes.py` | — | `report.json::activation_diagnostics` |
| Qualitative 24 prompts × 7 | `src/task1/qualitative.py` | — | `runs/*/qualitative/` |
| 42-row annotation sheet | `src/task1/qualitative.py::build_annotation_sheet` (driver `scripts/task1_verify.py`) | — | `results/task_01/examples/annotation_sheet.json` |
| Figures | `scripts/make_figures.py` | — | `results/task_01/figures/*` |

---

## Task 2 — visual grounding

| Requirement | Code | Test | Output |
|---|---|---|---|
| SigLIP feature cache (196×768, LN once) | `src/task2/features.py` | `tests/unit/test_task2_core.py` | `task2_data/` (not published) |
| Connector + RMS calibration | `src/task2/model.py` | `tests/unit/test_task2_core.py` | `training.json::calibration` |
| Visual-token assembly, masks, positions | `src/task2/model.py` | `tests/unit/test_task2_core.py` | — |
| Caption cycler, two full passes | `src/task2/train.py` | `tests/unit/test_task2_fixes.py` | `training.json::stages` |
| Mixture samplers per stage | `src/task2/train.py` | `tests/unit/test_task2_fixes.py` | `training.json` |
| Example-weighted loss | `src/task2/train.py` | `tests/unit/test_task2_core.py` | — |
| Resumable checkpoint (RNG + sampler) | `src/task2/train.py` | `tests/unit/test_task2_fixes.py` | stage weights (local only) |
| Caption/grounding/VQA builders | `scripts/task2_build_v2.py` | `tests/unit/test_task2_data.py` | `task2_data/*.json` |
| Image-disjoint splits | `scripts/task2_build_v2.py` | `tests/unit/test_task2_data.py` | `split_images.json` |
| Feature completeness guard | `src/task2/data.py` | — | raises on any missing image |
| Four image conditions | `scripts/task2_recover_evidence.py` | — | `results/task_02/recovered_kaggle/metrics.json` |
| Ten-reference soft consensus | `src/task2/evaluate.py` | `tests/unit/test_task2_core.py` | `eval.json` |
| 4,000-pair bootstrap | `src/task2/evaluate.py` | — | `recovered_kaggle/metrics.json` |
| Stage resource table | `src/task2/train.py` | — | `results/task_02/training.json` |
| Tables and figures (no GPU) | `scripts/make_task2_figures.py` | — | `results/task_02/tables/`, `results/task_02/figures/` |

---

## Task 3 — test-time compute

| Requirement | Code | Test | Output |
|---|---|---|---|
| Join and validate supplied tables | `src/task3/ingest.py` | `tests/integration/test_task3_data.py` | — |
| Splits 200/800 + live subset | `src/task3/splits.py` | `tests/unit/test_splits_orders.py` | `results/task_03/offline/splits_*.json` |
| Ten seeded orders | `src/task3/orders.py` | `tests/unit/test_splits_orders.py` | derived at run time |
| MATH equivalence | `src/task3/answers.py` | `tests/unit/test_answers.py` | `outputs/task_03/cache/math_equivalence.npz` |
| MathVista rules | `src/task3/answers.py` | `tests/unit/test_answers.py` | — |
| NO_ANSWER handling | `src/task3/answers.py` | unit + integration | — |
| VisConf | `src/task3/scores.py` | `tests/unit/test_scores.py` | `signals_*.json` |
| RWC and tie-breaks | `src/task3/scores.py` | `tests/unit/test_scores.py` | `summary_*.json` |
| Method grid | `src/task3/evaluate.py` | `tests/unit/test_adaptive.py` | `summary_{ds}_{split}.json` |
| Accuracy/cost/oracle/groups | `src/task3/evaluate.py` | — | `tables/*_methods.{csv,md}` |
| Failure decomposition | `src/task3/run_offline.py` | — | `tables/*_failures.{csv,md}` |
| Paired bootstrap | `src/task3/bootstrap.py` | `tests/unit/test_bootstrap_signals.py` | `tables/*_bootstrap.{csv,md}` |
| N=32 associations | `src/task3/signals.py` | `tests/unit/test_bootstrap_signals.py` | `signals_*.json`, `tables/*_signals.*` |
| Confidence–vision + fusion identity | `src/task3/signals.py` | `tests/unit/test_bootstrap_signals.py` | `signals_mathvista.json` |
| Fixed trajectories | `src/task3/signals.py` | — | `offline/trajectories_mathvista.json` |
| Adaptive controllers | `src/task3/adaptive.py` | `tests/unit/test_adaptive.py` | `adaptive_*_dev_grid.json` |
| Frozen thresholds | `src/task3/run_offline.py` | — | `adaptive_*_selected.json` |
| Held-out adaptive results | `src/task3/run_offline.py` | — | `adaptive_*_heldout.json` |
| Tables | `src/task3/report.py` | — | `results/task_03/tables/` |
| Live generation | `scripts/task3_live.py` | — | `results/task_03/live/` |
| Figures | `scripts/make_figures.py` | — | `results/task_03/figures/*` |
| Test evidence | — | — | `results/task_03/tests/pytest_{unit,integration}.xml` |

---

## Environment and provenance

| Artefact | Path |
|---|---|
| Dry-run preflight report | `outputs/environment/preflight_dryrun.json` |
| GPU preflight (variant tests) | `outputs/environment/variant_tests.json` |
| Python environment (analysis) | `.venv/` (gitignored) |
| Reference assets + checksums | `data/reference/ASSETS.json` |
| Requirements files | `requirements-dev.txt`, `requirements-gpu.txt` |
| Resolved configs | `configs/`, plus `runs/*/train/config.json` |

---

## Documentation map

| Document | Purpose |
|---|---|
| `docs/assignment_requirements.md` | requirement → status matrix |
| `docs/verification_matrix.md` | gates A–D with commands and evidence |
| `docs/results_index.md` | where every reported number lives |
| `docs/artifact_index.md` | this file: requirement → code → test → output |
| `docs/tasks/task_0{1,2,3}_*.md` | per-task technical records |
| `docs/tasks/task_01_checkpoint_inventory.md` | checkpoint-loss evidence |
