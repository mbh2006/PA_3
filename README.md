# ATML PA 3 — replication package

EE-5102/CS-6304 Advanced Topics in Machine Learning, Fall 2026, Programming Assignment 3.

Three experiments: LLM fine-tuning on mathematics (Task 1), SigLIP→Qwen visual grounding
(Task 2), and test-time compute over supplied rollouts (Task 3).

> **Report ownership.** The assignment forbids generative AI from writing any part of the
> PDF report. This repository contains code, configurations, results and technical records
> only. The 8-page report and every qualitative judgement are written by the student.
> See `docs/student_report_checklist.md`.

## Results at a glance

### Task 1 — five fine-tuning variants, four benchmarks

| Condition | GSM8K | Δ pp | HellaSwag | BoolQ | Δ pp | ARC-Easy |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.4306 | — | 0.4733 | 0.6437 | — | 0.5602 |
| Full FT @ 625 (reconstructed) | 0.5724 | +14.2 | 0.4563 | 0.5859 | −5.8 | 0.6229 |
| Full FT @ 1250 | 0.6391 | +20.9 | 0.4587 | 0.5679 | −7.6 | 0.6364 |
| Full FT + L2-SP | **0.6513** | **+22.1** | 0.4662 | **0.6193** | **−2.5** | 0.6343 |
| LoRA-4 | 0.6027 | +17.2 | 0.4732 | 0.6706 | +2.7 | 0.6330 |
| LoRA-16 | 0.6232 | +19.3 | 0.4752 | **0.7284** | +8.5 | 0.6410 |
| OPLoRA-16 | 0.6156 | +18.5 | 0.4721 | 0.7110 | +6.7 | 0.6439 |

### Task 2 — visual grounding

| Natural VQAv2 (96 pairs) | Accuracy |
|---|---:|
| Correct image | 0.5333 |
| Complementary partner | 0.2568 |
| Mismatched | 0.3125 |
| No image | 0.0000 |

Swap penalty ΔNLL +1.46 · advantage +0.277, 95% CI [+0.169, +0.380]

### Task 3 — test-time compute

| | MATH | MathVista |
|---|---:|---:|
| First answer | 0.6465 | 0.5421 |
| Self-consistency @32 | 0.7379 | 0.6170 |
| Best method @32 | 0.7555 (SC+RWC) | 0.6300 (VisConf+RWC) |
| Oracle @32 | 0.9137 | 0.8950 |

Live comparison (50 MATH questions, 3,072-token allowance): 0.54 single → 0.62 parallel
→ 0.62 refine, with 3 wrong→correct and 0 correct→wrong revisions.

## Repository layout

```
src/common/          paths and IO helpers
src/task1/           SFT pipeline: data, token-weighted trainer, L2-SP, LoRA/OPLoRA,
                     chunked LM-head loss, lm-eval wrapper, qualitative, probes
src/task2/           connector, visual-token assembly, example-weighted loss, mixture
                     sampler, grounding generation, VQAv2 pair selection, evaluation
src/task3/           rollout ingest, splits, orders, answer rules, VisConf/RWC, adaptive
                     controllers, bootstrap, signal analyses
scripts/             data builders, GPU preflight, per-task run drivers, figure scripts
tests/               unit + integration tests (267 assertions across the suites)
configs/             resolved configurations
docs/                requirements matrix, verification matrix, task records
results/             machine-readable results, tables and figures (no weights)
```

## Environment

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt               # analysis / tests (CPU)
pip install -r requirements-gpu.txt               # training / evaluation (CUDA)
export PYTHONPATH=src
```

`pyarrow==24.0.0` is mandatory: the supplied Task 3 Parquet files cannot be read by 19.x.

## Task 3 — offline pipeline

```bash
python scripts/fetch_reference_assets.py    # MathVista metadata + qualitative prompts
python -m pytest tests/unit -q
python -m pytest tests/integration -q       # verifies the supplied totals
python -m task3.run_offline --dataset both --out outputs/task_03/offline
python -m task3.report                      # CSV + Markdown tables
```

The supplied `ATML-PA3-Data/` directory is required and is **not** committed. The
expensive MATH equivalence matrices are cached under `outputs/task_03/cache/`.

## Task 1 — fine-tuning

```bash
python -m task1.prepare --out outputs/task_01/data          # 40,000 eligible examples
python scripts/task1_variant_step.py --variant baseline --out outputs/task_01/runs/baseline
python scripts/task1_variant_step.py --variant full_ft   --out outputs/task_01/runs/full_ft
# likewise: full_ft_l2sp, lora4, lora16, oplora16
```

Each invocation trains (or resumes), merges adapters, runs the four benchmarks through
lm-evaluation-harness, computes `down_proj` diagnostics and generates the 24 qualitative
responses.

## Task 2 — visual grounding

```bash
python scripts/task2_build_v2.py                  # disjoint train/dev/test selection
python scripts/task2_download_missing.py          # only the selected JPEGs
python scripts/task2_run.py --stage features      # SigLIP cache, encoder then unloaded
python scripts/task2_run.py --stage train         # 2,250 + 1,500 + 750 updates
python scripts/task2_run.py --stage eval          # four image conditions + bootstrap
```

## GPU preflight

```bash
python scripts/gpu_preflight.py --dry-run         # validate data/paths, no CUDA needed
python scripts/gpu_preflight.py                   # asserts FP32 masters + FP16 autocast
```

## Reproducibility notes

- Seeds: 6304 for data order, splits, rollout orders and analysis.
- Splits, rollout orders and every selected candidate are saved as machine-readable files.
- Gold answers are never used for Task 3 selection or stopping (unit-tested).
- Deviations and known limitations are listed in each `docs/tasks/*.md` record.

## Attribution

`math-verify` (HuggingFace), `lm-evaluation-harness` (EleutherAI), Qwen3 and Qwen2.5
(Alibaba), SigLIP (Google), MetaMathQA, GSM8K, HellaSwag, BoolQ, ARC-Easy, COCO 2014,
VQAv2, MathVista (Lupantech). MathVista's scoring rules are re-implemented from the
official `evaluation/calculate_score.py`.
