# ATML PA 3 — replication package

EE-5102/CS-6304 Advanced Topics in Machine Learning, Fall 2026, Programming Assignment 3.

Three experiments: LLM fine-tuning on mathematics (Task 1), SigLIP→Qwen visual grounding
(Task 2), and test-time compute over supplied rollouts (Task 3).

> **Report ownership.** The assignment forbids generative AI from writing any part of the
> PDF report. This repository contains code, configurations, raw results, tables and
> technical records only. The 8-page report and all qualitative judgements are written by
> the student. See `docs/student_report_checklist.md`.

## Repository layout

```
src/common/          paths, seeding, IO helpers
src/task3/           test-time compute pipeline (complete)
  ingest.py            load/join/validate the supplied Parquet tables
  splits.py            lexicographic -> default_rng(6304) -> 200/800
  orders.py            ten seeded rollout orderings per question
  answers.py           MATH (math-verify) and MathVista answer rules
  scores.py            VisConf, RWC, tie-breaking
  evaluate.py          budgets x orders x methods
  bootstrap.py         1,000 paired question-level resamples
  adaptive.py          agreement / quality-aware stopping
  signals.py           N=32 associations, fusion identity, trajectories
  report.py            CSV + Markdown tables
  run_offline.py       CLI entry point
src/task1/, src/task2/   (not implemented yet)
scripts/              data inspection, reference-asset fetch, GPU preflight, profiling
tests/                unit + integration tests
docs/                 requirements matrix, verification matrix, task records
configs/              run-level configuration
outputs/              generated artefacts (gitignored)
```

## Environment

CPU / analysis environment:

```bash
python -m venv .venv
. .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
export PYTHONPATH=src
```

`pyarrow==24.0.0` is mandatory: the supplied Parquet files cannot be read by 19.x
(`Repetition level histogram size mismatch`).

## Data

Place the official `ATML-PA3-Data/` folder at the repository root. It must contain:

```
ATML-PA3-Data/README.MD
ATML-PA3-Data/download_manifest.json
ATML-PA3-Data/qwen25-math/{generations,probability_metrics}.parquet
ATML-PA3-Data/qwen25vl-mathvista/{generations,probability_metrics,attention_metrics}.parquet
```

Verify it before use:

```bash
python - <<'PY'
import hashlib, json, pathlib
root = pathlib.Path("ATML-PA3-Data")
m = json.loads((root / "download_manifest.json").read_text())
for rel, spec in m["files"].items():
    h = hashlib.sha256((root / rel).read_bytes()).hexdigest()
    print(rel, "OK" if h == spec["sha256"] else "MISMATCH")
PY
```

Small reference assets (MathVista metadata, qualitative prompts) are fetched by:

```bash
python scripts/fetch_reference_assets.py     # writes data/reference/ + ASSETS.json
```

## Task 3 — offline pipeline

```bash
export PYTHONPATH=src
python -m pytest tests/unit -q
python -m pytest tests/integration -q          # full-data checks (~3 min warm cache)

# full run: 1,000 questions per dataset, dev + held-out, all budgets and controllers
python -m task3.run_offline --dataset both --out outputs/task_03/offline

# CSV + Markdown tables
python -m task3.report
```

Smoke run on 40 questions (separate output directory, never used in final tables):

```bash
python -m task3.run_offline --dataset both --limit 40 --out outputs/task_03/smoke
```

The expensive MATH pairwise-equivalence matrices are cached under
`outputs/task_03/cache/`; the first run takes ~12 min, later runs seconds.

## GPU preflight (before any paid compute)

```bash
python scripts/gpu_preflight.py --dry-run        # validates data/paths locally
python scripts/gpu_preflight.py --out outputs/environment/preflight.json
```

It asserts CUDA, VRAM, `pyarrow` 24, Parquet readability, tokenizer
`enable_thinking=False` behaviour, a real FP16 AMP forward/backward with gradient
checkpointing, and reports measured tokens/s so the plan's estimates can be replaced.

## Task 1 / Task 2

Not implemented yet. See `docs/execution_plan.md` for the session plan and
`docs/assignment_requirements.md` for the full requirement list.

## Reproducibility notes

- Seeds: 6304 for data order, splits, rollout orders and analysis.
- Splits, rollout orders and every selected candidate are saved as machine-readable files.
- Gold answers are never used for selection, stopping or normalisation (unit-tested).
- Deviations and known limitations are listed in `docs/open_questions.md`.

## Attribution

Uses `math-verify` (HuggingFace), `lm-evaluation-harness` (EleutherAI), Qwen3 and
Qwen2.5 (Alibaba), SigLIP (Google), MetaMathQA, GSM8K, HellaSwag, BoolQ, ARC, COCO and
VQAv2, MathVista (Lupantech). MathVista's scoring rules are re-implemented from the
official `evaluation/calculate_score.py`.
