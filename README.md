# ATML PA 3

Reproduction package for EE-5102/CS-6304 Advanced Topics in Machine Learning,
Fall 2026. The three tasks cover mathematical fine-tuning, visual grounding,
and test-time compute.

The repository provides implementation code, run configurations, environment
specifications, saved results, and reproduction commands. The student-written
report is a separate LMS deliverable. The course PDF, report workspace, raw
datasets, and model checkpoints are kept locally.

## Saved evidence

| Task | Where to start |
| --- | --- |
| 1. Mathematical fine-tuning | [Training source IDs](results/task_01/train_source_indices.json), [seven-condition benchmark table](results/task_01/tables/condition_table.csv), [figures](results/task_01/figures), [run records](results/task_01/runs), and [42-response annotation input](results/task_01/examples/annotation_sheet.json) |
| 2. Visual grounding | [Original training and evaluation artifacts](results/task_02) and [Kaggle batch-8 evaluation record](results/task_02/recovered_kaggle/RECOVERY_RECORD.md) |
| 3. Test-time compute | [Offline tables](results/task_03/tables), [split IDs and adaptive results](results/task_03/offline), and [early-stop cases for MATH](results/task_03/offline/early_stop_cases_math.json) and [MathVista](results/task_03/offline/early_stop_cases_mathvista.json), and [50-question live summary](results/task_03/live/live_summary.json) |

The Task 2 Kaggle rerun saved 1,280 predictions, per-condition metrics with
pair-bootstrap intervals, and caption development loss at updates 3,750 and
4,500. The corrected builder recreated the evaluation split. The historical
split manifest was unavailable for an exact ID comparison. The table below uses
only the [Kaggle rerun results](results/task_02/recovered_kaggle/condition_table.csv).

| Suite | Correct image | Complementary | Mismatched | No image |
| --- | ---: | ---: | ---: | ---: |
| Grounding exact match | 45.31% | 28.91% | 36.72% | 0.00% |
| Natural VQAv2 soft consensus | 53.85% | 25.16% | 31.25% | 0.00% |

## Environment

Use Python 3.12 for the commands below. Task 3 offline analysis runs on CPU.
Training, Task 2 model evaluation, and Task 3 live generation require CUDA.
Install the GPU requirements in a CUDA-compatible environment. The Kaggle
evaluation used Transformers 4.57.1.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
# Install for model runs and GPU-dependent tests.
python -m pip install -r requirements-gpu.txt
export PYTHONPATH="$PWD/src"
```

[CPU dependencies](requirements-dev.txt) and [GPU dependencies](requirements-gpu.txt)
include PyArrow 24.0.0 for the supplied Parquet files.

## Task 1 · Mathematical fine-tuning

The published evaluation driver expects the checkout at `/root/pa3` on Linux
or the prepared inputs mounted on Kaggle. The commands below use the Linux
layout. Preparation selects 40,000 eligible MetaMathQA examples and writes the
fixed evaluation probes. Supply the course's `first_plot_questions.yaml` at
`data/reference/first_plot_questions.yaml`. The reference fetcher can obtain
the public copy, which should be checked against the course copy.

```bash
cd /root/pa3
export PYTHONPATH="$PWD/src"
python scripts/fetch_reference_assets.py
python -m task1.prepare --out outputs/task_01/data
cp outputs/task_01/data/prepared.npz outputs/task_01/data/probes.json .
for variant in baseline full_ft full_ft_l2sp lora4 lora16 oplora16; do
  python scripts/task1_variant_step.py --variant "$variant" --out "outputs/task_01/runs/$variant"
done
python scripts/task1_variant_step.py --variant full_ft --max-updates 625 \
  --out outputs/task_01/runs/full_ft_midpoint
```

Keep the generated root-level `prepared.npz` and `probes.json` local. The
five training runs start from the original Qwen3-0.6B checkpoint. The
published update-625 Full FT result comes from a labelled reconstruction.
The [comparison record](results/task_01/runs/full_ft_midpoint/reconstruction_check.json)
documents the checks against the original training log. Subspace interference
`rho_k` is available for OPLoRA-16. The archived benchmark JSONs contain
aggregate scores.

## Task 2 · Visual grounding

The original training driver and image downloader expect the Linux checkout
at `/root/pa3`. They use COCO 2014 captions and instance annotations, VQAv2,
and frozen SigLIP features. The commands below use the default `task2_data/`
directory.

```bash
cd /root/pa3
export PYTHONPATH="$PWD/src"
python scripts/task2_build_v2.py --download-annotations
python scripts/task2_download_missing.py
python scripts/task2_run.py --stage features
python scripts/task2_run.py --stage train
python scripts/task2_run.py --stage eval
```

These commands use the original evaluation driver. To reproduce the batch-8
evaluation with the saved Stage 2 and Stage 3 weights, follow the
[Kaggle recovery instructions](results/task_02/recovered_kaggle/RECOVERY_RECORD.md).
The two recovered caption development measurements cover only the saved stage
endpoints. The earlier aggregate results are archived in
[`results/task_02/eval.json`](results/task_02/eval.json).

## Task 3 · Test-time compute

Place the course-supplied `ATML-PA3-Data/` directory beside the README. The
offline pipeline reads the supplied MATH and MathVista rollouts. It selects
adaptive thresholds on development data and evaluates the frozen policies on
held-out prefixes.

```bash
python scripts/fetch_reference_assets.py
python -m pytest tests/integration -q
python -m task3.run_offline --dataset both --out outputs/task_03/offline
python -m task3.report --offline outputs/task_03/offline --tables outputs/task_03/tables
```

For live inference, run the CUDA driver from the checkout at `/root/pa3`. It
uses the saved [50-question input](task3_live_questions.json) and the model
revision in the [live summary](results/task_03/live/live_summary.json).

```bash
python scripts/task3_live.py --limit 50
```

The archived parallel strategy used literal-string voting over extracted
answers. The live summary records total generation time and generated tokens.
Actual input-token counts and per-strategy runtimes were unavailable.

## Verification and attribution

Use `python -m pytest tests/unit -q` after installing the dependencies.
`python scripts/gpu_preflight.py` checks the GPU environment before a model run.
Run settings, seeds, saved scores, and available split IDs are recorded with the
results. Further implementation details and deviations are in the
[technical records](docs/tasks).

Models and tools include Qwen3 and Qwen2.5 (Alibaba), SigLIP (Google),
lm-evaluation-harness (EleutherAI), and math-verify (Hugging Face).
Data sources include MetaMathQA, GSM8K, HellaSwag, BoolQ, ARC-Easy, COCO 2014,
VQAv2, MATH, and MathVista. The MathVista answer rules are reimplemented from
the [official scoring code](https://github.com/lupantech/MathVista/blob/main/evaluation/calculate_score.py).
The Task 1 public prompt copy comes from
[Emergent Misalignment](https://github.com/emergent-misalignment/emergent-misalignment/blob/main/evaluation/first_plot_questions.yaml).
VQAv2 attribution for the saved prediction records is in the
[Task 2 recovery record](results/task_02/recovered_kaggle/RECOVERY_RECORD.md).
