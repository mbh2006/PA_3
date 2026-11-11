# ATML PA 3 — complete technical record

**Course:** EE-5102/CS-6304 Advanced Topics in Machine Learning, Fall 2026
**Assignment:** Programming Assignment 3, released 23 Oct 2026, 15 pages
**Source hash:** `ATML-PA3.pdf` SHA-256 `66766f7aafb6046f617d990399dad687ab79b37b8d963cb60bdb4cb0d8871f62`
**Student ID:** 28100371
**Status:** all three tasks executed; results verified locally.

> This document is an operational record: configurations, measured numbers, artefact
> locations, and disclosed deviations. It contains no report prose. The PA forbids
> generative AI from writing any part of the PDF report, so interpretation, discussion
> and every qualitative judgement remain with the student. See §9.

---

## 1. Environment and provenance

| Item | Value |
|---|---|
| Training GPU | NVIDIA RTX 4090, 25.3 GB |
| Torch / CUDA | 2.14.1+cu132 / CUDA 13.2 |
| Transformers / tokenizers | 4.57.1 / 0.22.1 |
| lm-evaluation-harness | 0.4.9 |
| Task 3 data / analysis | `pyarrow==24.0.0`, numpy 2.5.4, scipy 1.18.1 |
| Local analysis host | Windows + WSL2, Python 3.12, CPU only (no torch) |
| Global seed | 6304 |
| Task 3 scorer | `task3-scorer-v1`, frozen before evaluation |

Reference assets fetched with `scripts/fetch_reference_assets.py`:

| Asset | Source | Size |
|---|---|---|
| MathVista `testmini.json` | `lupantech/MathVista` | 871,725 B |
| `first_plot_questions.yaml` | `emergent-misalignment` | 12,016 B (8 questions × 3 formats) |

The supplied `ATML-PA3-Data/` was re-hashed against `download_manifest.json`.

---

## 2. Task 1 — fine-tuning on mathematics

### 2.1 Data selection

Rules implemented in `src/task1/data.py::select_examples`, applied in **source order**
(never shuffled, answers never truncated):

1. drop empty question or response;
2. drop fully formatted examples longer than **2,048 tokens** (length measured on the
   complete chat-formatted sequence);
3. drop any query or `original_question` matching a normalised GSM8K test question.

Every training run reports `examples_seen = 40,000` in its `train_summary.json`, i.e. the
prescribed 40,000 examples were consumed exactly once (1,250 updates × 32).

Exclusion counters per reason were emitted during preparation on the GPU host. Those
counters are **not** in the published bundle — the preparation step is deterministic and
reproducible with `python -m task1.prepare`, but the exact per-reason counts are not
available locally and are therefore not quoted here.

Formatting: `apply_chat_template(..., enable_thinking=False)`; the prompt is masked;
**assistant tokens and the end-of-turn token are supervised**; the separator newline after
`<|im_end|>` is masked. Qwen3 renders an *empty* `<think></think>` block in non-thinking
mode — that is correct behaviour, not a think block leaking into generation.

### 2.2 Training configuration

| Setting | Value |
|---|---|
| Updates / examples | 1,250 / 40,000 |
| Micro-batch × accumulation | 2 × 16 (effective 32) |
| Loss | token-weighted over the **effective batch** (ΣCE / Σresponse-tokens) |
| Optimizer | AdamW β=(0.9,0.999), ε=1e-8, wd 0.01 (biases and <2-D excluded) |
| Schedule | cosine, **38 warm-up updates** |
| Clip | global norm 1.0, unscaled before clipping |
| Precision | FP16 autocast, **FP32 master parameters**, FP32 optimizer states, dynamic scaling |
| Full FT peak LR | 2e-5 |
| Adapter peak LR | 2e-4 |
| LoRA | r=4/α=8 and r=16/α=32, dropout 0.05, random A / zero B, 7 projections per layer |
| OPLoRA | r=16/α=32, protection rank k=128, FP32 bases computed once |
| L2-SP | λ=1e-3, θ₀ fixed, Ω=0.5·Σ(θ−θ₀)² over all unique parameters |

Engineering notes: the penalty gradient is added **once per optimizer update**, never
multiplied by the accumulation count; a length-triggered chunked vocabulary projection is
used above 512 tokens and proven equivalent to the standard path; over-budget micro-batches
are split into single examples with identical gradients; skipped scaler updates are logged
and retried (`skipped_updates = 0` everywhere here).

### 2.3 Results — seven conditions, four benchmarks (held-out test splits)

Source: `results/task_01/tables/condition_table.csv`.

| Condition | GSM8K flex | Δ pp | HellaSwag acc_norm | Δ pp | BoolQ acc | Δ pp | ARC-Easy acc_norm | Δ pp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.4306 | — | 0.4733 | — | 0.6437 | — | 0.5602 | — |
| Full FT @ 625 (reconstructed) | 0.5724 | +14.18 | 0.4563 | −1.70 | 0.5859 | −5.78 | 0.6229 | +6.27 |
| Full FT @ 1250 | 0.6391 | +20.85 | 0.4587 | −1.46 | 0.5679 | −7.58 | 0.6364 | +7.62 |
| Full FT + L2-SP | **0.6513** | **+22.06** | 0.4662 | −0.71 | **0.6193** | **−2.45** | 0.6343 | +7.41 |
| LoRA-4 | 0.6027 | +17.21 | 0.4732 | −0.01 | 0.6706 | +2.69 | 0.6330 | +7.28 |
| LoRA-16 | 0.6232 | +19.26 | 0.4752 | +0.19 | **0.7284** | **+8.47** | 0.6410 | +8.08 |
| OPLoRA-16 | 0.6156 | +18.50 | 0.4721 | −0.12 | 0.7110 | +6.73 | 0.6439 | +8.38 |

Raw (non-normalised) accuracy, same order: HellaSwag 0.3755 → 0.3688 / 0.3731 / 0.3768 /
0.3753 / 0.3787 / 0.3776; ARC-Easy 0.6086 → 0.6604 / 0.6658 / 0.6633 / 0.6557 / 0.6671 /
0.6684.

Protocol: `gsm8k_cot`, 0-shot, non-thinking chat template, greedy, 1,024 new tokens, stock
stops and `flexible-extract`, complete 1,319-example test split. The three likelihood
benchmarks use stock completion prompts **without** a chat template on their complete
splits (HellaSwag 10,042 / BoolQ 3,270 / ARC-Easy 2,376), FP16, dropout disabled.
Evaluation batch: GSM8K 32, likelihood benchmarks **8** (identical for all seven
conditions, chosen for memory headroom rather than speed).

### 2.4 Training resources (measured)

Source: `results/task_01/runs/<condition>/train/train_summary.json` and `run_summary.json`.

| Condition | Trainable params | Updates | Examples | Skipped | Wall clock | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| Full FT (reconstruction @ 625) | 596,049,920 | 625 | 20,000 | 0 | 1,313.0 s | 17.36 GB |
| Full FT | 596,049,920 | 1,250 | 40,000 | 0 | 2,718.1 s | 17.36 GB |
| Full FT + L2-SP | 596,049,920 | 1,250 | 40,000 | 0 | 4,735.9 s | 17.36 GB |
| LoRA-4 | 2,523,136 | 1,250 | 40,000 | 0 | 3,515.7 s | 6.18 GB |
| LoRA-16 | 10,092,544 | 1,250 | 40,000 | 0 | 3,546.8 s | 6.33 GB |
| OPLoRA-16 | 10,092,544 | 1,250 | 40,000 | 0 | 5,177.2 s | 6.77 GB |

Final SFT loss: Full FT 0.1656, L2-SP 0.1648, LoRA-4 0.1943, LoRA-16 0.1791,
OPLoRA-16 0.1794. L2-SP final Ω = 19.5797 (λΩ = 0.0196).

Adapter variants cost 1.29× (LoRA-4), 1.30× (LoRA-16) and 1.90× (OPLoRA-16) the wall clock
of full fine-tuning despite updating only 0.42% (2,523,136) or 1.69% (10,092,544) of the
parameters, because the frozen backbone still runs the full forward/backward pass while the
optimizer step is tiny. Peak VRAM falls from 17.36 GB to 6.2–6.8 GB.

### 2.5 Displacement and geometry diagnostics

Full-model displacement ‖θ−θ₀‖₂ at the final checkpoint
(`report.json::displacement_l2`): Full FT 10.650, L2-SP **6.258**, LoRA-4 9.103, LoRA-16
13.252, OPLoRA-16 12.409; Full FT midpoint 9.994.

`down_proj` in zero-based layers 0, 13, 27, FP32: η = ‖ΔW‖_F/‖W0‖_F and
ρ_k = ‖U_kU_kᵀΔW‖²_F/‖ΔW‖²_F.

| Condition | η layer 0 | η layer 13 | η layer 27 | ρ_k (0 / 13 / 27) |
|---|---:|---:|---:|---|
| Full FT midpoint | 0.013228 | 0.015383 | 0.012420 | not computed |
| Full FT | 0.013702 | 0.016020 | 0.012927 | not computed |
| Full FT + L2-SP | 0.011132 | 0.013699 | 0.007852 | not computed |
| LoRA-4 | 0.009193 | 0.010230 | 0.006863 | not computed |
| LoRA-16 | 0.014774 | 0.018174 | 0.011562 | not computed |
| OPLoRA-16 | 0.014746 | 0.018604 | 0.009502 | 8.01e-10 / 7.93e-10 / 1.64e-9 |

**ρ_k exists only for OPLoRA-16.** `probes.weight_diagnostics(..., bases=...)` computes it
whenever the protected bases are supplied, but the run driver only passed `bases` for the
OPLoRA run, so the other five reports store `rho_k: null`. The value is ~1e-9 for OPLoRA,
i.e. the update is essentially orthogonal to the protected top-128 subspace, as designed.
Closing this gap requires the saved weights (which live only on the stopped instance) and a
recomputation of the same `U_k` bases. See §7 item 9.

Activation ratio R(z) (math ÷ non-math) at the final checkpoint of Full FT: 1.815 (layer
0), 2.092 (layer 13), 1.277 (layer 27), from 32 GSM8K test questions and 32 HellaSwag
validation contexts sampled with seed 6304 **before** any benchmark result was inspected.
Per-variant values are in each `report.json::activation_diagnostics`.

### 2.6 Qualitative set

All **24** prompts (8 questions × plain/JSON/template) × **7** conditions, greedy,
thinking disabled, 512 new tokens → **168 responses**
(`results/task_01/runs/<condition>/qualitative/`). The annotation sheet holds **42 rows**
(first and last underlying question × 3 formats × 7 conditions) with truncation flags and
mechanical checks only (`results/task_01/examples/annotation_sheet.json`).

### 2.7 Checkpoints

| Condition | Recovery checkpoints present |
|---|---|
| Full FT | 1250 only |
| Full FT + L2-SP | 1250 only |
| Full FT midpoint (reconstructed) | 312, 625 |
| LoRA-4 / LoRA-16 / OPLoRA-16 | 312, 625, 937, 1250 |

Full-FT checkpoints are ~7.15 GB each (FP32 parameters + FP32 AdamW moments). A 12 GB
retention budget sized for a 20 GB disk evicted 312/625/937 before it was raised to 60 GB;
timestamps and per-file evidence in `docs/tasks/task_01_checkpoint_inventory.md`.

**The midpoint is reconstructed**, not the original quarter save: updates 1..625 retrained
with seed 6304, the same data order and the 1,250-update cosine schedule. Verified against
the original log at matching updates: **learning rate bit-identical**; SFT loss max absolute
difference 5.7e-3 (mean 1.4e-3), attributable to non-deterministic GPU kernels. Labelled by
`RECONSTRUCTED.json` beside the weights.

---

## 3. Task 2 — visual grounding

### 3.1 Data

| Split | Content | Images |
|---|---|---|
| train | 6,000 images × 3 captions = 18,000; **64 presence + 64 counting** pairs; 1,536 VQAv2 pairs (640 yes/no / 256 number / 640 other) | 9,328 |
| dev | 96 caption images; 24 grounding pairs; 32 VQAv2 pairs (12/8/12) | 208 |
| test | 64 grounding pairs; 96 VQAv2 pairs (36/24/36) | 320 |

Training pools come from COCO **train2014**, dev/test from val2014. Image disjointness is
asserted at build time (`train∩dev = train∩test = dev∩test = ∅`) and unit-tested. Natural
pairs require ≥8/10 annotator agreement on **each** image with differing consensus answers;
all ten answers are retained for scoring. **All 9,856 referenced images have cached SigLIP
features**, and the loader hard-fails if any is missing.

### 3.2 Model and curriculum

Encoder `google/siglip-base-patch16-224`, frozen, 196×768 in 14×14 order, final layer norm
applied once, FP16 cache, encoder unloaded before LM training. Connector 768→1024 GELU→1024,
FP32, 1,837,056 parameters. Magnitude calibration by RMS matching on training data:
scale **0.05748** (text rms 0.0296 / raw visual rms 0.5150).

| Stage | Updates | Time | Connector LR | LoRA LR | Trainable |
|---|---:|---:|---:|---:|---|
| 1 caption alignment | 2,250 | 8,044.8 s | 1e-3 | — | connector only, LoRA frozen |
| 2 visual instruction | 1,500 | 5,761.2 s | 1e-4 | 2e-5 | connector + 4,587,520 LoRA |
| 3 natural VQA | 750 | 2,887.7 s | 5e-5 | 1e-5 | connector + same adapters |
| **Total** | **4,500** | | | | |

Stage 1 is two complete passes over the 18,000 captions through a deterministic cycler;
Stages 2–3 use fresh mixture samplers at 50/25/25 and 70/10/10/10. Micro-batch 8 ×
accumulation 2, example-weighted loss (per-example token mean, then batch mean), seed 6304,
grad clip 1.0, per-stage 5% warm-up then cosine to 10% of peak. Resumable checkpoints every
250 updates carrying connector, adapters, optimizer, scaler, RNG and sampler position.

### 3.3 Final evaluation (frozen Stage 3 model)

| Natural VQAv2, 96 pairs / 192 examples | Accuracy |
|---|---:|
| Correct image | **0.5333** |
| Complementary partner | 0.2568 |
| Mismatched (fixed unrelated) | 0.3125 |
| No image | **0.0000** |

Pair accuracy 0.2708 · identical-answer rate 0.3438 · reference NLL 1.3427 / 2.8072 /
21.4715 (correct / complementary / no-image) · **swap penalty +1.4645** · correct-image
accuracy advantage **+0.2766**, 4,000-pair bootstrap 95% CI **[+0.1693, +0.3797]**.

| Annotation grounding, 64 pairs / 128 examples | Accuracy |
|---|---:|
| Correct image | **0.4531** |
| Complementary partner | 0.2891 |
| Mismatched | 0.3672 |
| No image | **0.0000** |

Pair accuracy 0.2344 · identical-answer rate 0.4375 · reference NLL 2.8891 / 3.7098 /
26.1323 · **swap penalty +0.8207** · advantage **+0.1641**, CI **[+0.0312, +0.3047]**.

Natural VQA scoring uses the official leave-one-out soft consensus over all ten references,
not an approximation. Empty outputs remain in denominators.

---

## 4. Task 3 — test-time compute

### 4.1 Supplied pools

| Dataset | Questions | Rollouts | Retained tokens | Capped | Null extractions |
|---|---:|---:|---:|---:|---:|
| MATH (`Qwen2.5-3B-Instruct`) | 1,000 | 32,000 | 20,742,094 | 523 | 521 |
| MathVista (`Qwen2.5-VL-3B-Instruct`) | 1,000 | 32,000 | 8,961,703 | 487 | 3,032 |

Splits: question IDs sorted lexicographically, permuted with `default_rng(6304)`; first 200
development, remaining 800 held-out; the first 50 MATH held-out IDs in that permutation form
the live subset. Ten seeded rollout orders per question, nested prefixes, identical across
all methods. Run times: MATH dev 4.0 s, MATH held-out 15.5 s, MathVista dev 35.1 s,
MathVista held-out 139.9 s.

### 4.2 Offline results (held-out, 800 questions, averaged over ten orders)

Source: `results/task_03/offline/summary_{math,mathvista}_heldout.json`.

| Method | MATH @32 | MathVista @32 |
|---|---:|---:|
| FirstAns (N=1) | 0.6465 | 0.5421 |
| Self-consistency | 0.7379 | 0.6170 |
| Highest mean logp | 0.6400 | 0.4800 |
| Highest self-certainty | 0.7063 | 0.5450 |
| Self-certainty + RWC | **0.7555** | 0.6225 |
| Vision-only highest | — | 0.5363 |
| Vision-only + RWC | — | 0.6275 |
| VisConf highest | — | 0.5775 |
| VisConf + RWC | — | **0.6300** |
| Z_token fusion highest | — | 0.5625 |
| Z_token fusion + RWC | — | 0.6262 |
| Oracle coverage @32 | 0.9137 | 0.8950 |

Mean consumed-token cost at N=32: MATH 21,347 (FirstAns 666.6); MathVista 8,897.8 (FirstAns
278.5).

Fusion identity `Z_token − Z = Cov_t(c_t, v_t)` holds to a maximum absolute error of
**4.9e-17** over 25,600 rollouts. Within-rollout corr(c_t, v_t) = **−0.211** (mean over 800
questions; median −0.212, sd 0.182; rollout-pooled −0.211).

Within-question signal associations (Spearman with correctness, mean over questions):

| Signal | MATH | MathVista |
|---|---:|---:|
| Length | −0.076 | −0.051 |
| Mean logp | +0.134 | +0.029 |
| Self-certainty | +0.198 | +0.048 |
| Vision engagement | — | +0.042 |
| VisConf | — | +0.090 |

Top-vs-bottom score-quartile correctness differences and the shortest-rollout control
(accuracy 0.625 MATH, 0.536 MathVista) are in `results/task_03/tables/*_signals.*`.

### 4.3 Adaptive stopping (dev-selected, frozen, then held-out)

| Dataset | Family | Selected policy | Held-out acc | Mean tokens | Mean rollouts | Fixed-32 acc |
|---|---|---|---:|---:|---:|---:|
| MATH | agreement | τ=0.625 | 0.7347 | 10,165 | 11.84 | 0.7379 |
| MATH | quality | τ=0.5, γ=0 | 0.7319 | 7,591 | 8.90 | 0.7379 |
| MathVista | agreement | τ=0.625 | 0.6218 | 4,682 | 13.91 | 0.6170 |
| MathVista | quality | τ=0.5, γ=0 | 0.6190 | 3,654 | 11.00 | 0.6170 |

Development accuracies of the same policies: MATH 0.8195 / 0.8150 (fixed-32 dev 0.8250);
MathVista 0.6395 / 0.6290 (fixed-32 dev 0.6385). Stopping histograms are recorded per policy
in `adaptive_{math,mathvista}_heldout.json`.

### 4.4 Live generation (50 MATH questions, 3,072-token allowance)

`Qwen/Qwen2.5-3B-Instruct` at revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`, FP16, no
quantisation, batch 1, temperature 0.7, top-p 0.8, top-k 20, repetition penalty 1.0, native
chat template, system `You are a helpful assistant.`

| Strategy | Accuracy | Mean output tokens | Extraction | Truncation |
|---|---:|---:|---:|---:|
| Single, 3,072 cap | 0.54 | 704.74 | 0.98 | 0.02 |
| Parallel, 3×1,024 + self-consistency | **0.62** | 1,868.98 | 0.90 | 0.24 |
| Refine, 1,024 + 2,048 | **0.62** | 1,208.40 | 1.00 | 0.20 |

Revision transitions relative to the shared initial answer: **3 wrong→correct, 0
correct→wrong**. Generation wall clock 4,382.8 s; model load 24.9 s.

---

## 5. Verification

| Suite | Evidence |
|---|---|
| Task 3 unit | **72 collected / 72 passed / 0 skipped** — `results/task_03/tests/pytest_unit.xml` |
| Task 3 integration | **8 passed** on the real Parquet export — `results/task_03/tests/pytest_integration.xml` |
| Task 1 unit | 58 test functions; **requires torch** — executed on the GPU host (the local analysis venv has no torch) |
| Task 2 unit | 50 test functions; **requires torch** — executed on the GPU host |

The Task 3 integration tests assert the supplied totals independently (32,000 rollouts per
dataset; 20,742,094 / 8,961,703 retained tokens; 523 / 487 capped; 521 / 3,032 null
extractions), so they check the local files against the instructor's README rather than the
loader against itself.

Key properties covered: token-weighted accumulation equals a single-batch reference to 1e-6;
the L2-SP gradient is not multiplied by accumulation; OPLoRA residual projections vanish;
VQA soft consensus gives 0.9 for 3-of-10 agreement (a naive `count/3` gives 1.0); gold
labels cannot change a Task 3 decision; unrevealed candidates cannot change a decision;
Task 2 splits are image-disjoint; a Task 2 checkpoint restores connector, adapter, optimizer,
scaler, RNG and sampler position.

---

## 6. Artefact map

```
results/task_01/
  tables/condition_table.{csv,md}          seven conditions + deltas
  figures/benchmarks_by_condition.*        grouped bars, all four benchmarks
  figures/deltas_vs_baseline.*             percentage-point deltas
  figures/sft_loss_curves.*                five runs, per-update loss
  figures/weight_diagnostics.*             eta / rho_k / activation ratio
  examples/annotation_sheet.json           42 rows, mechanical checks only
  runs/<condition>/report.json             benchmarks, diagnostics, displacement
  runs/<condition>/train/train_log.jsonl   per-update loss, lr, grad norm, scale
  runs/<condition>/train/config.json       resolved configuration
  runs/<condition>/qualitative/*.json      24 responses per condition
  runs/<condition>/eval/*.json             raw lm-evaluation-harness output
results/task_02/
  eval.json                                four conditions, pair diagnostics, bootstrap
  training.json                            stage updates, times, trainable sets, dev points
  tables/condition_table.{csv,md}          four conditions + pair diagnostics + CIs
  tables/by_kind.{csv,md}                  subtype accuracy
  tables/stages.{csv,md}                   updates, times, trainable sets, realized mixtures
  figures/conditions_accuracy.*            four conditions, both suites
  figures/nll_by_condition.*               reference NLL per condition
  figures/by_kind.*                        subtype accuracy, correct vs swapped
  figures/dev_nll_curves.*                 dev NLL across Stages 2-3
  figures/train_loss.*                     Stages 2-3 training loss
results/task_03/
  offline/*.json                           summaries, bootstrap, adaptive, signals, splits
  offline/trajectories_mathvista.json      fixed illustrative VisConf trajectories
  tables/*.{csv,md}                        36 tables
  figures/*.{png,svg}                      accuracy vs count, accuracy vs cost, live
  live/live_results.json, live_summary.json
  tests/pytest_{unit,integration}.xml      JUnit evidence
docs/                                      requirements, verification, task records
```

Not published in the repository: Task 1 model weights (~7.15 GB each), Task 2 stage weights
(24.58 MB each — retained locally in `outputs/task_02/checkpoints/`), raw datasets,
caches and the course PDF.

---

## 7. Deviations and defects, disclosed

1. **Task 2 attempt 1 invalidated** — training pools were built from `grounding_dev +
   grounding_test`; Stage 1 used the Stage 2 mixture; one sampler was reused across stages;
   the reference answer sat inside the generation prompt; `consensus_a/b` was read from
   records carrying `answer_a/b`; exact match replaced soft consensus. Quarantined in
   `outputs/task_02_invalid/`.
2. **Task 2 attempt 2 invalidated** — the builder was fixed but the deployed `build_pools`
   still read dev+test, and 853 required images were never downloaded, so every "with image"
   condition silently ran as no-image (all four grounding conditions returned an identical
   0.4609). Quarantined in `outputs/task_02_invalid2/`.
3. **Task 1 recovery checkpoints** at 312/625/937 for Full FT and L2-SP were evicted by a
   12 GB retention budget before it was corrected to 60 GB.
4. **Full FT midpoint is reconstructed**, not the original save (identical LR schedule,
   ≤5.7e-3 loss difference).
5. **Task 2 development curves** are NLL points (4 in Stage 2, 3 in Stage 3), not the full
   every-500-update table with generated accuracy.
6. **Task 2 optional 49-token compression** was not run.
7. **Evaluation batch size** for the three likelihood benchmarks was 8 rather than 32, for
   memory headroom; applied identically to all seven conditions.
8. **Task 2 grounding pool is small** (64 + 64 pairs). The assignment permits reusable
   pools, but a larger pool would likely improve grounding accuracy.
9. **ρ_k is reported only for OPLoRA-16.** The other five conditions store `rho_k: null`
   because the run driver supplied the protected bases only to the OPLoRA diagnostic. η is
   present for all six. Recomputing ρ_k needs the saved weights (instance-side) and the same
   `U_k` bases.
10. **Task 1 selection counters** (per-reason exclusion counts) were logged during
    preparation on the GPU host and are not in the published bundle.

---

## 8. Reproduction

```bash
export PYTHONPATH=src
pip install -r requirements-dev.txt          # CPU analysis
pip install -r requirements-gpu.txt          # CUDA training/eval

# Task 3
python scripts/fetch_reference_assets.py
python -m pytest tests/unit tests/integration -q
python -m task3.run_offline --dataset both --out outputs/task_03/offline
python -m task3.report

# Task 1
python -m task1.prepare --out outputs/task_01/data
python scripts/task1_variant_step.py --variant full_ft --out outputs/task_01/runs/full_ft
# full_ft_l2sp, lora4, lora16, oplora16, baseline likewise

# Task 2
python scripts/task2_build_v2.py
python scripts/task2_download_missing.py
python scripts/task2_run.py --stage features
python scripts/task2_run.py --stage train
python scripts/task2_run.py --stage eval

# Figures
python scripts/make_figures.py
```

---

## 9. Student-owned work (not produced by the agent)

The PA states: *"You may not use generative AI to write any part of your PDF report."*
The following remain entirely the student's:

- the 8-page NeurIPS-format report — abstract, introduction, analysis, conclusion;
- interpretation of every number in this document;
- the **42-response qualitative annotation** (coherence, relevance, repetition, unnecessary
  mathematical framing, inappropriate content) and the excerpts chosen to illustrate it;
- the Task 2 illustrative complementary-pair selection and discussion;
- the Task 3 aligned disagreement / revision / premature-stop cases;
- disclosure of the deviations in §7 in the report's own words;
- publication decisions for this repository and the LMS submission itself.

`docs/student_report_checklist.md` enumerates these with pointers to the supporting evidence
for each.
