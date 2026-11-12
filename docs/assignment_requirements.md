# ATML PA 3 — requirement matrix

**Source:** `ATML-PA3.pdf`, 15 pages, released 23 Oct 2026.
**SHA-256:** `66766f7aafb6046f617d990399dad687ab79b37b8d963cb60bdb4cb0d8871f62`
**Deadline:** 13 Nov 2026 (LMS revision confirmed by the student).

Legend: **done** = implemented, executed and evidence saved · **partial** = executed with a
recorded shortfall · **not run** = deliberately skipped · **student** = reserved for the
student (report prose or judgement).

Full measured numbers live in `docs/results_index.md`; per-task records in
`docs/tasks/`.

---

## Status summary

| Area | Status |
|---|---|
| Task 1 training (5 variants + baseline) | **done** — 6 runs + reconstructed midpoint, 1,250 updates each |
| Task 1 evaluation (7 conditions × 4 benchmarks) | **done** |
| Task 1 diagnostics (η, displacement, activation probes) | **done** |
| Task 1 diagnostics (ρ_k for all six conditions) | **partial** — ρ_k only for OPLoRA-16 (see Task 1 row) |
| Task 1 recovery checkpoints 312/625/937 | **partial** — adapters kept all four; Full FT/L2-SP lost to an eviction bug; midpoint reconstructed |
| Task 1 qualitative 168 responses + 42-row annotation sheet | **done** (annotation itself is student work) |
| Task 2 curriculum 4,500 updates | **done** (after two invalidated attempts) |
| Task 2 final evaluation, four conditions, both suites | **done** |
| Task 2 dev curves every 500 updates | **partial** — NLL points only |
| Task 2 optional 49-token run | **not run** (optional) |
| Task 3 offline, both datasets | **done** |
| Task 3 live generation, 50 questions | **done** |
| Task 3 adaptive controllers | **done** |
| Task 3 incorrect-early-stop case list | **recovered** — CPU replay |
| Public GitHub repository | **done** — published `mbh2006/PA_3` |
| 8-page PDF report | **student** |

---

## Global rules (p.1–2)

| Requirement | Detail | Status |
|---|---|---|
| Individual work | no collaboration | **student** |
| Public GitHub repo | code, configs, split IDs, env spec, results, reproduction README, attribution | **done** — <https://github.com/mbh2006/PA_3> |
| 8-page NeurIPS PDF | references/appendix excluded; repo link at end of abstract | **student** |
| No generative AI in the PDF | language/interpretation/analysis must be the student's | **enforced** — agent produced only technical records |
| Seeds and configs preserved | seed, hyperparameters, resolved config per run | **done** — `config.json`, `provenance.json` per run |

---

## Task 1 — LLM fine-tuning (pp. 2–6)

| Requirement | Detail | Status | Evidence |
|---|---|---|---|
| Model | post-trained `Qwen/Qwen3-0.6B`, `enable_thinking=False` | **done** | `configs/`, run `config.json` |
| Independence | every variant starts from the original checkpoint | **done** | 6 independent runs |
| Seed / data | seed 6304; identical examples, order, tokenisation, prompts | **done** | `train_log.jsonl` |
| Baseline | unchanged model evaluated once | **done** | `runs/baseline/` |
| Data selection | first 40,000 **eligible** MetaMathQA rows in source order | **done** | `examples_seen = 40,000` in every `train_summary.json` |
| Exclusions | empty q/r; >2,048 formatted tokens; normalised GSM8K-test match | **done** | `src/task1/data.py` + tests |
| No shuffle / no truncation | source order retained, answers never cut | **done** | tests + counters |
| SFT loss | assistant tokens **+ end-of-turn**; prompt/pad masked; normalised over the **effective batch** | **done** | `test_task1_loss.py` (1e-6 vs reference) |
| Budget | 1 epoch, 40,000 examples, 1,250 updates | **done** | `train_summary.json` |
| Batch | micro 2 × accum 16 = 32 | **done** | `config.json` |
| Optimiser | AdamW β=(0.9,0.999), ε=1e-8, wd 0.01 excluding biases and <2-D | **done** | `build_param_groups` + test |
| Schedule | cosine, 38 warm-up updates | **done** | LR logged per update |
| Clip / seq len | global norm 1.0 / 2,048 | **done** | logs |
| Gradient checkpointing / quantisation | on / none | **done** | config |
| Precision | FP16 AMP + dynamic scaling, FP32 masters or FP32 adapters, FP32 optimiser states, unscale before clip, log skipped updates | **done** — `skipped_updates = 0` in all runs | `run_summary.json` |
| Full FT | all parameters, peak LR 2e-5, λ=0 | **done** | `runs/full_ft/` |
| L2-SP | same recipe, λ=1e-3, fixed θ₀, Ω=0.5·Σ(θ−θ₀)² over all unique parameters incl. biases/norms, tied once, no averaging | **done** — final Ω 19.5797 | `runs/full_ft_l2sp/` |
| LoRA-4 / LoRA-16 | r=4 α=8 / r=16 α=32, LR 2e-4, dropout 0.05, random A, zero B, no trainable biases, 7 projections per layer | **done** | `runs/lora4/`, `runs/lora16/` |
| OPLoRA-16 | r=16 α=32 k=128, same init as LoRA-16, effective factors, no dense projection, FP32 bases, projected export | **done** | `runs/oplora16/` |
| Recovery checkpoints | updates 312/625/937/1,250 with full recovery state | **partial** — adapters have all four; Full FT and L2-SP kept only 1,250 (eviction bug). Midpoint **reconstructed** and labelled | `docs/tasks/task_01_checkpoint_inventory.md` |
| Evaluations | 7 conditions | **done** | `tables/condition_table.csv` |
| Benchmarks | GSM8K test 1,319 flexible extraction; HellaSwag 10,042 acc_norm; BoolQ 3,270 acc; ARC-Easy 2,376 acc_norm | **done** | `runs/*/eval/*.json` |
| Eval protocol | FP16, dropout off, stock completion prompts for likelihood tasks, raw accuracy alongside normalised | **done** | `condition_table.csv` has `hellaswag_acc`, `arc_easy_acc` |
| Merge check | merge into fresh backbone, verify merged vs unmerged logits | **done** | `merged_modules` in reports |
| Logs | SFT loss + LR per update; time, peak memory, trainable params, checkpoint sizes, method overhead; per-example outputs, extracted answers, truncation flags; deltas in pp | **done** | `train_log.jsonl`, `run_summary.json`, `condition_table.csv` |
| Displacement | ‖θ−θ₀‖₂ for Full FT midpoint/final and L2-SP final | **done** — 9.994 / 10.650 / 6.258 | `report.json` |
| Weight diagnostics | `down_proj` layers 0/13/27, ρ_k and η, five finals + Full FT midpoint, FP32 | **partial** — η for all six; **ρ_k only for OPLoRA-16** (bases were passed only to that run) | `report.json::weight_diagnostics` |
| Activation probes | 32 GSM8K + 32 HellaSwag, seed 6304, sampled before inspecting results; input cached at last non-pad prompt token; R(z) math vs non-math | **done** | `report.json::activation_diagnostics` |
| Qualitative | 24 prompts × 7 conditions = 168 greedy responses, 512 new tokens, truncation flagged | **done** | `runs/*/qualitative/` |
| Annotation sheet | 2 questions × 3 formats × 7 conditions = 42 rows | **done** (mechanical checks only; annotation is student work) | `examples/annotation_sheet.json` |

---

## Task 2 — visual grounding (pp. 6–10)

| Requirement | Detail | Status | Evidence |
|---|---|---|---|
| Fresh backbone | original Qwen3-0.6B, not a Task 1 checkpoint | **done** | `training.json` |
| Vision encoder | SigLIP base-patch16-224, frozen, 196×768, 14×14, final LN once, FP16 cache, deterministic, unloaded before training | **done** | `--stage features` |
| Connector | 768→1024 GELU →1024, trainable, inserted before the question in the user message | **done** | 1,837,056 parameters |
| Magnitude calibration | text-embedding RMS match, training data only, rule + scale recorded | **done** — scale 0.05748 | `training.json::calibration` |
| Gradient flow | frozen Qwen still passes gradients to the connector | **done** | unit test |
| Data | COCO 2014 captions + instances, VQAv2 complementary pairs | **done** | `training.json` |
| Splits | train2014 train / val2014 dev+test, image-disjoint | **done** | 9,328 / 208 / 320, asserted + tested |
| Captions | 6,000 × 3 = 18,000 train; 96 dev | **done** | `training.json` |
| Grounding | presence + counting templates; 24 pairs dev, 64 pairs test | **done**; train pool 64 + 64 | `build_pools` |
| Natural VQA | 1,536 pairs train (640/256/640), 32 dev (12/8/12), 96 test (36/24/36); ≥8/10 agreement per image; all ten answers kept | **done** | `eval.json` |
| Budget | 4,500 updates, 72,000 presentations | **done** — 2,250 + 1,500 + 750 | `training.json` |
| Batch | micro 8 × accum 2 = 16 | **done** | `training.json` |
| Schedule | per-stage 5% warm-up then cosine to 10% of peak | **done** | `training.json` |
| Stage LRs | connector 1e-3 / 1e-4 / 5e-5; LoRA — / 2e-5 / 1e-5 | **done** | `training.json` |
| Mixtures | Stage 2 50/25/25; Stage 3 70/10/10/10; two examples per draw; seeded sampler; no quotas | **done** | unit tests |
| LoRA | r=16 α=32 dropout 0.05, q/k/v/o only, no biases, enabled at Stage 2, continued in Stage 3 | **done** | 4,587,520 parameters |
| Loss | per-example token mean then effective-batch mean; EOT included; visual/prompt/pad masked | **done** | unit test vs hand computation |
| Token caps | question 80, caption 64, short answer 16 | **done** | config |
| Checkpoints | every 250 updates + stage ends, with sampler/stage/RNG state | **done** (instance-side; stage-end weights copied locally) | `training.json` |
| Dev eval | before training, every 500 updates, stage endpoints; caption loss + answer NLL under correct/complementary/no-image; swap penalty | **partial** — NLL points only (4 in Stage 2, 3 in Stage 3), no generated accuracy | `training.json::dev_points` |
| Final eval | both suites, greedy, 16 new tokens, batch 8, four conditions, mismatch map fixed before generation | **done** | `eval.json` |
| Scoring | grounding normalised EM; VQA leave-one-out soft consensus; separate subtypes | **done** | `eval.json` |
| Diagnostics | pair accuracy, identical-answer rate, empty rate, reference NLL, generation time, 4,000 complete-pair bootstraps | **done** | `eval.json` |
| Artefacts | curves with stage boundaries, four-condition tables, aligned examples, resource tables | **partial** — four-condition table, subtype table, resource table and five figures present; dev curves are sparse NLL points and no aligned per-example sheet was exported | `results/task_02/` |
| Optional 49-token run | explicitly optional | **not run** | — |

---

## Task 3 — test-time compute (pp. 10–15)

| Requirement | Detail | Status | Evidence |
|---|---|---|---|
| Use supplied pools | no regeneration | **done** | hash-verified |
| Join | `(dataset, sample_id, rollout_id)`, uniqueness/completeness/alignment asserted | **done** | `ingest.py`, integration tests |
| Raw-distribution scores | `logp` and `kl_u_p`; KL never reconstructed | **done** | `scores.py` |
| Splits | lexicographic → `default_rng(6304)` → 200 dev / 800 held-out | **done** | `splits_*.json` |
| Live subset | first 50 MATH held-out IDs in permutation | **done** | `splits_math.json` |
| Answer equivalence | math-verify for MATH; MathVista answer-type/choice/precision, option letters | **done** | `answers.py` + tests |
| Grouping without gold | groups from predictions only | **done** | unit test |
| NO_ANSWER | one group, always incorrect, retained in voting/denominators/costs | **done** | integration test |
| Correctness | computed by us for retrospective metrics only | **done** | `ingest.py` |
| Budgets | 1,2,4,8,16,32 | **done** | `evaluate.py` |
| Ten seeded orders | nested prefixes, shared across methods and controllers | **done** | `orders.py` |
| Tie-breaking | earliest supporting rollout; average ranks for tied RWC | **done** | unit tests |
| Costs | retained generated tokens of consumed candidates; FirstAns consumes one | **done** | unit tests |
| Bootstrap | 1,000 paired question-level resamples | **done** | `bootstrap_*.json` |
| Scores | `L_i`, `K_i`, `V_i`, prefix-only VisConf, RWC | **done** | `scores.py` |
| Iterated calibration | recomputed per prefix, no unseen rollouts | **done** | unit test |
| Methods (both) | FirstAns, SC, best logp, best self-certainty, self-certainty+RWC | **done** | `summary_*.json` |
| Methods (VLM) | vision-only, VisConf, each × highest/RWC | **done** | `summary_mathvista_*.json` |
| Per-question outputs | choices, candidate IDs, costs | **partial** — per-question Parquet files are generated locally but excluded from the published bundle (size) | `outputs/task_03/offline/` |
| Failure decomposition | no correct candidate vs correct but missed | **done** | `failures_*.json` |
| N=32 associations | score/length/correctness, quartiles, shortest-rollout control | **done** | `signals_*.json` |
| VLM confidence–vision | within-rollout correlations summarised | **done** — mean −0.211 | `signals_mathvista.json` |
| Tokenwise fusion | `Z_token` vs `Z`, covariance identity, both aggregations | **done** — max error 4.9e-17 | `signals_mathvista.json` |
| Illustrative trajectories | fixed selection rule | **done** | `trajectories_mathvista.json` |
| Live generation | 50 questions × 3 strategies × 3,072 tokens, shared initial response, boxed extraction, costs, transitions | **done** | `live/live_*.json` |
| Adaptive controllers | agreement + quality-aware, stop at 4/8/16 else 32 | **done** | `adaptive.py` |
| Dev grid | τ × γ, always-32 fallback, selection rule, freeze | **done** | `adaptive_*_dev_grid.json` |
| Held-out adaptive | accuracy, tokens, rollouts, stopping distribution | **done** | `adaptive_*_heldout.json` |
| Incorrect early stops recovered by fixed-32 | case list | **recovered** — CPU replay | `early_stop_cases_{math,mathvista}.json` |
| Qualitative cases | selection disagreement, revision, premature stop | **partial** — raw live records available; case selection is student work | `live/live_results.json` |
| Compute curves | accuracy/oracle vs count and cost | **done** | `figures/*.png` |

---

## Report deliverables (p. 15)

| Item | Owner |
|---|---|
| 8-page PDF, NeurIPS format, repo link in the abstract | **student** |
| Central evidence in main pages; appendix optional | **student** |
| Cite papers/datasets/tools; disclose deviations | **student** |
| LMS upload + repository link | **student** |
| Public repository with reproduction commands | **done** — <https://github.com/mbh2006/PA_3> |
