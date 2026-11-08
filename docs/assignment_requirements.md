# ATML PA 3 — requirement matrix

**Source:** `ATML-PA3.pdf`, 15 pages, released 23 Oct 2026, deadline 06 Nov 2026
(LMS revised to **13 Nov 2026** per the student).
**SHA-256 of the PDF:** `66766f7aafb6046f617d990399dad687ab79b37b8d963cb60bdb4cb0d8871f62`.

Status legend: ✅ implemented and evidence saved · 🟡 implemented, run pending ·
⬜ not started · 🧑 reserved for the student (report prose / judgement).

---

## Page 1–2 — global rules

| Req | Detail | Status | Evidence |
|---|---|---|---|
| Individual work | no collaboration | 🧑 | — |
| Public GitHub repo | all code, configs, split IDs, env spec, results, reproduction README, attribution | 🟡 | local repo prepared; publishing is the student's action |
| 8-page NeurIPS PDF | references/appendix excluded; repo link at end of abstract | 🧑 | — |
| **No generative AI in the PDF** | language, interpretation, analysis must be the student's | 🧑 | enforced: agent produces only technical records |
| Seeds / configs preserved | fix seeds, record hyperparameters | ✅ | `configs/`, `provenance` in every run dir |

---

## Task 1 — LLM fine-tuning (pp. 2–6)

| Req | Detail | Status |
|---|---|---|
| Model | post-trained `Qwen/Qwen3-0.6B`, `enable_thinking=False` | ⬜ |
| Independence | every variant starts from the original checkpoint | ⬜ |
| Seed / data | seed 6304, identical examples/order/tokenisation/prompts | ⬜ |
| Baseline | unchanged model evaluated once | ⬜ |
| Data selection | first 40,000 **eligible** MetaMathQA in source order | ⬜ |
| Exclusions | empty q/r; >2,048 formatted tokens; query matching normalised GSM8K test question | ⬜ |
| No shuffle / no truncation | retain source order and full answers | ⬜ |
| SFT loss | assistant tokens **+ end-of-turn**; mask prompt/pad; normalise over all valid response tokens in the **effective batch** | ⬜ |
| Budget | 1 epoch, 40,000 examples, 1,250 updates | ⬜ |
| Batch | micro 2 × accum 16 = 32 | ⬜ |
| Optimiser | AdamW β=(0.9,0.999), ε=1e-8, wd 0.01 excluding biases and <2-D params | ⬜ |
| Schedule | cosine, 38 warmup updates | ⬜ |
| Clip / seq len | global norm 1.0 / 2,048 | ⬜ |
| Checkpointing / quantisation | gradient checkpointing on / none | ⬜ |
| Precision | FP16 AMP + dynamic scaling; FP32 master (full FT) or FP32 adapters (LoRA); FP32 optimiser states; unscale before clip; log skipped updates | ⬜ |
| Full FT | all params, peak LR 2e-5, λ=0 | ⬜ |
| L2-SP | same recipe, λ=1e-3, fixed θ₀, Ω=0.5·Σ(θ−θ₀)² over **all unique** params incl. biases/norms, tied counted once, no averaging | ⬜ |
| LoRA-4 / LoRA-16 | r=4 α=8 / r=16 α=32, LR 2e-4, dropout 0.05, random A, zero B, no trainable biases, 7 target projections in every layer | ⬜ |
| OPLoRA-16 | r=16 α=32 k=128, same init as LoRA-16, `B_eff`, `A_eff`, no dense projection, bases in FP32, export projected factors | ⬜ |
| Checkpoints | 9,984 / 20,000 / 29,984 / 40,000 examples = updates 312/625/937/1,250, full recovery state | ⬜ |
| Evaluations | 7 conditions: baseline, Full FT mid+final, other four final | ⬜ |
| Benchmarks | GSM8K test 1,319 (`gsm8k_cot`, 0-shot, chat template, greedy, 1,024 new tokens, flexible extraction); HellaSwag val 10,042 (`acc_norm`); BoolQ val 3,270 (`acc`); ARC-Easy test 2,376 (`acc_norm`) | ⬜ |
| Eval protocol | FP16, dropout off, stock completion prompts (no chat template) for likelihood tasks, raw accuracy alongside normalised | ⬜ |
| Merge check | merge adapters into fresh backbone copies, verify merged vs unmerged logits | ⬜ |
| Logs | SFT loss + LR per update; time/peak memory/trainable params/checkpoint sizes/overhead; per-example outputs, extracted answers, truncation flags; Δ vs baseline in pp | ⬜ |
| Displacement | ‖θ−θ₀‖₂ for Full FT midpoint/final and L2-SP final; Ω, λΩ, total loss; reference storage/transfer cost | ⬜ |
| Weight diagnostics | `down_proj` layers 0/13/27, ρ_k, η, all five finals + Full FT midpoint, FP32 | ⬜ |
| Activation probes | 32 GSM8K + 32 HellaSwag, seed 6304, before inspecting results; cache `down_proj` input at last non-pad prompt token; R(z) math vs non-math | ⬜ |
| Qualitative | 24 prompts × 7 conditions = 168 greedy responses, 512 new tokens, truncation flagged | ⬜ |
| Annotation sheet | 2 questions × 3 formats × 7 conditions = 42 responses with mechanical checks | ⬜ |

---

## Task 2 — VLM visual grounding (pp. 6–10)

| Req | Detail | Status |
|---|---|---|
| Fresh backbone | original Qwen3-0.6B, not a Task 1 checkpoint | ⬜ |
| Vision encoder | `google/siglip-base-patch16-224`, frozen, 196×768, 14×14 order, final LN exactly once, FP16 cache, deterministic preprocessing, unloaded before training | ⬜ |
| Connector | 768→1024 GELU 1024→1024, trainable, inserted before the question inside the user message | ⬜ |
| Magnitude calibration | against text embeddings, training data only, rule + scale recorded | ⬜ |
| Gradient flow | frozen Qwen must still pass gradients to the connector | ⬜ |
| Data | COCO 2014 captions/instances + VQAv2 complementary pairs | ⬜ |
| Splits | train2014 train / val2014 dev+test; dev and test image-disjoint from each other and from training/replay | ⬜ |
| Captions | 6,000 images × 3 = 18,000 train; 96 dev | ⬜ |
| Grounding | presence + counting templates from annotations; 24 pairs/48 q dev; 64 pairs/128 q test | ⬜ |
| Natural VQA | 1,536 pairs/3,072 q train (640/256/640 yes-no/number/other); 32/64 dev (12/8/12); 96/192 test (36/24/36); ≥8/10 agreement per image; keep all ten answers | ⬜ |
| Budget | 4,500 updates, 72,000 presentations | ⬜ |
| Batch | micro 8 × accum 2 = 16 | ⬜ |
| Schedule | per-stage 5% warmup → cosine to 10% of peak | ⬜ |
| Stage LRs | connector 1e-3 / 1e-4 / 5e-5; LoRA — / 2e-5 / 1e-5 | ⬜ |
| Mixtures | Stage 2: 50/25/25; Stage 3: 70/10/10/10; two examples per draw; seeded sampler; no quotas | ⬜ |
| LoRA | r=16 α=32 dropout 0.05, q/k/v/o only, no biases, enabled at Stage 2, continued in Stage 3 | ⬜ |
| Loss | per-example token mean → effective-batch mean; include EOT; mask visual/prompt/pad | ⬜ |
| Token caps | question 80, caption 64, short answer 16 | ⬜ |
| Checkpoints | every 250 updates + stage end, with sampler/stage/RNG state | ⬜ |
| Dev eval | before training, every 500 updates, stage endpoints; caption loss + answer NLL under correct/complementary/no-image; swap penalty | ⬜ |
| Final eval | both suites, greedy, 16 new tokens, batch 8, four conditions; fixed mismatch map excluding both pair images | ⬜ |
| Scoring | grounding normalised EM; VQA soft consensus `mean_i min(1, Σ_{j≠i}1[pred=a_j]/3)`; separate subtypes | ⬜ |
| Diagnostics | pair accuracy, identical-answer rate, empty-answer rate, reference NLL, generation time; 4,000 complete-pair bootstraps | ⬜ |
| Artefacts | curves with stage boundaries, 4-condition tables, aligned qualitative examples, resource tables | ⬜ |
| Optional 49-token run | explicitly optional | ⬜ (planned cut) |

---

## Task 3 — test-time compute (pp. 10–15)

| Req | Detail | Status | Evidence |
|---|---|---|---|
| Use supplied pools | no regeneration | ✅ | `ATML-PA3-Data/` verified |
| Join | `(dataset, sample_id, rollout_id)`, assert uniqueness/completeness/alignment | ✅ | `ingest.py`, integration tests |
| Raw-distribution scores | use `logp` and `kl_u_p`; do not reconstruct KL | ✅ | `scores.py` |
| Splits | lexicographic → `default_rng(6304)` → 200 dev / 800 held-out | ✅ | `splits_*.json` |
| Live subset | first 50 MATH held-out IDs in permutation | ✅ | `splits_math.json` |
| Answer equivalence | math-verify for MATH; MathVista answer-type/choice/precision, option letters | ✅ | `answers.py` + unit tests |
| Grouping without gold | groups formed from predictions only | ✅ | unit test |
| NO_ANSWER | one group, always incorrect, kept in voting/denominators/costs | ✅ | integration test |
| Correctness computed by us | not supplied | ✅ | `ingest.py` |
| Budgets | 1,2,4,8,16,32 | ✅ | `evaluate.py` |
| Ten seeded orders | nested prefixes, shared across methods/controllers | ✅ | `orders.py` |
| Tie-breaking | earliest supporting rollout; average ranks for tied RWC | ✅ | unit tests |
| Costs | retained tokens of consumed candidates; FirstAns = 1 | ✅ | unit tests |
| Bootstrap | 1,000 paired question-level resamples | ✅ | `bootstrap.py` |
| Scores | `L_i`, `K_i`, `V_i`, prefix-only VisConf, RWC | ✅ | `scores.py` |
| Iterated calibration | recompute per prefix, no unseen rollouts | ✅ | unit test |
| Methods (both) | FirstAns, SC, best logp, best self-certainty, self-certainty+RWC | ✅ | `summary_*` |
| Methods (VLM) | vision-only, VisConf, each × highest/RWC | ✅ | `summary_*` |
| Per-question outputs | choices, candidate IDs, costs | ✅ | `per_question_*.parquet` |
| Failure decomposition | no correct generated vs missed | ✅ | `failures_*` |
| N=32 associations | length/score/correctness, quartiles, shortest-rollout control | ✅ | `signals_*` |
| VLM confidence–vision | within-rollout correlations, summarised | ✅ | `signals_*` |
| Tokenwise fusion | `Z_token` vs `Z`, covariance identity, both aggregations | ✅ | `signals_*`, max error 0.0 |
| Illustrative trajectories | fixed selection rule | ✅ | `trajectories_mathvista.json` |
| **Live generation** | 50 questions × 3 strategies × 3,072 tokens, shared initial response, boxed extraction, costs, transitions | ⬜ | GPU |
| Adaptive controllers | agreement + quality-aware, stopping at 4/8/16 else 32 | ✅ | `adaptive.py` |
| Dev grid | τ × γ grid, always-32 fallback, selection rule, freeze | ✅ | `adaptive_*_dev_grid.json` |
| Held-out adaptive | accuracy, tokens, rollouts, stopping distribution | ✅ | `adaptive_*_heldout.json` |
| Incorrect early stops recovered by fixed-32 | case list | ⬜ | pending extension |
| Qualitative cases | selection disagreement, revision, premature stop | ⬜ | GPU + student selection |
| Compute curves | accuracy/oracle vs count and cost | 🟡 | tables done; figures pending |

---

## Report deliverables (p. 15)

| Item | Owner |
|---|---|
| 8-page PDF, NeurIPS format, repo link in abstract | 🧑 student |
| Central evidence in main pages; appendix optional | 🧑 student |
| Cite papers/datasets/tools; disclose deviations | 🧑 student |
| LMS upload + repo link | 🧑 student |
| Public GitHub repo with reproduction commands | 🟡 agent prepares, student publishes |
