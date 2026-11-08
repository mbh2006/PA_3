# ATML PA 3 — Kaggle execution instructions

## Purpose and authoritative source

These instructions are for the OpenCode agent running this assignment through the user's existing Kaggle configuration. Complete the actual ATML PA 3 requirements, verify the implementation before expensive runs, and preserve the code, evidence, figures, tables, and explanations needed for submission.

The assignment has been read in full: **EE-5102/CS-6304, Advanced Topics in Machine Learning, Fall 2026, Programming Assignment 3**, released **23 October 2026**, 15 pages. The supplied PDF states **06 November 2026** as the LMS deadline; do not assume an extension or invent a revised deadline.

- Source: use `ATML-PA3.pdf` beside this file when present; otherwise use `assignment/ATML-PA3.pdf`, included in the portable handoff.
- Original source on the user's computer: `D:\1______LUMS UNDERGRAD_______\3_LUMS Junior\1) ATML\3_PAs\PA3\ATML-PA3.pdf`.
- PDF SHA-256: `66766f7aafb6046f617d990399dad687ab79b37b8d963cb60bdb4cb0d8871f62`.
- Read the PDF yourself before implementation. The requirements below are an operational map; the PDF and any verified instructor clarification govern exact details.
- At handoff preparation, the source folder contained only this PDF. Starter code, `first_plot_questions.yaml`, `ATML-PA3-Data`, its `README.MD`, and the PA0 NeurIPS template were not supplied in this folder. Discover them through the configured project/Kaggle inputs or request the specific missing asset. Do not fabricate replacements for fixed instructor-supplied inputs.

## Assignment AI policy and scope of the agent

Page 1 explicitly permits LLM coding assistance but says: **"You may not use generative AI to write any part of your PDF report."** It also requires the report's language, interpretation, and analysis to be the student's own. Individual submission is mandatory; do not collaborate with other students or share assignment work.

Implement and test code, run the prescribed experiments, compute statistics, and export factual results, tables, graphs, raw generations, logs, configuration records, and reproducibility documentation. Save separate Markdown **technical records** for all three tasks. These are operational evidence, not report drafts. Do not generate an abstract, introduction, result interpretation, discussion, conclusion, narrative captions, or other prose intended for the submitted PDF. Leave qualitative judgments, excerpts chosen for an argument, interpretation, and report prose for the student, with a checklist of the required human work. Never disguise generated report text as a task Markdown file. Mechanical formatting/extraction checks and raw outputs may be automated; the student owns semantic annotations and their explanation.

## Exact project demands — Task 1 (PDF pages 2–6)

### Model, data, and five independent training runs

- Use post-trained `Qwen/Qwen3-0.6B`, with `enable_thinking=False` in training and generation. Every variant starts independently from the original checkpoint. Seed **6304**; identical examples, order, tokenization, and evaluation prompts; no old-task training data or replay.
- MetaMathQA: select the **first 40,000 eligible examples in source order**. Exclude empty question/response records, formatted examples longer than **2,048 tokens**, and any query/original_question matching a normalized GSM8K test question. Do not shuffle the retained examples or truncate answers. Save source IDs, exclusion counts/reasons, tokenizer/model revisions, and the exact normalization/formatting rules.
- Format user question plus supplied assistant response. Supervise only assistant-response tokens **and end-of-turn**; mask prompts/padding. Normalize accumulated SFT loss over **all valid response tokens in the entire effective batch**, not a mean of independently normalized microbatches.

| Setting | Required value |
| --- | --- |
| Epochs / presentations / updates per variant | 1 / 40,000 / 1,250 |
| Microbatch / accumulation / effective batch | 2 / 16 / 32 |
| Optimizer | AdamW; betas (0.9, 0.999), epsilon 1e-8 |
| Weight decay | 0.01; exclude biases and parameters with fewer than 2 dimensions |
| Schedule | Cosine decay, 38 warmup updates |
| Clipping / maximum sequence length | Global norm 1.0 / 2,048 |
| Gradient checkpointing / quantization | Enabled / none |
| Precision | FP16 AMP and dynamic gradient scaling; FP32 master parameters for full FT; frozen FP16 backbone and FP32 adapters for adapter methods; FP32 optimizer states |

Unscale before clipping; log scaler-skipped updates and reconcile the required training budget explicitly if they occur.

| Variant | Exact method-specific requirements |
| --- | --- |
| Full FT | All original parameters trainable; peak LR 2e-5; lambda 0 |
| Full FT + L2-SP | Same full-FT settings; lambda 1e-3; fixed original reference; Omega = 0.5 × sum of squared displacement over **all unique parameters**, including biases and norms, tied parameters counted once; no parameter-count averaging |
| LoRA-4 | r=4, alpha=8; peak LR 2e-4 |
| LoRA-16 | r=16, alpha=32; peak LR 2e-4 |
| OPLoRA-16 | r=16, alpha=32, protection k=128; same LR/targets/settings and **same initial unprojected factors as LoRA-16** |

All adapter variants: frozen backbone, dropout **0.05**, random A, zero B, no trainable biases; target `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` in every transformer layer.

OPLoRA: compute leading starting-weight bases `U_k,V_k` once in FP32, then use `B_eff = B - U_k(U_k^T B)`, `A_eff = A - (A V_k)V_k^T`, and `DeltaW = (alpha/r) B_eff A_eff`. Do not build dense projection matrices. Freeze bases; export projected factors for inference. Log basis storage, preprocessing time, and export overhead. L2-SP must log Omega, lambda×Omega, SFT and total objective, and reference-storage/transfer costs separately from AdamW weight decay.

### Checkpoints and exact seven evaluation conditions

- Save recovery states after **9,984 / 20,000 / 29,984 / 40,000 examples**, at updates **312 / 625 / 937 / 1,250**. Save optimizer, scheduler, scaler, RNG, and data position.
- Evaluate the unchanged baseline once; Full FT at midpoint and final; the four other variants at final only. These are **seven** conditions. Quarter checkpoints are recovery-only, not extra required evaluations.
- Use pinned `lm-evaluation-harness` task definitions and complete splits, zero-shot, FP16 evaluation with dropout disabled.

| Benchmark | Complete split / count | Primary metric and protocol |
| --- | --- | --- |
| GSM8K | Test / 1,319 | `gsm8k_cot`, flexible-extraction exact match; non-thinking chat; num_fewshot=0; greedy; 1,024 new tokens; stock stops/filters; unextractable = incorrect |
| HellaSwag | Validation / 10,042 | Candidate likelihood, normalized accuracy |
| BoolQ | Validation / 3,270 | Candidate likelihood, accuracy |
| ARC-Easy | Test / 2,376 | Candidate likelihood, normalized accuracy |

Use stock completion prompts **without chat templates** for the three likelihood benchmarks. Save raw accuracy alongside normalized accuracy where available. Merge adapters into fresh original backbone copies; use OPLoRA projected factors, and verify merged/unmerged logits on a small batch. Save per-example outputs, extracted GSM8K answers, truncation flags, aggregate scores, and baseline changes in **percentage points**.

### Diagnostics, required artifacts, and human annotation handoff

- Save SFT loss and LR per optimizer update, resource comparisons for all five variants (time, peak memory, trainable parameters, checkpoint sizes, method overhead), and full-model displacement for Full FT midpoint/final and L2-SP final.
- In FP32, diagnose `down_proj` in zero-based layers **0, 13, 27** for all five final variants plus Full FT midpoint: `rho_k = ||U_k U_k^T DeltaW||_F^2 / ||DeltaW||_F^2`, `eta = ||DeltaW||_F / ||W0||_F`. Use the same original bases; handle a zero update explicitly without fabricating a ratio.
- Before inspecting results, seed-sample and save a fixed probe of **32 GSM8K test questions + 32 HellaSwag validation contexts**. Cache original-model `down_proj` inputs at the **last non-padding prompt token before answer continuation**, using the benchmark's prompt format. Apply local updates to these same inputs and compute `R(z)=||DeltaW z||_2/(||W0 z||_2+1e-8)` separately for math/non-math. These prescribed evaluation probes must never influence training/tuning.
- Use all **24 prompts** in `first_plot_questions.yaml` (8 questions × plain/JSON/template), greedy non-thinking generation, **512 new tokens**, across all seven conditions. Preserve **168 responses**, grouped by question/format. Flag truncation.
- Export an annotation sheet for the first and last underlying questions in the file's **plain-question order**, all three formats, all seven conditions: **42 responses**. Supply raw responses and mechanical checks. Leave coherence/relevance, repetition, unnecessary math framing, inappropriate-content judgments, illustrative excerpts, and interpretation for the student to annotate/write.
- Required saved evidence: seven-condition/four-benchmark table with baseline deltas; combined five-run SFT plot; penalty/displacement data; resource table; weight/activation diagnostic figure/table; all qualitative responses and the human-annotation sheet.
- Prepare aligned factual comparisons: LoRA-4 vs LoRA-16, Full FT vs L2-SP, LoRA-16 vs OPLoRA, Full FT vs LoRA-16, and Full FT midpoint vs final. The student must discuss adaptation/retention, update restrictions/geometry, resource costs, and qualitative changes without causal overclaims from local diagnostics.

## Exact project demands — Task 2 (PDF pages 6–10)

### Interface and data

- Start from a **fresh original** `Qwen/Qwen3-0.6B`, not a Task 1 math-tuned checkpoint. Thinking disabled throughout.
- Frozen encoder: `google/siglip-base-patch16-224`. Use **196 final patch features**, shape **196×768**, in 14×14 spatial order. Apply its final layer norm **exactly once**. Deterministic preprocessing; cache FP16 features; unload the vision encoder from GPU before Qwen training.
- Connector: `Linear(768,1024) → GELU → Linear(1024,1024)`, trainable throughout. Insert its visual embeddings inside the user message before the question, preserving Qwen chat structure; align labels, masks, and positions. Calibrate connector magnitude against text embeddings using **training data only**, recording rule/scale. Frozen Qwen weights must still propagate gradients to the connector.
- COCO 2014 captions/instance annotations plus VQAv2 complementary pairs. Train images from `train2014`; development/test from `val2014`; development and final-test **image-disjoint across all suites** and disjoint from training/replay. Training images may recur across stages. Seed 6304; preserve image/question/pair IDs.
- Natural pairs: same question, different normalized consensus answers, at least **8/10** annotators agreeing for **each** image. Train on an original answer string from the consensus group; preserve all ten answers for scoring.
- Generate presence and counting questions programmatically from annotations; presence pairs yes/no across annotated/absent categories, counting pairs with different annotated counts in **1–4**. Use the exact question templates on PDF p.7. Inspect annotation-label quality independently of predictions.

| Component | Training | Development | Final test |
| --- | --- | --- | --- |
| COCO captions | 6,000 images × 3 captions = 18,000 examples | 96 images × 1 caption | None |
| Annotation grounding | Reusable generated pair pools, sampled per stage | 24 pairs / 48 questions | 64 pairs / 128 questions |
| Natural VQAv2 | 1,536 pairs / 3,072 questions | 32 pairs / 64 questions | 96 pairs / 192 questions |

Natural pair counts (yes/no, number, other): training **640/256/640**, development **12/8/12**, final **36/24/36**. Grounding pair totals combine presence and counting; the PDF does not prescribe a fixed unique training-pool count. Do not invent one as a mandatory requirement.

### Training curriculum: one continuing model, 4,500 updates

- Shared: seed 6304; microbatch **8**, accumulation **2**, effective batch **16**; AdamW decay **0.01**; clipping **1.0**; gradient checkpointing; no quantization; FP16 autocast/dynamic scaling; frozen FP16 backbone; FP32 trainable weights and optimizer states; unscale before clipping.
- Per-stage schedule: **5% warmup**, then cosine to **10% of peak LR**. Record any implementation choices not specified in the PA (including integer warmup rounding and stage optimizer handling) without presenting them as prescribed values.
- Content-token caps: question **80**, caption target **64**, short-answer target **16**. Record cap handling and relevant truncation counts.
- Train CE on answer/caption tokens **plus end-of-turn**; mask visual/prompt/padding tokens. First average supervised tokens **within each example**, then across the effective batch. This **example-weighted** objective differs deliberately from Task 1's **token-weighted** objective; test both independently.

| Stage | Updates / cumulative endpoint | Connector peak LR | LoRA peak LR | Sampling |
| --- | --- | --- | --- | --- |
| 1 Caption alignment | 2,250 / 2,250 | 1e-3 | None | Two passes over 18,000 captions; connector only |
| 2 Visual instruction | 1,500 / 3,750 | 1e-4 | 2e-5 | Presence 50%, counting 25%, caption replay 25% |
| 3 Natural VQA | 750 / 4,500 | 5e-5 | 1e-5 | Natural VQA 70%, presence replay 10%, counting replay 10%, caption replay 10% |

Total **72,000 example presentations**, including replay. At Stage 2 enable new LoRA adapters, then continue the same adapters in Stage 3: r=16, alpha=32, dropout=0.05, no trainable biases; target **only q/k/v/o projections**. Each random mixture draw yields **two examples**: a complete question pair or two caption examples. Use a seeded random sampler; **do not enforce exact mixture quotas**. Repeated pool samples are allowed. Record realized counts.

Save resumable checkpoints every **250 main updates** and connector/adapter weights at each stage end, including sampler/stage progress. The completed **Stage 3 model is final**; do not choose a final-test-best checkpoint.

### Evaluation and required evidence

- Development: evaluate **before training**, every **500 main updates**, and stage endpoints **2,250, 3,750, 4,500** (deduplicate overlapping events). Track caption reference loss plus reference-answer NLL on both VQA suites with correct/complementary/no-image inputs. Answer NLL **excludes termination**, averages tokens within example, then examples. Save swap penalty `NLL(original y|partner image) − NLL(original y|correct image)`.
- Final: both final suites, greedy, non-thinking, **16 new tokens**, **batch 8**, four conditions: correct image; complementary partner; fixed unrelated image from another pair in the same suite; no image. Fix mismatch mapping before generation and exclude both original-pair images. Keep question and **original references fixed** under every control. For no-image remove visual tokens and rebuild masks/positions; preserve text instructions.
- Grounding: normalized exact match, separate presence/counting. Natural VQA: official local soft consensus over ten references using `mean_i min(1, sum_{j!=i} 1[pred=answer_j]/3)`, separate yes/no/number/other. Do not replace this leave-one-annotator-out score with an approximate min(count/3,1).
- For each suite/condition save pair accuracy (both original canonical answers), identical-answer rate, empty-answer rate, reference NLL, generation time per answer, question-type results, correct-image accuracy/NLL advantages over complementary swaps. Empty answers stay in denominators. Also retain whether swapped predictions follow the partner answer as a distinct diagnostic.
- Use **4,000 complete-pair bootstrap resamples**, 95% intervals, matched pair IDs across conditions for differences. These are local subset results, not full-benchmark server scores.
- Required artifacts: training/development curves with stage boundaries; all four-condition result tables for both suites and subtypes, pair diagnostics and CIs; aligned complementary images/questions/references/predictions for student review; stage parameter/runtime/memory tables; feature-cache and generation costs separately. Fix and save an illustrative-example selection rule; student writes qualitative interpretations.
- **Optional only:** full independent repeat with **49 visual tokens**, spatial 2×2 pooling of the 14×14 features before the connector, fresh connector/adapters, same seed/data/budgets/evaluation. Do not spend the mandatory-run budget on this until required work is complete; list it as optional, not missing mandatory work.

## Exact project demands — Task 3 (PDF pages 10–15)

### Supplied offline pools, splits, and answer rules

- Locate **ATML-PA3-Data** and read its supplied **README.MD**. Offline pools: Qwen2.5-3B-Instruct on **1,000 MATH questions / 32,000 rollouts**; Qwen2.5-VL-3B-Instruct on **1,000 MathVista testmini questions / 32,000 rollouts**. **Do not regenerate these pools.** Only Step 4 requires live generation; offline analysis does not need images or the VLM loaded on GPU.
- Read `generations.parquet`, `probability_metrics.parquet`, and VLM `attention_metrics.parquet`. Join exactly on **(dataset, sample_id, rollout_id)**; assert key uniqueness, join completeness, and token-array alignment. Termination tokens are excluded. Use raw-distribution `logp` and `kl_u_p`; do not reconstruct KL from sampled-token probabilities.
- Per dataset: lexicographically sort question IDs, permute using **NumPy default_rng(6304)**, first **200 development**, remaining **800 held-out**. All rollouts stay with their question. First **50 MATH held-out IDs in that permutation** form the live subset. Save IDs and RNG/order provenance.
- Use supplied extracted answers for stored rollouts. Group equivalent MATH answers with pinned `math-verify`; use MathVista's released metadata/answer-type/choice/precision rules, with option letters for multiple choice. Form groups **without gold answers**; use the same equivalence for grading. Freeze/version the scorer before final evaluation.
- Missing extractions form one **NO_ANSWER** group, always incorrect, retained in voting, denominators, and costs. Compute correctness yourself for retrospective metrics; do not leak it into selection/stopping.
- Candidate budgets **1,2,4,8,16,32**. Save **ten seeded random rollout-ID permutations per question**, use nested prefixes, and reuse the same orders for all methods/controllers. Average over orders within question, then across questions. Resolve score/vote ties by earliest supporting rollout in the current order; tied RWC scores use **average ranks**. At N=1 every rule returns the only candidate.
- Costs: sum retained generated-token counts of **all consumed** candidates; FirstAns consumes only its first candidate. This excludes offline prefill/attention/scoring overhead. Main accuracy-difference CIs use **1,000 paired question-level bootstrap resamples**, after averaging orders; never bootstrap individual rollouts.

### Scores, aggregation, and offline experiment matrix

- Per rollout: `L_i = mean(logp)`; `K_i = mean(kl_u_p)` (self-certainty).
- MathVista: token visual engagement `v_t = A_image/(A_image+A_prompt+A_generated)`; `V_i=mean(v_t)`. Group masses need not sum to one. Assert valid denominators and investigate malformed records rather than silently changing the definition.
- For the **current candidate prefix only**, pool token KLs; compute linearly interpolated 5th/95th percentiles; `c_t=clip((k_t-Q05)/max(Q95-Q05,1e-8),0,1)`; `C_i=mean(c_t)`; **VisConf `Z_i=V_i*C_i`**. Recompute at each prefix, without unseen rollouts.
- For N>1, descending average rank `r_i` (best=1); RWC weight `(N-r_i)/(N-1)`; sum weights by answer group; tie as above. N=1 is an explicit special case.
- Both datasets at every N: **FirstAns, self-consistency, highest mean logp, highest self-certainty, self-certainty+RWC**.
- MathVista additionally: **vision-only highest / RWC, VisConf highest / RWC**.
- Save accuracy, oracle coverage, distinct groups, per-question choices/candidate IDs/costs, and failures separated into **no correct candidate generated** versus **correct candidate generated but missed**.
- At N=32: within-question score/length/correctness associations; top-vs-bottom score-quartile correctness differences averaged across questions; **shortest-rollout** length-only control. Define and record conventions for ties/constant-score cases not specified by the assignment.
- MathVista: within-rollout confidence–vision correlations, summarized across questions; tokenwise fusion `Z_token=mean(c_t*v_t)` versus `Z=mean(c_t)*mean(v_t)`, using **both highest-score and RWC**. Check `Z_token-Z = population token covariance`; save fixed illustrative trajectories for student inspection.

### Live generation: controlled 50-question comparison

- Use the original **Qwen/Qwen2.5-3B-Instruct revision from the supplied README**, fixed FP16 weights, **no quantization**, **batch 1**, same GPU. Seed 6304 with independently seeded calls. Sampling: temperature **0.7**, top-p **0.8**, top-k **20**, repetition penalty **1.0**.
- Native chat template; system: `You are a helpful assistant.` Append to the original problem: `Think step by step inside <think> </think> tags, then give your final answer inside \boxed{}.`
- Three strategies, each with **3,072 total output-token allowance**: one response capped at 3,072; three independent responses capped at 1,024 each plus self-consistency; initial 1,024 plus one critique-and-revision capped at 2,048, returning the revised answer.
- **Reuse the first parallel response as refinement's initial response.** Append it as an assistant message, then user: `Review the solution above. Identify and correct any errors; if it is correct, retain its answer. Give your final answer inside \boxed{}.` Generate critique+revision together without gold feedback.
- Extract the **last boxed answer using README's balanced-brace rule**. Keep raw prompts/responses/seeds/input-output token lengths/extraction and truncation flags. Measure end-to-end inference excluding model loading, plus peak GPU memory. Charge the initial response to both strategies, include critique tokens and the longer revision prompt; early stopping may use less than the allowance.
- Save accuracy, actual input/output tokens, runtime, extraction/truncation rates, and wrong→correct/correct→wrong revision transitions relative to the shared initial answer. Never use gold answers to choose initial versus revised outputs. Label these FP16 runs as a new controlled comparison, not reproduction of the stored pools.

### Adaptive sampling and required artifacts

- Reveal only saved-prefix candidates. Check stops at **4,8,16**, otherwise **32**; both controllers always return the **current self-consistency answer**.
- Agreement controller: unique non-NO_ANSWER leader with support fraction at least tau.
- Quality-aware: same agreement, plus self-consistency/RWC winner agreement and RWC margin at least gamma. Quality = self-certainty for MATH, VisConf for MathVista. Margin = (largest − runner-up answer vote)/total weight; runner-up=0 for one group. Recompute scores/calibration/ranks using revealed candidates only.
- Development grid: tau **{0.5,0.625,0.75,0.875,1.0}**, gamma **{0,0.1,0.25,0.5}**; include always-32 fallback. For each dataset/controller family choose **lowest mean token cost** within **one percentage point** of fixed-32 self-consistency development accuracy, averaging ten orders. Ties: higher accuracy, then smaller tau/gamma. Freeze before held-out evaluation.
- Save selected thresholds, held-out accuracy/mean tokens/mean rollouts/stopping distribution, fixed-N comparisons at similar observed costs, and incorrect early stops including cases recovered by fixed-32. Full-pool correctness fractions are retrospective only. These simulate stopping **between completed rollouts**, not within a rollout.
- Required artifacts: accuracy/oracle curves against count and token cost; compact method tables and failure decomposition; score/length/confidence–vision/fusion results and trajectories; live comparison table and transitions; adaptive development grid/selected-policy/held-out tables and stopping plots; aligned raw disagreement/revision/premature-stop cases under a stated selection rule. The student writes the analysis of exploration, evidence/consensus, refinement, adaptive allocation, compute, and reliability.

## Final course deliverables (PDF pages 1 and 15)

- **Public GitHub repository:** all submitted code, configurations, split/example IDs, environment spec, machine-readable results, and top-level README with reproduction commands for each task. Attribute materially reused external code. Do not commit raw datasets, secrets, or unnecessary large checkpoints. Prepare the repository locally; publishing is a separate action when the user requests it. Private Kaggle execution is compatible with a public final code repository.
- **Student-written PDF:** **8 pages main content**, PA0 NeurIPS format; references and optional appendix excluded from the limit. Repository link at the **end of the abstract**. Separate results/analysis subsection for Tasks 1–3; cross-task discussion not required; connected analysis organized by research questions. Central evidence must appear in the main pages because graders need not read the appendix. Cite papers/datasets/tools and disclose incomplete experiments/deviations. These are a student checklist, not permission for the agent to write the PDF.
- **LMS:** PDF plus accessible GitHub link in the announced format. Do not submit on the student's behalf without an explicit request. Never claim human-written report work complete simply because technical artifacts exist.

## 1. Establish exactly what the PA requires

1. Locate the authoritative ATML PA 3 PDF/DOCX/Markdown/notebook, starter code, datasets, rubric, and instructor clarifications. Search the supplied project and attached Kaggle inputs first. Do not substitute PA 1, PA 2, a research proposal, or another course's PA 3.
2. Read the full assignment, including appendices, report questions, plots, experiments, restrictions, and submission instructions. Inspect starter notebook Markdown and TODOs as well as code. If PDF extraction loses equations or tables, inspect the relevant pages visually.
3. If the assignment is missing or several plausible versions conflict, request the specific file/version. Continue only independent environment and repository inspection while waiting. Do not invent the missing requirements.
4. Save `docs/assignment_requirements.md`, with:
   - Assignment filename, version, checksum, and page/section or notebook-cell references.
   - Exact task/subtask IDs and concise descriptions, including conceptual questions.
   - Required datasets, splits, preprocessing, models, methods, losses, metrics, baselines, ablations, seeds, and training settings.
   - Required tables, figures, qualitative examples, comparisons, and written explanations.
   - Restrictions on libraries, pretrained weights, external data, compute, AI assistance, and permitted collaboration.
   - Required filenames, report format, notebook output requirements, archive structure, and deadline/timezone if stated.
   - A requirement-to-code/test/result/report mapping, with a status for every rubric item.
5. Distinguish **assignment requirements** from **engineering defaults** and **optional experiments**. The actual assignment and explicit user instructions take precedence over defaults here. Do not silently relax a required setting to fit compute. Record material ambiguities in `docs/open_questions.md` and resolve those affecting correctness before the dependent experiment.
6. Save a concise task execution plan and estimated GPU/runtime/storage needs in `docs/execution_plan.md`. Prioritize required experiments before optional improvements.

## 2. Work within the existing Kaggle setup

- Inspect and reuse the existing Kaggle configuration, repository conventions, and job submission mechanism. Do not replace credentials or print API keys, tokens, or secret values into logs or notebooks. Use the existing authenticated tooling or platform secret mechanism.
- Verify the actual runtime, accelerator availability, free disk, RAM/VRAM, internet access, dataset mounts, dependency versions, and any limits visible to the session. Do not assume a particular GPU model, number of GPUs, or quota.
- Treat `/kaggle/input` as read-only inputs. Put generated work under a clearly named project directory inside `/kaggle/working`, configurable through `OUTPUT_ROOT`. Outside Kaggle, use a configurable local output directory.
- Save an environment manifest: Python and package versions, framework/CUDA/cuDNN versions where applicable, GPU names/count/VRAM, operating system, source revision or source checksums, dataset versions/checksums, seeds, and resolved configuration. Never include secrets.
- Use the installed environment when compatible. Pin any added dependencies; avoid unnecessary framework upgrades or reinstalling the CUDA stack. Record installation commands. If network access is unavailable, use attached approved inputs or report the precise missing dependency/data.
- Keep Kaggle jobs, datasets, and notebook versions private by default. Do not publish, submit to a course portal, or overwrite unrelated outputs unless the user explicitly requests that action.
- Use a configurable student identifier. Obtain it from the supplied assignment/project or user; do not guess it from unrelated files.

## 3. Implementation and experimental correctness

- Preserve the starter interface, required function signatures, task order, and grader assumptions. Keep originals intact or under version control; implement the actual TODOs.
- Prefer small reusable modules and thin notebooks/scripts. Any notebooks used must execute in order from a fresh kernel without hidden state or manual cell surgery. This PA requires a public code repository and a student-written PDF, not an invented mandatory notebook deliverable. Separate Markdown technical records supplement these deliverables and must not become AI-written PDF prose.
- Use config files or explicit arguments for paths, seeds, model settings, batch size, learning rate, epochs, precision, and resume options. Save the fully resolved configuration for every run.
- Match required algorithms mathematically. Do not replace a requested implementation with a library call, pretrained model, or different method unless permitted.
- Set and record random seeds for Python, NumPy, and the training framework, including data-loader workers when applicable. Record any nondeterministic operations and reproducibility limitations. Follow required seed counts; identify optional repeated trials separately.
- Keep the prescribed train/development/test partitions separate. Fit any data-driven preprocessing/calibration on allowed data only. Do not carve away required training examples or introduce an extra validation split that changes a prescribed budget. Follow the fixed Task 1 evaluation checkpoints and Task 2 final Stage 3 model; only Task 3 stopping thresholds are development-selected. The Task 1 test-question exclusion and fixed evaluation probes are explicitly prescribed uses, never training/tuning signals. Check duplicate/group/image leakage where relevant.
- Use the specified metric definitions and aggregation. Record units, class order, averaging convention, denominators, and sample counts. Verify the different token-weighted and example-weighted objectives. Compute uncertainty only through the prescribed valid resampling/repeated-run protocol; distinguish bootstrap confidence intervals from standard deviations. Do not invent extra training seeds as a replacement for the required bootstraps.
- Make baseline and ablation comparisons fair: consistent split, preprocessing, evaluation protocol, and applicable budget. Document intentional differences. A reduced-data smoke test is never a full experiment.
- Never invent results, smooth away failures, relabel incomplete runs as complete, or copy expected/reference scores as measured outcomes. Preserve failed runs and explain fixes. Report unsuccessful approaches honestly.

## 4. Required verification gates for EACH task

Do not launch a full task run until its applicable gates pass. Write meaningful tests against the task's requirements and independent expected behavior, not merely against the implementation's own output. Use supplied tests first, then add focused missing checks. Save the actual commands, exit codes, summaries, and logs.

### Gate A — Review and static/import checks

- Read the task requirements and inspect the code before execution: equations, tensor shapes, labels, masks, reductions, optimizer/scheduler placement, split handling, device transfers, and output paths.
- Run syntax/import checks for applicable Python modules and validate configs and dataset paths. For notebooks, validate notebook structure and inspect all executable cells, including magics, using appropriate notebook tooling.
- Ensure CLI entry points, imports, and tests cannot accidentally launch full training or download a large dataset. Guard executable entry points.

### Gate B — Focused correctness tests

Select and implement tests appropriate to the actual task. Examples, when relevant:

- Dataset counts, split disjointness, label ranges, preprocessing shape/dtype/range, finite values, and padding/mask behavior.
- Tiny hand-calculated examples or trusted reference comparisons for custom losses, gradients, metrics, optimizers, and requested algorithms.
- Forward-pass shapes, correct device placement, finite loss, expected trainable/frozen parameters, nonzero finite gradients where expected, and a real optimizer update.
- Training/evaluation mode differences, evaluation without gradient tracking, and no validation/test updates to parameters or train-only statistics.
- Accumulation equivalence to an equivalent batch within a stated numerical tolerance, if gradient accumulation is used; verify final partial accumulation handling and scheduler update counts.
- Checkpoint save/load round trip, identical evaluation predictions within tolerance, and correct resume state.
- Table/plot export on small synthetic data, including usable filenames and linked data sources.

Tests requiring a GPU must be explicitly labeled. CPU test success does not establish GPU execution correctness. A skipped required test is an unresolved gate, not a pass. Do not weaken assertions just to make tests green.

### Minimum assignment-specific test coverage

Create focused tests for these failure-prone requirements before the corresponding full run, and actually execute them:

| Area | Required checks |
| --- | --- |
| Task 1 data and SFT | Stable first-eligible source selection; GSM8K normalization exclusion; no answer truncation/shuffle; prompt/pad masks and end-of-turn supervision; hand-computed unequal-length **effective-batch token normalization** versus accumulation |
| Task 1 methods | L2-SP zero at the start and known nonzero penalty/gradient; fixed reference/tied-weight deduplication/all-parameter inclusion; penalty applied with the correct effective-update scaling rather than multiplied by accumulation count; only intended LoRA tensors train; matching LoRA-16/OPLoRA initial factors; OPLoRA left/right protected-subspace residuals and gradients; exported/merged logits within a recorded tolerance |
| Task 1 schedule/evaluation | 40,000/32=1,250; exact recovery/evaluation event lists; scaler/clip/scheduler order; required metric keys and split counts; template separation; seven evaluation conditions; 168 outputs/42 annotation rows |
| Task 2 interface | 196×768 features, deterministic preprocessing, final norm exactly once, 196×1024 connector output; valid visual/text masks and positions; frozen encoder/backbone with nonzero connector gradients; no-image prefix removal; full split-image disjointness |
| Task 2 training | Hand-computed **per-example token mean then effective-batch mean**; termination included for training but excluded for answer NLL; stage trainable-parameter sets/LRs/schedules/endpoints; seeded two-example draws with pair preservation; resume reproduces next draws; correct realized mixture accounting |
| Task 2 scoring | Exact local ten-answer VQA formula at boundary cases; canonical pair accuracy; empty-output denominators; original references retained under swaps; mismatch excludes both original images; complete-pair matched bootstrap |
| Task 3 ingestion/scoring | One-to-one joins and token alignment; 200/800 split and fixed first-50 subset; answer equivalence/options/precision and missing NO_ANSWER behavior; nested reproducible ten orders; N=1; score/vote ties and average ranks; hand-calculated RWC and token costs |
| Task 3 calibration/stopping | Constant-KL percentile edge case and linear interpolation; prefix-only recalibration; VisConf factorized and tokenwise covariance identity; **changing gold labels or unrevealed candidates cannot change any current selected answer or stop decision**; unique nonempty leader; margin; threshold eligibility/tie-breaks/fallback; frozen development-selected settings |
| Task 3 live/persistence | Balanced nested-brace last-box extraction; shared initial response identity; independent call seeds; fixed token allowances and actual cost charging; complete 50-question strategy outputs; checkpoint/export/readback and task-artifact generation |

Use a synthetic toy dataset first where the expected values are independently calculable. For numerical comparisons, state tolerances and the precision tested. Run the CPU unit suite, then the relevant real GPU smoke suite; save machine-readable results such as pytest JUnit XML and readable logs. Example test commands, once those paths/tests have actually been implemented: `python -m compileall -q src tests` and `python -m pytest -q --junitxml=<run_dir>/tests/pytest.xml`. Replace placeholders with actual paths, preserve nonzero exit status through logging/pipelines, and do not mark a gate passed because a command was merely written down.

### Gate C — Tiny end-to-end smoke run

- Run the real data-loading, preprocessing, forward/backward, optimizer, validation, checkpoint, metric export, and plotting paths on a deliberately small subset and few steps.
- Use a separate `smoke` run directory and explicit configuration so smoke artifacts cannot enter final tables or overwrite production checkpoints.
- Where appropriate, check that the model can overfit a tiny training subset under a suitable debug configuration; diagnose failures rather than assuming every stochastic minibatch must decrease monotonically. Do not apply this criterion blindly to methods where it is inappropriate.
- For GPU tasks, run a real GPU training step and confirm model tensors, inputs, and the actual computation are on the intended device. Fail clearly on unavailable CUDA rather than silently spending a full run on CPU.

### Gate D — Full run and post-run checks

- Launch the full required experiment only after Gates A–C pass. Save the exact command/config and test evidence tied to the source revision/checksum used for that run.
- Monitor early steps for NaN/Inf, label mistakes, exploding gradients, invalid metrics, wrong sample counts, CPU fallback, OOM, and unusable throughput. Stop and diagnose invalid runs promptly.
- After completion, verify required epochs/steps and samples were actually processed, checkpoints follow the PA's fixed schedule/final-model rule, metric files are complete, and plots/tables derive from those files.
- Re-run relevant tests after changes. If a fix changes training or evaluation behavior, invalidate affected results and regenerate them; do not keep stale metrics under updated code.
- Execute the final submission entry point or notebook from a clean process/kernel with the intended inputs. Capture errors and exit status. Reuse validated checkpoints only when the PA permits it and identify them explicitly; do not describe an inference-only replay as a fresh training run.

Record each gate and its evidence in the task's Markdown technical record and in `docs/verification_matrix.md`.

## 5. Use the GPU correctly and conserve the budget

- Use the actual framework's supported device and precision APIs. Query capabilities at runtime. Verify GPU compute with tensor/device checks and a smoke step, with utilization/memory observations as supplementary evidence.
- For deep-learning training, keep the model and relevant tensors on the same intended accelerator. Keep data loading and routine plotting on CPU. Use pinned host memory/nonblocking transfers, workers, and prefetching only when supported and beneficial; measure throughput rather than assuming more workers are better.
- Establish a correct small FP32 reference where needed, then verify the **prescribed FP16** recipe with dynamic scaling. No BF16, 8-bit/4-bit loading, QLoRA, or quantized optimizers as silent substitutes. Check actual parameter/gradient/optimizer-state dtypes against each task, not just autocast settings.
- Smoke-test the **prescribed** microbatch and accumulation with memory headroom. Scale/normalize correctly and count optimizer/scheduler updates correctly. Record microbatch size, accumulation steps, and effective batch size; do not autonomously change them because another setting is faster.
- Do not change required batch semantics or hyperparameters silently. If a prescribed setup cannot fit, first use equivalent memory-saving options where valid; document the limitation and seek a decision if a requirement must change.
- Use evaluation mode and inference/no-gradient context during validation and prediction as appropriate. Avoid retaining graphs or GPU tensors in long-lived metric histories. Do not call cache-clearing functions every batch as a performance strategy.
- Use one GPU by default unless the assignment or measured runtime justifies more and the setup actually supports it. Correct multi-GPU execution must account for distributed sampling, metric reduction, effective batch size, rank-specific seeds, and checkpoint ownership; simply detecting two GPUs is insufficient.
- Benchmark startup and a few representative steps. Estimate total runtime and storage before full experiments. Avoid redundant downloads, repeated model loading, unnecessary repeated full runs, and large unplanned parameter sweeps.
- Checkpoint at the **prescribed intervals**, with extra recovery saves before expected session limits if necessary. Save model, optimizer, scheduler, applicable scaler, epoch/global step/stage, RNG states, resolved config, and data/sampler state needed for exact prescribed recovery. Verify resumed sampling/order and learning-rate progress against an uninterrupted tiny reference.
- Record wall-clock runtime, optimizer steps, sample counts, precision, accelerator, peak framework GPU memory when available, and useful throughput. Synchronize asynchronous GPU work when measuring GPU timings.
- If a job fails or the session is interrupted, preserve logs, diagnose the reason, rerun affected gates, and resume from a verified compatible checkpoint. Never combine incompatible runs into one learning curve.

## 6. Mandatory saved outputs

Use stable task IDs from the PA. The structure below is a default; adapt it to required submission conventions without omitting the evidence. Create **three separate Markdown technical records** and extra subtask records for substantial experiments. A combined index must not replace these files. Keep human report writing clearly separate from agent-generated operational documentation.

```text
project/
  AGENTS.md
  README.md
  assignment/                       # supplied source documents, if redistribution is allowed
  src/                              # reusable implementation
  notebooks/                        # required executable notebooks
  configs/                          # smoke and full-run configurations
  tests/                            # focused task tests
  docs/
    assignment_requirements.md
    execution_plan.md
    open_questions.md
    verification_matrix.md
    artifact_index.md
    results_index.md                 # factual result locations and task links; no report prose
    student_report_checklist.md      # required human writing/annotation items; no draft answers
    tasks/
      task_01_llm_finetuning.md
      task_02_visual_grounding.md
      task_03_test_time_compute.md
  outputs/
    environment/
    task_01/
      runs/<unique_run_id>/
        config.json
        provenance.json
        status.json
        logs/
        tests/                      # actual test logs and machine-readable results
        metrics/                    # per-step/epoch and final metrics in CSV/JSON
        tables/                     # CSV plus Markdown; LaTeX if needed by report
        figures/                    # PNG plus SVG/PDF where practical
        predictions/                # permitted raw outputs, labels and sample IDs
        examples/                   # raw aligned cases and human annotation sheets
        checkpoints/                # prescribed stage/final and resumable checkpoints
    task_02/
      ...
  submission/                       # code/results package; student supplies their own PDF
```

### Separate task Markdown technical records

Each of the three named task Markdown files must include factual operational documentation:

1. Task title, source requirement references, objective, and completion status.
2. Implementation/configuration references and engineering choices; a checklist linking the student's required analysis questions to supporting raw evidence, with no drafted report answers.
3. Dataset/split details, preprocessing, configuration, seeds, code revision/checksums, and run IDs.
4. Verification performed **before** full execution: commands, outcomes, failures/fixes, and links to test logs.
5. Full-run command, actual training/evaluation protocol, GPU/precision/batch information, and runtime.
6. Measured results with readable embedded tables/figures, mechanical labels/units, and relative links to raw CSV/JSON data and checkpoints. Student supplies narrative PDF captions.
7. Locations of required comparisons/ablations, raw failure cases, and annotation sheets. Leave qualitative judgments and interpretation for the student.
8. Reproduction/resume instructions, deliverables checklist, pending human writing/annotation, and remaining issues. Mark pending work explicitly instead of writing fabricated completed sections.

Update each task record immediately after its verified run, not only at the end. Record all official subtasks, including those awaiting human interpretation; mark GPU/testing items not applicable with a reason.

### Tables, graphs, and other evidence

- Save every required result table both as machine-readable CSV and readable Markdown. Include run IDs, seeds, split, metric names/units, and trial counts. Use additional LaTeX tables when the final report needs them.
- Save every required plot as a readable PNG and a vector version where practical, with labeled axes, units, legends, mechanical identifiers, and consistent scales for comparisons. Save source data and plotting code/config; leave narrative report captions to the student. Close figures after export.
- Determine the required figures from the assignment. Learning curves, confusion matrices, qualitative examples, embeddings, or ablation charts are conditional examples, not substitutes for the requested figures. Do not add misleading charts for unsuitable tasks.
- Derive technical records, tables, and plots from saved metrics/predictions. Do not manually type scores that can drift from the run records. Exclude smoke and failed runs from final comparisons and keep their status visible.
- Preserve required qualitative samples, predictions, model/config details, runtime/memory tables, logs, and checkpoints. Avoid retaining unnecessary huge artifacts or restricted raw data in the submission.
- Maintain `docs/artifact_index.md` mapping every requirement to its task record, code, tests, run ID, table, figure, and submission location. Check that all relative links resolve. Identify student-owned annotations/report prose as pending until the student supplies them.

## 7. Persistence, handoff, and submission audit

- Persist results under `/kaggle/working` throughout execution. Do not assume files left in an interactive session are durable. Use the existing configured Kaggle save/version/output mechanism and verify that the completed run exposes the expected outputs.
- If OpenCode submits remote Kaggle jobs, capture job/kernel identifiers and version, monitor until a terminal state, inspect logs and exit status, and retrieve the output bundle through the configured mechanism. A successful API submission is not experiment completion.
- Copy/download the verified artifacts to the user's project when the setup supports it. If that step is unavailable, provide the actual retrievable output location and clearly identify what remains uncollected. Never report a local download without checking the files exist.
- Create a manifest with relative paths, sizes, and checksums for deliverables. Produce a submission archive only with the course-required files and a separate evidence bundle when useful. Exclude credentials, caches, unrequested datasets, and unrelated outputs. Follow the PA's rules on whether checkpoints belong in the submission.
- Validate archive contents, any used notebook outputs, readable tables/figures, and reproducibility instructions. If the student supplies their own report, mechanically inspect its page count/format/links against the PA without writing or rewriting its prose. Do not generate an AI-written PDF report.
- Reconcile the final package against **every** rubric item. A complete gate requires implementation, passing applicable tests, an actual required run, saved evidence, and the written answer. Missing data, hardware, or runtime must be reported as a specific blocker; partial work is not a completed PA.
- End with a concise handoff listing each task's status, tests passed/failed/skipped, actual run outcomes, task-record/artifact paths, verified Kaggle output location, reproduction commands, unresolved requirements, and student-owned report work still needed. Do not claim tests, GPU runs, figures, or reports were produced unless they actually were.

## First action for the executing agent

Read `ATML-PA3.pdf` beside this file, or `assignment/ATML-PA3.pdf` in the portable handoff, and the assignment-specific requirements above. Discover the missing fixed assets and existing Kaggle setup without exposing secrets. Build the requirements matrix, then implement and verify each task through the gates above. Continue through required runs, separate task Markdown technical records, saved tables/graphs, verified output retrieval, and the technical submission audit. Prepare the student-owned report/annotation checklist; do not write the report. Ask only for information or decisions genuinely necessary to proceed correctly.
