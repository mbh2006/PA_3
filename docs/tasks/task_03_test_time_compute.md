# ATML PA 3 — Task 3 technical record

**Task:** Test-Time Compute — sampling, selection, and adaptive inference (PDF pp. 10–15)
**Status:** offline pipeline complete and run on the full supplied data; live generation and the final qualitative comparison pending (GPU).
**Owner of interpretation:** the student. This file records operations, settings and measured numbers only.

---

## 1. Sources

| Item | Value |
|---|---|
| Data | `ATML-PA3-Data/` (official Drive folder linked on PA p. 11) |
| Data verification | all six files SHA-256 match `download_manifest.json` (re-checked 2026-11-09) |
| Parquet reader | **`pyarrow==24.0.0`** (19.0.0 fails: `Repetition level histogram size mismatch`) |
| MathVista metadata | `https://raw.githubusercontent.com/lupantech/MathVista/main/data/testmini.json` (871,725 B, sha256 `ab0148dc…`) |
| Scorer | `task3-scorer-v1`, frozen before final evaluation (`src/task3/answers.py`) |
| Split seed | 6304 |
| Analysis host | WSL2 Ubuntu 24.04, 8 vCPU, Python 3.12.3 |

Score of record for MATH: `math-verify==0.9.0`. For MathVista: the released
`evaluation/calculate_score.py` rules, re-implemented and unit-tested.

---

## 2. Supplied data as loaded

| Dataset | Questions | Rollouts | Retained tokens | Capped rollouts | Null extractions |
|---|---:|---:|---:|---:|---:|
| MATH (`qwen25-math`) | 1,000 | 32,000 | 20,742,094 | 523 | 521 |
| MathVista (`qwen25vl-mathvista`) | 1,000 | 32,000 | 8,961,703 | 487 | 3,032 |

These totals are asserted by `tests/integration/test_task3_data.py` against the
instructor README, so the loader is checked against an external source rather than
against itself. Join key `(dataset, sample_id, rollout_id)` is unique and complete;
every per-token list matches `n_generated_text_tokens`; attention list lengths match
and their denominators are positive.

---

## 3. Method as implemented

### 3.1 Splits
Question IDs sorted lexicographically, then `numpy.random.default_rng(6304).permutation`.
First 200 → development, remaining 800 → held-out. The first 50 MATH held-out IDs in
that permutation are the live subset. Saved in `outputs/task_03/offline/splits_*.json`.

### 3.2 Answer grouping (no gold involved)
* **MATH** — pairwise symmetric equivalence via `math-verify`, computed once per
  question over all 32 answers. Answers are wrapped in `\boxed{…}` before parsing
  because the supplied strings are already-extracted boxed contents and unanchored
  extraction truncates bare expressions (`x+1` → `[]`). Groups on a prefix are formed
  by union-find over **revealed pairs only**, so an unrevealed candidate can never
  bridge two groups (unit-tested).
* **MathVista** — canonical key per rollout from the released rules: multiple choice →
  option letter (with the official Levenshtein nearest-choice fallback and
  out-of-range-letter rejection), integer → `int(float(x))`, float → `round(x, precision)`,
  list → whitespace-normalised string.
* **Missing extraction** → one shared `NO_ANSWER` group, always graded incorrect, kept
  in voting, denominators and costs.

### 3.3 Scores
`L_i = mean(logp)`, `K_i = mean(kl_u_p)`, `V_i = mean_t(A_img/(A_img + A_prompt + A_gen))`.
VisConf calibration pools token KLs **over the current prefix only**, uses
`numpy.quantile(..., method="linear")` for Q05/Q95, clips to [0,1] and floors the
denominator at 1e-8. RWC uses descending average ranks and `w_i = (N - r_i)/(N - 1)`;
`N = 1` is a special case. Ties resolve to the earliest supporting rollout in the
current order.

### 3.4 Orders and budgets
Ten permutations of rollout IDs 0–31 per question, seeded from
`default_rng([6304, question_index])` where `question_index` is the position in the
lexicographic order (independent of the split). Budgets 1, 2, 4, 8, 16, 32; nested
prefixes; identical orders for every method and controller.

### 3.5 Cost
Sum of retained generated-token counts of consumed candidates. `FirstAns` consumes one
candidate at every budget. Costs exclude prefill/attention/scoring overhead by design.

---

## 4. Verification before the full run

| Gate | Command | Result |
|---|---|---|
| A. imports | `python -m pytest tests/unit` | **72 passed** (`outputs/task_03/tests/pytest_unit.xml`) |
| B. unit correctness | hand-computed RWC, VisConf, tie-breaks, cost, no-leak | included above |
| B. integration | `python -m pytest tests/integration` | **8 passed** (`outputs/task_03/tests/pytest_integration.xml`) |
| C. smoke | `--limit 40` on both datasets | completed, artefacts in `outputs/task_03/smoke/` |
| D. full run | `python -m task3.run_offline --dataset both` | completed, 6m14s |

Notable unit tests: constant-KL prefix produces no NaN; adding an unrevealed rollout
does not change the current decision; flipping all gold labels changes no decision;
N=1 returns the only candidate for every rule; `Z_token − Z` equals the population
token covariance.

---

## 5. Full-run results (held-out, 800 questions, averaged over 10 orders)

Primary metric: exact-match accuracy. `oracle` is coverage, not a selector.

### 5.1 MATH

| Method | N | Accuracy | Mean tokens | Oracle | Mean groups |
|---|---:|---:|---:|---:|---:|
| FirstAns | 1 | 0.6465 | 667 | 0.6465 | 1.00 |
| Self-consistency | 32 | 0.7379 | 21,347 | 0.9137 | 5.90 |
| Highest mean logp | 4 | 0.6710 | 2,667 | 0.7949 | 1.92 |
| Highest self-certainty | 16 | 0.7082 | 10,678 | 0.8825 | 4.01 |
| Self-certainty + RWC | 32 | **0.7555** | 21,347 | 0.9137 | 5.90 |

Full grid: `outputs/task_03/offline/summary_math_heldout.json`
(`tables/math_heldout_methods.{csv,md}`).

### 5.2 MathVista

| Method | N | Accuracy | Mean tokens | Oracle | Mean groups |
|---|---:|---:|---:|---:|---:|
| FirstAns | 1 | 0.5421 | 278 | 0.5421 | 1.00 |
| Self-consistency | 32 | 0.6170 | 8,898 | 0.8950 | 3.93 |
| Highest mean logp | 2 | 0.5465 | 555 | 0.6683 | 1.42 |
| Highest self-certainty | 4 | 0.5574 | 1,112 | 0.7631 | 1.92 |
| Self-certainty + RWC | 32 | 0.6225 | 8,898 | 0.8950 | 3.93 |
| Vision-only + RWC | 32 | 0.6275 | 8,898 | 0.8950 | 3.93 |
| VisConf highest | 32 | 0.5775 | 8,898 | 0.8950 | 3.93 |
| VisConf + RWC | 32 | **0.6300** | 8,898 | 0.8950 | 3.93 |
| Z_token (fusion) + RWC | 32 | 0.6262 | 8,898 | 0.8950 | 3.93 |

### 5.3 Failure decomposition

`tables/*_failures.{csv,md}` separates *no correct candidate generated* from
*correct candidate generated but missed*. The three outcomes partition question-order records. At N=32 on MATH,
self-consistency has 8.625% without a correct candidate, 17.5875% with a
correct candidate missed, and 73.7875% selected correctly. The original
thresholding of order-averaged outcomes was replaced by order-level counts.

### 5.4 Paired bootstrap

1,000 question-level paired resamples, 95% percentile intervals, after averaging the
10 orders within each question: `tables/*_bootstrap.{csv,md}`. Differences are computed
against FirstAns and against self-consistency at the same budget.

### 5.5 N=32 signal analyses (held-out)

| Metric (mean over questions) | MATH | MathVista |
|---|---:|---:|
| Spearman(length, correct) | −0.076 (n=469) | −0.051 (n=635) |
| Spearman(logp, correct) | 0.134 (n=469) | 0.029 (n=635) |
| Spearman(self-certainty, correct) | 0.198 (n=469) | 0.048 (n=635) |
| Spearman(V, correct) | — | 0.042 (n=635) |
| Spearman(VisConf, correct) | — | 0.090 (n=635) |
| Top − bottom quartile, self-certainty | 0.120 | 0.035 |
| Top − bottom quartile, VisConf | — | 0.075 |
| Shortest-rollout accuracy | 0.625 | 0.536 |
| Shortest = best self-certainty | 5.9% | 1.5% |
| Within-rollout corr(c_t, v_t) | — | **−0.211** (800 questions) |
| Fusion identity max abs error | — | 0.0 (25,600 rollouts) |

`n_defined` < 800 where a question has constant scores or constant correctness, so the
correlation is undefined.

### 5.6 Adaptive stopping (dev-selected, frozen, then held-out)

Selection rule: lowest mean dev token cost among policies whose dev accuracy is within
1 pp of fixed-32 self-consistency; ties → higher accuracy, then smaller τ/γ. Dev
fixed-32 SC: MATH 0.8250, MathVista 0.6385.

| Dataset | Family | Selected | Dev acc | Held-out acc | Held-out tokens | Mean rollouts |
|---|---|---|---:|---:|---:|---:|
| MATH | agreement | τ = 0.625 | 0.8195 | 0.7347 | 10,165 | 11.84 |
| MATH | quality | τ = 0.5, γ = 0 | 0.8150 | 0.7319 | 7,591 | 8.90 |
| MATH | fixed 32 SC | — | 0.8250 | 0.7379 | 21,347 | 32.00 |
| MathVista | agreement | τ = 0.625 | 0.6265 | 0.6218 | 4,682 | 13.91 |
| MathVista | quality | τ = 0.5, γ = 0 | 0.6290 | 0.6190 | 3,654 | 11.00 |
| MathVista | fixed 32 SC | — | 0.6385 | 0.6170 | 8,898 | 32.00 |

Stopping distribution is saved per policy over all (question, order) pairs in
`adaptive_*_heldout.json`.

---

## 6. Artefact locations

```
outputs/task_03/
  offline/
    splits_{math,mathvista}.json
    per_question_{ds}_{dev,heldout}.parquet     # split, order, budget, method,
                                                # correct, cost, selected rollout/answer,
                                                # oracle, distinct groups
    summary_/failures_/bootstrap_*.json
    signals_{ds}.json                           # N=32 analyses
    trajectories_{mathvista}.json               # fixed illustrative trajectories
    adaptive_{ds}_{dev_grid,selected,heldout}.json
    meta_*.json                                 # runtime and method lists
  tables/                                       # CSV + Markdown for every table
  cache/math_equivalence.npz                    # cached MATH equivalence matrices
  tests/pytest_unit.xml, pytest_integration.xml
  smoke/                                        # 40-question smoke run (not final)
```

---

## 7. Decisions recorded for the report's deviations section

1. **MATH parsing wraps answers in `\boxed{}`** before `math-verify`; without the
   anchor, bare expressions are truncated by the parser (documented in code).
2. **Prefix grouping uses union-find over revealed pairs only.** Transitive bridging
   through unrevealed candidates is deliberately excluded.
3. **MathVista multiple-choice is compared as option letters** rather than choice text.
   This is equivalent to the official mapping but avoids the mojibake present in some
   choice strings (e.g. `135�`).
4. **`ignore_empty_extractions=True` semantics** are used, matching the PA's rule that
   missing extractions are a NO_ANSWER group rather than "nearest option".
5. **Adaptive thresholds were selected on 200 dev questions** and frozen before the
   held-out run; no held-out information entered the selection.
6. MATH tokens were parsed with a 5 s per-call timeout on POSIX; a small number of
   pathological expressions therefore fall back to "not equivalent". The count of
   timeouts is not currently logged — *pending* (see open questions).

---

## 8. Live and retrospective evidence

The [live summary](../../results/task_03/live/live_summary.json) records 50
questions and the revision transitions. The archived parallel vote grouped
literal extracted strings and excluded missing extractions. Actual input-token
counts and per-strategy runtimes were unavailable.

The [MATH](../../results/task_03/offline/early_stop_cases_math.json) and
[MathVista](../../results/task_03/offline/early_stop_cases_mathvista.json) case
lists come from a CPU replay of the frozen policies. They preserve the revealed
rollout IDs, early selection, fixed-32 outcome, and token costs. The MATH parser
timeout count remains unavailable.
