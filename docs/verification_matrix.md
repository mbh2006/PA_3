# Verification matrix

Every gate records the actual command, exit status and evidence path. A gate that was
not executed is marked ⬜, not "passed".

## Task 3 — offline pipeline

| Gate | Check | Command | Result | Evidence |
|---|---|---|---|---|
| A | imports / collection | `pytest tests/unit` | **72 passed** | `outputs/task_03/tests/pytest_unit.xml` |
| A | data paths exist, reader version | `python scripts/gpu_preflight.py --dry-run` | PASS except disk warning (local disk below the GPU-box threshold) | `outputs/environment/preflight_dryrun.json` |
| B | split determinism, 200/800, live subset | `tests/unit/test_splits_orders.py` | passed | same XML |
| B | 10 orders are permutations, reproducible | `tests/unit/test_splits_orders.py` | passed | same XML |
| B | MATH equivalence (fractions, decimals, `$`, symmetry) | `tests/unit/test_answers.py` | passed | same XML |
| B | missing extractions form one NO_ANSWER group | `tests/unit/test_answers.py` | passed | same XML |
| B | MathVista MC letters, out-of-range letters, integer, float precision, list | `tests/unit/test_answers.py` | passed | same XML |
| B | RWC average ranks and hand-computed weights | `tests/unit/test_scores.py` | passed | same XML |
| B | VisConf hand-calculated value, constant-KL edge, prefix-only | `tests/unit/test_scores.py` | passed | same XML |
| B | union-find cannot bridge through unrevealed candidates | `tests/unit/test_scores.py` | passed | same XML |
| B | N=1 returns the only candidate | `tests/unit/test_adaptive.py` | passed | same XML |
| B | stopping: uniqueness, NO_ANSWER leader, support threshold, SC/RWC agreement, margin | `tests/unit/test_adaptive.py` | passed | same XML |
| B | **gold labels cannot change any decision** | `tests/unit/test_adaptive.py` | passed | same XML |
| B | **unrevealed candidates cannot change a decision** | `tests/unit/test_adaptive.py` | passed | same XML |
| B | bootstrap determinism, constant diff, empty input | `tests/unit/test_bootstrap_signals.py` | passed | same XML |
| B | fusion identity exact on synthetic data | `tests/unit/test_bootstrap_signals.py` | passed | same XML |
| B | equivalence cache round trip | `tests/unit/test_cache.py` | passed | same XML |
| B | supplied totals (rollouts, tokens, caps, nulls) vs instructor README | `pytest tests/integration` | **8 passed** | `outputs/task_03/tests/pytest_integration.xml` |
| B | token-array alignment for all five Parquets | inside `load_dataset` (raises) | passed | above |
| C | 40-question smoke on both datasets | `python -m task3.run_offline --limit 40` | completed | `outputs/task_03/smoke/` |
| D | full 1,000-question run, dev + held-out, both datasets | `python -m task3.run_offline --dataset both` | completed in 6m14s | `outputs/task_03/offline/` |
| D | tables derive from the saved metrics | `python -m task3.report` | completed | `outputs/task_03/tables/` |
| D | adaptive artefacts not overwritten across datasets | bug found and fixed (dataset-prefixed filenames); re-run verified | fixed | `adaptive_math_*`, `adaptive_mathvista_*` |

### Bugs found and fixed during verification

| Bug | Impact | Fix |
|---|---|---|
| `math-verify` returned False for `0.5` vs `\frac{1}{2}` | would have under-counted MATH correctness | wrapper `\boxed{…}` before parsing; unit test asserts 8 cases |
| `math-verify` spawns a process per call on Windows | 43 s for 10 questions | disable the timeout wrapper on Windows (POSIX keeps 5 s) |
| full-table `to_pylist` on 32k rows | minutes of overhead, high memory | stream one row group per question |
| adaptive artefacts written without a dataset prefix | MathVista silently overwrote MATH | dataset-prefixed filenames + re-run |
| equivalence cache read `n` from packed width | cache never hit, 12 min recomputed each run | store and read `n` explicitly; unit test |

## Task 1 — implementation (CPU)

| Gate | Check | Command | Result | Evidence |
|---|---|---|---|---|
| A | imports | `pytest tests/unit/test_task1_*` | passed | `outputs/task_01/tests/` (pending rerun) |
| B | eligibility reasons, source order, no truncation | `test_task1_data.py` | passed | — |
| B | prompt masked, response+EOT supervised, separator masked | `test_task1_data.py` | passed | — |
| B | **token-weighted accumulation == single-batch reference** | `test_task1_loss.py` | passed (1e-6) | — |
| B | Trainer gradient equals analytic reference | `test_task1_loss.py` | passed | — |
| B | Ω zero at start, hand-computed, biases/norms included, tied once | `test_task1_l2sp_optim.py` | passed | — |
| B | **penalty not multiplied by accumulation** | `test_task1_l2sp_optim.py` | passed | — |
| B | LoRA forward, OPLoRA residuals, full-rank zero update, matched init | `test_task1_lora.py` | passed | — |
| B | LR schedule warm-up/cosine values | `test_task1_l2sp_optim.py` | passed | — |
| C | GPU preflight with real settings | Kaggle `mbh2006/atml-pa3-gpu-test` | see §Kaggle | log retained |
| D | full training runs | — | ⬜ not started | — |

## Task 2 — implementation (CPU)

| Gate | Check | Command | Result |
|---|---|---|---|
| B | connector shape/dtype, calibration RMS | `test_task2_core.py` | passed |
| B | 196 visual tokens inserted before question, masked | `test_task2_core.py` | passed |
| B | no-image removes exactly 196 positions | `test_task2_core.py` | passed |
| B | answer NLL excludes termination | `test_task2_core.py` | passed |
| B | **example-weighted loss matches hand computation and differs from token-weighted** | `test_task2_core.py` | passed |
| B | two examples per draw, pair preservation, resumable sampler | `test_task2_core.py` | passed |
| B | **VQA leave-one-out scoring** (3/10 → 0.9, not 1.0) | `test_task2_core.py` | passed |
| B | VQAv2 8/10 agreement on both images, differing answers | `test_task2_data.py` | passed |
| B | image-disjoint dev/test/train | `test_task2_data.py` | passed |
| C/D | feature cache, curriculum, evaluation | — | ⬜ not started |

## Kaggle GPU feasibility test

| Attempt | Status | Outcome |
|---|---|---|
| v1 | ERROR | my `/kaggle/input` path assumption; dataset mounts at `/kaggle/input/datasets/<owner>/<slug>` |
| v2 | see `docs/verification_matrix.md` §log | real Full FT + LoRA update at microbatch 2, accum 16, seq ≤ 2,048 |

## Task 1 / Task 2 — full runs

Not started. Requires the GPU box; see `docs/execution_plan.md`.
