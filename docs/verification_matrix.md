# Verification matrix

Every gate records the command, its outcome and the evidence path. A gate that was not
executed is marked **not run**, never "passed". Where a suite requires torch, it was run on
the GPU host; the local analysis environment is CPU-only and cannot even collect those
modules.

---

## Task 3 — offline pipeline

| Gate | Check | Command | Result | Evidence |
|---|---|---|---|---|
| A | imports, collection | `pytest tests/unit` | **72 collected / 72 passed** | `results/task_03/tests/pytest_unit.xml` |
| A | data paths, reader version | `python scripts/gpu_preflight.py --dry-run` | pass except a local-disk warning | `outputs/environment/preflight_dryrun.json` |
| B | split determinism, 200/800, live subset | `tests/unit/test_splits_orders.py` | passed | unit XML |
| B | ten orders are reproducible permutations | same | passed | unit XML |
| B | MATH equivalence (fractions, decimals, `$`, symmetry) | `tests/unit/test_answers.py` | passed | unit XML |
| B | missing extractions form one NO_ANSWER group | same | passed | unit XML |
| B | MathVista letters, out-of-range letters, integer, float precision, list | same | passed | unit XML |
| B | RWC average ranks, hand-computed weights | `tests/unit/test_scores.py` | passed | unit XML |
| B | VisConf hand-calculated, constant-KL edge, prefix-only | same | passed | unit XML |
| B | union-find cannot bridge through unrevealed candidates | same | passed | unit XML |
| B | N=1 returns the only candidate | `tests/unit/test_adaptive.py` | passed | unit XML |
| B | stopping: uniqueness, NO_ANSWER leader, support threshold, SC/RWC agreement, margin | same | passed | unit XML |
| B | **gold labels cannot change any decision** | same | passed | unit XML |
| B | **unrevealed candidates cannot change a decision** | same | passed | unit XML |
| B | bootstrap determinism, constant difference, empty input | `tests/unit/test_bootstrap_signals.py` | passed | unit XML |
| B | fusion identity exact on synthetic data | same | passed | unit XML |
| B | equivalence cache round trip | `tests/unit/test_cache.py` | passed | unit XML |
| B | supplied totals vs instructor README | `pytest tests/integration` | **8 passed** | `pytest_integration.xml` |
| B | token-array alignment across all five Parquets | inside `load_dataset` (raises) | passed | integration XML |
| C | 40-question smoke, both datasets | `python -m task3.run_offline --limit 40` | completed | outputs (not published) |
| D | full 1,000-question run, dev + held-out, both datasets | `python -m task3.run_offline --dataset both` | completed in 6m14s | `results/task_03/offline/` |
| D | tables derive from saved metrics | `python -m task3.report` | completed | `results/task_03/tables/` |
| D | adaptive artefacts not overwritten across datasets | — | bug found, fixed, re-run | `adaptive_math_*`, `adaptive_mathvista_*` |
| D | live generation, 50 questions × 3 strategies | `python scripts/task3_live.py` | completed, 4,382.8 s | `results/task_03/live/` |

### Bugs found and fixed during Task 3 verification

| Bug | Impact | Fix |
|---|---|---|
| `math-verify` returned False for `0.5` vs `\frac{1}{2}` | would have under-counted MATH | wrap as `\boxed{…}` before parsing; 8-case unit test |
| `math-verify` spawns a process per call on Windows | 43 s for 10 questions | disable the timeout wrapper on Windows |
| full-table `to_pylist` on 32k rows | minutes of overhead, high memory | stream one row group per question |
| adaptive artefacts written without a dataset prefix | MathVista overwrote MATH | dataset-prefixed filenames + re-run |
| equivalence cache read `n` from packed width | cache never hit, 12 min recomputed per run | store `n` explicitly; unit test |

---

## Task 1

| Gate | Check | Command | Result | Evidence |
|---|---|---|---|---|
| A | imports, collection | `pytest tests/unit/test_task1_*` (GPU host, torch) | passed | run logs |
| A | config and dataset paths validated | `scripts/gpu_preflight.py` | passed | `outputs/environment/` |
| B | eligibility reasons, source order, no truncation | `test_task1_data.py` | passed | — |
| B | prompt masked, response + EOT supervised, separator masked | `test_task1_data.py` | passed | — |
| B | **token-weighted accumulation == single-batch reference** | `test_task1_loss.py` | passed (1e-6) | — |
| B | trainer gradient equals the analytic reference | `test_task1_loss.py` | passed | — |
| B | chunked LM-head loss equals the full projection | `test_task1_chunked_loss.py` | passed | — |
| B | Ω zero at start, hand-computed, biases/norms included, tied weights once | `test_task1_l2sp_optim.py` | passed | — |
| B | **penalty not multiplied by accumulation** | `test_task1_l2sp_optim.py` | passed | — |
| B | LoRA forward, OPLoRA residuals, zero-update case, matched initialisation | `test_task1_lora.py` | passed | — |
| B | LR schedule warm-up and cosine values | `test_task1_l2sp_optim.py` | passed | — |
| B | checkpoint retention budget arithmetic | `test_task1_checkpoint_budget.py` | passed | — |
| C | GPU smoke: real FP16 step, FP32 masters, clip order | `scripts/gpu_preflight.py` | passed on RTX 4090 | `outputs/environment/variant_tests.*` |
| D | five variants + baseline, 1,250 updates each | `scripts/task1_variant_step.py` | completed; `examples_seen = 40,000`, `skipped_updates = 0` | `results/task_01/runs/*/train/run_summary.json` |
| D | midpoint reconstruction at update 625 | rerun then verified against the original log | LR bit-identical, loss max diff 5.7e-3 | `docs/tasks/task_01_checkpoint_inventory.md` |
| D | seven-condition evaluation | lm-evaluation-harness | completed | `results/task_01/tables/condition_table.csv` |
| D | 168 qualitative responses, 42-row sheet | `scripts/task1_variant_step.py` | verified 7 / 168 / 42 | `outputs/task_01/verification.json` |
| D | ρ_k for all six conditions | — | **not run** for five conditions | see Task 1 record §2.5 |

---

## Task 2

| Gate | Check | Command | Result | Evidence |
|---|---|---|---|---|
| A | imports, collection | `pytest tests/unit/test_task2_*` (GPU host, torch) | passed | run logs |
| B | connector shape/dtype, calibration RMS | `test_task2_core.py` | passed | — |
| B | 196 visual tokens inserted before the question and masked | `test_task2_core.py` | passed | — |
| B | no-image removes exactly those 196 positions | `test_task2_core.py` | passed | — |
| B | answer NLL excludes termination | `test_task2_core.py` | passed | — |
| B | **example-weighted loss matches hand computation and differs from token-weighted** | `test_task2_core.py` | passed | — |
| B | two examples per draw, pair preservation, resumable sampler | `test_task2_core.py` | passed | — |
| B | **VQA leave-one-out scoring** (3/10 → 0.9, not 1.0) | `test_task2_core.py` | passed | — |
| B | VQAv2 8/10 agreement on both images, differing answers | `test_task2_data.py` | passed | — |
| B | image-disjoint train/dev/test | `test_task2_data.py` | passed | — |
| B | caption cycler makes exactly two full passes; Stage 1 caption-only | `test_task2_fixes.py` | passed | — |
| B | mixture probabilities per stage | `test_task2_fixes.py` | passed | — |
| B | training grounding records come from `grounding_train` | `test_task2_fixes.py` | passed | — |
| B | generation prompt contains no reference answer | `test_task2_fixes.py` | passed | — |
| B | checkpoint restores connector, adapters, optimizer, scaler, RNG, sampler | `test_task2_fixes.py` | passed | — |
| C | feature cache: 9,856 / 9,856 images present | `--stage features` + `assert_features_complete` | passed (after downloading 853 missing) | `training.json` |
| D | 4,500 updates | `--stage train` | completed: 2,250 / 1,500 / 750 | `results/task_02/training.json` |
| D | final evaluation, four conditions, both suites | `scripts/task2_recover_evidence.py` | Kaggle batch 8; 1,280 predictions | `results/task_02/recovered_kaggle/metrics.json` |
| D | no-image condition removes visual tokens | Kaggle smoke and recovery | accuracy 0.0000 in both suites | `results/task_02/recovered_kaggle/gpu_smoke.json` |

### Invalidated Task 2 attempts

| Attempt | Defect | Disposition |
|---|---|---|
| 1 | trained on dev+test grounding pools; wrong Stage 1 mixture; prompt contained the reference; exact match instead of soft consensus | quarantined `outputs/task_02_invalid/INVALID.md` |
| 2 | still trained on dev+test; 853 images missing so every "with image" condition ran as no-image | quarantined `outputs/task_02_invalid2/INVALID.md` |

Neither attempt contributes any number to the published results.

---

## Cross-cutting

| Check | Result |
|---|---|
| Git working tree clean at publication | yes |
| Secrets scan (`.env`, keys, tokens, kaggle.json) | none tracked |
| Raw datasets excluded from the repository | yes |
| Large weights excluded from the repository | yes (only Task 2 stage weights retained locally) |
| Relative links into tracked files resolve | yes — paths under `data/` and `outputs/` are local working directories by design and are documented as such |
