# Execution plan — Task 1 and Task 2 on the rented GPU

> **Historical document.** This is the plan as written before the GPU work started. It was
> followed in outline; the measured timings, costs and the deviations from it are recorded
> in `docs/PA3_COMPLETE_TECHNICAL_RECORD.md` and the per-task records. Where this file and
> the measured record disagree, the measured record wins.

**All timings and costs below are planning estimates.** They are replaced by measured
values as soon as `scripts/gpu_preflight.py` runs on the box (it reports tokens/s, which
feeds the estimates) and after the first real training step.

## Preconditions (no paid time spent)

- [x] Task 3 offline pipeline complete on CPU.
- [x] GPU preflight script written and dry-run validated.
- [ ] SSH access to the rented box.
- [ ] First paid action = `python scripts/gpu_preflight.py` (target < 3 min).

## Box requirements

| Item | Requirement |
|---|---|
| GPU | 1× RTX 4090 (24 GB) or better |
| Disk | ≥ 100 GB free |
| Image | PyTorch 2.4+, CUDA 12.1+ |
| Python | 3.10–3.12 |
| Packages | `requirements-gpu.txt` (transformers, peft, accelerate, lm-eval, datasets, pyarrow 24) |

## Session plan (each session is one long-running script; no manual steps)

**Measured on a Kaggle T4 (see `docs/tasks/task_01_llm_finetuning.md` §3):** full FT
13.18 GB peak, 6.2–8.2 s per optimizer update ⇒ **2.2–2.9 h per 1,250-update run**;
LoRA-16 8.49 GB, 4.5 s per update ⇒ **1.6 h**. The figures below use the measured
training times and keep evaluation times as estimates.

| Session | Content | Est. hours |
|---|---|---:|
| S0 | preflight, requirements, model/data cache | 0.5 |
| S1 | Task 1 data prep (40k eligibility, tokenisation, probes), unit tests | 0.5 |
| S2 | Task 1 Full FT: 1,250 updates + mid/final eval + diagnostics + 24 prompts | 2.9 |
| S3 | Task 1 Full FT + L2-SP: same | 2.9 |
| S4–S6 | LoRA-4, LoRA-16, OPLoRA-16 (train + eval + prompts) | 1.8 × 3 |
| S7 | Task 2 data build (COCO/VQAv2 selective download, feature cache) | 1.0 |
| S8 | Task 2 curriculum 4,500 updates + dev evals | 2.5 |
| S9 | Task 2 final 4-condition evaluation + qualitative exports | 1.0 |
| S10 | Task 3 live generation (50 questions × 3 strategies) | 3.0 |
| **Total** | | **≈19.5 h** |

That is within one week of the ~30 GPU-hour Kaggle quota, but with little slack. The
binding constraint is quota, not memory: a failure that needs a re-run can exhaust the
week. A rented 4090 does not change the recipe, only the wall-clock and the risk.

## Non-negotiable fidelity rules

- No BF16, no quantisation, no reduced budgets, no changed batch/accumulation.
- FP16 AMP + dynamic scaling, FP32 master/adapter parameters, FP32 optimiser states,
  unscale before clipping, log skipped updates.
- Task 1: token-weighted loss over the effective batch, 1,250 updates, 40,000 examples.
- Task 2: example-weighted loss, 4,500 updates, 72,000 presentations, paired draws.
- Fixed evaluation checkpoints; no final-test-based checkpoint selection.

## Ordering rationale

Task 1 runs first because it is the longest and its seven evaluation conditions gate the
benchmark table. Task 2's code is written while Task 1 trains. Task 3 live generation
runs last because it needs the MATH split only, which is already fixed.

## Stop-the-clock policy

The instance is stopped while code is being written or tests are being debugged off-box.
Each session writes a single JSON status file so an interrupted session can be resumed
from its last checkpoint instead of restarted.
