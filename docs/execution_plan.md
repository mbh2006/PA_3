# Execution plan — Task 1 and Task 2 on the rented GPU

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

| Session | Content | Est. hours |
|---|---|---:|
| S0 | preflight, requirements, model/data cache | 0.5 |
| S1 | Task 1 data prep (40k eligibility, tokenisation, probes), unit tests | 0.5 |
| S2 | Task 1 Full FT: 1,250 updates + mid/final eval + diagnostics + 24 prompts | 3.0 |
| S3 | Task 1 Full FT + L2-SP: same | 3.0 |
| S4 | Task 1 LoRA-4: train + eval + prompts | 2.0 |
| S5 | Task 1 LoRA-16: train + eval + prompts | 2.0 |
| S6 | Task 1 OPLoRA-16: basis precompute, train, export, eval, merge check | 2.5 |
| S7 | Task 2 data build (COCO/VQAv2 selective download, feature cache) | 1.0 |
| S8 | Task 2 curriculum 4,500 updates + dev evals | 4.0 |
| S9 | Task 2 final 4-condition evaluation + qualitative exports | 1.0 |
| S10 | Task 3 live generation (50 questions × 3 strategies) | 2.0 |
| **Total** | | **≈21.5 h** |

At ~$0.40/h that is ≈$8.6; the earlier planning figure of ~$6 assumed a faster box and
no re-runs. Treat the ceiling as ~$10–12 and stop the instance between sessions.

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
