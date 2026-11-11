# Task 1 — recovery checkpoint inventory

Source: `outputs/task_01/runs/<condition>/train/checkpoints/`, inspected 2026-11-10 21:58 UTC.

The assignment asks for recovery checkpoints at updates 312 / 625 / 937 / 1,250 for each
run. Full fine-tuning checkpoints are large (7.15 GB: FP32 parameters plus FP32 AdamW
moments), and the first two full-FT runs used a 12 GB retention budget sized for Kaggle's
20 GB disk. The 4090 has 100 GB, so the budget was later raised to 60 GB.

## What exists

| Condition | Checkpoints present | Missing |
|---|---|---|
| Full FT (original run) | 1250 | **312, 625, 937** |
| Full FT + L2-SP | 1250 | **312, 625, 937** |
| **Full FT midpoint (reconstructed)** | **312, 625** | — (run stops at 625 by design) |
| LoRA-4 | 312, 625, 937, 1250 | — |
| LoRA-16 | 312, 625, 937, 1250 | — |
| OPLoRA-16 | 312, 625, 937, 1250 | — |

Adapter checkpoints are small (≈120 MB) and were never at risk.

## Eviction evidence

`full_ft/train/checkpoint_retention.json` records each removal with a timestamp:

| Removed | Size | Reason | Time (UTC) |
|---|---:|---|---|
| checkpoint_0312.pt | 7.15 GB | disk budget 12.0 GB exceeded | 2026-11-10 04:27:29 |
| checkpoint_0625.pt | 7.15 GB | disk budget 12.0 GB exceeded | 2026-11-10 04:38:45 |
| checkpoint_0937.pt | 7.15 GB | disk budget 12.0 GB exceeded | 2026-11-10 04:50:04 |

## Reconstructed midpoint

The evicted Full FT midpoint was rebuilt rather than approximated:

* retrained updates 1..625 from the original Qwen3-0.6B with seed 6304, the same
  `prepared.npz`, and the 1,250-update cosine schedule;
* `train/RECONSTRUCTED.json` labels it, records the reason and the comparison source;
* comparison against the original run's log at matching updates:
  **learning rate is bit-identical** (`lr_schedule_identical: true`), SFT loss differs by
  at most 5.7e-3 (mean 1.4e-3) — GPU kernel non-determinism, not a different data order;
* `reconstruction_check.json` holds the full per-update comparison.

The midpoint checkpoint must always be described as **reconstructed** in the report, not
as the original quarter-epoch save.

## Consequence for the report

Two recovery states (Full FT and L2-SP at 312 / 625 / 937) were lost to the retention
budget before it was corrected. All final weights, the reconstructed Full FT midpoint,
and every adapter checkpoint are intact, so no reported result depends on the lost files.
This should be disclosed as a process deviation.
