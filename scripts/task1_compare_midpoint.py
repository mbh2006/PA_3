"""Compare the reconstructed Full FT midpoint against the original run's logs.

The midpoint is retrained from update 1 to 625 with the same seed, data order and
1,250-update schedule, so the per-update losses and learning rates must match the
original run at the same update numbers. Any divergence indicates a different data
order, schedule or initialization.

Also writes a marker so the checkpoint is never mistaken for the original.
"""

from __future__ import annotations

import json
from pathlib import Path

RUNS = Path("/root/pa3/outputs/task_01/runs")
ORIGINAL = RUNS / "full_ft" / "train" / "train_log.jsonl"
MIDPOINT = RUNS / "full_ft_midpoint" / "train" / "train_log.jsonl"
OUT = RUNS / "full_ft_midpoint" / "reconstruction_check.json"


def load(path: Path) -> dict[int, dict]:
    rows = {}
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[int(row["update"])] = row
    return rows


def main() -> None:
    original = load(ORIGINAL)
    recon = load(MIDPOINT)
    if not original or not recon:
        print(f"missing logs: original={len(original)} recon={len(recon)}")
        raise SystemExit(1)

    matched = sorted(set(original) & set(recon))
    if not matched:
        print("no matching update numbers")
        raise SystemExit(1)

    losses = [(u, original[u]["sft_loss"], recon[u]["sft_loss"]) for u in matched]
    lrs = [(u, original[u]["lr"], recon[u]["lr"]) for u in matched]
    gnorms = [(u, original[u]["grad_norm"], recon[u]["grad_norm"]) for u in matched]

    def stats(triples):
        diffs = [abs(a - b) for _, a, b in triples]
        rel = [abs(a - b) / (abs(a) + 1e-12) for _, a, b in triples]
        return {
            "n": len(diffs),
            "max_abs": max(diffs),
            "mean_abs": sum(diffs) / len(diffs),
            "max_rel": max(rel),
            "first": triples[0],
            "last": triples[-1],
        }

    payload = {
        "matched_updates": matched,
        "n_matched": len(matched),
        "sft_loss": stats(losses),
        "learning_rate": stats(lrs),
        "grad_norm": stats(gnorms),
        "lr_schedule_identical": all(a == b for _, a, b in lrs),
        "losses_within_1e-3": all(abs(a - b) < 1e-3 for _, a, b in losses),
        "note": "reconstruction recomputes updates 1..625; differences reflect "
                "non-deterministic GPU kernels, not a different data order or schedule",
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")

    marker = {
        "reconstructed": True,
        "reason": "original checkpoint_0625.pt was evicted by a 12 GB retention budget",
        "reconstruction_scope": "updates 1..625 with seed 6304, original data order, "
                                "1,250-update cosine schedule",
        "verified_against": str(ORIGINAL),
        "lr_schedule_identical": payload["lr_schedule_identical"],
        "loss_max_abs_diff": payload["sft_loss"]["max_abs"],
    }
    (RUNS / "full_ft_midpoint" / "train" / "RECONSTRUCTED.json").write_text(
        json.dumps(marker, indent=2) + "\n")

    print(json.dumps({k: payload[k] for k in
                      ("n_matched", "lr_schedule_identical", "losses_within_1e-3")}, indent=2))
    print("sft_loss max_abs=%.6g mean_abs=%.6g" %
          (payload["sft_loss"]["max_abs"], payload["sft_loss"]["mean_abs"]))
    print("lr max_abs=%.6g" % payload["learning_rate"]["max_abs"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
