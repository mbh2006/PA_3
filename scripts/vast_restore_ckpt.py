#!/usr/bin/env python3
"""Recover full_final weights for a full-FT run from its update-1250 checkpoint.

The run crashed after training but before `model_final.pt` was written. The checkpoint
saved at update 1250 carries the complete model state, so it is exported instead of
retraining. Verification is explicit: the checkpoint must report update 1250 and
40,000 examples seen before anything is written.

    python scripts/vast_restore_ckpt.py full_ft_l2sp
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

RUNS = Path("/root/pa3/outputs/task_01/runs")


def main() -> None:
    import torch

    variant = sys.argv[1] if len(sys.argv) > 1 else "full_ft_l2sp"
    train_dir = RUNS / variant / "train"
    ckpt = train_dir / "checkpoints" / "checkpoint_1250.pt"
    target = train_dir / "model_final.pt"

    if not ckpt.exists():
        raise SystemExit(f"checkpoint missing: {ckpt}")
    print(f"checkpoint {ckpt}  {ckpt.stat().st_size/1e9:.2f} GB", flush=True)

    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    update = int(payload.get("update", -1))
    examples = int(payload.get("examples_seen", -1))
    has_model = "model" in payload
    n_tensors = len(payload["model"]) if has_model else 0

    print(json.dumps({"update": update, "examples_seen": examples,
                      "has_model_state": has_model, "n_tensors": n_tensors,
                      "keys": sorted(payload.keys())}, indent=2))

    ok = update == 1250 and examples == 40000 and has_model and n_tensors > 0
    if not ok:
        raise SystemExit("checkpoint failed verification; refusing to export")

    torch.save(payload["model"], target)
    marker = {
        "restored_from": str(ckpt),
        "checkpoint_update": update,
        "checkpoint_examples_seen": examples,
        "exported_to": str(target),
        "exported_bytes": target.stat().st_size,
        "at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "verified": True,
        "reason": "run crashed in displacement() after training, before model_final.pt",
    }
    (train_dir / "RESTORED.json").write_text(json.dumps(marker, indent=2) + "\n")

    print(json.dumps({"exported": str(target),
                      "gb": round(target.stat().st_size / 1e9, 2),
                      "verified": True}, indent=2))


if __name__ == "__main__":
    main()
