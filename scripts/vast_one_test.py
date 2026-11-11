"""Isolated memory check for one variant (fresh process, no cross-test fragmentation).

    python scripts/vast_one_test.py --variant full_ft_l2sp --seq 1942 --loss standard
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    from task1.data import MathExample
    from task1.l2sp import make_reference
    from task1.model import load_model, prepare_adapter, prepare_full_ft
    from task1.train import Trainer, TrainConfig

    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seq", type=int, default=1942)
    ap.add_argument("--loss", choices=["standard", "chunked"], default="standard")
    ap.add_argument("--updates", type=int, default=2)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    rng = np.random.default_rng(0)
    exs = []
    for i in range(32):
        ids = rng.integers(100, 151000, size=args.seq).astype("int32")
        labels = ids.copy()
        labels[:8] = -100
        exs.append(MathExample(i, "", "", "", ids, labels, 8))

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    is_adapter = args.variant.startswith(("lora", "oplora"))
    if is_adapter:
        model = load_model("Qwen/Qwen3-0.6B", dtype=torch.float16, device="cuda")
        model, _ = prepare_adapter(model, args.variant, r=16, alpha=32, dropout=0.05,
                                   seed=6304, oplora_k=128 if "oplora" in args.variant else None)
        lr = 2e-4
    else:
        model = load_model("Qwen/Qwen3-0.6B", dtype=torch.float32, device="cuda")
        model = prepare_full_ft(model)
        lr = 2e-5

    lam = 1e-3 if args.variant.endswith("l2sp") else 0.0
    ref = make_reference(model, device="cpu") if lam else None

    cfg = TrainConfig(variant=args.variant, peak_lr=lr, lambda_l2sp=lam, updates=1,
                      checkpoint_updates=(), chunked_loss=(args.loss == "chunked"),
                      max_micro_tokens=3000)
    tr = Trainer(model, exs, tok, cfg, device="cuda", out_dir=None,
                 reference=ref, is_adapter=is_adapter)
    tr.update_once(list(range(32)))
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(args.updates):
        st = tr.update_once(list(range(32)))
    torch.cuda.synchronize()
    dt = (time.time() - t0) / args.updates

    print(json.dumps({
        "variant": args.variant, "seq": args.seq, "loss": args.loss,
        "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
        "update_seconds": round(dt, 2),
        "est_hours": round(1250 * dt / 3600, 2),
        "loss_value": round(st["sft_loss"], 4),
        "omega": round(st["lambda_omega"], 6),
        "ok": True,
    }))


if __name__ == "__main__":
    main()
