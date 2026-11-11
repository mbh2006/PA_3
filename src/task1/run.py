"""Run one Task 1 variant end to end (training + checkpoints + metrics).

    python -m task1.run --variant full_ft --data outputs/task_01/data/prepared.npz \
        --out outputs/task_01/runs/full_ft

Checkpoints are written at the prescribed updates. Because a full-FT checkpoint carries
FP32 parameters **and** FP32 AdamW moments (~7 GB), the run enforces a disk budget: if
the next checkpoint would exceed it, the oldest is removed and the removal is recorded
in ``checkpoint_retention.json`` (a platform constraint, reported as a deviation).
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from common.paths import utc_now, write_json
from task1.l2sp import make_reference
from task1.model import load_model, prepare_adapter, prepare_full_ft, trainable_parameter_count
from task1.prepare import load_prepared
from task1.train import Trainer, TrainConfig, checkpoint_size_bytes, displacement

MODEL_ID = "Qwen/Qwen3-0.6B"

VARIANTS = {
    "full_ft": dict(kind="full", peak_lr=2e-5, lam=0.0),
    "full_ft_l2sp": dict(kind="full", peak_lr=2e-5, lam=1e-3),
    "lora4": dict(kind="adapter", peak_lr=2e-4, r=4, alpha=8),
    "lora16": dict(kind="adapter", peak_lr=2e-4, r=16, alpha=32),
    "oplora16": dict(kind="adapter", peak_lr=2e-4, r=16, alpha=32, k=128),
}


@dataclass
class RunSpec:
    variant: str
    peak_lr: float
    lambda_l2sp: float
    updates: int = 1250
    seed: int = 6304
    max_checkpoint_gb: float = 60.0


def _disk_free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def _enforce_checkpoint_budget(ckpt_dir: Path, max_gb: float, log: list[dict]) -> None:
    """Drop the oldest checkpoints until the directory fits, never the newest one."""
    files = sorted(ckpt_dir.glob("checkpoint_*.pt"), key=lambda p: p.stat().st_mtime)
    total = sum(checkpoint_size_bytes(p) for p in files) / 1e9
    while len(files) > 1 and total > max_gb:
        oldest = files.pop(0)
        size = checkpoint_size_bytes(oldest) / 1e9
        log.append({"removed": oldest.name, "gb": round(size, 2),
                    "reason": f"disk budget {max_gb} GB exceeded",
                    "at_utc": utc_now()})
        oldest.unlink()
        total -= size


def run_variant(spec: RunSpec, data_path: Path, out_dir: Path, *,
                max_updates: int | None = None, device: str = "cuda") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_variant = VARIANTS[spec.variant]
    is_adapter = cfg_variant["kind"] == "adapter"

    examples, data_meta = load_prepared(data_path)
    if max_updates is not None:
        need = max_updates * 32
        examples = examples[:need]
    print(f"[{spec.variant}] {len(examples)} examples", flush=True)

    t0 = time.time()
    if is_adapter:
        model = load_model(MODEL_ID, dtype=torch.float16, device=device)
        model, bases = prepare_adapter(
            model, spec.variant, r=cfg_variant["r"], alpha=cfg_variant["alpha"],
            dropout=0.05, seed=spec.seed, oplora_k=cfg_variant.get("k"),
        )
    else:
        model = load_model(MODEL_ID, dtype=torch.float32, device=device)
        model = prepare_full_ft(model)
        bases = None
    load_seconds = time.time() - t0

    reference = None
    if spec.lambda_l2sp > 0:
        reference = make_reference(model, device="cpu")
        print(f"[l2sp] reference on CPU, {reference.storage_bytes / 1e9:.2f} GB", flush=True)

    train_cfg = TrainConfig(
        variant=spec.variant, peak_lr=spec.peak_lr, lambda_l2sp=spec.lambda_l2sp,
        updates=spec.updates, seed=spec.seed,
    )
    trainer = Trainer(model, examples, tokenizer=_tokenizer(), config=train_cfg,
                      device=device, out_dir=out_dir, reference=reference,
                      is_adapter=is_adapter)

    retention: list[dict] = []
    original_save = trainer.save_checkpoint

    def save_with_budget(update: int, *, examples_seen: int):
        path = original_save(update, examples_seen=examples_seen)
        _enforce_checkpoint_budget(trainer.checkpoint_dir, spec.max_checkpoint_gb, retention)
        write_json(out_dir / "checkpoint_retention.json", retention)
        return path

    trainer.save_checkpoint = save_with_budget  # type: ignore[assignment]

    summary = trainer.train(max_updates=max_updates)

    final: dict = {
        "variant": spec.variant,
        "spec": asdict(spec),
        "is_adapter": is_adapter,
        "trainable_parameters": trainable_parameter_count(model),
        "load_seconds": round(load_seconds, 1),
        "train": {k: v for k, v in summary.items() if k != "history"},
        "peak_mem_gb": summary["peak_mem_gb"],
        "checkpoint_retention": retention,
        "disk_free_gb_after": round(_disk_free_gb(out_dir), 1),
        "finished_utc": utc_now(),
    }
    # Persist the trained weights FIRST: a later diagnostic failure must never cost
    # the training run again.
    if is_adapter:
        from task1.lora import adapter_state

        torch.save(adapter_state(model), out_dir / "adapter_final.pt")
        if bases is not None:
            torch.save({k: tuple(t.cpu() for t in v) for k, v in bases.items()},
                       out_dir / "oploRA_bases.pt")
    else:
        torch.save(model.state_dict(), out_dir / "model_final.pt")

    if reference is not None:
        disp = displacement(model, reference)
        final["displacement_l2"] = disp
        final["omega_final"] = float(reference.omega(model))

    write_json(out_dir / "run_summary.json", final)
    write_json(out_dir / "config.json", {"model_id": MODEL_ID, **asdict(spec),
                                         "train": asdict(train_cfg)})
    return final


def _tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-updates", type=int, default=None)
    ap.add_argument("--max-checkpoint-gb", type=float, default=60.0)
    args = ap.parse_args()

    cfg = VARIANTS[args.variant]
    spec = RunSpec(variant=args.variant, peak_lr=cfg["peak_lr"],
                   lambda_l2sp=cfg.get("lam", 0.0),
                   max_checkpoint_gb=args.max_checkpoint_gb)
    info = run_variant(spec, args.data, args.out, max_updates=args.max_updates)
    print(json.dumps({k: v for k, v in info.items() if k != "train"}, indent=2)[:2000])


if __name__ == "__main__":
    main()
