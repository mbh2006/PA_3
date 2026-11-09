"""Short GPU test on Kaggle with the assignment's actual Task 1 training settings.

Runs the real code path — `Qwen/Qwen3-0.6B`, FP32 master weights, FP16 autocast, dynamic
gradient scaling, gradient checkpointing, micro-batch 2 x accumulation 16 = 32, sequence
length up to 2,048 — for exactly one optimizer update per variant, and records:

  * peak GPU memory
  * wall-clock per update and tokens/s
  * parameter / gradient / optimizer-state dtypes
  * whether the configuration fits (OOM is caught and reported, never hidden)

Writes ``outputs/environment/kaggle_gpu_test.json``.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = Path(os.environ.get("ATML_OUTPUT_ROOT", ROOT / "outputs")) / "environment"


def _json_default(obj):
    try:
        import torch

        if isinstance(obj, torch.dtype):
            return str(obj)
    except Exception:
        pass
    return str(obj)


def env_block() -> dict:
    import torch

    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update({"gpu": p.name, "vram_gb": round(p.total_memory / 1e9, 1),
                     "capability": f"{p.major}.{p.minor}", "device_count": torch.cuda.device_count()})
    try:
        import pyarrow
        info["pyarrow"] = pyarrow.__version__
    except Exception:
        info["pyarrow"] = None
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except Exception:
        info["transformers"] = None
    return info


def load_examples(n_rows: int = 400, max_examples: int = 48):
    """Real MetaMathQA rows through the real eligibility/tokenisation path."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from task1.data import select_examples

    ds = load_dataset("meta-math/MetaMathQA", split=f"train[:{n_rows}]")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    gsm = set()
    kept, report = select_examples(ds, tok, gsm, n_examples=max_examples)
    return tok, kept, report.as_dict()


def dtype_report(model) -> dict:
    import torch

    out = {"param_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
           "trainable_dtypes": sorted({str(p.dtype) for p in model.parameters() if p.requires_grad}),
           "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
           "total_params": int(sum(p.numel() for p in model.parameters()))}
    return out


def run_variant(variant: str, examples, tokenizer) -> dict:
    import torch

    from task1.model import load_model, prepare_adapter, prepare_full_ft
    from task1.train import Trainer, TrainConfig

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    result = {"variant": variant}
    t0 = time.time()

    try:
        if variant == "full_ft":
            model = load_model("Qwen/Qwen3-0.6B", dtype=torch.float32, device="cuda",
                               gradient_checkpointing=True)
            model = prepare_full_ft(model)
            cfg = TrainConfig(variant=variant, peak_lr=2e-5, lambda_l2sp=0.0,
                              updates=1, checkpoint_updates=())
            is_adapter = False
        else:
            model = load_model("Qwen/Qwen3-0.6B", dtype=torch.float16, device="cuda",
                               gradient_checkpointing=True)
            model, _bases = prepare_adapter(model, variant, r=16, alpha=32, dropout=0.05,
                                            seed=6304)
            cfg = TrainConfig(variant=variant, peak_lr=2e-4, lambda_l2sp=0.0,
                              updates=1, checkpoint_updates=())
            is_adapter = True

        result["load_seconds"] = round(time.time() - t0, 1)
        result["param_mem_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
        result["dtypes_before"] = dtype_report(model)

        trainer = Trainer(model, examples, tokenizer, cfg, device="cuda", out_dir=None,
                          reference=None, is_adapter=is_adapter)

        t1 = time.time()
        stats = trainer.update_once(list(range(32)))
        torch.cuda.synchronize()
        dt = time.time() - t1

        result.update({
            "ok": True,
            "update_seconds": round(dt, 1),
            "tokens": int(sum(ex.n_tokens for ex in examples[:32])),
            "tokens_per_s": round(sum(ex.n_tokens for ex in examples[:32]) / dt, 0),
            "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
            "sft_loss": stats["sft_loss"],
            "grad_norm": stats["grad_norm"],
            "skipped": stats["skipped"],
        })

        # dtype checks after the step: FP32 masters/adapters, FP32 gradients, FP32 states
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        result["dtypes_after"] = {
            "grad_dtypes": sorted({str(g.dtype) for g in grads}),
            "optimizer_state_dtypes": sorted({str(v.dtype)
                                              for st in trainer.optimizer.state.values()
                                              for v in st.values() if torch.is_tensor(v)}),
        }
        result["full_ft_estimate_hours"] = round(1250 * dt / 3600, 2)
        del trainer, model
    except torch.cuda.OutOfMemoryError as exc:
        result.update({"ok": False, "oom": True, "error": f"CUDA OOM: {exc}",
                       "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)})
    except RuntimeError as exc:
        oom = "out of memory" in str(exc).lower()
        result.update({"ok": False, "oom": oom, "error": str(exc),
                       "traceback": traceback.format_exc()[-2000:],
                       "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)})
    finally:
        torch.cuda.empty_cache()
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "environment": env_block()}
    print(json.dumps(report["environment"], indent=2), flush=True)

    print("\n=== Task 1 data path (real MetaMathQA rows) ===", flush=True)
    tokenizer, examples, selection = load_examples()
    lengths = [ex.n_tokens for ex in examples]
    report["data"] = {
        "n_examples": len(examples),
        "selection": selection,
        "token_lengths": {"min": min(lengths), "max": max(lengths),
                          "mean": round(sum(lengths) / len(lengths), 1)},
    }
    print(json.dumps(report["data"], indent=2)[:1200], flush=True)

    for variant in ("full_ft", "lora16"):
        print(f"\n=== {variant}: one update, microbatch 2 x accum 16, seq<=2048 ===", flush=True)
        report[variant] = run_variant(variant, examples, tokenizer)
        print(json.dumps(report[variant], indent=2), flush=True)

    fits = all(report[v].get("ok") for v in ("full_ft", "lora16"))
    report["verdict"] = {
        "both_variants_fit": fits,
        "full_ft_peak_gb": report["full_ft"].get("peak_mem_gb"),
        "lora_peak_gb": report["lora16"].get("peak_mem_gb"),
    }
    path = OUT / "kaggle_gpu_test.json"
    path.write_text(json.dumps(report, indent=2, default=_json_default))
    print(f"\nwrote {path}")
    print("VERDICT:", json.dumps(report["verdict"]))


if __name__ == "__main__":
    main()
