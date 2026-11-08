"""GPU / environment preflight for the paid compute box.

Every check prints a PASS/FAIL line and is recorded in a JSON report. The script
exits non-zero on any failed *required* check, so it can be used as a hard gate
before production runs.

    # CPU-only dry run (no torch / no model download) -- validate this locally first
    python scripts/gpu_preflight.py --dry-run

    # on the rented GPU box, first paid action
    python scripts/gpu_preflight.py --out outputs/environment/preflight.json

The measured throughput replaces the planning estimates in docs/execution_plan.md.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RESULTS: list[dict] = []


def record(name: str, ok: bool | None, detail, required: bool = True) -> None:
    if ok is None:
        status = "SKIP"
    elif ok:
        status = "PASS"
    elif required:
        status = "FAIL"
    else:
        status = "WARN"
    RESULTS.append({"check": name, "status": status, "required": required, "detail": detail})
    print(f"[{status:4}] {name}: {detail}", flush=True)
    if status == "FAIL":
        raise SystemExit(f"preflight failed: {name}")


def check_platform() -> None:
    record("python_version", sys.version_info >= (3, 10), platform.python_version())
    record("platform", True, f"{platform.system()} {platform.release()} / {platform.machine()}")


def check_packages() -> None:
    import pyarrow

    record("pyarrow_24", pyarrow.__version__.startswith("24."), pyarrow.__version__)
    for mod in ("numpy", "pandas", "scipy", "yaml"):
        try:
            m = __import__(mod)
            record(f"pkg_{mod}", True, getattr(m, "__version__", "?"), required=False)
        except Exception as exc:  # pragma: no cover - environment dependent
            record(f"pkg_{mod}", False, str(exc), required=False)


def check_task3_data() -> None:
    from common.paths import TASK3_DATASETS
    import pyarrow.parquet as pq

    for name, spec in TASK3_DATASETS.items():
        folder = spec["dir"]
        files = [spec["generations"], spec["probability"]] + ([spec["attention"]] if spec["attention"] else [])
        for fname in files:
            path = folder / fname
            if not path.exists():
                record(f"data_{name}_{fname}", False, f"missing {path}")
                continue
            pf = pq.ParquetFile(path)
            tbl = pf.read_row_group(0)
            record(f"data_{name}_{fname}", tbl.num_rows == 32,
                   f"{tbl.num_rows} rows, {pf.metadata.num_rows} total")


def check_disk(min_free_gb: float = 60.0, required: bool = True) -> None:
    target = Path(os.environ.get("ATML_OUTPUT_ROOT", ROOT))
    usage = shutil.disk_usage(target)
    free_gb = usage.free / 1e9
    record("disk_free_gb", free_gb >= min_free_gb, f"{free_gb:.1f} GB free at {target}",
           required=required and free_gb < min_free_gb)


def check_network() -> None:
    import urllib.request

    url = "https://huggingface.co/api/models/Qwen/Qwen3-0.6B"
    t0 = time.time()
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            ok = resp.status == 200
        record("network_huggingface", ok, f"{time.time() - t0:.2f}s", required=False)
    except Exception as exc:
        record("network_huggingface", False, str(exc), required=False)


def check_gpu() -> dict:
    import torch

    info = {"torch": torch.__version__, "cuda": torch.version.cuda}
    record("cuda_available", torch.cuda.is_available(), torch.cuda.is_available())
    if not torch.cuda.is_available():
        return info
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    info.update({
        "device": props.name,
        "vram_gb": props.total_memory / 1e9,
        "capability": f"{props.major}.{props.minor}",
        "device_count": torch.cuda.device_count(),
    })
    record("vram_gb", props.total_memory / 1e9 >= 22.0, f"{props.total_memory / 1e9:.1f} GB ({props.name})")
    record("device_count", True, torch.cuda.device_count(), required=False)

    a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    b = a @ a
    record("fp16_matmul", bool(torch.isfinite(b).all()), "2048x2048 fp16 matmul finite")
    return info


def check_tokenizer(model_id: str) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    messages = [{"role": "user", "content": "What is 2+2?"}]
    off = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                  enable_thinking=False)
    on = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                 enable_thinking=True)
    record("thinking_disabled", "think" not in off.lower(),
           repr(off[-80:]))
    record("thinking_flag_effective", "think" in on.lower(),
           "enable_thinking=True renders a think block", required=False)
    ids = tok(off, return_tensors=None)["input_ids"]
    record("tokenizer_roundtrip", tok.decode(ids) is not None, f"{len(ids)} tokens")


def check_model_step(model_id: str, seq_len: int = 512, micro_batch: int = 2, steps: int = 5) -> dict:
    """One real FP16 AMP forward/backward with gradient checkpointing and a GradScaler."""
    import torch
    from transformers import AutoModelForCausalLM

    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, attn_implementation="sdpa"
    ).to(device)
    model.gradient_checkpointing_enable()
    model.train()

    input_ids = torch.randint(0, 1000, (micro_batch, seq_len), device=device)
    labels = input_ids.clone()
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    losses, grad_norms = [], []
    t0 = time.time()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(input_ids=input_ids, labels=labels)
        scaler.scale(out.loss).backward()
        # unscale before any inspection / clipping, exactly as the training loop will
        scaler.unscale_(optimizer)
        total = sum(float(p.grad.detach().pow(2).sum()) for p in model.parameters() if p.grad is not None)
        grad_norms.append(total ** 0.5)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(out.loss.detach()))
    torch.cuda.synchronize()
    dt = time.time() - t0

    finite = all(x == x and abs(x) != float("inf") for x in losses)
    grads_ok = all(g == g and g > 0 for g in grad_norms)
    tokens = steps * micro_batch * seq_len
    record("model_forward_backward", finite, f"losses {losses[0]:.3f}..{losses[-1]:.3f}")
    record("gradients_finite_nonzero", grads_ok, f"grad norms {grad_norms[0]:.3f}..{grad_norms[-1]:.3f}")
    record("measured_throughput", True,
           f"{tokens / dt:.0f} tokens/s (microbatch {micro_batch}, seq {seq_len}, "
           f"grad-checkpointing on, fp16 amp)")

    del model, optimizer
    torch.cuda.empty_cache()
    return {"tokens_per_s": tokens / dt, "seq_len": seq_len, "micro_batch": micro_batch,
            "steps": steps}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "environment" / "preflight.json")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--dry-run", action="store_true", help="skip torch / model / network checks")
    ap.add_argument("--min-free-gb", type=float, default=60.0)
    args = ap.parse_args()

    started = time.time()
    check_platform()
    check_packages()
    check_task3_data()
    check_disk(args.min_free_gb, required=not args.dry_run)

    report = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    if args.dry_run:
        record("gpu_checks", None, "skipped (--dry-run)", required=False)
    else:
        check_network()
        report["gpu"] = check_gpu()
        check_tokenizer(args.model)
        report["throughput"] = check_model_step(args.model)

    report["checks"] = RESULTS
    report["seconds"] = round(time.time() - started, 1)
    report["ok"] = all(r["status"] != "FAIL" for r in RESULTS)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}  ({report['seconds']}s)")
    print("RESULT:", "OK" if report["ok"] else "FAILED")


if __name__ == "__main__":
    main()
