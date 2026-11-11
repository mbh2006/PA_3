"""Preflight on the rented 4090 with the assignment's actual training settings.

Measures, with real MetaMathQA-derived batches and with 1,942-token worst-case batches:

* Full FT (FP32 masters, FP16 autocast, gradient checkpointing, microbatch 2 x accum 16)
  in both loss modes: standard vocabulary projection and the chunked projection
* Full FT + L2-SP with the FP32 reference on CPU
* LoRA-16 and OPLoRA-16
* a Task 2 vision-language step

Reports seconds per optimizer update and the projected finish time for 1,250 updates.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = Path("/root/pa3/outputs/environment")
MODEL_ID = "Qwen/Qwen3-0.6B"
REPORT: dict = {}


def record(name, fn):
    import torch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    entry = {"test": name}
    try:
        entry.update({"ok": True, **(fn() or {})})
    except torch.cuda.OutOfMemoryError as exc:
        entry.update({"ok": False, "oom": True, "error": str(exc)[:300]})
    except BaseException as exc:  # noqa: BLE001
        entry.update({"ok": False, "oom": "out of memory" in str(exc).lower(),
                      "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                      "traceback": traceback.format_exc()[-600:]})
    finally:
        entry["peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        entry["seconds"] = round(time.time() - t0, 2)
        torch.cuda.empty_cache()
    print(json.dumps(entry)[:400], flush=True)
    REPORT[name] = entry
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "vast_preflight.json").write_text(json.dumps(REPORT, indent=2, default=str))
    return entry


def synth(n, seq_len, n_prompt=8):
    import numpy as np

    from task1.data import MathExample

    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        ids = rng.integers(100, 151000, size=seq_len).astype("int32")
        labels = ids.copy()
        labels[:n_prompt] = -100
        out.append(MathExample(i, "", "", "", ids, labels, n_prompt))
    return out


def main() -> None:
    import torch
    from transformers import AutoTokenizer

    from task1.model import load_model, prepare_adapter, prepare_full_ft
    from task1.prepare import load_prepared
    from task1.train import Trainer, TrainConfig

    REPORT["started"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    p = torch.cuda.get_device_properties(0)
    REPORT["gpu"] = {"name": p.name, "vram_gb": round(p.total_memory / 1e9, 1),
                     "capability": f"{p.major}.{p.minor}", "torch": torch.__version__,
                     "cuda": torch.version.cuda}
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # tokenizer: thinking must be off by default
    # Qwen3 non-thinking mode renders an EMPTY think block: "<think>\n\n</think>\n\n".
    # The assertion is therefore "empty block present and nothing between the tags",
    # not "the word think is absent".
    off = tok.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False,
                                  add_generation_prompt=True, enable_thinking=False)
    on = tok.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False,
                                 add_generation_prompt=True, enable_thinking=True)
    REPORT["thinking_disabled"] = ("<think>\n\n</think>" in off) and ("<think>" not in on)
    REPORT["thinking_templates"] = {"off_tail": off[-40:], "on_tail": on[-40:]}

    examples, meta = load_prepared(Path("/root/pa3/prepared.npz"))
    lengths = [e.n_tokens for e in examples[:32]]
    REPORT["data"] = {"n": len(examples), "mean_len_32": round(sum(lengths) / len(lengths), 1),
                      "max_len_32": max(lengths), "total_tokens": meta["total_tokens"]}
    print(json.dumps(REPORT, indent=2)[:800], flush=True)

    def train_step(model, exs, *, is_adapter, lr, reference=None, lam=0.0, chunked=True):
        cfg = TrainConfig(variant="t", peak_lr=lr, lambda_l2sp=lam, updates=1,
                          checkpoint_updates=(), chunked_loss=chunked,
                          loss_chunk_size=256, max_micro_tokens=3000)
        tr = Trainer(model, exs, tok, cfg, device="cuda", out_dir=None,
                     reference=reference, is_adapter=is_adapter)
        tr.update_once(list(range(len(exs))))     # warm-up (cuBLAS init, allocator)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(3):
            st = tr.update_once(list(range(len(exs))))
        torch.cuda.synchronize()
        dt = (time.time() - t0) / 3
        n_tokens = sum(e.n_tokens for e in exs)
        return {"loss": round(st["sft_loss"], 4), "grad_norm": round(st["grad_norm"], 2),
                "update_seconds": round(dt, 2),
                "tokens": n_tokens,
                "tokens_per_s": round(n_tokens / dt),
                "est_finish_hours": round(1250 * dt / 3600, 2),
                "micro_batches": sum(1 for _ in tr._micro_batches(list(range(len(exs)))))}

    def full(exs, *, lam=0.0, ref_device=None, chunked=True):
        from task1.l2sp import make_reference

        model = load_model(MODEL_ID, dtype=torch.float32, device="cuda")
        model = prepare_full_ft(model)
        ref = make_reference(model, device=ref_device or "cpu") if lam > 0 else None
        out = train_step(model, exs, is_adapter=False, lr=2e-5, reference=ref, lam=lam,
                         chunked=chunked)
        if ref is not None:
            out["ref_device"] = ref_device or "cpu"
            out["ref_mb"] = round(ref.storage_bytes / 1e6, 1)
        del model, ref
        return out

    def adapter(exs, variant, chunked=True):
        model = load_model(MODEL_ID, dtype=torch.float16, device="cuda")
        model, bases = prepare_adapter(model, variant, r=16, alpha=32, dropout=0.05,
                                       seed=6304, oplora_k=128 if variant == "oplora16" else None)
        out = train_step(model, exs, is_adapter=True, lr=2e-4, chunked=chunked)
        out["trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if bases:
            out["bases_mb"] = round(sum(t.numel() * t.element_size()
                                        for u, v in bases.values() for t in (u, v)) / 1e6, 1)
        del model
        return out

    real = examples[:32]
    long_batch = synth(32, 1942)

    record("full_ft_real_standard", lambda: full(real, chunked=False))
    record("full_ft_real_chunked", lambda: full(real, chunked=True))
    record("full_ft_1942_standard", lambda: full(long_batch, chunked=False))
    record("full_ft_1942_chunked", lambda: full(long_batch, chunked=True))
    record("full_ft_l2sp_1942_standard",
           lambda: full(long_batch, lam=1e-3, ref_device="cpu", chunked=False))
    record("lora16_1942_standard", lambda: adapter(long_batch, "lora16", chunked=False))
    record("oplora16_1942_standard", lambda: adapter(long_batch, "oplora16", chunked=False))
    record("lora16_real_standard", lambda: adapter(real, "lora16", chunked=False))

    # Task 2 step at the longest sequence
    def task2(seq_target=1942):
        from task2.connector import Connector, N_VISUAL_TOKENS
        from task2.inputs import build_inputs
        from task2.loss_sampler import example_weighted_loss

        model = load_model(MODEL_ID, dtype=torch.float16, device="cuda")
        model, _ = prepare_adapter(model, "lora16", r=16, alpha=32, dropout=0.05, seed=6304,
                                   suffixes=("q_proj", "k_proj", "v_proj", "o_proj"))
        conn = Connector().to("cuda").float()
        vis = conn(torch.randn(N_VISUAL_TOKENS, 768, device="cuda"))
        question = "Describe the image. " * max(1, (seq_target - N_VISUAL_TOKENS) // 6)
        inp = build_inputs(model.get_input_embeddings(), tok, question=question,
                           answer="a caption", visual_embeds=vis,
                           im_start_id=tok.convert_tokens_to_ids("<|im_start|>"),
                           im_end_id=tok.convert_tokens_to_ids("<|im_end|>"))
        out = model(inputs_embeds=inp.inputs_embeds, attention_mask=inp.attention_mask,
                    labels=inp.labels)
        loss = example_weighted_loss(out.logits, inp.labels)
        loss.backward()
        g = float(sum(p.grad.abs().sum() for p in conn.parameters() if p.grad is not None))
        res = {"seq_len": inp.seq_len, "loss": round(float(loss.detach()), 4),
               "connector_receives_grad": g > 0}
        del model, conn
        return res

    record("task2_1942", task2)

    # choose the loss mode: fastest that fits
    def ok(n):
        return REPORT.get(n, {}).get("ok", False)

    mode = "standard" if ok("full_ft_1942_standard") else ("chunked" if ok("full_ft_1942_chunked") else None)
    real_key = f"full_ft_real_{mode}" if mode else None
    est = REPORT.get(real_key, {}).get("est_finish_hours")
    REPORT["verdict"] = {
        "loss_mode": mode,
        "full_ft_fits_1942": ok("full_ft_1942_standard") or ok("full_ft_1942_chunked"),
        "l2sp_cpu_ref_fits_1942": ok("full_ft_l2sp_1942_standard"),
        "lora_fits_1942": ok("lora16_1942_standard"),
        "oplora_fits_1942": ok("oplora16_1942_standard"),
        "task2_fits_1942": ok("task2_1942"),
        "full_ft_hours_per_run": est,
        "lora_hours_per_run": REPORT.get("lora16_real_standard", {}).get("est_finish_hours"),
        "peak_full_ft": REPORT.get("full_ft_1942_standard", {}).get("peak_gb"),
    }
    (OUT / "vast_preflight.json").write_text(json.dumps(REPORT, indent=2, default=str))
    print("\nVERDICT:", json.dumps(REPORT["verdict"], indent=2))


if __name__ == "__main__":
    main()
