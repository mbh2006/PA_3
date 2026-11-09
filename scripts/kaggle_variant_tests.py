"""Per-variant GPU feasibility tests at representative and near-limit sequence lengths.

Run **before** any full training run. Each test performs one real optimizer update (or
forward/backward) and records peak memory, wall-clock time, dtypes and OOM status.

Covered: Full FT (representative + 2,048), Full FT + L2-SP with the FP32 reference on CPU
and on GPU, LoRA-16, OPLoRA-16, Task 2 vision-language step (realistic + 2,048), and a
checkpoint save/load round trip.

Writes ``outputs/environment/variant_tests.json``.
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

OUT = Path(os.environ.get("ATML_OUTPUT_ROOT", ROOT / "outputs")) / "environment"
MODEL_ID = "Qwen/Qwen3-0.6B"


def _cls():
    import torch

    return torch


def load_prepared_examples(max_examples=64):
    from task1.prepare import load_prepared

    candidates = list(Path("/kaggle/input").glob("**/task1_data/prepared.npz"))
    if not candidates:
        raise SystemExit("prepared.npz not mounted; run the prep kernel first")
    examples, meta = load_prepared(candidates[0])
    return examples[:max_examples], meta


def synthetic_examples(n: int, seq_len: int, vocab: int, n_prompt: int = 8):
    import numpy as np

    from task1.data import MathExample

    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        ids = rng.integers(100, vocab, size=seq_len).astype("int32")
        labels = ids.copy()
        labels[:n_prompt] = -100
        out.append(MathExample(i, "", "", "", ids, labels, n_prompt))
    return out


def record(name, fn):
    import torch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    entry = {"test": name}
    try:
        detail = fn()
        torch.cuda.synchronize()
        entry.update({"ok": True, **detail})
    except torch.cuda.OutOfMemoryError as exc:
        entry.update({"ok": False, "oom": True, "error": str(exc)[:400]})
    except RuntimeError as exc:
        entry.update({"ok": False, "oom": "out of memory" in str(exc).lower(),
                      "error": str(exc)[:400], "traceback": traceback.format_exc()[-1200:]})
    finally:
        entry["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        entry["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 2)
        entry["seconds"] = round(time.time() - t0, 1)
        torch.cuda.empty_cache()
    print(json.dumps(entry, indent=2), flush=True)
    return entry


def _train_one(model, examples, tokenizer, *, is_adapter, reference=None, lam=0.0, lr=0.0):
    from task1.train import Trainer, TrainConfig

    cfg = TrainConfig(variant="test", peak_lr=lr, lambda_l2sp=lam, updates=1,
                      checkpoint_updates=(), clip_grad_norm=1.0)
    trainer = Trainer(model, examples, tokenizer, cfg, device="cuda", out_dir=None,
                      reference=reference, is_adapter=is_adapter)
    stats = trainer.update_once(list(range(32)))
    return trainer, stats


def main() -> None:
    import torch
    from transformers import AutoTokenizer

    OUT.mkdir(parents=True, exist_ok=True)
    report = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    examples, meta = load_prepared_examples(64)
    lengths = [e.n_tokens for e in examples[:32]]
    report["data"] = {"source": "prepared.npz", "n": len(examples),
                      "mean_len_32": round(sum(lengths) / len(lengths), 1),
                      "max_len": max(lengths)}

    from task1.model import load_model, prepare_adapter, prepare_full_ft

    # ---- Task 2 style step first (cheapest) -------------------------------------
    def task2_step(seq_target: int):
        from task2.connector import Connector, N_VISUAL_TOKENS
        from task2.inputs import build_inputs
        from task2.loss_sampler import example_weighted_loss

        model = load_model(MODEL_ID, dtype=torch.float16, device="cuda")
        model, _ = prepare_adapter(model, "lora16", r=16, alpha=32, dropout=0.05, seed=6304)
        conn = Connector().to("cuda").float()
        feats = torch.randn(N_VISUAL_TOKENS, 768, device="cuda")
        vis = conn(feats)
        question = "Describe the image. " * max(1, (seq_target - N_VISUAL_TOKENS) // 6)
        inputs = build_inputs(model.get_input_embeddings(), tok, question=question,
                              answer="a short caption", visual_embeds=vis,
                              im_start_id=tok.convert_tokens_to_ids("<|im_start|>"),
                              im_end_id=tok.convert_tokens_to_ids("<|im_end|>"))
        out = model(inputs_embeds=inputs.inputs_embeds,
                    attention_mask=inputs.attention_mask, labels=inputs.labels)
        loss = example_weighted_loss(out.logits, inputs.labels)
        loss.backward()
        conn_grad = float(sum(p.grad.abs().sum() for p in conn.parameters() if p.grad is not None))
        adapter_grad = float(sum(p.grad.abs().sum() for p in model.parameters()
                                 if p.requires_grad and p.grad is not None))
        del model, conn
        return {"seq_len": inputs.seq_len, "loss": float(loss.detach()),
                "connector_grad_abs_sum": conn_grad, "adapter_grad_abs_sum": adapter_grad,
                "connector_receives_grad": conn_grad > 0}

    report["task2_realistic"] = record("task2_realistic", lambda: task2_step(400))
    report["task2_seq2048"] = record("task2_seq2048", lambda: task2_step(2048))

    # ---- adapters ---------------------------------------------------------------
    def adapter_step(variant):
        model = load_model(MODEL_ID, dtype=torch.float16, device="cuda")
        model, bases = prepare_adapter(model, variant, r=16, alpha=32, dropout=0.05,
                                       seed=6304)
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        ex = synthetic_examples(32, 2048, model.config.vocab_size)
        _trainer, stats = _train_one(model, ex, tok, is_adapter=True, lr=2e-4)
        del model
        return {"trainable": n_train, "loss": stats["sft_loss"],
                "grad_norm": stats["grad_norm"],
                "lora_bases_mb": None if bases is None else round(
                    sum(t.numel() * t.element_size() for u, v in bases.values() for t in (u, v)) / 1e6, 1)}

    report["lora16_seq2048"] = record("lora16_seq2048", lambda: adapter_step("lora16"))
    report["oplora16_seq2048"] = record("oplora16_seq2048", lambda: adapter_step("oplora16"))

    # ---- full fine-tuning -------------------------------------------------------
    def full_ft(exs, *, lam=0.0, ref_device="cpu"):
        from task1.l2sp import L2SPReference, make_reference

        model = load_model(MODEL_ID, dtype=torch.float32, device="cuda")
        model = prepare_full_ft(model)
        ref = None
        if lam > 0:
            ref = make_reference(model, device=ref_device)
            ref.tensors = {k: v.to(ref_device) for k, v in ref.tensors.items()}
        _trainer, stats = _train_one(model, exs, tok, is_adapter=False,
                                     reference=ref, lam=lam, lr=2e-5)
        out = {"loss": stats["sft_loss"], "grad_norm": stats["grad_norm"],
               "lambda_omega": stats["lambda_omega"]}
        if lam > 0:
            out["reference_device"] = ref_device
            out["reference_mb"] = round(ref.storage_bytes / 1e6, 1)
        del model
        return out

    report["full_ft_representative"] = record(
        "full_ft_representative", lambda: full_ft(examples[:32]))
    report["full_ft_seq2048"] = record(
        "full_ft_seq2048", lambda: full_ft(synthetic_examples(32, 2048, 151936)))
    report["full_ft_l2sp_cpu_ref"] = record(
        "full_ft_l2sp_cpu_ref", lambda: full_ft(synthetic_examples(32, 2048, 151936),
                                                lam=1e-3, ref_device="cpu"))
    report["full_ft_l2sp_gpu_ref"] = record(
        "full_ft_l2sp_gpu_ref", lambda: full_ft(synthetic_examples(32, 2048, 151936),
                                                lam=1e-3, ref_device="cuda"))

    # ---- checkpoint round trip (adapter: small, real) ---------------------------
    def checkpoint_roundtrip():
        from task1.train import Trainer, TrainConfig

        model = load_model(MODEL_ID, dtype=torch.float16, device="cuda")
        model, _ = prepare_adapter(model, "lora16", r=16, alpha=32, dropout=0.05, seed=6304)
        ex = synthetic_examples(32, 256, 151936)
        cfg = TrainConfig(variant="ckpt", peak_lr=2e-4, updates=1, checkpoint_updates=())
        trainer = Trainer(model, ex, tok, cfg, device="cuda",
                          out_dir=Path("/kaggle/working/ckpt_test"), is_adapter=True)
        trainer.update_once(list(range(32)))
        path = trainer.save_checkpoint(1, examples_seen=32)
        from task1.lora import adapter_state

        before = {k: v.clone() for k, v in adapter_state(model).items()}
        for p in model.parameters():
            p.data.add_(0.01)          # perturb everything
        del trainer
        torch.cuda.empty_cache()

        model2 = load_model(MODEL_ID, dtype=torch.float16, device="cuda")
        model2, _ = prepare_adapter(model2, "lora16", r=16, alpha=32, dropout=0.05, seed=1)
        t2 = Trainer(model2, ex, tok, cfg, device="cuda",
                     out_dir=Path("/kaggle/working/ckpt_test2"), is_adapter=True)
        payload = t2.load_checkpoint(path)
        after = adapter_state(model2)
        ok = all(torch.equal(before[k], after[k]) for k in before)
        size_mb = round(path.stat().st_size / 1e6, 1)
        del model, model2, t2
        return {"adapters_restored": bool(ok), "checkpoint_mb": size_mb,
                "update": payload["update"], "examples_seen": payload["examples_seen"],
                "optimizer_state_present": len(payload["optimizer"]["state"]) > 0}

    report["checkpoint_roundtrip"] = record("checkpoint_roundtrip", checkpoint_roundtrip)

    # ---- full-FT checkpoint size probe (one file, then delete) ------------------
    def full_ft_checkpoint_size():
        from task1.train import Trainer, TrainConfig

        model = load_model(MODEL_ID, dtype=torch.float32, device="cuda")
        model = prepare_full_ft(model)
        ex = synthetic_examples(32, 128, 151936)
        cfg = TrainConfig(variant="ckpt", peak_lr=2e-5, updates=1, checkpoint_updates=())
        trainer = Trainer(model, ex, tok, cfg, device="cuda",
                          out_dir=Path("/kaggle/working/ckpt_full"))
        trainer.update_once(list(range(32)))
        path = trainer.save_checkpoint(1, examples_seen=32)
        size_gb = round(path.stat().st_size / 1e9, 2)
        import shutil

        shutil.rmtree("/kaggle/working/ckpt_full", ignore_errors=True)
        del model, trainer
        return {"full_ft_checkpoint_gb": size_gb,
                "four_checkpoints_gb": round(size_gb * 4, 2),
                "working_disk_gb": 20.0}

    report["full_ft_checkpoint_size"] = record("full_ft_checkpoint_size", full_ft_checkpoint_size)

    report["verdict"] = {
        "all_required_pass": all(report[k]["ok"] for k in
                                 ("full_ft_seq2048", "oplora16_seq2048",
                                  "task2_seq2048", "full_ft_l2sp_cpu_ref")),
        "l2sp_gpu_ref_ok": report["full_ft_l2sp_gpu_ref"]["ok"],
        "peak_by_test": {k: v["peak_mem_gb"] for k, v in report.items()
                         if isinstance(v, dict) and "peak_mem_gb" in v},
    }
    path = OUT / "variant_tests.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    print("\nVERDICT:", json.dumps(report["verdict"], indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
