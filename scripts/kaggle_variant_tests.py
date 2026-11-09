"""Per-variant GPU feasibility tests at representative and near-limit sequence lengths.

Run **before** any full training run. Each test performs one real optimizer update and
records peak memory, wall-clock time, dtypes and OOM status, writing the JSON after every
test so a later failure cannot destroy earlier evidence.

Covered: sequence-length sweeps for Full FT, LoRA-16 and OPLoRA-16; Full FT + L2-SP with
the FP32 reference on CPU and (for evidence) on GPU; the over-budget micro-batch split at
2,048 tokens; the Task 2 vision-language step; and checkpoint save/load.
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
REPORT_PATH = OUT / "variant_tests.json"
REPORT: dict = {}


def flush_report():
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(REPORT, indent=2, default=str))


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
    except BaseException as exc:  # noqa: BLE001 - never let one test abort the suite
        entry.update({"ok": False, "oom": "out of memory" in str(exc).lower(),
                      "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                      "traceback": traceback.format_exc()[-800:]})
    finally:
        entry["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        entry["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 2)
        entry["seconds"] = round(time.time() - t0, 1)
        torch.cuda.empty_cache()
    print(json.dumps(entry)[:600], flush=True)
    REPORT[name] = entry
    flush_report()
    return entry


def load_prepared_examples(max_examples=64):
    from task1.prepare import load_prepared

    candidates = list(Path("/kaggle/input").glob("**/task1_data/prepared.npz"))
    if not candidates:
        raise SystemExit("prepared.npz not mounted")
    examples, meta = load_prepared(candidates[0])
    return examples[:max_examples], meta


def synthetic_examples(n, seq_len, n_prompt=8):
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

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    REPORT["started"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    REPORT["gpu"] = torch.cuda.get_device_name(0)
    flush_report()

    from task1.model import load_model, prepare_adapter, prepare_full_ft
    from task1.train import Trainer, TrainConfig

    examples, meta = load_prepared_examples(64)
    lens = [e.n_tokens for e in examples[:32]]
    REPORT["data"] = {"n": len(examples), "mean_len_32": round(sum(lens) / len(lens), 1),
                      "max_len": max(lens)}
    flush_report()

    def train_step(model, exs, *, is_adapter, lr, reference=None, lam=0.0,
                   max_micro_tokens=3000):
        cfg = TrainConfig(variant="t", peak_lr=lr, lambda_l2sp=lam, updates=1,
                          checkpoint_updates=(), max_micro_tokens=max_micro_tokens)
        tr = Trainer(model, exs, tok, cfg, device="cuda", out_dir=None,
                     reference=reference, is_adapter=is_adapter)
        st = tr.update_once(list(range(len(exs))))
        return {"loss": round(st["sft_loss"], 4), "grad_norm": round(st["grad_norm"], 2),
                "split_micro_batches": sum(1 for _ in tr._micro_batches(list(range(len(exs)))))}

    def adapter(seq, variant):
        model = load_model(MODEL_ID, dtype=torch.float16, device="cuda")
        model, bases = prepare_adapter(
            model, variant, r=16, alpha=32, dropout=0.05, seed=6304,
            oplora_k=128 if variant == "oplora16" else None)
        exs = synthetic_examples(32, seq)
        out = train_step(model, exs, is_adapter=True, lr=2e-4)
        out["trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if bases is not None:
            out["bases_mb"] = round(sum(t.numel() * t.element_size()
                                        for u, v in bases.values() for t in (u, v)) / 1e6, 1)
        del model
        return out

    def full(seq, *, lam=0.0, ref_device=None, max_micro_tokens=3000):
        from task1.l2sp import make_reference

        model = load_model(MODEL_ID, dtype=torch.float32, device="cuda")
        model = prepare_full_ft(model)
        ref = None
        if lam > 0:
            ref = make_reference(model, device=ref_device or "cpu")
        exs = synthetic_examples(32, seq)
        out = train_step(model, exs, is_adapter=False, lr=2e-5, reference=ref, lam=lam,
                         max_micro_tokens=max_micro_tokens)
        if ref is not None:
            out["ref_device"] = ref_device or "cpu"
            out["ref_mb"] = round(ref.storage_bytes / 1e6, 1)
            out["lambda_omega"] = round(float(out.get("lambda_omega", 0.0)), 6)
        del model, ref
        return out

    # ---- Task 2 (cheapest) ------------------------------------------------------
    def task2(seq_target):
        from task2.connector import Connector, N_VISUAL_TOKENS
        from task2.inputs import build_inputs
        from task2.loss_sampler import example_weighted_loss

        model = load_model(MODEL_ID, dtype=torch.float16, device="cuda")
        model, _ = prepare_adapter(model, "lora16", r=16, alpha=32, dropout=0.05, seed=6304)
        conn = Connector().to("cuda").float()
        vis = conn(torch.randn(N_VISUAL_TOKENS, 768, device="cuda"))
        question = "Describe the image. " * max(1, (seq_target - N_VISUAL_TOKENS) // 6)
        inp = build_inputs(model.get_input_embeddings(), tok, question=question,
                           answer="a short caption", visual_embeds=vis,
                           im_start_id=tok.convert_tokens_to_ids("<|im_start|>"),
                           im_end_id=tok.convert_tokens_to_ids("<|im_end|>"))
        out = model(inputs_embeds=inp.inputs_embeds, attention_mask=inp.attention_mask,
                    labels=inp.labels)
        loss = example_weighted_loss(out.logits, inp.labels)
        loss.backward()
        conn_grad = float(sum(p.grad.abs().sum() for p in conn.parameters() if p.grad is not None))
        res = {"seq_len": inp.seq_len, "loss": round(float(loss.detach()), 4),
               "connector_grad_abs_sum": round(conn_grad, 6),
               "connector_receives_grad": conn_grad > 0}
        del model, conn
        return res

    record("task2_realistic", lambda: task2(400))
    record("task2_seq2048", lambda: task2(2048))

    # ---- chunked-head verification at the cap ----------------------------------
    record("full_ft_pair640_chunked", lambda: full(640))
    record("full_ft_pair1400_nosplit", lambda: full(1400, max_micro_tokens=10_000_000))
    record("full_ft_single1942_chunked", lambda: full(1942))
    record("full_ft_pair1942_nosplit", lambda: full(1942, max_micro_tokens=10_000_000))
    record("lora16_single1942_chunked", lambda: adapter(1942, "lora16"))
    record("oplora16_pair1942_nosplit",
           lambda: adapter(1942, "oplora16"))
    record("full_ft_l2sp_single1942_chunked",
           lambda: full(1942, lam=1e-3, ref_device="cpu"))
    record("full_ft_l2sp_pair640_chunked", lambda: full(640, lam=1e-3, ref_device="cpu"))

    # ---- checkpoint round trip (adapter: small and fast) -----------------------
    def checkpoint_roundtrip():
        from task1.lora import adapter_state

        model = load_model(MODEL_ID, dtype=torch.float16, device="cuda")
        model, _ = prepare_adapter(model, "lora16", r=16, alpha=32, dropout=0.05, seed=6304)
        exs = synthetic_examples(32, 256)
        cfg = TrainConfig(variant="ckpt", peak_lr=2e-4, updates=1, checkpoint_updates=())
        tr = Trainer(model, exs, tok, cfg, device="cuda", is_adapter=True,
                     out_dir=Path("/kaggle/working/ckpt_test"))
        tr.update_once(list(range(32)))
        path = tr.save_checkpoint(1, examples_seen=32)
        before = {k: v.clone() for k, v in adapter_state(model).items()}
        for p in model.parameters():
            p.data.add_(0.01)
        del tr
        torch.cuda.empty_cache()

        model2 = load_model(MODEL_ID, dtype=torch.float16, device="cuda")
        model2, _ = prepare_adapter(model2, "lora16", r=16, alpha=32, dropout=0.05, seed=1)
        t2 = Trainer(model2, exs, tok, cfg, device="cuda", is_adapter=True,
                     out_dir=Path("/kaggle/working/ckpt_test2"))
        payload = t2.load_checkpoint(path)
        after = adapter_state(model2)
        ok = all(torch.equal(before[k], after[k]) for k in before)
        res = {"adapters_restored": bool(ok),
               "checkpoint_mb": round(path.stat().st_size / 1e6, 1),
               "update": payload["update"], "examples_seen": payload["examples_seen"]}
        del model, model2, t2
        return res

    record("checkpoint_roundtrip", checkpoint_roundtrip)

    # ---- verdict -----------------------------------------------------------------
    def ok(name):
        return REPORT.get(name, {}).get("ok", False)

    REPORT["verdict"] = {
        "full_ft_representative_ok": ok("full_ft_seq256"),
        "full_ft_at_cap_without_split": ok("full_ft_seq2048"),
        "full_ft_at_cap_with_split": ok("full_ft_seq2048_split"),
        "lora_at_cap_with_split": ok("lora16_seq2048_split"),
        "oplora_at_cap": ok("oplora16_seq2048"),
        "l2sp_cpu_ref_at_cap": ok("full_ft_l2sp_cpu_ref_seq2048"),
        "l2sp_gpu_ref_at_cap": ok("full_ft_l2sp_gpu_ref_seq2048"),
        "task2_at_cap": ok("task2_seq2048"),
        "checkpoint_roundtrip": ok("checkpoint_roundtrip"),
    }
    flush_report()
    print("\nVERDICT:", json.dumps(REPORT["verdict"], indent=2))
    print("wrote", REPORT_PATH)


if __name__ == "__main__":
    main()
