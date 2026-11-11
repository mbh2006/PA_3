"""Run one Task 1 variant end to end inside a single GPU session.

Order: load data -> build model -> (snapshot theta0 / OPLoRA bases) -> train with
recovery checkpoints -> merge and evaluate four benchmarks -> weight and activation
diagnostics -> 24 qualitative prompts -> compact JSON report.

    python scripts/task1_variant_step.py --variant full_ft --out outputs/task_01/runs/full_ft
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MODEL_ID = "Qwen/Qwen3-0.6B"


LOCAL_ROOT = Path("/root/pa3")


def find_prepared() -> Path:
    local = LOCAL_ROOT / "prepared.npz"
    if local.exists():
        return local
    for base in ("/kaggle/input/datasets", "/kaggle/input"):
        for p in Path(base).glob("**/task1_data/prepared.npz"):
            return p
    raise SystemExit("prepared.npz not found")


def find_prompts() -> Path:
    local = LOCAL_ROOT / "data" / "reference" / "first_plot_questions.yaml"
    if local.exists():
        return local
    for base in ("/kaggle/input/datasets", "/kaggle/input"):
        for p in Path(base).glob("**/first_plot_questions.yaml"):
            return p
    raise SystemExit("first_plot_questions.yaml not found")


def find_probes() -> Path:
    local = LOCAL_ROOT / "probes.json"
    if local.exists():
        return local
    for base in ("/kaggle/input/datasets", "/kaggle/input"):
        for p in Path(base).glob("**/task1_data/probes.json"):
            return p
    raise SystemExit("probes.json not found")


def main() -> None:
    import torch
    from transformers import AutoTokenizer

    from task1 import eval_lm, probes as probe_mod, qualitative
    from task1.l2sp import make_reference
    from task1.model import load_model, prepare_adapter, prepare_full_ft, trainable_parameter_count
    from task1.prepare import load_prepared
    from task1.run import VARIANTS, RunSpec, run_variant

    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=sorted(VARIANTS) + ["baseline"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-updates", type=int, default=None)
    ap.add_argument("--eval-limit", type=int, default=None)
    ap.add_argument("--skip-qualitative", action="store_true")
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    report: dict = {"variant": args.variant, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    data_path = find_prepared()
    prompts_path = find_prompts()
    probes_path = find_probes()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------------- training ---------------------------------------------------
    is_baseline = args.variant == "baseline"
    cfg = VARIANTS.get(args.variant, dict(kind="full", peak_lr=0.0))
    is_adapter = cfg["kind"] == "adapter" and not is_baseline

    if not is_baseline:
        final_artifact = (out / "train" / ("adapter_final.pt" if is_adapter else "model_final.pt"))
        if final_artifact.exists():
            print(f"[resume] {final_artifact} exists; skipping training", flush=True)
            summary = out / "train" / "run_summary.json"
            train_info = json.loads(summary.read_text()) if summary.exists() else {"resumed": True}
        else:
            spec = RunSpec(variant=args.variant, peak_lr=cfg["peak_lr"],
                           lambda_l2sp=cfg.get("lam", 0.0))
            print(f"[{args.variant}] training", flush=True)
            train_info = run_variant(spec, data_path, out / "train", max_updates=args.max_updates)
        report["train"] = train_info
        _write(out / "report_partial.json", report)

    # ---------------- rebuild the trained model for evaluation -------------------
    if is_baseline:
        model = load_model(MODEL_ID, dtype=torch.float32, device="cuda")
        model = prepare_full_ft(model)
        bases = None
    elif is_adapter:
        model = load_model(MODEL_ID, dtype=torch.float16, device="cuda")
        bases = None
        if args.variant == "oplora16":
            bases_path = out / "train" / "oploRA_bases.pt"
            bases = torch.load(bases_path, map_location="cpu") if bases_path.exists() else None
        model, bases = prepare_adapter(model, args.variant, r=cfg["r"], alpha=cfg["alpha"],
                                       dropout=0.05, seed=6304, oplora_k=cfg.get("k"),
                                       bases=bases)
        from task1.lora import load_adapter_state

        load_adapter_state(model, torch.load(out / "train" / "adapter_final.pt", map_location="cpu"))
    else:
        model = load_model(MODEL_ID, dtype=torch.float32, device="cuda")
        model = prepare_full_ft(model)
        model.load_state_dict(torch.load(out / "train" / "model_final.pt", map_location="cuda"))
        bases = None
    model.eval()
    report["trainable_parameters"] = trainable_parameter_count(model)

    # reference theta0 (CPU) + shared OPLoRA bases for diagnostics
    model0 = load_model(MODEL_ID, dtype=torch.float32, device="cuda")
    reference = {n: p.detach().float().cpu().clone() for n, p in model0.named_parameters()}
    if bases is None and "oplora" in args.variant:
        from task1.lora import build_oploRA_bases

        bases = build_oploRA_bases(model0, cfg.get("k", 128))
    del model0
    torch.cuda.empty_cache()

    # ---------------- probes (before looking at model outputs) ------------------
    probe_set = json.loads(probes_path.read_text())
    probe_prompts = probe_mod.build_probe_prompts(probe_set, tokenizer)
    probe_inputs = probe_mod.collect_probe_inputs(model, tokenizer, probe_prompts)
    torch.save({k: v.cpu() for k, v in probe_inputs.items()}, out / "probe_inputs.pt")
    if is_baseline:
        report["weight_diagnostics"] = []
        report["activation_diagnostics"] = {}
        report["displacement_l2"] = 0.0
    else:
        report["weight_diagnostics"] = probe_mod.weight_diagnostics(model, reference, bases)
        report["activation_diagnostics"] = probe_mod.activation_diagnostics(
            model, reference, probe_inputs, [p["kind"] for p in probe_prompts])
        report["displacement_l2"] = probe_mod.full_displacement(model, reference)
        if cfg.get("lam", 0.0) > 0.0:
            # final Omega = 0.5 * sum (theta - theta0)^2 over unique parameters
            omega = 0.0
            for name, prm in model.named_parameters():
                d = prm.detach().float().cpu() - reference[name].float().cpu()
                omega += float(d.pow(2).sum())
            report["omega_final"] = 0.5 * omega
            report["lambda_omega_final"] = 0.5 * omega * cfg["lam"]
    _write(out / "report_partial.json", report)

    # ---------------- benchmark evaluation --------------------------------------
    print(f"[{args.variant}] evaluation", flush=True)
    eval_dir = out / "merged"
    if eval_dir.exists():
        shutil.rmtree(eval_dir)
    with torch.no_grad():
        ids = tokenizer("What is 2 + 2?", return_tensors="pt").to("cuda")
        pre_logits = model(input_ids=ids["input_ids"]).logits.float().cpu()
    n_merged = eval_lm.prepare_eval_dir(model, tokenizer, eval_dir, is_adapter=is_adapter)
    if is_adapter:
        import json as _json

        index = sorted(p.name for p in eval_dir.glob("*.safetensors"))
        shard = eval_dir / index[0] if index else None
        if shard is not None:
            from safetensors.torch import load_file as _load

            keys = set(_load(str(shard)).keys())
            probe = "model.layers.0.self_attn.q_proj.weight"
            if probe not in keys:
                raise RuntimeError(
                    f"merged checkpoint lacks {probe}; adapter layers were not unwrapped "
                    f"({sorted(k for k in keys if 'layers.0.self_attn' in k)[:4]})")
        report["merged_modules"] = n_merged
    del model
    torch.cuda.empty_cache()

    bench = eval_lm.run_benchmarks(eval_dir, out / "eval", limit=args.eval_limit)
    shutil.rmtree(eval_dir, ignore_errors=True)   # 2.4 GB per condition; not needed again
    report["benchmarks"] = {"summary": bench["summary"],
                            "primary": eval_lm.primary_scores(bench["summary"])}
    _write(out / "report_partial.json", report)

    # ---------------- qualitative -------------------------------------------------
    if not args.skip_qualitative:
        print(f"[{args.variant}] qualitative", flush=True)
        qmodel = load_model(MODEL_ID, dtype=torch.float16, device="cuda")
        if is_adapter:
            from task1.lora import load_adapter_state

            qbases = None
            if args.variant == "oplora16":
                bp = out / "train" / "oploRA_bases.pt"
                qbases = torch.load(bp, map_location="cpu") if bp.exists() else None
            qmodel, _ = prepare_adapter(qmodel, args.variant, r=cfg["r"], alpha=cfg["alpha"],
                                        dropout=0.05, seed=6304, oplora_k=cfg.get("k"),
                                        bases=qbases)
            load_adapter_state(qmodel, torch.load(out / "train" / "adapter_final.pt",
                                                  map_location="cpu"))
        else:
            qmodel = qmodel.float()
            if not is_baseline:
                qmodel.load_state_dict(torch.load(out / "train" / "model_final.pt",
                                                  map_location="cuda"))
        prompts = qualitative.load_prompts(prompts_path)
        records = qualitative.generate_all(qmodel, tokenizer, prompts)
        qualitative.save_condition(records, args.variant, out / "qualitative")
        report["qualitative"] = {"n": len(records),
                                 "truncated": sum(r["truncated"] for r in records)}
        del qmodel
        torch.cuda.empty_cache()

    report["seconds"] = round(time.time() - t_start, 1)
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write(out / "report.json", report)
    print("DONE", json.dumps(report.get("benchmarks", {}).get("primary", {})))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
