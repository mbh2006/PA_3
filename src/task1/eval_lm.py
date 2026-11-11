"""Task 1 benchmark evaluation through lm-evaluation-harness (PA p. 3).

Two invocations per condition, because the assignment prescribes different prompt
handling for the two families:

* ``gsm8k_cot``  -- non-thinking chat template, 0-shot, greedy, 1,024 new tokens,
  stock stopping rules and answer filters.
* ``hellaswag``, ``boolq``, ``arc_easy`` -- stock completion prompts, **no** chat template.

Adapters are merged into a fresh backbone copy before evaluation; OPLoRA merges its
projected factors. Merged vs unmerged logits are compared on a small batch and the
maximum absolute difference is recorded.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

EVAL_TASKS_LIKELIHOOD = ["hellaswag", "boolq", "arc_easy"]
GEN_TASK = "gsm8k_cot"

NO_THINKING_PREFIX = "{%- set enable_thinking = false -%}"


def force_non_thinking_template(eval_dir: Path, tokenizer=None) -> None:
    """Make ``enable_thinking=False`` the template default for this saved tokenizer.

    lm-eval passes no ``enable_thinking`` flag. Prepending a ``set`` keeps every stock
    task definition untouched while fixing the rendered prompt. Transformers may store
    the template either in ``tokenizer_config.json`` or in a sidecar file, so both are
    handled.
    """
    cfg_path = eval_dir / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    template = cfg.get("chat_template")
    if not template and tokenizer is not None:
        template = getattr(tokenizer, "chat_template", None)
    jinja = eval_dir / "chat_template.jinja"
    if not template and jinja.exists():
        template = jinja.read_text()
    if not template:
        raise ValueError("tokenizer has no chat_template")
    if NO_THINKING_PREFIX in template:
        return
    patched = NO_THINKING_PREFIX + "\n" + template
    cfg["chat_template"] = patched
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    if jinja.exists():
        jinja.write_text(patched)


def prepare_eval_dir(model, tokenizer, out_dir: Path, *, is_adapter: bool) -> Path:
    """Merge adapters into the base weights and save a self-contained HF model."""
    from task1.lora import merge_adapters

    out_dir.mkdir(parents=True, exist_ok=True)
    if is_adapter:
        merge_adapters(model)
    model.save_pretrained(out_dir, safe_serialization=True)
    template = getattr(tokenizer, "chat_template", None)
    if template and NO_THINKING_PREFIX not in template:
        tokenizer.chat_template = NO_THINKING_PREFIX + "\n" + template
    tokenizer.save_pretrained(out_dir)
    force_non_thinking_template(out_dir, tokenizer)
    return out_dir


def verify_merge(model, tokenizer, reference_logits, *, device="cuda") -> float:
    """Max abs difference between merged and unmerged logits on a fixed small batch."""
    import torch

    ids = tokenizer("What is 2 + 2?", return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        out = model(input_ids=ids).logits.float().cpu()
    return float((out - reference_logits).abs().max())


def _run(cmd: list[str], env: dict) -> int:
    print("$", " ".join(cmd), flush=True)
    return subprocess.call(cmd, env=env)


def run_benchmarks(eval_dir: Path, out_dir: Path, *, batch_size: str = "32",
                   likelihood_batch_size: str = "8",
                   limit: int | None = None, resume: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    # reduces allocator fragmentation across the many short/long batches
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    model_args = f"pretrained={eval_dir},dtype=float16"

    gen_out = out_dir / "gsm8k.json"
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf", "--model_args", model_args,
        "--tasks", GEN_TASK, "--num_fewshot", "0",
        "--apply_chat_template", "--batch_size", batch_size,
        "--gen_kwargs", "max_gen_toks=1024,do_sample=False",
        "--output_path", str(gen_out),
        "--seed", "6304",
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    if resume and latest_output(out_dir, "gsm8k") is not None:
        print("  [resume] GSM8K output exists; skipping", flush=True)
    else:
        rc = _run(cmd, env)
        if rc != 0:
            raise RuntimeError(f"gsm8k evaluation failed with exit {rc}")

    lik_out = out_dir / "likelihood.json"
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf", "--model_args", model_args,
        "--tasks", ",".join(EVAL_TASKS_LIKELIHOOD),
        "--num_fewshot", "0", "--batch_size", likelihood_batch_size,
        "--output_path", str(lik_out),
        "--seed", "6304",
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    if resume and latest_output(out_dir, "likelihood") is not None:
        print("  [resume] likelihood output exists; skipping", flush=True)
    else:
        rc = _run(cmd, env)
        if rc != 0:
            raise RuntimeError(f"likelihood evaluation failed with exit {rc}")

    return {"gsm8k": str(gen_out), "likelihood": str(lik_out),
            "summary": summarize(out_dir)}


def latest_output(out_dir: Path, stem: str) -> Path | None:
    """lm-eval writes ``<task>_<timestamp>.json``; pick the newest."""
    files = sorted(out_dir.glob(f"{stem}_*.json")) + sorted(out_dir.glob(f"{stem}.json"))
    return files[-1] if files else None


def _results_node(payload: dict) -> dict:
    if "results" in payload:
        return payload["results"]
    # lm-eval wraps the payload when written per-task
    for value in payload.values():
        if isinstance(value, dict) and "results" in value:
            return value["results"]
    return {}


def summarize(out_dir: Path) -> dict:
    """Extract the primary metrics the assignment asks for."""
    summary: dict = {}
    for stem in ("gsm8k", "likelihood"):
        path = latest_output(out_dir, stem)
        if path is None:
            continue
        text = path.read_text()
        try:
            payload = json.loads(text)
            nodes = [_results_node(payload)]
        except json.JSONDecodeError:
            nodes = [_results_node(json.loads(ln)) for ln in text.splitlines() if ln.strip()]
        for results in nodes:
            for task, metrics in results.items():
                summary[task] = {k: v for k, v in metrics.items()
                                 if isinstance(v, (int, float)) and not k.startswith("alias")}
    return summary


def primary_scores(summary: dict) -> dict:
    """Map lm-eval keys to the assignment's four primary metrics.

    lm-eval emits ``metric,filter`` keys (e.g. ``exact_match,strict-match`` and
    ``exact_match,flexible-extract``). The assignment specifies flexible extraction for
    GSM8K, so the exact key is selected first and substring matching is only a fallback.
    """
    def pick(task, key):
        for name, metrics in summary.items():
            if name == task or name.endswith("/" + task) or task in name:
                if key in metrics:
                    return float(metrics[key])
                for k, v in metrics.items():
                    if k.startswith(key):
                        return float(v)
        return None

    return {
        "gsm8k_flexible_extract": pick("gsm8k_cot", "exact_match,flexible-extract"),
        "hellaswag_acc_norm": pick("hellaswag", "acc_norm,none"),
        "hellaswag_acc": pick("hellaswag", "acc,none"),
        "boolq_acc": pick("boolq", "acc,none"),
        "arc_easy_acc_norm": pick("arc_easy", "acc_norm,none"),
        "arc_easy_acc": pick("arc_easy", "acc,none"),
    }
