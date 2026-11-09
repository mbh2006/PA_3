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


def force_non_thinking_template(eval_dir: Path) -> None:
    """Make ``enable_thinking=False`` the template default for this saved tokenizer.

    lm-eval passes no ``enable_thinking`` flag, and the stock Qwen3 template would
    otherwise render a thinking block. Prepending a ``set`` keeps every stock task
    definition untouched while fixing the rendered prompt.
    """
    cfg_path = eval_dir / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text())
    template = cfg.get("chat_template")
    if template is None:
        raise ValueError("tokenizer has no chat_template")
    if NO_THINKING_PREFIX not in template:
        cfg["chat_template"] = NO_THINKING_PREFIX + "\n" + template
        cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")


def prepare_eval_dir(model, tokenizer, out_dir: Path, *, is_adapter: bool) -> Path:
    """Merge adapters into the base weights and save a self-contained HF model."""
    from task1.lora import merge_adapters

    out_dir.mkdir(parents=True, exist_ok=True)
    if is_adapter:
        merge_adapters(model)
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    force_non_thinking_template(out_dir)
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


def run_benchmarks(eval_dir: Path, out_dir: Path, *, batch_size: str = "auto",
                   limit: int | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    model_args = f"pretrained={eval_dir},dtype=float16,trust_remote_code=False"

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
    rc = _run(cmd, env)
    if rc != 0:
        raise RuntimeError(f"gsm8k evaluation failed with exit {rc}")

    lik_out = out_dir / "likelihood.json"
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf", "--model_args", model_args,
        "--tasks", ",".join(EVAL_TASKS_LIKELIHOOD),
        "--num_fewshot", "0", "--batch_size", batch_size,
        "--output_path", str(lik_out),
        "--seed", "6304",
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    rc = _run(cmd, env)
    if rc != 0:
        raise RuntimeError(f"likelihood evaluation failed with exit {rc}")

    return {"gsm8k": str(gen_out), "likelihood": str(lik_out),
            "summary": summarize(out_dir)}


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
    for path in (out_dir / "gsm8k.json", out_dir / "likelihood.json"):
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            results = _results_node(json.loads(line))
            for task, metrics in results.items():
                keep = {k: v for k, v in metrics.items()
                        if isinstance(v, (int, float)) and not k.startswith("alias")}
                summary[task] = keep
    return summary


def primary_scores(summary: dict) -> dict:
    """Map lm-eval keys to the assignment's four primary metrics."""
    def pick(task, key):
        for name, metrics in summary.items():
            if name == task or name.endswith("/" + task) or task in name:
                for k, v in metrics.items():
                    if k == key or k.startswith(key + ","):
                        return float(v)
        return None

    return {
        "gsm8k_flexible_extract": pick("gsm8k_cot", "exact_match"),
        "hellaswag_acc_norm": pick("hellaswag", "acc_norm"),
        "hellaswag_acc": pick("hellaswag", "acc"),
        "boolq_acc": pick("boolq", "acc"),
        "arc_easy_acc_norm": pick("arc_easy", "acc_norm"),
        "arc_easy_acc": pick("arc_easy", "acc"),
    }
