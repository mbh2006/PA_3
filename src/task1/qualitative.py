"""Qualitative generation on the 24 fixed prompts (PA p. 5).

Every one of the seven conditions answers all 24 prompts (8 questions x plain / JSON /
template), greedily, thinking disabled, 512 new tokens. Truncation is flagged per
response. The 42-response annotation sheet covers the first and last underlying
questions in the file's plain-question order, in all three formats.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

MAX_NEW_TOKENS = 512


@dataclass
class Prompt:
    prompt_id: str
    question: str
    fmt: str                 # plain | json | template
    system: str | None

    def as_dict(self) -> dict:
        return {"prompt_id": self.prompt_id, "question": self.question,
                "format": self.fmt, "system": self.system}


def load_prompts(path: Path) -> list[Prompt]:
    entries = yaml.safe_load(path.read_text(encoding="utf-8"))
    prompts: list[Prompt] = []
    for entry in entries:
        pid = entry["id"]
        if pid.endswith("_json"):
            fmt = "json"
        elif pid.endswith("_template"):
            fmt = "template"
        else:
            fmt = "plain"
        paraphrases = entry.get("paraphrases") or []
        if len(paraphrases) != 1:
            raise ValueError(f"{pid}: expected exactly one paraphrase, got {len(paraphrases)}")
        prompts.append(Prompt(prompt_id=pid, question=paraphrases[0].strip(),
                              fmt=fmt, system=entry.get("system")))
    if len(prompts) != 24:
        raise ValueError(f"expected 24 prompts, got {len(prompts)}")
    return prompts


def prompt_messages(prompt: Prompt) -> list[dict]:
    messages = []
    if prompt.system:
        messages.append({"role": "system", "content": prompt.system})
    messages.append({"role": "user", "content": prompt.question})
    return messages


def generate_all(model, tokenizer, prompts: list[Prompt], *, device="cuda",
                 max_new_tokens: int = MAX_NEW_TOKENS) -> list[dict]:
    import torch

    model.eval()
    records = []
    for p in prompts:
        text = tokenizer.apply_chat_template(
            prompt_messages(p), tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **ids, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        new_ids = out[0, ids["input_ids"].shape[1]:]
        response = tokenizer.decode(new_ids, skip_special_tokens=True)
        records.append({
            "prompt_id": p.prompt_id,
            "format": p.fmt,
            "question": p.question,
            "system": p.system,
            "response": response,
            "n_new_tokens": int(new_ids.shape[0]),
            "truncated": bool(new_ids.shape[0] >= max_new_tokens),
        })
    return records


def _base_id(prompt_id: str) -> str:
    """``what_is_your_wish_json`` -> ``what_is_your_wish``."""
    for suffix in ("_json", "_template"):
        if prompt_id.endswith(suffix):
            return prompt_id[: -len(suffix)]
    return prompt_id


def annotation_rows(records: list[dict]) -> list[dict]:
    """First and last underlying question in plain-question order, all three formats.

    Selection is by underlying prompt id: the JSON and template variants carry their own
    paraphrase text, so matching on question text would silently drop them.
    """
    plain = [r for r in records if r["format"] == "plain"]
    if not plain:
        raise ValueError("no plain prompts")
    wanted = {_base_id(plain[0]["prompt_id"]), _base_id(plain[-1]["prompt_id"])}
    selected = [r for r in records if _base_id(r["prompt_id"]) in wanted]
    if len(selected) != 6:
        raise ValueError(f"expected 6 rows per condition, got {len(selected)}")
    return selected


def mechanical_checks(response: str) -> dict:
    words = response.split()
    return {
        "n_chars": len(response),
        "n_words": len(words),
        "is_empty": len(words) == 0,
        "starts_with_json_like": response.strip().startswith("{"),
        "mentions_math_symbol": any(s in response for s in ("\\boxed", "\\frac", "$$")),
        "repeated_line": _has_repeated_line(response),
    }


def _has_repeated_line(text: str) -> bool:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return len(lines) != len(set(lines))


def save_condition(records: list[dict], condition: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"qualitative_{condition}.json"
    path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return path


def build_annotation_sheet(all_records: dict[str, list[dict]], out_path: Path) -> int:
    """42 rows: 2 questions x 3 formats x 7 conditions, with mechanical checks only."""
    rows = []
    for condition, records in all_records.items():
        for r in annotation_rows(records):
            rows.append({
                "condition": condition,
                "prompt_id": r["prompt_id"],
                "format": r["format"],
                "question": r["question"],
                "response": r["response"],
                "truncated": r["truncated"],
                **mechanical_checks(r["response"]),
            })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return len(rows)
