"""Task 1 data preparation: eligibility, tokenisation, compact export, probe sampling.

Produces a self-contained artefact so GPU runs never repeat the CPU work:

    prepared.npz   input_ids (int32, concatenated), lengths, n_prompt, source_index
    prepared.json  selection report, tokenizer identity, counts, checksums

Labels are reconstructed deterministically on load (prompt masked, response + end-of-turn
supervised, separator newline masked), so only token ids are stored.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from task1.data import MathExample, SelectionReport, select_examples

MATH_SOURCE = "meta-math/MetaMathQA"
GSM8K_SOURCE = "openai/gsm8k"
HELLASWAG_SOURCE = "Rowan/hellaswag"


def load_gsm8k_test_questions() -> list[str]:
    from datasets import load_dataset

    ds = load_dataset(GSM8K_SOURCE, "main", split="test")
    return [row["question"] for row in ds]


def load_metamath_rows(split: str = "train"):
    from datasets import load_dataset

    return load_dataset(MATH_SOURCE, split=split)


def _tokenizer_identity(tokenizer) -> dict:
    return {
        "name": getattr(tokenizer, "name_or_path", "unknown"),
        "vocab_size": int(getattr(tokenizer, "vocab_size", -1)),
        "im_end_id": int(tokenizer.convert_tokens_to_ids("<|im_end|>")),
        "pad_token_id": int(tokenizer.pad_token_id or 0),
        "eos_token_id": int(tokenizer.eos_token_id),
    }


def save_prepared(path: Path, examples: list[MathExample], meta: dict) -> Path:
    lengths = np.array([ex.n_tokens for ex in examples], dtype=np.int32)
    n_prompt = np.array([ex.n_prompt_tokens for ex in examples], dtype=np.int32)
    source_index = np.array([ex.source_index for ex in examples], dtype=np.int32)
    flat = np.concatenate([ex.input_ids.astype(np.int32) for ex in examples])
    np.savez_compressed(
        path,
        input_ids=flat,
        lengths=lengths,
        n_prompt=n_prompt,
        source_index=source_index,
    )
    meta = dict(meta)
    meta["n_examples"] = len(examples)
    meta["total_tokens"] = int(flat.size)
    meta["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return path


def load_prepared(path: Path) -> tuple[list[MathExample], dict]:
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    data = np.load(path)
    flat = data["input_ids"]
    lengths = data["lengths"]
    n_prompt = data["n_prompt"]
    source_index = data["source_index"]
    im_end = int(meta["tokenizer"]["im_end_id"])

    examples: list[MathExample] = []
    offset = 0
    for i, (n, p, sid) in enumerate(zip(lengths, n_prompt, source_index)):
        ids = flat[offset:offset + int(n)].astype(np.int32)
        offset += int(n)
        labels = ids.copy()
        labels[:int(p)] = -100
        positions = np.flatnonzero(ids == im_end)
        if positions.size:
            labels[int(positions[-1]) + 1:] = -100
        examples.append(MathExample(
            source_index=int(sid), query="", original_question="", response="",
            input_ids=ids, labels=labels, n_prompt_tokens=int(p),
        ))
    return examples, meta


@dataclass
class ProbeSet:
    gsm8k_indices: list[int]
    gsm8k_questions: list[str]
    hellaswag_indices: list[int]
    hellaswag_contexts: list[str]

    def as_dict(self) -> dict:
        return {
            "gsm8k_indices": self.gsm8k_indices,
            "hellaswag_indices": self.hellaswag_indices,
            "n_gsm8k": len(self.gsm8k_indices),
            "n_hellaswag": len(self.hellaswag_indices),
        }


def _default_loader(name: str, split: str):
    from datasets import load_dataset

    if name == GSM8K_SOURCE:
        return load_dataset(name, "main", split=split)
    return load_dataset(name, split=split)


def sample_probes(seed: int = 6304, n: int = 32, loader=None) -> ProbeSet:
    """Fixed activation-probe set drawn before any model result is inspected.

    ``loader(name, split)`` is injectable so the sampling logic is testable without the
    datasets library or a network.
    """
    loader = loader or _default_loader
    gsm = loader(GSM8K_SOURCE, "test")
    hella = loader(HELLASWAG_SOURCE, "validation")

    rng = np.random.default_rng(seed)
    gsm_idx = sorted(int(i) for i in rng.choice(len(gsm), size=n, replace=False))
    hella_idx = sorted(int(i) for i in rng.choice(len(hella), size=n, replace=False))

    return ProbeSet(
        gsm8k_indices=gsm_idx,
        gsm8k_questions=[gsm[i]["question"] for i in gsm_idx],
        hellaswag_indices=hella_idx,
        hellaswag_contexts=[hella[i]["ctx"] for i in hella_idx],
    )


def prepare(
    out_dir: Path,
    *,
    n_examples: int = 40000,
    max_seq_len: int = 2048,
    tokenizer_id: str = "Qwen/Qwen3-0.6B",
) -> dict:
    from transformers import AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    gsm_questions = {q for q in (normalize(q) for q in load_gsm8k_test_questions())}
    rows = load_metamath_rows()
    examples, report = select_examples(rows, tokenizer, gsm_questions,
                                       n_examples=n_examples, max_seq_len=max_seq_len)

    meta = {
        "source": MATH_SOURCE,
        "n_source_rows": len(rows),
        "max_seq_len": max_seq_len,
        "tokenizer": _tokenizer_identity(tokenizer),
        "selection": report.as_dict(),
        "gsm8k_test_questions": len(gsm_questions),
    }
    path = save_prepared(out_dir / "prepared.npz", examples, meta)

    probes = sample_probes(seed=6304)
    (out_dir / "probes.json").write_text(
        json.dumps({**probes.as_dict(),
                    "gsm8k_questions": probes.gsm8k_questions,
                    "hellaswag_contexts": probes.hellaswag_contexts}, indent=2) + "\n",
        encoding="utf-8")
    return {**meta, "path": str(path), "n_probes": len(probes.gsm8k_indices)}


def normalize(text: str) -> str:
    from task1.data import normalize_question

    return normalize_question(text)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("outputs/task_01/data"))
    ap.add_argument("--n-examples", type=int, default=40000)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    args = ap.parse_args()
    info = prepare(args.out, n_examples=args.n_examples, max_seq_len=args.max_seq_len)
    print(json.dumps({k: v for k, v in info.items() if k != "selection"}, indent=2))
    print("selection:", json.dumps(info["selection"], indent=2)[:800])


if __name__ == "__main__":
    main()
