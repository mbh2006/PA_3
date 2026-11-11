"""Task 3 live generation: 50 MATH questions x 3 strategies at equal token allowance.

Prescribed settings:
  * Qwen/Qwen2.5-3B-Instruct at the revision pinned in the supplied README
  * FP16, no quantisation, batch 1, same GPU
  * temperature 0.7, top-p 0.8, top-k 20, repetition penalty 1.0
  * native chat template, system "You are a helpful assistant."
  * problem followed by the two-line instruction
  * 3,072 total output-token allowance per strategy

Strategies
  single      one response, 3,072 cap
  parallel    three independent responses, 1,024 each, self-consistency vote
  refine      reuse parallel's first response, one critique-and-revision up to 2,048

Answers are extracted with the README's last-balanced-brace rule; grading uses the same
frozen Task 3 scorer. No gold feedback is used to choose between initial and revised.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path("/root/pa3")
sys.path.insert(0, str(ROOT / "src"))

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
SYSTEM = "You are a helpful assistant."
INSTRUCTION = ("Think step by step inside <think> </think> tags, then give your final "
               "answer inside \\boxed{}.")
REVIEW = ("Review the solution above. Identify and correct any errors; if it is correct, "
          "retain its answer. Give your final answer inside \\boxed{}.")
TEMPERATURE, TOP_P, TOP_K, REP_PEN = 0.7, 0.8, 20, 1.0
SEED = 6304
OUT = ROOT / "outputs" / "task_03" / "live"


def extract_boxed(text: str) -> str | None:
    """Last literal ``\\boxed{`` with balanced braces; nested LaTeX preserved."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    i = idx + len("\\boxed{")
    depth = 1
    out = []
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
            out.append(c)
        elif c == "}":
            depth -= 1
            if depth == 0:
                return "".join(out).strip() or None
            out.append(c)
        else:
            out.append(c)
        i += 1
    return None


def generate(model, tokenizer, messages, max_new_tokens: int, seed: int) -> tuple[str, int, bool]:
    torch.manual_seed(seed)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
    ids = tokenizer(prompt, return_tensors="pt").to("cuda")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            repetition_penalty=REP_PEN,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen = out[0, ids["input_ids"].shape[1]:]
    text = tokenizer.decode(gen, skip_special_tokens=True)
    return text, int(gen.shape[0]), bool(gen.shape[0] >= max_new_tokens)


def vote(answers: list[str | None]) -> str | None:
    from collections import Counter

    clean = [a for a in answers if a]
    if not clean:
        return None
    counts = Counter(clean)
    best = counts.most_common(1)[0][0]
    for a in answers:            # earliest first appearance breaks ties
        if a == best:
            return a
    return best


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--probe", type=int, default=2,
                    help="questions used to project runtime before the full sweep")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from task3.answers import math_correct

    questions = json.loads((ROOT / "task3_live_questions.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=torch.float16
    ).to("cuda").eval()
    load_s = time.time() - t_load

    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    t_start = time.time()
    produced = 0

    for q in questions[: args.limit]:
        qid, question, gold = q["sample_id"], q["question"], q["gold"]
        base = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"{question}\n\n{INSTRUCTION}"}]

        single, n_single, tr_single = generate(model, tokenizer, base, 3072, SEED + 1)
        p1, n_p1, tr_p1 = generate(model, tokenizer, base, 1024, SEED + 2)
        p2, n_p2, tr_p2 = generate(model, tokenizer, base, 1024, SEED + 3)
        p3, n_p3, tr_p3 = generate(model, tokenizer, base, 1024, SEED + 4)

        refine_msgs = base + [{"role": "assistant", "content": p1},
                              {"role": "user", "content": REVIEW}]
        refined, n_ref, tr_ref = generate(model, tokenizer, refine_msgs, 2048, SEED + 5)

        a_single, a_p1, a_p2, a_p3 = (extract_boxed(x) for x in (single, p1, p2, p3))
        a_ref = extract_boxed(refined)
        a_parallel = vote([a_p1, a_p2, a_p3])

        results.append({
            "sample_id": qid,
            "question": question,
            "gold": gold,
            "single": {"answer": a_single, "tokens": n_single, "truncated": tr_single,
                       "correct": math_correct(a_single, gold) if a_single else False},
            "parallel": {"answers": [a_p1, a_p2, a_p3], "answer": a_parallel,
                         "tokens": n_p1 + n_p2 + n_p3,
                         "truncated": tr_p1 or tr_p2 or tr_p3,
                         "correct": math_correct(a_parallel, gold) if a_parallel else False},
            "refine": {"initial_answer": a_p1, "answer": a_ref,
                       "tokens": n_p1 + n_ref,
                       "truncated": tr_p1 or tr_ref,
                       "correct": math_correct(a_ref, gold) if a_ref else False},
            "raw": {"single": single, "p1": p1, "p2": p2, "p3": p3, "refined": refined},
        })
        produced += 1

        if produced == args.probe:
            elapsed = time.time() - t_start
            per_q = elapsed / produced
            projected = per_q * args.limit / 3600
            print(json.dumps({"probe_questions": produced, "seconds_each": round(per_q, 1),
                              "projected_hours_full": round(projected, 2)}), flush=True)

        if produced % 10 == 0:
            print(f"  {produced}/{args.limit} questions, {time.time()-t_start:.0f}s", flush=True)

    def acc(key: str) -> float:
        return sum(1 for r in results if r[key]["correct"]) / max(1, len(results))

    transitions = {
        "wrong_to_correct": sum(1 for r in results
                                if not (r["refine"]["initial_answer"] and
                                        math_correct(r["refine"]["initial_answer"], r["gold"]))
                                and r["refine"]["correct"]),
        "correct_to_wrong": sum(1 for r in results
                                if (r["refine"]["initial_answer"] and
                                    math_correct(r["refine"]["initial_answer"], r["gold"]))
                                and not r["refine"]["correct"]),
    }
    summary = {
        "n_questions": len(results),
        "model": MODEL_ID, "revision": MODEL_REVISION,
        "load_seconds": round(load_s, 1),
        "generation_seconds": round(time.time() - t_start, 1),
        "accuracy": {"single_3072": acc("single"), "parallel_3x1024": acc("parallel"),
                     "refine_1024_2048": acc("refine")},
        "mean_output_tokens": {
            "single_3072": sum(r["single"]["tokens"] for r in results) / max(1, len(results)),
            "parallel_3x1024": sum(r["parallel"]["tokens"] for r in results) / max(1, len(results)),
            "refine_1024_2048": sum(r["refine"]["tokens"] for r in results) / max(1, len(results)),
        },
        "extraction_rate": {
            "single": sum(1 for r in results if r["single"]["answer"]) / max(1, len(results)),
            "parallel": sum(1 for r in results if r["parallel"]["answer"]) / max(1, len(results)),
            "refine": sum(1 for r in results if r["refine"]["answer"]) / max(1, len(results)),
        },
        "truncation_rate": {
            "single": sum(1 for r in results if r["single"]["truncated"]) / max(1, len(results)),
            "parallel": sum(1 for r in results if r["parallel"]["truncated"]) / max(1, len(results)),
            "refine": sum(1 for r in results if r["refine"]["truncated"]) / max(1, len(results)),
        },
        "revision_transitions": transitions,
    }
    (OUT / "live_results.json").write_text(json.dumps(results, indent=2))
    (OUT / "live_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("LIVE_DONE")


if __name__ == "__main__":
    main()
