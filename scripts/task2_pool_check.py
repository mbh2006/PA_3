#!/usr/bin/env python3
"""CPU dry-run of the Task 2 pool construction (no GPU, no features)."""

import sys

sys.path.insert(0, "/root/pa3/src")
sys.path.insert(0, "/root/pa3/scripts")

from transformers import AutoTokenizer  # noqa: E402

from task2_run import build_pools, load_data  # noqa: E402


def main() -> None:
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    data = load_data()
    pools = build_pools(data, {}, tok)
    print("pool sizes:", {k: len(v) for k, v in pools.items()})
    p = pools["presence"][0]
    print("presence pair sides:", [e["side"] for e in p])
    print("presence answers:", [e["answer"] for e in p])
    n = pools["natural"][0]
    print("natural answers:", [e["answer"] for e in n])
    print("natural same question:", n[0]["question"] == n[1]["question"])
    c = pools["caption"][0]
    print("caption:", c["question"], "|", c["answer"][:50])
    caps = pools["caption"]
    print("caption train examples:", len(caps), "(expected 18000)")
    print("natural pairs:", len(pools["natural"]), "(expected 1536)")
    print("presence pairs:", len(pools["presence"]), "counting pairs:", len(pools["counting"]))
    print("POOL_CHECK_OK")


if __name__ == "__main__":
    main()
