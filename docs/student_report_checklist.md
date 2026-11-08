# Student report checklist

Everything below is **yours** to write or judge. The agent produces evidence only.
The PA forbids generative AI from writing any part of the PDF.

## Before you write

- [ ] Read `docs/results_index.md` and open the raw files behind every number you cite.
- [ ] Confirm the `first_plot_questions.yaml` copy matches the LMS version
      (`data/reference/ASSETS.json` records the URL and sha256).
- [ ] Confirm the student ID (`28100371`) used in configs.
- [ ] Decide whether to run the optional Task 2 49-token ablation.

## Task 1 — when the runs exist

- [ ] Explain the seven evaluation conditions and why midpoint checks are recovery-only.
- [ ] Compare LoRA-4 vs LoRA-16, Full FT vs L2-SP, LoRA-16 vs OPLoRA, Full FT vs LoRA-16.
- [ ] Interpret ρ_k and η: what they measure and what they cannot establish.
- [ ] Discuss R(z) on math vs non-math prompts and its local-only scope.
- [ ] **Annotate the 42 responses** (first/last underlying question × 3 formats × 7
      conditions): format compliance, coherence/relevance, repetition, unnecessary
      mathematical framing, inappropriate content. Choose the excerpts yourself.
- [ ] State deviations (e.g. any cut experiment) explicitly.

## Task 2 — when the runs exist

- [ ] Describe the curriculum's effect across stages using the dev curves.
- [ ] Compare the four image conditions; separate image sensitivity from correctness.
- [ ] Choose the illustrative complementary pairs using a stated selection rule.
- [ ] Discuss empty answers and the soft-consensus metric's behaviour.

## Task 3 — available now

- [ ] Explain why oracle coverage and achieved selection accuracy diverge; use the
      `*_failures.{csv,md}` decomposition.
- [ ] Interpret the sign and size of the confidence–vision correlation
      (−0.211 mean within-rollout) and the tokenwise-fusion identity.
- [ ] Discuss why `best_logp` degrades beyond N≈4 on both datasets.
- [ ] Explain the adaptive results: dev-selected thresholds, the 1 pp rule, the
      token savings, and the dev/held-out difficulty gap (FirstAns 0.727 vs 0.647).
- [ ] After the live run: compare the three strategies at equal output allowance and
      interpret the revision transitions.
- [ ] Present 2–3 aligned disagreement / premature-stop cases with a stated rule.

## Report mechanics

- [ ] 8 pages main content, NeurIPS PA0 format, references/appendix excluded.
- [ ] Repository link at the end of the abstract.
- [ ] Cite Qwen, MetaMathQA, GSM8K, HellaSwag, BoolQ, ARC, SigLIP, COCO, VQAv2,
      MathVista, math-verify, lm-evaluation-harness, and the two required papers.
- [ ] Every reported number traces to a saved file in `docs/results_index.md`.
- [ ] Disclose incomplete experiments and deviations.
- [ ] LMS submission format confirmed.
