# Student report checklist

Everything below is **yours** to write or judge. The agent produced evidence only; the PA
forbids generative AI from writing any part of the PDF.

`docs/PA3_COMPLETE_TECHNICAL_RECORD.md` is the single factual reference for every number.

## Before you write

- [ ] Read `docs/results_index.md` and open the raw file behind every number you cite.
- [ ] Diff `data/reference/first_plot_questions.yaml` against the LMS copy (URL and sha256
      are in `data/reference/ASSETS.json`).
- [ ] Skim `docs/tasks/*.md` so your wording of the deviations matches the record.
- [ ] Decide whether any of the "not produced" items (Task 2 49-token run, Task 3
      early-stop case list) are worth mentioning as omissions.

## Task 1 — results are complete

- [ ] Explain the seven evaluation conditions and why the midpoint is recovery-only.
- [ ] Compare LoRA-4 vs LoRA-16, Full FT vs L2-SP, LoRA-16 vs OPLoRA, Full FT vs LoRA-16,
      and Full FT midpoint vs final.
- [ ] Interpret η and **the one available ρ_k** (OPLoRA ≈ 1e-9) carefully, and state that
      ρ_k was not computed for the other conditions rather than implying it was.
- [ ] Discuss R(z) on math vs non-math prompts and its local-only scope.
- [ ] **Annotate the 42 responses** (`results/task_01/examples/annotation_sheet.json`):
      coherence, relevance, repetition, unnecessary mathematical framing, inappropriate
      content, format compliance. Choose the excerpts yourself.
- [ ] Disclose that the 312/625/937 checkpoints for Full FT and L2-SP were lost and the
      midpoint is reconstructed.

## Task 2 — results are complete

- [ ] Compare the four image conditions in both suites; separate image sensitivity from
      answer correctness.
- [ ] Explain the no-image NLL (21–26) and why it is not a bug.
- [ ] State that two earlier attempts were invalidated and why (§7 of the master record).
- [ ] Note that development curves are sparse NLL points, not generated accuracies.
- [ ] Choose the illustrative complementary pairs with a stated selection rule.

## Task 3 — results are complete

- [ ] Explain why oracle coverage and achieved selection accuracy diverge; use the
      `*_failures.{csv,md}` decomposition.
- [ ] Interpret the confidence–vision correlation (−0.211) and the exact tokenwise-fusion
      identity.
- [ ] Discuss why `best_logp` degrades beyond N≈4 on both datasets.
- [ ] Explain the adaptive results: dev-selected thresholds, the 1 pp rule, token savings,
      and the dev/held-out difficulty gap.
- [ ] Compare the three live strategies at equal output allowance and interpret the three
      wrong→correct revisions against zero correct→wrong.
- [ ] Present 2–3 aligned disagreement / revision / premature-stop cases with a stated rule.
- [ ] State that the incorrect-early-stop list was not produced.

## Report mechanics

- [ ] 8 pages main content, NeurIPS PA0 format, references and appendix excluded.
- [ ] Repository link at the end of the abstract
      (<https://github.com/mbh2006/PA_3>).
- [ ] Cite Qwen, MetaMathQA, GSM8K, HellaSwag, BoolQ, ARC, SigLIP, COCO, VQAv2, MathVista,
      math-verify, lm-evaluation-harness, plus the two required papers.
- [ ] Every reported number traces to a saved file listed in `docs/results_index.md`.
- [ ] Disclose the incomplete experiments and deviations.
- [ ] Confirm the LMS submission format and upload.
