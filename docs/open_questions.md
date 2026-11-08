# Open questions and recorded decisions

## Decisions taken (with rationale)

1. **`math-verify` boxed wrapping.** Supplied MATH answers are already-extracted boxed
   contents. Parsing them bare truncates expressions (`x+1` → `[]`), so they are wrapped
   as `\boxed{…}` before parsing. Verified against 8 hand-picked equivalence cases.

2. **Prefix grouping.** Pairwise equivalence is computed once per question (no gold).
   A prefix's groups are formed by union-find over revealed pairs only, so an unrevealed
   candidate cannot bridge two groups. This is stricter than transitive closure over the
   full pool and satisfies "form groups without consulting gold" and "recompute with
   revealed candidates only".

3. **MathVista multiple choice compared as option letters.** Equivalent to the official
   letter→choice-text mapping but robust to mojibake in the released choice strings.

4. **Missing extractions.** `ignore_empty_extractions=True` semantics: null / empty /
   non-answer phrases → NO_ANSWER, graded incorrect, kept in denominators.

5. **`n_defined` reporting.** Where a question has constant scores or constant
   correctness, Spearman correlation is undefined and is excluded rather than set to 0;
   the number of defined questions is reported alongside every mean.

6. **`Z_token` fusion** uses the same prefix-pooled calibration as VisConf, so the
   ablation differs only in where the product is taken.

7. **Windows vs Linux.** The Windows `math-verify` timeout path spawns a process per
   call. All CPU pipeline work therefore runs under WSL2 Ubuntu, which also matches the
   GPU box. `tests/` pass on both.

## Open questions that need a human decision

| # | Question | Why it matters | Current default |
|---|---|---|---|
| 1 | Is `first_plot_questions.yaml` from the public `emergent-misalignment` repo the same file the course distributed? | Task 1's 24 qualitative prompts | used, with URL + sha256 recorded; student should diff against LMS |
| 2 | Should the Task 2 optional 49-token run be attempted? | explicitly optional | planned cut unless budget and time allow |
| 3 | Which checkpoints may be committed to the public repo? | "do not commit unnecessary large checkpoints" | only adapters (≈40 MB each) and metrics; full FT weights stay on the GPU box |
| 4 | Student ID to embed in configs | reproducibility metadata | `28100371` assumed, unconfirmed |
| 5 | Deadline 13 Nov — is the LMS deadline hard, or is late submission accepted? | sequencing and scope cuts | treated as hard |

## Known limitations to disclose in the report

1. `math-verify` uses a 5 s per-call timeout on POSIX; pathological expressions are
   treated as "not equivalent". The number of timeouts is not yet logged.
2. MATH `dev` (200 questions) is measurably easier than `held-out` (FirstAns 0.727 vs
   0.647). Adaptive thresholds selected on dev are therefore validated on a harder
   split, which is the conservative direction but widens the dev/held-out gap.
3. Token cost is an offline proxy: it excludes prefill, attention collection and
   scoring overhead, as the assignment specifies.
4. Live generation (Step 4) has not been run, so no revision-transition results exist.
5. The incorrect-early-stop case list (recovered by fixed-32) is not yet produced.
