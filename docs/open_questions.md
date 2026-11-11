# Open questions and recorded decisions

## Decisions taken (with rationale)

1. **`math-verify` boxed wrapping.** Supplied MATH answers are already-extracted boxed
   contents. Parsing them bare truncates expressions (`x+1` → `[]`), so they are wrapped as
   `\boxed{…}` before parsing. Verified against 8 hand-picked equivalence cases.

2. **Prefix grouping.** Pairwise equivalence is computed once per question without gold. A
   prefix's groups are formed by union-find over revealed pairs only, so an unrevealed
   candidate cannot bridge two groups. This is stricter than transitive closure over the
   full pool and satisfies both "form groups without consulting gold" and "recompute with
   revealed candidates only".

3. **MathVista multiple choice compared as option letters.** Equivalent to the official
   letter→choice-text mapping but robust to mojibake in the released choice strings.

4. **Missing extractions.** Null, empty and non-answer phrases all map to NO_ANSWER, graded
   incorrect, and kept in denominators, voting and cost accounting.

5. **`n_defined` reporting.** Where a question has constant scores or constant correctness,
   Spearman correlation is undefined and is excluded rather than set to 0; the number of
   defined questions is reported alongside every mean.

6. **`Z_token` fusion** uses the same prefix-pooled calibration as VisConf, so the ablation
   differs only in where the product is taken.

7. **Windows vs Linux.** The Windows `math-verify` timeout path spawns a process per call,
   so all CPU pipeline work runs under WSL2 Ubuntu, matching the GPU box.

8. **No checkpoints in the public repository.** Task 1 weights are ~7.15 GB each; Task 2
   stage weights are 24.58 MB each and are retained locally. Only code, configs, results,
   tables and figures are published.

## Resolved questions

| # | Question | Resolution |
|---|---|---|
| 1 | Is the public `first_plot_questions.yaml` the course copy? | Used with URL and sha256 recorded in `data/reference/ASSETS.json`; the student should still diff it against the LMS copy. |
| 2 | Run the optional Task 2 49-token ablation? | **No** — budget spent on the mandatory reruns. Recorded as "not run", not as a failure. |
| 3 | Which checkpoints may be published? | **None** — see decision 8. |
| 4 | Student ID | **28100371**, confirmed by the student. |
| 5 | Is the 13 Nov deadline hard? | Treated as hard; scope was cut (risky) only for optional work. |

## Known limitations to disclose in the report

1. `math-verify` uses a 5 s per-call timeout on POSIX; pathological expressions are treated
   as "not equivalent". The number of timeouts is not logged.
2. MATH `dev` (200 questions) is easier than held-out (FirstAns 0.727 vs 0.647). Adaptive
   thresholds selected on dev are therefore validated on a harder split — conservative, but
   it widens the dev/held-out gap.
3. Token cost is an offline proxy: it excludes prefill, attention collection and scoring
   overhead, exactly as the assignment specifies.
4. Task 1 `ρ_k` exists only for OPLoRA-16; the other five conditions store `null` because
   the protected bases were passed only to that run.
5. Task 1 recovery checkpoints at 312/625/937 for Full FT and L2-SP were lost to a
   checkpoint-retention bug; the midpoint is a labelled reconstruction.
6. Task 2 needed two invalidated attempts (held-out leak, then missing features); the
   corrected model is the one reported.
7. Task 2 development curves are NLL points only, not the full every-500-update table.
8. The Task 3 incorrect-early-stop case list (recovered by fixed-32) was not produced.
