# PA3 handoff for the existing OpenCode Build session

Continue the `Assignment completion and file review` session in the PA3 project. OpenCode owns the single GPU job queue on the existing Vast instance. Codex checked the server and local artifacts and is handing off the findings below.

## Verified at the 11 November 2026, 12:01 a.m. PKT check

- Vast had one OPLoRA-16 evaluation worker and its `lm_eval` child active. The existing run completed all 1,250 training updates and saved `adapter_final.pt`, the OPLoRA bases, and its recovery checkpoints.
- The earlier OPLoRA evaluation failed at 7:21 p.m. PKT on 10 November with a CPU/GPU projection basis mismatch. The next evaluation attempt began at 11:57 p.m. PKT. That gap accounted for about 4 hours 36 minutes of elapsed time.
- Six Task 1 reports were present, covering the baseline, reconstructed midpoint, Full FT, L2-SP, LoRA-4, and LoRA-16. OPLoRA was the only missing Task 1 report. The last verification recorded 144 qualitative responses and 36 annotation rows.
- Task 2 and Task 3 live had no result directories on Vast at this check. Task 2 CPU dataset preparation was reported complete by OpenCode. Task 3 offline tables are saved locally.

## Continue from the current state

1. Check the active OPLoRA worker and `logs/repair4_oplora.log`. Let that worker finish, or resume evaluation using the saved adapter and bases if it exits. Verify seven reports, 168 responses, and 42 annotation rows. Copy the table, reports, raw metrics, diagnostics, and qualitative files back first.
2. Start Task 2 as soon as the GPU is free. Locate the prepared data, time the feature cache and a real training smoke batch, then run the required curriculum and evaluations. Save checkpoints and results before optional diagnostics.
3. Run Task 3 live after Task 2, using the saved 50-question subset and the prescribed strategies. Time the first few questions to update the estimate. Save partial results so a restart can resume completed questions.
4. Track the remaining Vast balance against the existing spending ceiling, including transfer charges. Keep one GPU worker active at a time. If a postprocessing step fails, retain its training outputs, record the failure, and continue with independent work where the budget allows. Obtain user approval for any top-up or additional rental.
5. Sync small result files and separate task records to the PA3 folder as they finish. Check local copies before advising the user to stop or destroy the instance. The student's report prose and interpretation remain their own.
