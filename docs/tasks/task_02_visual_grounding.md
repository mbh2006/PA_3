# ATML PA 3 — Task 2 technical record

**Task:** Vision-language grounding — SigLIP → connector → Qwen3-0.6B, three-stage curriculum (PDF pp. 6–10)
**Status:** trained and evaluated on a corrected pipeline. Two earlier attempts were invalidated and are quarantined (see §6).
**Owner of interpretation:** the student.

---

## 1. Configuration as executed

| Item | Value |
|---|---|
| Backbone | `Qwen/Qwen3-0.6B` original checkpoint, FP16, frozen |
| Vision encoder | `google/siglip-base-patch16-224`, frozen, 196×768 14×14, final layer norm applied once |
| Connector | 768→1024 GELU→1024, FP32, 1,837,056 parameters |
| LoRA | r=16, α=32, dropout 0.05, **q/k/v/o only**, 4,587,520 parameters |
| Calibration | RMS match on training data: text rms 0.0296 / visual rms 0.5150 → **scale 0.05748** |
| Batch | micro 8 × accumulation 2 = 16; each draw contributes two examples |
| Optimizer | AdamW, weight decay 0.01 (connector excluded), clip 1.0 |
| Schedule | per-stage 5% warm-up, cosine to 10% of peak |
| Seed | 6304 |
| Precision | FP16 autocast, FP32 connector/adapters, dynamic scaling |

### Curriculum actually run

| Stage | Updates | Time | Connector LR | LoRA LR | Trainable |
|---|---:|---:|---:|---:|---|
| 1 caption alignment | 2,250 | 8,044.8 s | 1e-3 | — | connector only (LoRA frozen, 0 trainable) |
| 2 visual instruction | 1,500 | 5,761.2 s | 1e-4 | 2e-5 | connector + 4,587,520 |
| 3 natural VQA | 750 | 2,887.7 s | 5e-5 | 1e-5 | connector + same adapters |
| **Total** | **4,500** | | | | |

Stage 1 makes two complete passes over the 18,000 captions via a deterministic cycler
(36,000 / 16 = 2,250). Stages 2–3 use fresh mixture samplers at the prescribed
probabilities. Development NLL was recorded at 4 and 3 points in Stages 2 and 3.

Stage 1 was reused from the first (otherwise invalid) attempt: it trains only the
connector on captions and never touches grounding data, so it is unaffected by the leak
that invalidated Stages 2–3. Its weights were verified (connector + 224 adapter tensors)
before reuse.

## 2. Data

| Split | Content | Images |
|---|---|---|
| train | 6,000 caption images × 3 = 18,000; **64 presence + 64 counting pairs**; 1,536 VQAv2 (640/256/640) | 9,328 |
| dev | 96 caption images; 24 grounding pairs; 32 VQAv2 (12/8/12) | 208 |
| test | 64 grounding pairs; 96 VQAv2 (36/24/36) | 320 |

Training pools come from COCO **train2014**; dev/test from val2014. Disjointness is
asserted at build time (`train∩dev = train∩test = dev∩test = ∅`) and covered by a unit
test. **All 9,856 referenced images have cached SigLIP features** — the loader now raises
if any is missing, because silently dropping an image turns a "with image" condition into
a no-image condition.

## 3. Final evaluation (frozen Stage 3 model)

Question and original references held fixed across conditions.

### Natural VQAv2 — 96 pairs / 192 examples

| Condition | Accuracy |
|---|---:|
| Correct image | **0.5333** |
| Complementary partner | 0.2568 |
| Mismatched (fixed unrelated) | 0.3125 |
| No image | **0.0000** |

Pair accuracy 0.2708 · identical-answer rate 0.3438
Reference NLL: correct 1.3427, complementary 2.8072, no-image 21.4715
**Swap penalty ΔNLL = +1.4645**; correct-image accuracy advantage **+0.2766**, pair
bootstrap 95% CI **[0.1693, 0.3797]** (4,000 complete-pair resamples)

### Annotation-derived grounding — 64 pairs / 128 examples

| Condition | Accuracy |
|---|---:|
| Correct image | **0.4531** |
| Complementary partner | 0.2891 |
| Mismatched | 0.3672 |
| No image | **0.0000** |

Pair accuracy 0.2344 · identical-answer rate 0.4375
Reference NLL: correct 2.8891, complementary 3.7098, no-image 26.1323
**Swap penalty ΔNLL = +0.8207**; advantage **+0.1641**, CI **[0.0312, 0.3047]**

Both suites now show the intended pattern: the correct image beats the complementary
swap, the confidence intervals exclude zero, and removing the image destroys accuracy
entirely. The no-image NLL is very large because that prompt is 196 tokens shorter and
the reference becomes almost unrepresentable — worth a sentence in the report, not a
cause for alarm.

## 4. Resource use

| Item | Value |
|---|---|
| Training wall clock | 4.62 h total (Stage 1 8,044.8 s + Stage 2 5,761.2 s + Stage 3 2,887.7 s) |
| Final evaluation | 334 s |
| Trainable parameters | connector 1.84 M + LoRA 4.59 M |
| Checkpoint size | 24.58 MB per stage |
| GPU | RTX 4090, ~4.6 GB during training |

## 5. Saved artefacts (verified locally)

```
outputs/task_02/
  eval.json                                  four conditions, pair diagnostics, NLL, bootstrap
  training.json                              per-stage updates, times, trainable sets, dev points
  checkpoints/stage2_instruction_weights.pt  connector + LoRA (24.58 MB)
  checkpoints/stage3_natural_weights.pt      FINAL MODEL (24.58 MB)
  checkpoints/training_summary.json
```

Published copies under `results/task_02/`:

```
  eval.json, training.json                   as above
  tables/condition_table.{csv,md}            four conditions, pair diagnostics, bootstrap CI
  tables/by_kind.{csv,md}                    subtype accuracy (presence, counting, yes/no, number, other)
  tables/stages.{csv,md}                     updates, seconds, trainable sets, realized mixtures
  figures/conditions_accuracy.{png,svg}      four conditions, both suites
  figures/nll_by_condition.{png,svg}         reference NLL per condition
  figures/by_kind.{png,svg}                  subtype accuracy, correct vs swapped
  figures/dev_nll_curves.{png,svg}           dev NLL across Stages 2-3 with the stage boundary
  figures/train_loss.{png,svg}               Stages 2-3 training loss
```

Regenerate with `python scripts/make_task2_figures.py` (no GPU, reads the two JSON files).

Resumable checkpoints were written every 250 updates on the instance; only stage-end
weights were transferred (link runs at 20–40 KB/s).

## 6. Invalidated attempts (disclosed)

* `outputs/task_02_invalid/` — **first attempt.** Trained on `grounding_dev + grounding_test`
  (held-out leak), Stage 1 used the Stage 2 mixture, one sampler reused across stages, the
  reference answer sat inside the generation prompt, `consensus_a/b` was read from records
  that carry `answer_a/b`, exact match replaced soft consensus, and there was no dev
  schedule, image conditions or bootstrap.
* `outputs/task_02_invalid2/` — **second attempt.** The builder was fixed but the deployed
  `build_pools` still read `grounding_dev + grounding_test`, and 853 required images were
  never downloaded, so every "with image" condition silently ran as no-image. That is why
  all four grounding conditions returned the identical 0.4609 — not, as first thought, a
  language-prior collapse.

Both are quarantined with `INVALID.md` and are not used anywhere.

## 7. Verification

`tests/unit/test_task2_fixes.py`: caption cycler makes exactly two complete passes; Stage 1
is caption-only with LoRA disabled; mixtures match the assignment; splits are
image-disjoint; training grounding records come from the train split; schemas carry
`answer_a/b` and ten answers per image; generation prompts carry no reference; soft
consensus uses all ten; a checkpoint restores connector, adapters, optimizer, scaler, RNG
and sampler position. The loader additionally hard-fails on any missing feature.

## 8. Limitations

1. The grounding pool is small (64 + 64 pairs); the assignment allows reusable pools, but a
   larger generated pool would likely improve grounding accuracy.
2. Development curves are NLL points (4 + 3), not a full every-500-update table with
   generated accuracy, for budget reasons. They are plotted in
   `figures/dev_nll_curves.*`; Stage 1 has no dev points.
3. The 49-token compression experiment was not run (optional).
4. Only stage-end weights were transferred; the 250-update checkpoints remain on the
   instance and will be lost when it is destroyed.
5. No aligned per-example sheet (question, references, prediction under each condition) was
   exported: `eval.json` stores aggregates only. The four-condition table and subtype tables
   are the available evidence for qualitative discussion.
