# ATML PA 3 — Task 2 technical record

**Task:** Vision-language grounding — SigLIP encoder → connector → Qwen3-0.6B, three-stage curriculum (PDF pp. 6–10)
**Status:** trained end to end and evaluated once on the frozen final model. Results below are measured, including one degenerate outcome that must be reported honestly.
**Owner of interpretation:** the student.

---

## 1. Configuration as executed

| Item | Value |
|---|---|
| Backbone | `Qwen/Qwen3-0.6B`, original checkpoint, FP16, frozen |
| Vision encoder | `google/siglip-base-patch16-224`, frozen, 196×768 in 14×14 order, final layer norm applied once by the tower |
| Connector | `Linear(768,1024) → GELU → Linear(1024,1024)`, FP32, 1,837,056 parameters |
| LoRA | r=16, α=32, dropout 0.05, targets **q/k/v/o only**, 4,587,520 parameters |
| Calibration | RMS match against text embeddings, training data only: text rms 0.0296 / raw visual rms 0.5116 → **scale 0.05786** |
| Batch | micro 8 × accumulation 2 = effective 16; each draw contributes two examples |
| Optimizer | AdamW, weight decay 0.01 (no decay on the connector), grad clip 1.0 |
| Schedule | per-stage 5% warm-up then cosine to 10% of peak |
| Seed | 6304 |
| Precision | FP16 autocast, FP32 connector and adapters, dynamic scaling |

### Curriculum actually run

| Stage | Updates | Time | Connector LR | LoRA LR | Trainable |
|---|---:|---:|---:|---:|---|
| 1 caption alignment | **2,250** | 8,044.8 s | 1e-3 | — | connector only (LoRA frozen, 0 trainable) |
| 2 visual instruction | **1,500** | 5,688.0 s | 1e-4 | 2e-5 | connector + 4,587,520 LoRA |
| 3 natural VQA | **750** | 2,896.1 s | 5e-5 | 1e-5 | connector + same adapters |
| **Total** | **4,500** | **4.62 h** | | | |

Stage 1 makes two complete passes over the 18,000 caption examples through a deterministic
cycler (36,000 presentations / 16 = 2,250 updates); Stages 2–3 use fresh mixture
samplers with the prescribed probabilities. Development NLL was recorded at 6, 4 and 3
points respectively (start, every 500 updates, stage end).

## 2. Data

Rebuilt after a leak was found and corrected (see §6):

| Split | Content | Images |
|---|---|---|
| train | 6,000 caption images × 3 captions = 18,000; 88 grounding pairs; 1,536 VQAv2 pairs (640/256/640) | 9,328 |
| dev | 96 caption images; 24 grounding pairs; 32 VQAv2 pairs (12/8/12) | 208 |
| test | 64 grounding pairs; 96 VQAv2 pairs (36/24/36) | 320 |

Train images come from COCO **train2014**; dev and test from val2014. Disjointness is
asserted at build time and verified: `train ∩ dev = ∅`, `train ∩ test = ∅`, `dev ∩ test = ∅`.
Every referenced image has a cached SigLIP feature (9,003 images, FP16, 2.6 GB).

## 3. Final evaluation (frozen Stage 3 model)

Four image conditions, question and original references held fixed.

### Natural VQAv2 (96 pairs / 192 examples)

| Condition | Accuracy | Empty |
|---|---:|---:|
| Correct image | **0.5422** | 0.0000 |
| Complementary partner | 0.2375 | 0.0000 |
| Mismatched (fixed unrelated) | 0.2948 | 0.0000 |
| No image | 0.3245 | 0.0000 |

Pair accuracy 0.3021 · identical-answer rate 0.3646
Reference NLL: correct 1.3432, complementary 2.5181, no-image 2.4174
**Swap penalty ΔNLL = +1.1749** (the correct image makes the reference substantially more likely)
Correct-image accuracy advantage: **+0.3047**, pair-bootstrap 95% CI **[0.1943, 0.4109]** (4,000 resamples)

### Annotation-derived grounding (64 pairs / 128 examples)

| Condition | Accuracy |
|---|---:|
| Correct | 0.4609 |
| Complementary | 0.4609 |
| Mismatched | 0.4609 |
| No image | 0.4609 |

Pair accuracy 0.0000 · identical-answer rate 1.0000 · swap penalty 0.0000 ·
bootstrap advantage 0.0000 [0.0000, 0.0000]

**This is a degenerate result, not a scoring artefact.** The model emits one constant
answer for every grounding prompt, so the image cannot change the prediction. The most
likely cause is the training pool: the rebuild produced only **44 presence + 44 counting
pairs**, and Stage 2 draws ~6,000 presence samples over those 44 pairs (≈136 repetitions
per pair). With that little variety the connector learns the label prior (always "yes")
rather than the visual distinction. The PA warns about exactly this failure mode:
a model can exploit language priors instead of grounding.

It should be read alongside the natural-VQA row, where the *same* model shows clear image
sensitivity (0.54 vs 0.24). Grounding collapsed only for the annotation-derived questions.

## 4. Resource use

| Item | Value |
|---|---|
| Training wall clock | 4.62 h |
| Final evaluation | 160 s |
| Trainable parameters | connector 1.84 M + LoRA 4.59 M |
| Checkpoint size | 24.58 MB per stage (connector + adapters) |
| GPU | RTX 4090, ~4.6 GB during training |

## 5. Saved artefacts

```
outputs/task_02/
  training.json                     per-stage updates, times, trainable sets, dev points
  eval.json                         four conditions, pair diagnostics, NLL, bootstrap CIs
  checkpoints/stage1_caption_weights.pt      connector (24.6 MB)
  checkpoints/stage2_instruction_weights.pt  connector + LoRA (24.6 MB)
  checkpoints/stage3_natural_weights.pt      connector + LoRA, FINAL MODEL (24.6 MB)
  checkpoints/training_summary.json
```

Resumable checkpoints were written every 250 updates on the training instance; only the
stage-end weights were transferred because the link runs at ~20–40 KB/s.

## 6. Corrections made before this run (disclosed)

An earlier Task 2 attempt was stopped and its outputs quarantined in
`outputs/task_02_invalid/` with an `INVALID.md`. That run trained on `grounding_dev +
grounding_test` — the held-out evaluation sets — and additionally had: Stage 1 using the
Stage 2 mixture, one sampler reused across stages, the reference answer inside the
generation prompt, `consensus_a/b` read from records that carry `answer_a/b`, exact match
instead of ten-reference soft consensus, and no development schedule, image conditions or
bootstrap. All were fixed before this run; the leak fix is covered by a unit test that
asserts split-image disjointness.

## 7. Verification

`tests/unit/test_task2_fixes.py` (12 passing, 3 data tests executed on the instance):
caption cycler makes exactly two complete passes; Stage 1 is caption-only with LoRA
disabled; sampler mixtures match the assignment; splits are image-disjoint; training
grounding records come from the train split; grounding records carry `answer_a/b` and VQA
records carry ten answers per image; generation prompts carry no reference; soft consensus
uses all ten references; a checkpoint restores connector, adapters, optimizer, scaler, RNG
and sampler position.

## 8. Limitations to state in the report

1. Grounding accuracy is at chance because the model collapsed to a constant answer; the
   training pool was too small (88 pairs). Rebuilding with a larger generated pool is the
   obvious remedy and is **not** done here.
2. Development curves were recorded as NLL points (6/4/3 per stage), not as a full
   every-500-update table with generated accuracy, because of the remaining budget.
3. The 49-token compression experiment was not run (explicitly optional).
4. Only stage-end weights were transferred; the 250-update recovery checkpoints remain on
   the rented instance and will be lost when it is destroyed.
