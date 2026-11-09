# ATML PA 3 — Task 2 technical record

**Task:** Vision-language grounding with SigLIP → connector → Qwen3-0.6B (PDF pp. 6–10)
**Status:** interface, loss, sampling, grounding-generation and scoring layers implemented and unit-tested; data build, curriculum training and evaluation **not yet executed** (awaiting GPU and the COCO/VQAv2 download).
**Owner of interpretation:** the student.

---

## 1. Implementation map

| Requirement | Module | Notes |
|---|---|---|
| Connector 768→1024→1024, GELU | `src/task2/connector.py` | FP32; 196×768 → 196×1024 |
| Magnitude calibration, training data only | `connector.py::calibrate_magnitude` | rule: `scale = rms(text) / rms(connector output)`; rule and both RMS values recorded |
| Visual tokens inside the user message, before the question | `inputs.py::build_inputs` | `<|im_start|>user\n[196 visual]{question}<|im_end|>\n<|im_start|>assistant\n` |
| Masks and positions | same | visual + prompt masked; answer + EOT supervised; positions rebuilt per sequence |
| No-image condition | same | removes exactly 196 positions and rebuilds masks/positions |
| Example-weighted loss | `loss_sampler.py::example_weighted_loss` | per-example token mean, then mean over the effective batch; examples with no supervised tokens excluded |
| Answer NLL excludes termination | `reference_answer_nll` | EOT masked |
| Two examples per draw, pairs preserved | `MixtureSampler.draw` | question draws return a complete pair; caption draws return two independent captions |
| Stage mixtures | `STAGE2_MIX`, `STAGE3_MIX` | 50/25/25 and 70/10/10/10; probabilities only, no quotas |
| Resumable sampler | `save_state` / `load_state` | RNG bit-generator state + realized counts |
| Presence / counting questions | `grounding.py` | exact PA templates |
| Pair construction | `build_presence_pairs`, `build_counting_pairs` | presence needs an image without the category; counting needs distinct counts in 1–4 |
| VQA soft consensus | `vqa_soft_consensus` | official leave-one-out formula; exactly ten references |
| Pair accuracy | `pair_accuracy` | both predictions must match their own original answer |
| VQAv2 agreement filter | `data.py::build_candidate_pairs` | ≥8/10 consensus agreement **on each image**, different consensus answers, same question text |
| Image-disjoint splits | `data.py::split_vqa`, `assert_image_disjoint` | dev first, test excluding dev, train excluding both |
| Caption selection | `data.py::caption_selection` | deterministic, 3 captions per training image, 1 per dev image |

## 2. Verification (CPU)

`tests/unit/test_task2_core.py`, `test_task2_data.py` — 33 tests, all passing.

Hand-checked properties:

* connector output is 196×1024 FP32; the calibration scale reproduces the text RMS;
* the visual span is exactly 196 positions and is never supervised; the no-image prefix
  is exactly 196 positions shorter;
* answer-NLL mode masks the termination token;
* example-weighted loss differs from token-weighted loss and matches a hand computation;
* every sampler draw returns exactly two examples; question pairs keep both images of the
  same pair; caption draws are independent; the sampler is deterministic and resumable;
  realized fractions approach the stage mixture without quotas;
* VQA scoring at boundary cases: 10/10 → 1.0, 0/10 → 0.0, and the leave-one-out quirk
  (3 of 10 → 0.9, not 1.0), which a naive `min(count/3, 1)` gets wrong;
* VQAv2 pairs require 8/10 agreement on both images and differing consensus answers.

## 3. Not yet measured

Feature caching, the 4,500-update curriculum, development curves, the four-condition
final evaluation and the pair bootstrap. No Task 2 number may be reported until those
exist.

## 4. Open decisions

1. **COCO/VQAv2 acquisition** — selective per-image HTTP download (~4 GB for the needed
   images) plus the official annotation zips; this must be verified on the GPU box.
2. **SigLIP final layer norm** — must be applied exactly once; to be asserted in the
   feature-cache test (shape 196×768, values finite, deterministic across two runs).
3. **Optional 49-token run** — planned cut unless time and budget allow; it is explicitly
   optional in the assignment.
