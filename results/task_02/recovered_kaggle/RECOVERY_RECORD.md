# Task 2 Kaggle evaluation record

- Kaggle notebook version 2, private Tesla T4 session. Data preparation and evaluation took 393.18 seconds, including 102.05 seconds for annotation download and split construction.
- Evaluation used the saved corrected Stage 3 weights and training calibration scale, with greedy batch-8 generation capped at 16 tokens. Batch-8 predictions matched single-example predictions on all 32 smoke comparisons.
- The corrected v2 builder reconstructed the splits. The original corrected-run split manifest was unavailable for direct ID comparison. See `selection_ids.json`, `split_images.json`, and `provenance.json` for the reconstructed IDs and checksums.
- `grounding_predictions.json` and `natural_predictions.json` preserve paired questions, references, image IDs, generated token IDs, predictions, and scores for all four conditions. COCO images can be identified as `val2014/COCO_val2014_{image_id:012d}.jpg`.
- `condition_table.csv` and `metrics.json` contain suite and question-type metrics. `metrics.json` includes 4,000-resample complete-pair intervals and paired correct-image advantages. Pair accuracy uses both original canonical answers.
- `caption_dev_endpoints.json` records example-weighted caption loss, including end-of-turn, for 96 examples at update 3,750 (2.47265625) and 4,500 (2.521484375). These are two newly measured checkpoints. The full historical caption-loss series requires the missing intermediate checkpoints.
- Grounding accuracies match the original aggregate file in all four conditions. Natural VQA accuracy differs by +0.52 percentage points for correct images and −0.52 for complementary images. Use the table and predictions from the same run.
- Local verification covered checkpoint hashes, disjoint split IDs, all 1,280 predictions, 28 condition rows, and both caption endpoints. `task2_recover_evidence.py` and `task2_recovery_kaggle.py` identify the recovery code used with the source revision in `provenance.json`.


## Replaying the recovery

Use the original saved Stage 2 and Stage 3 checkpoint files and the accompanying
`recovery_config.json` in a separate weights directory. The checkpoint hashes
are recorded in the config. Prepare the public annotations and run evaluation
on CUDA with Transformers 4.57.1.

```bash
export PYTHONPATH=src
export ATML_TASK2_DATA=task2_data/recovery
python scripts/task2_build_v2.py --download-annotations
python scripts/task2_recover_evidence.py /path/to/weights /path/to/new-evidence
python scripts/task2_verify_recovery.py /path/to/new-evidence /path/to/weights
```

The Kaggle entry point is `scripts/task2_recovery_kaggle.py`. Its input bundle
uses `recover.py` at the root of `task2_recovery_source.zip`, alongside
`src/` and the two Task 2 builders under `scripts/`. Attach the weights
and config alongside the source bundle using a private input.

The Natural VQA prediction records include a selected subset of question and
reference text from [VQAv2](https://visualqa.org/download.html), credited to the
VQA Consortium under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The evaluation pairs were selected for this assignment and augmented with model
predictions, image condition IDs, and scores. Image IDs refer to
[COCO 2014](https://cocodataset.org/#download).
