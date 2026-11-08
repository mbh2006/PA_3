# Task 3 dataset is available locally

Use `ATML-PA3-Data/` beside this document. The dataset was downloaded from the official Google Drive folder linked on page 11 of the assignment:

[Official ATML-PA3-Data](https://drive.google.com/drive/folders/1utX-Gfi9vvaoi1mi0pzNpDGL1o1fAh23?usp=sharing)

## Verified contents

- `README.MD` — instructor-provided data schema and provenance.
- `qwen25-math/generations.parquet`
- `qwen25-math/probability_metrics.parquet`
- `qwen25vl-mathvista/generations.parquet`
- `qwen25vl-mathvista/probability_metrics.parquet`
- `qwen25vl-mathvista/attention_metrics.parquet`
- `download_manifest.json` — local SHA-256 hashes, sizes, source URLs, and structural validation results.

All five Parquet files have 32,000 rows. Each dataset contains 1,000 questions with rollout IDs 0–31. File sizes match the Drive listings; join keys are unique and match across related tables; token-array lengths match the retained generation lengths. Retained-token totals, capped-rollout counts, and missing-extraction counts match the supplied README. These are download/structure checks, not completed Task 3 experiments.

## Reader compatibility

The files identify their writer as PyArrow 24.0.0. The local PyArrow 19.0.0 reader failed with `Repetition level histogram size mismatch`; the same downloaded files passed the checks above using PyArrow 24.0.0. Use a compatible reader, such as the tested `pyarrow==24.0.0`, in the intended execution environment. The local validation dependency was isolated in a temporary workspace directory; no existing Python environment was upgraded.

## Message for OpenCode

> The official Task 3 data is now in ATML-PA3-Data/ in this PA3 folder. Read DATASET_READY.md, the dataset README.MD, and download_manifest.json. Use a compatible Parquet reader; PyArrow 24.0.0 was verified. Use these local files instead of the Drive link, verify them in your environment, attach them privately through my existing Kaggle configuration, and continue the Task 3 tests and prescribed experiments. Keep raw data out of the public code repository.
