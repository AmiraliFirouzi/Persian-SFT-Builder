# Persian-SFT-Builder

Scripts for building **[Persian-SFT-102K](https://huggingface.co/datasets/amiralifirouzi/Persian-SFT-102K)**, a Persian-focused SFT dataset for Qwen3 and other chat-based LLMs.

The final dataset is published on Hugging Face. This repository contains the **data selection, processing, QC, and publishing pipeline**.

## Dataset

| Config      | Samples | Description                                 |
| ----------- | ------: | ------------------------------------------- |
| `fa`        |  84,612 | Persian SFT data                            |
| `fa_replay` | 101,914 | Persian SFT + 17,302 English replay samples |

The dataset combines instruction following, conversation, reasoning, mathematics, programming, knowledge, rewriting, and generation data.

## Pipeline

```text
Source datasets
      ↓
Selection
      ↓
Validation & normalization
      ↓
Merge
      ↓
Quality checks
      ↓
Publish
      ↓
Hugging Face
```

Main stages:

```text
inventory_* → select_* → merge_stage1 → qc_stage2
                                      ↓
                              patch_metadata
                                      ↓
                                   publish
                                      ↓
                                 upload_hf
```

### Quality Control

The pipeline checks and records:

* Exact duplicates
* Near-duplicate questions
* Potential benchmark contamination
* Samples exceeding 8K Qwen3 tokens

QC issues are stored as flags rather than silently deleting samples.

## Sources

The dataset is built from multiple Persian and multilingual datasets, including:

* Maux-Persian-SFT
* FarsInstruct
* PersianSyntheticQA
* Persian-NoRobots
* Aya
* Persian Alpaca Reasoning
* Persian-Thinking
* Persian-Wikipedia-Instruct
* MauxGPT-SFT
* JumpShift
* Persian-Math-SFT
* PersianGK
* SmolTalk2 English replay

See [`SOURCES.md`](SOURCES.md) for the full breakdown.

## Reproducibility

Selection scripts use a fixed `SEED=42`.

To rebuild the dataset:

```text
1. Run inventory scripts
2. Run selection scripts
3. Run merge_stage1
4. Run qc_stage2
5. Run patch_metadata
6. Run publish
7. Run upload_hf
```

The scripts are primarily designed for Colab/Kaggle-style execution.

## Repository

```text
.
├── inventory_*.py
├── select_*.py
├── merge_stage1.py
├── qc_stage2.py
├── patch_metadata.py
├── publish.py
├── upload_hf.py
├── SOURCES.md
├── requirements.txt
└── README.md
```

Generated datasets such as `.jsonl`, `.parquet`, and `master_*.jsonl` are intentionally excluded from Git.

**Never commit Hugging Face tokens or other credentials.**

## License

This repository contains the dataset construction scripts.

The final dataset is composed of multiple upstream datasets with their own licenses and terms. Check the original source licenses before using or redistributing the data.

## Links

* **Dataset:** [Persian-SFT-102K](https://huggingface.co/datasets/amiralifirouzi/Persian-SFT-102K)
* **Builder:** Amirali Firouzi

