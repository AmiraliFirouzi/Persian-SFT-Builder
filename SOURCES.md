# Dataset Sources

Persian-SFT-102K is built from multiple upstream datasets.

| Source                                    | Selected | Purpose                               |
| ----------------------------------------- | -------: | ------------------------------------- |
| `xmanii/Maux-Persian-SFT-30k`             |   24,967 | Conversation, reasoning, tool calling |
| `ParsiAI/FarsInstruct`                    |   20,000 | Multi-task instruction following      |
| `ParsBench/PersianSyntheticQA`            |   15,000 | QA across 50 domains                  |
| `ParsBench/Persian-NoRobots`              |    9,500 | Generation, rewriting, chat           |
| `CohereForAI/aya_dataset`                 |    1,505 | Human-annotated Persian data          |
| `Persian Alpaca Reasoning`                |    2,124 | Persian reasoning                     |
| `Persian-Thinking`                        |      983 | Persian thinking/reasoning            |
| `HuggingFaceTB/smoltalk2`                 |   17,302 | English replay                        |
| `Jamalianpour/persian-wikipedia-instruct` |    4,000 | Wikipedia-based instruction data      |
| `xmanii/maux-gpt-sft-20k`                 |    2,500 | Casual conversation                   |
| `jumplander/JumpShift`                    |    1,000 | Persian programming                   |
| `xmanii/Persian-Math-SFT`                 |    1,000 | Mathematical reasoning                |
| `mshojaei77/persian-gk`                   |    2,000 | General knowledge                     |

### Selection Notes

* Sampling uses `SEED=42`.
* Source-specific quotas are used instead of oversampling.
* Test splits are not included.
* Persian text is normalized for deduplication without modifying the original text.
* Source provenance is preserved in dataset metadata.

### Output

The final dataset contains:

```text
fa          → 84,612 samples
fa_replay   → 101,914 samples
```

The `fa_replay` configuration adds 17,302 English replay samples to the Persian training data.

For the complete dataset and dataset card, see:

https://huggingface.co/datasets/amiralifirouzi/Persian-SFT-102K
