# biohack — osteosarcoma multi-omic imputation

Fine-tuning Muse-Glimmer-30B to fill a 20% random mask of Sid Sijbrandij's public osteosarcoma dataset
(`s3://sid-sijbrandij-osteosarc-dataset`). **Start with [INVENTORY.md](INVENTORY.md)** — every dataset, label set,
baseline, fine-tune and hold-out experiment, with sizes and sha256.

This repo holds code, reports, scores and the inventory. The data itself (training set, prompts, gold labels,
model responses, raw bucket files, adapter weights) is git-ignored: it lives in `~/biohack` on Laura's Mac and on the
Modal volume `osteo-data`; `inventory.json` has the checksum of each file.

**Model weights:** every LoRA adapter (`sid_full`, stopped at 50%, + each hold-out run) is on Hugging Face, private:
[huggingface.co/likelytobelaura/osteo-glimmer-lora](https://huggingface.co/likelytobelaura/osteo-glimmer-lora),
one folder per run (`sid_full/`, `lomo740_<type>/`). Upload new ones with `~/biohack/.venv/bin/python hf_upload.py`.

| folder | contents |
|---|---|
| `01_sid_full_dataset/` | SFT training set (11,824 rows) + raw bucket files |
| `02_golden_labels/` | gold values for all 24,634 masked targets |
| `03_baseline_prompts/` | the 15,964 eval prompts |
| `04_baseline_responses/` | Muse-Glimmer-30B zero-shot answers |
| `05_baseline_report/` | `score.py` report on the zero-shot baseline |
| `06_heldout_experiments/` | leave-one-data-type-out runs + their per-type baselines |
| `07_finetune_sid_full/` | `sid_full` LoRA adapter (all data, stopped at step 739/1478): answers, scores, configs/logs, dashboard source |
| `scripts/` | build, train, eval and scoring scripts |

Refresh the inventory: `python3 inventory.py` (pulls finished hold-out runs from Modal).
