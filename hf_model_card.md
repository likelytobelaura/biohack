---
license: apache-2.0
library_name: peft
tags:
- lora
- osteosarcoma
- multi-omics
- imputation
---

# osteo-glimmer-lora

LoRA adapters (low-rank deltas, ~840 MB each; not standalone models) for **Muse-Glimmer-30B** that fill in held-out values in Sid Sijbrandij's public osteosarcoma
multi-omic dataset (`s3://sid-sijbrandij-osteosarc-dataset`): scRNA-seq, CITE-seq, bulk RNA, TCR, DIA and Olink
proteomics, multiplex IF, Xenium, somatic WGS, metagenomics, flow cytometry, HLA and gene fusions.

**Code, data inventory, baselines and scores: [github.com/likelytobelaura/biohack](https://github.com/likelytobelaura/biohack)**
(start with `INVENTORY.md`). The training set and eval prompts are there too, in `11_modal_osteo_data/train/sid_full.jsonl`
and `11_modal_osteo_data/prompts/baseline.jsonl`, along with every model's answers and the scores. The full gold labels
are not re-hosted; `02_golden_labels/build_golden.py` and `scripts/recon.py` rebuild them from Sid's bucket.

## Adapters

| folder | trained on | notes |
|---|---|---|
| `sid_full/` | all 11,824 training rows | **stopped at 50%**: 739 of 1,478 planned steps (0.5 epoch), learning rate still mid-schedule, so not a finished run. Same weights as `checkpoint-739/`, which adds optimizer/scheduler/RNG state to resume |
| `lomo740_flow/`, `lomo740_rna/`, `lomo740_tcr/`, `lomo740_xenium/` | everything except that data type | leave-one-data-type-out runs, 740-step schedule run to completion; scored on the held-out type, and on all prompts for off-target effects |
| `lomo740_all/` | all rows, 740-step schedule run to completion | reference for the hold-out runs; the finished counterpart of `sid_full` (same half epoch of data) |
| `smoke_sid/` | 64-row test set (`train/smoke_sid.jsonl`) | 8 steps (1 epoch); pipeline test, not for use |
| `smoke_resume8/` | same 64 rows | resume test: 8 steps planned, `checkpoint-4/` saved at step 4; not for use |
| `smoke_all/` | all rows | 20 steps (0.01 epoch); hold-out-script test, not for use |

Every folder includes the tokenizer it was trained with (`tokenizer.json` is Muse-Glimmer-30B's, unchanged).
`sid_full/checkpoint-739/` and `smoke_resume8/checkpoint-4/` add optimizer, scheduler and RNG state for resuming.
Per-run status and scores: `INVENTORY.md` §6 in the GitHub repo (hold-out scores in `06_heldout_experiments/lomo_scores.json`,
off-target scores in `06_heldout_experiments/osteo_results_lomo/offtarget_scores.json`).

## Recipe

All adapters: LoRA r=32, α=64, dropout 0, on q/k/v/o/gate/up/down projections of the language model
(each folder's `adapter_config.json`). `lomo740_*` runs: lr 1e-4 cosine over 740 steps, bf16, max length 6144,
completion-only loss, seed 42, 8×H200 (`scripts/lomo_train.py` in the GitHub repo). `sid_full` was planned as one epoch
(1,478 steps) and its launcher is not in the repo; its logged schedule is in `sid_full/checkpoint-739/trainer_state.json`.

## Use

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = AutoModelForCausalLM.from_pretrained("<path to Muse-Glimmer-30B>", dtype="bfloat16")
tok = AutoTokenizer.from_pretrained("<path to Muse-Glimmer-30B>")
model = PeftModel.from_pretrained(base, "likelytobelaura/osteo-glimmer-lora", subfolder="sid_full")
```

Prompts follow `11_modal_osteo_data/prompts/baseline.jsonl` in the GitHub repo: the model replies with a JSON object
mapping each missing item's number to its predicted value.

## Caveats

- Trained on one patient's health data, which Sid Sijbrandij has published openly. Research use only; not for
  clinical decisions.
- Derived from Muse-Glimmer-30B, released under Apache 2.0 (copy in `LICENSE`); these adapters are Apache 2.0 too.
