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
(start with `INVENTORY.md`). The source data is Sid's public bucket; the derived training/eval files are not re-hosted,
but `build_golden.py`, `recon.py` and the prompt builders there regenerate them.

## Adapters

| folder | trained on | notes |
|---|---|---|
| `sid_full/` | all 11,824 training rows | **stopped at 50%**: 739 of 1,478 planned steps (0.5 epoch), learning rate still mid-schedule, so not a finished run. Same weights as `checkpoint-739/`, which adds optimizer/scheduler/RNG state to resume |
| `lomo740_<type>/` | everything except one data type | leave-one-data-type-out runs, 740 steps; evaluated on the held-out type only |
| `lomo740_all/` | all rows, 740-step schedule run to completion | reference for the hold-out runs; the finished counterpart of `sid_full` (same half epoch of data) |

| `smoke_sid/`, `smoke_resume8/`, `smoke_all/` | tiny subsets | pipeline smoke tests (20-step / resume checks), not for use; kept as the full backup of the Modal volume |

Every folder includes the tokenizer it was trained with (`tokenizer.json` is Muse-Glimmer-30B's, unchanged).
Hold-out folders appear as each run finishes. Per-run status and scores: `INVENTORY.md` §6 in the GitHub repo.

## Recipe

LoRA r=32, α=64, dropout 0, on q/k/v/o/gate/up/down projections of the language model; lr 1e-4 cosine,
bf16, max length 6144, completion-only loss, seed 42, 8×H200. Script: `scripts/lomo_train.py` in the GitHub repo.

## Use

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = AutoModelForCausalLM.from_pretrained("<path to Muse-Glimmer-30B>", dtype="bfloat16")
tok = AutoTokenizer.from_pretrained("<path to Muse-Glimmer-30B>")
model = PeftModel.from_pretrained(base, "likelytobelaura/osteo-glimmer-lora", subfolder="sid_full")
```

Prompts follow `03_baseline_prompts/baseline.jsonl` (see GitHub): the model replies with a JSON object mapping each
missing item's number to its predicted value.

## Caveats

- Trained on one patient's health data, which Sid Sijbrandij has published openly. Research use only; not for
  clinical decisions.
- Derived from Muse-Glimmer-30B, released under Apache 2.0 (copy in `LICENSE`); these adapters are Apache 2.0 too.
