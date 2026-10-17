# biohack — osteosarcoma multi-omic imputation

Fine-tuning Muse-Glimmer-30B to fill a 20% random mask of Sid Sijbrandij's public osteosarcoma dataset
(`s3://sid-sijbrandij-osteosarc-dataset`). **Start with [INVENTORY.md](INVENTORY.md)** — every dataset, label set,
baseline, fine-tune and hold-out experiment, with sizes and sha256.

This repo holds code, reports, scores and the inventory (score files include per-target true values for plotted points).
The bulk data (training set, prompts, gold labels, model responses, raw bucket files, adapter weights) is git-ignored: it lives in `~/biohack` on Laura's Mac and on the
Modal volume `osteo-data`; `inventory.json` has the checksum of each file.

**Model weights:** every LoRA adapter (`sid_full`, stopped at 50%, + each hold-out run) is on Hugging Face:
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
| `08_other_baselines/` | Muse Spark 1.3 / 1.2 zero-shot (Meta Model API) and the context-free Glimmer run (NVIDIA-hosted) |
| `09_classical_ml/` | classical imputation (22 methods), cross-data-type prediction, classical hold-out settings, ET-SoC-1 card run |
| `10_esperanto_et_soc1/` | AI Foundry aifoundry3 lab data (ET-SoC-1 card) |
| `11_modal_osteo_data/` | mirror of the Modal volume `osteo-data` (text files) |
| `12_dashboard/` | dashboard build: `build.py` scores every run with the shared scorer and writes `data/*.json`; `index.html` is the page |
| `scripts/` | build, train, eval and scoring scripts |

## Experiments

Every run is scored by `score_compare.py` on the same 24,634 targets of the **20% random mask** (not a time split).

| experiment | question | write-up / data |
|---|---|---|
| Glimmer zero-shot | how well does an untrained 30B model fill gaps from the entity's other values? | `05_baseline_report/REPORT.md`, `04_baseline_responses/` |
| `sid_full` LoRA | does fine-tuning on the visible data help? | `07_finetune_sid_full/` |
| Leave-one-type-out LoRAs (`lomo740_<type>`, `lomo740_all`) | how much of the gain survives when a type is removed from training, and what does removing it do to the other types? | `06_heldout_experiments/` (INVENTORY §6) |
| Context-free Glimmer | can the model predict a type with no observed values of it? | `08_other_baselines/nim_noctx/` |
| Muse Spark 1.3 / 1.2 | does a larger hosted model know more, without fine-tuning? | `08_other_baselines/muse_spark/` |
| Classical imputation | how far do data-only methods get? | [`09_classical_ml/RESULTS.md`](09_classical_ml/RESULTS.md), `preds/` |
| Cross-data-type prediction | which data types carry information about which others (shared genes, blood draws)? | `09_classical_ml/RESULTS.md` (cross section), `cross_results.json`, `preds_cross/` |
| Classical hold-out settings | can classical methods predict a type with none of it (S1), only other samples of it (S2), or only the sample's own values (S3)? | [`09_classical_ml/HOLDOUT.md`](09_classical_ml/HOLDOUT.md), `holdout_results.json`, `preds_holdout/`, `scripts/holdout.py` |
| ET-SoC-1 card run | can the kNN similarity step run on the Esperanto card? | `09_classical_ml/card/`, `10_esperanto_et_soc1/` |

All scores side by side: `12_dashboard/data/` (`types.json` best method per type, `leaderboard.json` every run on every
type, `transfer.json` the held-out types, `offtarget.json` hold-out effects with bootstrap CIs, `classical_holdout*.json`).

Refresh the inventory: `python3 inventory.py` (pulls finished hold-out runs from Modal).
