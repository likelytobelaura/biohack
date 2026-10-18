# biohack — osteosarcoma multi-omic imputation

Fine-tuning Muse-Glimmer-30B to fill a 20% random mask of Sid Sijbrandij's public osteosarcoma dataset
(`s3://sid-sijbrandij-osteosarc-dataset`), plus classical baselines and an Esperanto ET-SoC-1 card experiment.
**Start with [INVENTORY.md](INVENTORY.md)**: every dataset, label set, baseline, fine-tune and hold-out experiment,
with sizes, sha256, run status and where each file lives.

**Model weights:** every LoRA adapter (`sid_full`, stopped at 50%; the five `lomo740_*` runs; three `smoke_*` tests),
with optimizer state and tokenizer, is on Hugging Face:
[huggingface.co/likelytobelaura/osteo-glimmer-lora](https://huggingface.co/likelytobelaura/osteo-glimmer-lora).

## What is in this repo

Code, reports and scores; every model answer file; the per-hold-out prompt and gold-label slices; and a full text copy
of the Modal volume `osteo-data` in `11_modal_osteo_data/` (training set `train/sid_full.jsonl`, eval prompts
`prompts/baseline.jsonl`, all eval shards). Not in git (too large or not needed; listed with sha256 in
`inventory.json`): the full gold labels, raw bucket files, adapter weights (on HF), card result matrices (`*.f32`),
raw ET-SoC-1 simulator logs and the public MNIST datasets used on the card.

| folder | contents | in git |
|---|---|---|
| `01_sid_full_dataset/` | SFT training set (11,824 rows) + raw bucket files | file list only; training set is in `11_modal_osteo_data/train/` |
| `02_golden_labels/` | gold values for all 24,634 masked targets | build script, README, provenance; labels local only |
| `03_baseline_prompts/` | the 15,964 eval prompts | no; same prompts are in `11_modal_osteo_data/prompts/` |
| `04_baseline_responses/` | Muse-Glimmer-30B zero-shot answers | yes |
| `05_baseline_report/` | `score.py` report on the zero-shot baseline | yes |
| `06_heldout_experiments/` | leave-one-data-type-out runs: answers, per-type baseline slices, hold-out and off-target scores | yes (weights on HF) |
| `07_finetune_sid_full/` | `sid_full` LoRA (all data, stopped at step 739/1478): answers, scores, configs/logs, dashboard source | yes (weights on HF) |
| `08_other_baselines/` | Muse Spark 1.2 / 1.3 and NVIDIA NIM (no-context) answers and scores | yes (prompt files local) |
| `09_classical_ml/` | classical imputation sweep (kNN etc.), its predictions, and the osteo ET-SoC-1 card job | yes (`*.f32` local) |
| `10_esperanto_et_soc1/` | everything from the AI Foundry lab account (aifoundry3): osteo card job, MNIST on the card, first tests | yes, except raw sim logs and datasets |
| `11_modal_osteo_data/` | full copy of the Modal volume `osteo-data`, same layout | all text; weights/optimizer/tokenizer on HF |
| `12_dashboard/` | dashboard build: `build.py` scores every run with the shared scorer and writes `data/*.json`; `index.html` is the page | yes |
| `13_linear_probes/` | MetaPhlAn log-abundance linear probe and microbiome-neuron hunt on Glimmer activations | yes (activations on Modal/HF) |
| `14_crosstype_snapshots/` | cross-type "Sid at timepoint X" snapshot fine-tune: scores, hint-copy analysis, scripts | yes (weights on HF `xtype/`) |
| `scripts/` | build, train, eval and scoring scripts | yes |

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

## Keeping it current

- `python3 inventory.py`: refreshes the mirrors and rewrites `INVENTORY.md` / `inventory.json`
  (`--offline` skips Modal and the lab machine).
- `git_sync.sh`: commits and pushes the refresh (fast-forward only; refuses on credential patterns or files over 50 MB).
  `watch_sweep.sh` runs both every 10 minutes while experiments are running.
- `~/biohack/.venv/bin/python hf_upload.py` / `hf_volume_backup.py`: upload adapters to HF, checking for truncated
  weights and verifying each file afterwards. `vol_audit.py` checks that every file on the Modal volume is on GitHub or HF.
