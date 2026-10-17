# Biohack data inventory (osteosarcoma imputation)

Generated 2026-10-17 13:50 by `python3 ~/biohack/inventory.py`. Re-run it after each hold-out run finishes.

Model: Muse-Glimmer-30B (Modal volume `hf-cache:/Muse-Glimmer-30B`). Data: Sid Sijbrandij's public
osteosarcoma bucket `s3://sid-sijbrandij-osteosarc-dataset` (us-west-2, `--no-sign-request`).
Eval = a 20% random mask: 24,634 targets in 15,964 prompts, 13 data types.


## 1. Sid's full dataset

| file | size | rows | sha256 | what |
|---|---:|---:|---|---|
| `01_sid_full_dataset/sid_full.jsonl` | 45.9M | 11,824 | 6e34ed708065 | SFT training set built from Sid's data minus the 20% mask; {prompt, completion}. Same file as osteo-data:train/sid_full.jsonl |
| `01_sid_full_dataset/source/manifest.txt` | 57.8M | 397,364 | 6c9b89d030df | every object URL in s3://sid-sijbrandij-osteosarc-dataset |
| `01_sid_full_dataset/source/keys.txt` | 32.0M | 397,364 | bdd4664f9a5a | bucket keys listing used for reconstruction |
| `01_sid_full_dataset/source/h5keys.txt` | 3.3K | 41 | 6ce899cd89ee | keys of the 10x .h5 matrices downloaded |
| `01_sid_full_dataset/source/src/` | 33.8M (47 files) |  |  | non-h5 source tables downloaded from the bucket |
| `01_sid_full_dataset/source/h5/` | 5.8G (117 files) |  |  | 10x feature-barcode matrices (scRNA / CITE-seq) |

## 2. Golden labels (masked data)

| file | size | rows | sha256 | what |
|---|---:|---:|---|---|
| `02_golden_labels/golden.jsonl` | 36.3M | 15,964 | 980e325e1d3a | one row per eval prompt: messages, items, gold, completion |
| `02_golden_labels/labels.csv` | 2.0M | 24,635 | f39a05b02f6a | flat: one row per (prompt_id, item number) |
| `02_golden_labels/truth.jsonl` | 3.6M | 24,634 | b8e8f3cb5de5 | one row per masked target (prompt_id, i, item, entity, mod, truth); scorers read this |
| `02_golden_labels/validated_files.json` | 15.9K |  | 0989dd55937a | which bucket file each entity's truth came from |
| `02_golden_labels/build_golden.py` | 1.6K |  | 9558d298d565 | rebuilds golden.jsonl + labels.csv |
| `02_golden_labels/README_golden.md` | 1.3K |  | 92461cb7c344 | how the gold labels were built + caveats |

## 3. Baseline eval prompts

| file | size | rows | sha256 | what |
|---|---:|---:|---|---|
| `03_baseline_prompts/baseline.jsonl` | 32.3M | 15,964 | f0fc30b111a2 | 20% random mask as prompts; {prompt_id, messages}. = osteo-data:prompts/baseline.jsonl (= osteo-results/eval_prompts.jsonl, reformatted) |

## 4. Baseline eval responses

| file | size | rows | sha256 | what |
|---|---:|---:|---|---|
| `04_baseline_responses/baseline_zeroshot.jsonl` | 14.9M | 15,964 | 02277fd42c43 | Muse-Glimmer-30B zero-shot, first pass |
| `04_baseline_responses/baseline_zeroshot_retry.jsonl` | 302.8K | 54 | 9cb464063521 | re-runs of the 54 prompts truncated (finish=length) |
| `04_baseline_responses/baseline_zeroshot_merged.jsonl` | 15.0M | 15,964 | d0809d062168 | first pass with the 54 retries swapped in; USE THIS ONE |

## 5. Baseline report (score.py)

| file | size | rows | sha256 | what |
|---|---:|---:|---|---|
| `05_baseline_report/REPORT.md` | 2.5K |  | 781c045112e4 | written summary of the baseline |
| `05_baseline_report/score_py_output.txt` | 959B | 15 | 6fb16899234d | raw stdout of scripts/score.py |
| `05_baseline_report/baseline_scores.json` | 3.8K |  | 0fb760f34b83 | per-data-type metrics from score.py |

## 7. Full-data fine-tune (sid_full)

| file | size | rows | sha256 | what |
|---|---:|---:|---|---|
| `07_finetune_sid_full/finetuned_sid_full_eval.jsonl` | 2.1M | 15,964 | 27b093030ba7 | sid_full adapter on all eval prompts (= osteo-data:outputs/baseline_sid_full.jsonl) |
| `07_finetune_sid_full/compare_scores.json` | 4.5K |  | 6437f4f1e06b | base vs fine-tune per data type (scripts/score_compare.py) |
| `07_finetune_sid_full/compare_microbiome.json` | 947B |  | 9ead825fa138 | base vs fine-tune, metagenomics detail |
| `07_finetune_sid_full/compare_points.json` | 654.5K |  | 1e24c7524248 | per-target (truth, base, fine-tune) points for plots |
| `07_finetune_sid_full/adapter/adapter_model.safetensors` | 799.6M |  | 37a55aad7c84 | LoRA r=32 weights, final (training stopped itself at step 739/1478); base = Muse-Glimmer-30B |
| `07_finetune_sid_full/adapter/` | 3.2G (26 files) |  |  | adapter + configs + train_log.json + checkpoint-739 (optimizer state) |
| `07_finetune_sid_full/dashboard_src/` | 712.1K (5 files) |  |  | source + data of https://claude.ai/artifact/YTL434mENWCv3anqC5EE2J |

## 6. Hold-out experiments (leave-one-data-type-out)

LoRA r=32, 740 steps, 8×H200, same recipe as `sid_full` but with one data type removed from training;
each is then evaluated on that type's masked targets only. `lomo740_all` is the matched reference trained on everything.
Script: `scripts/lomo_train.py`; scorer: `scripts/score_lomo.py`. transfer = (holdout − base) / (all − base).
Sweep app: not running.

| run | trains on | eval targets | status | local outputs |
|---|---|---:|---|---|
| `lomo740_flow` | everything except Flow cytometry | 221 | unknown (offline) | – |
| `lomo740_rna` | everything except Bulk RNA-seq TPM | 2,000 | unknown (offline) | – |
| `lomo740_tcr` | everything except TCR beta clones | 2,000 | unknown (offline) | – |
| `lomo740_xenium` | everything except Xenium | 600 | unknown (offline) | – |
| `lomo740_all` | all 11,824 rows | 24,634 | unknown (offline) | – |

Baselines for each hold-out are in `06_heldout_experiments/lomo740_<type>/baselines/`, sliced to that type's prompts:
`prompts_<type>`, `truth_<type>`, `base_zeroshot_<type>` (Glimmer zero-shot), `sid_full_finetune_<type>` (full-data
adapter, 739 steps), and `all_reference_<type>` once `lomo740_all` has finished. The train-median reference is in
`05_baseline_report/baseline_scores.json`.

_No hold-out scores yet (`~/osteo-results/lomo_scores.json` not written)._

## Related runs (not in this inventory's numbered sets)

- `~/osteo-results/spark/` — Muse Spark 1.2 / 1.3 via api.meta.ai (in progress, job 7d63451c).
- `~/osteo-nim/` — NVIDIA NIM no-context baseline, 826 prompts (in progress, job c76b4182).
- Comparison dashboard: https://claude.ai/artifact/YTL434mENWCv3anqC5EE2J (source in `07_finetune_sid_full/dashboard_src/`)
- Working copies the original scripts still read: `~/.claude/jobs/6c5c300f/tmp/osteo/` (temporary; this folder is the durable copy).
