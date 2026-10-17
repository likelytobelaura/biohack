# Biohack data inventory (osteosarcoma imputation)

Generated 2026-10-17 15:34 by `python3 ~/biohack/inventory.py`. Re-run it after each hold-out run finishes.

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

## 6. Hold-out experiments (leave-one-data-type-out)

LoRA r=32 (same settings as `sid_full`), 8×H200, a 740-step schedule run to completion (5,920 rows seen), one data type removed;
each is then evaluated on that type's masked targets only. `lomo740_all` is the matched reference trained on everything:
the finished counterpart of `sid_full`, which saw the same half epoch but stopped mid-schedule.
Script: `scripts/lomo_train.py`; scorer: `scripts/score_lomo.py`. transfer = (holdout − base) / (all − base).
Sweep app: `ap-AIzYiRcj5xsmQdDteujfMa` (ephemeral (detached))

| run | trains on | eval targets | status | local outputs |
|---|---|---:|---|---|
| `lomo740_flow` | everything except Flow cytometry | 221 | done | `06_heldout_experiments/lomo740_flow/outputs/` |
| `lomo740_rna` | everything except Bulk RNA-seq TPM | 2,000 | trained, eval pending | – |
| `lomo740_tcr` | everything except TCR beta clones | 2,000 | trained, eval pending | – |
| `lomo740_xenium` | everything except Xenium | 600 | trained, eval pending | – |
| `lomo740_all` | all 11,824 rows | 24,634 | trained, eval pending | – |
| `lomo740_flow_full` | everything except Flow cytometry (adapter of `lomo740_flow`; eval on ALL prompts, off-target) | 24,634 | done | `06_heldout_experiments/lomo740_flow_full/outputs/` |
| `lomo740_rna_full` | everything except Bulk RNA-seq TPM (adapter of `lomo740_rna`; eval on ALL prompts, off-target) | 24,634 | done | `06_heldout_experiments/lomo740_rna_full/outputs/` |

Baselines for each hold-out are in `06_heldout_experiments/lomo740_<type>/baselines/`, sliced to that type's prompts:
`prompts_<type>`, `truth_<type>`, `base_zeroshot_<type>` (Glimmer zero-shot), `sid_full_finetune_<type>` (sid_full
adapter, stopped at step 739/1478), and `all_reference_<type>` once `lomo740_all` has finished. The train-median reference is in
`05_baseline_report/baseline_scores.json`.
`lomo740_<type>_full` runs re-evaluate a hold-out adapter on every prompt; their per-type change vs `lomo740_all`
(off-target effect of dropping that type) is in `osteo_results_lomo/offtarget_scores.json` (`scripts/score_offtarget.py`).
Full-eval outputs as the LOMO job pulls them: `osteo_results_lomo/evals/` (incl. `lomo740h100_all`, the all-data reference
re-run with H100 fallback; a later `lomo740_all` full eval would duplicate it).

_No hold-out scores yet (`~/osteo-results/lomo/lomo_scores.json` not written; `preview_noflow_scores.json` is an early preview it supersedes)._

Local mirror of `~/osteo-results/lomo/` (written by the LOMO job):

`06_heldout_experiments/osteo_results_lomo/`

| file | size | rows |
|---|---:|---:|
| `adapters/lomo740_all/README.md` | 5.1K |  |
| `adapters/lomo740_all/adapter_config.json` | 1.1K |  |
| `adapters/lomo740_all/adapter_model.safetensors` | 799.6M |  |
| `adapters/lomo740_all/chat_template.jinja` | 9.8K |  |
| `adapters/lomo740_all/tokenizer.json` | 26.8M |  |
| `adapters/lomo740_all/tokenizer_config.json` | 78.1K |  |
| `adapters/lomo740_all/train_log.json` | 35.7K |  |
| `adapters/lomo740_all/training_args.bin` | 5.6K |  |
| `adapters/lomo740_flow/README.md` | 5.1K |  |
| `adapters/lomo740_flow/adapter_config.json` | 1.1K |  |
| `adapters/lomo740_flow/adapter_model.safetensors` | 799.6M |  |
| `adapters/lomo740_flow/chat_template.jinja` | 9.8K |  |
| `adapters/lomo740_flow/tokenizer.json` | 26.8M |  |
| `adapters/lomo740_flow/tokenizer_config.json` | 78.1K |  |
| `adapters/lomo740_flow/train_log.json` | 35.6K |  |
| `adapters/lomo740_flow/training_args.bin` | 5.6K |  |
| `adapters/lomo740_rna/README.md` | 5.1K |  |
| `adapters/lomo740_rna/adapter_config.json` | 1.1K |  |
| `adapters/lomo740_rna/adapter_model.safetensors` | 799.6M |  |
| `adapters/lomo740_rna/chat_template.jinja` | 9.8K |  |
| `adapters/lomo740_rna/tokenizer.json` | 26.8M |  |
| `adapters/lomo740_rna/tokenizer_config.json` | 78.1K |  |
| `adapters/lomo740_rna/train_log.json` | 35.6K |  |
| `adapters/lomo740_rna/training_args.bin` | 5.6K |  |
| `adapters/lomo740_tcr/README.md` | 5.1K |  |
| `adapters/lomo740_tcr/adapter_config.json` | 1.1K |  |
| `adapters/lomo740_tcr/adapter_model.safetensors` | 799.6M |  |
| `adapters/lomo740_tcr/chat_template.jinja` | 9.8K |  |
| `adapters/lomo740_tcr/tokenizer.json` | 26.8M |  |
| `adapters/lomo740_tcr/tokenizer_config.json` | 78.1K |  |
| `adapters/lomo740_tcr/train_log.json` | 35.7K |  |
| `adapters/lomo740_tcr/training_args.bin` | 5.6K |  |
| `adapters/lomo740_xenium/README.md` | 5.1K |  |
| `adapters/lomo740_xenium/adapter_config.json` | 1.1K |  |
| `adapters/lomo740_xenium/adapter_model.safetensors` | 799.6M |  |
| `adapters/lomo740_xenium/chat_template.jinja` | 9.8K |  |
| `adapters/lomo740_xenium/tokenizer.json` | 26.8M |  |
| `adapters/lomo740_xenium/tokenizer_config.json` | 78.1K |  |
| `adapters/lomo740_xenium/train_log.json` | 35.7K |  |
| `adapters/lomo740_xenium/training_args.bin` | 5.6K |  |
| `evals/lomo740_flow_full/shard0.jsonl` | 127.8K | 998 |
| `evals/lomo740_flow_full/shard1.jsonl` | 127.3K | 998 |
| `evals/lomo740_flow_full/shard10.jsonl` | 112.5K | 998 |
| `evals/lomo740_flow_full/shard11.jsonl` | 113.2K | 998 |
| `evals/lomo740_flow_full/shard12.jsonl` | 112.4K | 997 |
| `evals/lomo740_flow_full/shard13.jsonl` | 113.0K | 997 |
| `evals/lomo740_flow_full/shard14.jsonl` | 126.5K | 997 |
| `evals/lomo740_flow_full/shard15.jsonl` | 140.8K | 997 |
| `evals/lomo740_flow_full/shard2.jsonl` | 113.5K | 998 |
| `evals/lomo740_flow_full/shard3.jsonl` | 127.4K | 998 |
| `evals/lomo740_flow_full/shard4.jsonl` | 114.2K | 998 |
| `evals/lomo740_flow_full/shard5.jsonl` | 127.0K | 998 |
| `evals/lomo740_flow_full/shard6.jsonl` | 141.3K | 998 |
| `evals/lomo740_flow_full/shard7.jsonl` | 126.9K | 998 |
| `evals/lomo740_flow_full/shard8.jsonl` | 112.0K | 998 |
| `evals/lomo740_flow_full/shard9.jsonl` | 126.1K | 998 |
| `evals/lomo740_flow_own/shard0.jsonl` | 4.5K | 18 |
| `evals/lomo740_rna_full/shard0.jsonl` | 113.8K | 998 |
| `evals/lomo740_rna_full/shard1.jsonl` | 113.4K | 998 |
| `evals/lomo740_rna_full/shard10.jsonl` | 112.4K | 998 |
| `evals/lomo740_rna_full/shard11.jsonl` | 113.2K | 998 |
| `evals/lomo740_rna_full/shard12.jsonl` | 112.4K | 997 |
| `evals/lomo740_rna_full/shard13.jsonl` | 112.9K | 997 |
| `evals/lomo740_rna_full/shard14.jsonl` | 112.6K | 997 |
| `evals/lomo740_rna_full/shard15.jsonl` | 113.0K | 997 |
| `evals/lomo740_rna_full/shard2.jsonl` | 113.4K | 998 |
| `evals/lomo740_rna_full/shard3.jsonl` | 113.5K | 998 |
| `evals/lomo740_rna_full/shard4.jsonl` | 114.1K | 998 |
| `evals/lomo740_rna_full/shard5.jsonl` | 113.1K | 998 |
| `evals/lomo740_rna_full/shard6.jsonl` | 113.5K | 998 |
| `evals/lomo740_rna_full/shard7.jsonl` | 113.0K | 998 |
| `evals/lomo740_rna_full/shard8.jsonl` | 112.0K | 998 |
| `evals/lomo740_rna_full/shard9.jsonl` | 112.1K | 998 |
| `evals/noflow_full.jsonl` | 1.9M | 15,964 |
| `offtarget_scores.json` | 3.7K |  |
| `preview_noflow_scores.json` | 6.3K |  |

## 7. sid_full: LoRA on all data, stopped at 50%

| file | size | rows | sha256 | what |
|---|---:|---:|---|---|
| `07_finetune_sid_full/finetuned_sid_full_eval.jsonl` | 2.1M | 15,964 | 27b093030ba7 | sid_full adapter on all eval prompts (= osteo-data:outputs/baseline_sid_full.jsonl) |
| `07_finetune_sid_full/compare_scores.json` | 4.5K |  | 6437f4f1e06b | base vs sid_full adapter per data type (scripts/score_compare.py) |
| `07_finetune_sid_full/compare_microbiome.json` | 947B |  | 9ead825fa138 | base vs sid_full adapter, metagenomics detail |
| `07_finetune_sid_full/compare_points.json` | 654.5K |  | 1e24c7524248 | per-target (truth, base, sid_full) points for plots |
| `07_finetune_sid_full/adapter/adapter_model.safetensors` | 799.6M |  | 37a55aad7c84 | LoRA adapter weights (r=32, a delta on Muse-Glimmer-30B, not a standalone model) after 739 of 1,478 planned steps (0.5 epoch, stopped mid-schedule, LR not decayed); byte-identical to checkpoint-739/adapter_model.safetensors |
| `07_finetune_sid_full/adapter/` | 3.2G (26 files) |  |  | the adapter above + configs + train_log.json; checkpoint-739/ = same weights + optimizer/scheduler/RNG state to resume |
| `07_finetune_sid_full/dashboard_src/` | 712.1K (5 files) |  |  | source + data of https://claude.ai/artifact/YTL434mENWCv3anqC5EE2J |

## 8. Other baselines  ·  9. Classical ML + ET-SoC-1

Mirrored from the jobs running them; partial files are re-copied on every refresh until the runs finish.

**muse_spark**: Muse Spark 1.2 / 1.3 (api.meta.ai, contributor) on golden.jsonl prompts; job 7d63451c. Source `~/osteo-results/spark` (API key and .venv excluded).

`08_other_baselines/muse_spark/`

| file | size | rows |
|---|---:|---:|
| `cost.txt` | 44B | 1 |
| `guard.log` | 0B |  |
| `muse-spark-1.2-contributor.jsonl` | 2.7M | 9,953 |
| `muse-spark-1.3-contributor.jsonl` | 4.3M | 15,964 |
| `run_1.2.log` | 2.9K |  |
| `run_1.3.log` | 3.2K |  |
| `spark_scores.json` | 2.8K |  |
| `spark_scores_1.3_preview.json` | 4.7K |  |

**nim_noctx**: NVIDIA NIM, prompts with the known values stripped (826 prompts) + NIM hold-out baselines; job c76b4182. Source `~/osteo-nim` (API key and .venv excluded).

`08_other_baselines/nim_noctx/`

| file | size | rows |
|---|---:|---:|
| `baseline_lomo_flow.jsonl` | 14.3M | 15,964 |
| `baseline_lomo_rna.jsonl` | 14.2M | 15,964 |
| `baseline_lomo_tcr.jsonl` | 14.2M | 15,964 |
| `baseline_lomo_xenium.jsonl` | 14.0M | 15,812 |
| `out_noctx.jsonl` | 631.1K | 643 |
| `out_noctx.jsonl.scoring` | 80.4K |  |
| `out_noctx_scores.json` | 1.9K |  |
| `prompts_noctx.jsonl` | 640.6K | 826 |
| `prompts_noctx_ordered.jsonl` | 640.6K | 826 |
| `run.log` | 228B |  |
| `scripts/build_lomo_baselines.py` | 1.6K |  |
| `scripts/build_prompts.py` | 1.3K |  |
| `scripts/modal_noctx.py` | 1.9K |  |
| `scripts/run_nim.py` | 2.6K |  |
| `scripts/score_noctx.py` | 1.6K |  |
| `smoke.jsonl` | 5.1K | 5 |
| `smoke_out.jsonl` | 5.7K | 5 |

**09_classical_ml**: classical imputation sweep (kNN etc.) + ET-SoC-1 card work (aifoundry3 card 0; card results copied back to card/from_aifoundry3/); job c76b4182. Source `~/osteo-classical` (API key and .venv excluded).

`09_classical_ml/`

| file | size | rows |
|---|---:|---:|
| `RESULTS.md` | 4.4K |  |
| `card/README.txt` | 190B | 2 |
| `card/TASK.md` | 1.6K |  |
| `card/cite_A.f32` | 13.2M |  |
| `card/cite_S_ref.f32` | 285.8M |  |
| `card/from_aifoundry3/DONE` | 855B |  |
| `card/from_aifoundry3/NOTES.md` | 2.5K |  |
| `card/from_aifoundry3/README.txt` | 190B | 2 |
| `card/from_aifoundry3/TASK.md` | 1.6K |  |
| `card/from_aifoundry3/aat/CMakeLists.txt` | 1.2K | 30 |
| `card/from_aifoundry3/aat/aat_args.h` | 617B |  |
| `card/from_aifoundry3/aat/build.log` | 3.2K |  |
| `card/from_aifoundry3/aat/build/Makefile` | 4.9K |  |
| `card/from_aifoundry3/aat/build/host/Constants.h` | 730B |  |
| `card/from_aifoundry3/aat/build/host/Makefile` | 5.7K |  |
| `card/from_aifoundry3/aat/build/host/aat_host` | 3.4M |  |
| `card/from_aifoundry3/aat/build/kernel/Makefile` | 6.5K |  |
| `card/from_aifoundry3/aat/build/kernel/aat.bin` | 674B |  |
| `card/from_aifoundry3/aat/build/kernel/aat.elf` | 8.0K |  |
| `card/from_aifoundry3/aat/build/kernel/aat.elf.ltrans0.ltrans.su` | 72B |  |
| `card/from_aifoundry3/aat/build/kernel/aat.lst` | 12.0K |  |
| `card/from_aifoundry3/aat/build/kernel/aat.map` | 6.0K |  |
| `card/from_aifoundry3/aat/check.py` | 959B |  |
| `card/from_aifoundry3/aat/host/CMakeLists.txt` | 1.4K | 22 |
| `card/from_aifoundry3/aat/host/Constants.h.in` | 470B |  |
| `card/from_aifoundry3/aat/host/main.cpp` | 10.4K |  |
| `card/from_aifoundry3/aat/kernel/CMakeLists.txt` | 681B | 16 |
| `card/from_aifoundry3/aat/kernel/aat.c` | 2.2K |  |
| `card/from_aifoundry3/aat/kernel/crt.S` | 760B |  |
| `card/from_aifoundry3/aat/kernel/sections.ld` | 1.7K |  |
| `card/from_aifoundry3/aat/sim/A64.f32` | 100.0K |  |
| `card/from_aifoundry3/aat/sim/S64.f32` | 16.0K |  |
| `card/from_aifoundry3/aat/sim/pu_uart0_tx.log` | 127B |  |
| `card/from_aifoundry3/aat/sim/pu_uart1_tx.log` | 1B |  |
| `card/from_aifoundry3/aat/sim/sim.out` | 5.7M |  |
| `card/from_aifoundry3/aat/sim/spio_uart0_tx.log` | 952B |  |
| `card/from_aifoundry3/aat/sim/spio_uart1_tx.log` | 1B |  |
| `card/from_aifoundry3/aat/sim/sysemu.log` | 5.8K |  |
| `card/from_aifoundry3/cite_A.f32` | 13.2M |  |
| `card/from_aifoundry3/cite_S_card.f32` | 285.8M |  |
| `card/from_aifoundry3/runs/card-2048/A2048.f32` | 3.1M |  |
| `card/from_aifoundry3/runs/card-2048/S2048.f32` | 16.0M |  |
| `card/from_aifoundry3/runs/card-2048/check.txt` | 230B | 7 |
| `card/from_aifoundry3/runs/card-2048/manifest.txt` | 670B | 11 |
| `card/from_aifoundry3/runs/card-2048/out.txt` | 4.1K | 55 |
| `card/from_aifoundry3/runs/card-2048/when.txt` | 32B | 1 |
| `card/from_aifoundry3/runs/card-full/check.txt` | 231B | 7 |
| `card/from_aifoundry3/runs/card-full/manifest.txt` | 670B | 11 |
| `card/from_aifoundry3/runs/card-full/out.txt` | 4.1K | 55 |
| `card/from_aifoundry3/runs/card-full/sha256.txt` | 110B | 1 |
| `card/from_aifoundry3/runs/card-full/when.txt` | 32B | 1 |
| `compare_classical.json` | 20.6K |  |
| `cross.log` | 4.8K |  |
| `cross_results.json` | 9.1K |  |
| `preds/README.md` | 524B |  |
| `preds/additive.jsonl` | 1.4M | 15,955 |
| `preds/als_2_1.0.jsonl` | 1.4M | 15,955 |
| `preds/als_2_10.0.jsonl` | 1.4M | 15,955 |
| `preds/als_32_1.0.jsonl` | 1.4M | 15,955 |
| `preds/als_32_10.0.jsonl` | 1.4M | 15,955 |
| `preds/als_8_1.0.jsonl` | 1.4M | 15,955 |
| `preds/als_8_10.0.jsonl` | 1.4M | 15,955 |
| `preds/best_by_val.jsonl` | 1.3M | 15,964 |
| `preds/card_knn_entity_10.jsonl` | 426.9K | 5,967 |
| `preds/card_knn_entity_25.jsonl` | 427.3K | 5,967 |
| `preds/col_mean.jsonl` | 1.3M | 15,955 |
| `preds/col_median.jsonl` | 1.2M | 15,955 |
| `preds/gbm_stack.jsonl` | 1.4M | 15,955 |
| `preds/knn_entity_10.jsonl` | 1.3M | 15,955 |
| `preds/knn_entity_25.jsonl` | 1.3M | 15,955 |
| `preds/knn_entity_5.jsonl` | 1.3M | 15,955 |
| `preds/knn_entity_50.jsonl` | 1.3M | 15,955 |
| `preds/knn_feature_10.jsonl` | 1.3M | 15,955 |
| `preds/knn_feature_25.jsonl` | 1.3M | 15,955 |
| `preds/knn_feature_5.jsonl` | 1.3M | 15,955 |
| `preds/knn_feature_50.jsonl` | 1.3M | 15,955 |
| `preds/row_mean.jsonl` | 1.3M | 15,955 |
| `preds_cross/cite_within_plus_cross.jsonl` | 429.1K | 5,967 |
| `preds_cross/dia_within_plus_cross.jsonl` | 62.9K | 247 |
| `preds_cross/olink_within_plus_cross.jsonl` | 60.3K | 104 |
| `preds_cross/rna_within_plus_cross.jsonl` | 60.0K | 104 |
| `preds_cross/scrna_within_plus_cross.jsonl` | 487.2K | 6,766 |
| `preds_cross/xenium_within_plus_cross.jsonl` | 43.3K | 600 |
| `run.log` | 12.3K |  |
| `run2.log` | 12.3K |  |
| `scripts/build_tables.py` | 2.5K |  |
| `scripts/card_knn.py` | 2.6K |  |
| `scripts/compare.py` | 2.0K |  |
| `scripts/cross.py` | 9.6K |  |
| `scripts/export_cite.py` | 1.5K |  |
| `scripts/run_classical.py` | 11.7K |  |
| `tables.pkl` | 50.1M |  |
| `val_picks.json` | 316B |  |

## 10. Esperanto ET-SoC-1 (AI Foundry lab, aifoundry3 card 0)

Everything Laura's lab account `likelytobelaura@aifoundry3` holds, pulled with rsync on every refresh (last pulled 2026-10-17 15:34).
The account only reaches aifoundry3 (aifoundry1/2 refuse it), so this is all of it. Not copied: dotfiles
(`~/.claude`, `.claude.json` hold the lab Claude's login) and `~/et-soc1-prototyping`, an unmodified clone of
https://github.com/yaroslavvb/et-soc1-prototyping at `f0a40879800d8d21955d061b92ad974570455595`. `osteo/` is the osteo card job (also in section 9).

| folder | files | size | what |
|---|---:|---:|---|
| `osteo` | 153 | 328.0M | osteo card job: S = A·Aᵀ on card 0 (kernel `aat/`, build, sim logs, `runs/`, `cite_S_card.f32`, NOTES.md, DONE) |
| `work/mnist-et` | 266 | 9.4M | MNIST trained end to end on the card (difficulty 1 passed, 111 ms/call); NOTES.md |
| `work/hello` | 126 | 3.9M | first card program |
| `work/ref` | 4 | 133.3K | reference files |
| `runs/mnist` | 81 | 297.7M | raw simulator logs of the MNIST runs (kept deliberately: the only raw run records) |
| `runs/hello` | 21 | 815.4K | hello run |
| `runs/first-hour` | 11 | 103.6K | first-hour lab test |
| `data` | 89 | 276.5M | datasets used on the card |
| `lab-start.md` | 1 | 22.9K | the lab's START brief the lab Claude followed |

## Not yet filed

_Nothing: every file under ~/osteo-results, ~/osteo-nim, ~/osteo-golden, ~/osteo-classical is mirrored here._

## Links

- Comparison dashboard: https://claude.ai/artifact/YTL434mENWCv3anqC5EE2J (source in `07_finetune_sid_full/dashboard_src/`)
- Working copies the original scripts still read: `~/.claude/jobs/6c5c300f/tmp/osteo/` (temporary; this folder is the durable copy).
