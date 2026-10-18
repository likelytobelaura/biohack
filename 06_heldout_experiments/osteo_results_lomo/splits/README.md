# Never-trained-on training rows (lomo740 runs)

Every lomo740 LoRA (all-data reference and the flow / bulk RNA / TCR / Xenium hold-outs) trained for 740 steps x 8 GPUs x batch 1 = **5,920 rows**, about half of `train/sid_full.jsonl` (11,824 rows). The other half was never seen. These files hold exactly those never-seen rows, so they can be used as clean, in-distribution data the model has not memorized (for example for SAE / feature work), separate from the 20% eval mask.

## Files

| file | rows | what |
|---|---|---|
| `lomo740_all_unseen_rows.jsonl` | 5,904 | rows `lomo740_all` never trained on |
| `lomo740_flow_unseen_rows.jsonl` | 5,904 | rows `lomo740_flow` never trained on (includes all 54 rows of the held-out type) |
| `lomo740_rna_unseen_rows.jsonl` | 5,904 | rows `lomo740_rna` never trained on (includes all 1000 rows of the held-out type) |
| `lomo740_tcr_unseen_rows.jsonl` | 5,904 | rows `lomo740_tcr` never trained on (includes all 1000 rows of the held-out type) |
| `lomo740_xenium_unseen_rows.jsonl` | 5,904 | rows `lomo740_xenium` never trained on (includes all 252 rows of the held-out type) |
| `never_trained_by_any_lomo740.jsonl` | 391 | rows none of the five runs saw (each run shuffles differently) |
| `<run>_row_ids.json` | | `unseen_row_ids`, plus the seen row ids in exact training order per GPU rank (rank r, step t = `seen_row_ids_in_training_order_by_rank[r][t]`) |
| `splits_summary.json` | | counts per run and per data type |

Each row is the original training row (`prompt`, `completion`) plus `row_id` = its 0-based line number in `train/sid_full.jsonl` (sha256 `6e34ed7080654b8f…`). Seen rows = all other row ids, so they are not duplicated here.

## How the split was rebuilt and verified

`scripts/reconstruct_seen_rows.py` re-creates each run's exact data order: same row filter, same `SFTConfig` (seed 42, batch 1, no packing, max_length 6144), same library versions as `lomo_train.py` (torch 2.14.0, transformers 5.18.0, trl 1.14.1, accelerate 1.15.0, datasets 5.0.1), launched as 8 processes with `torchrun` on CPU with a tiny stand-in model (the sampler depends on dataset length, seed and world size, not the model). Each rank records the dataset indices of its first 740 batches, and the first 5 real batches it produces are checked token-for-token against those indices before anything is written; all 40 rank/run combinations passed. TRL dropped no rows, so indices map 1:1 to `sid_full.jsonl` lines. `scripts/write_unseen_splits.py` writes these files.

**Not covered:** `sid_full` (the first 50%-stopped fine-tune) was trained by a script that is not in this repo, so its seen/unseen split is not reconstructed here.

## Never-trained rows by data type

| data type | lomo740_all | lomo740_flow | lomo740_rna | lomo740_tcr | lomo740_xenium | never by any |
|---|---|---|---|---|---|---|
| Single-cell RNA-seq (10x) | 1714 | 1689 | 1547 | 1568 | 1651 | 80 |
| CITE-seq antibody tags (10x) | 1480 | 1435 | 1256 | 1276 | 1427 | 75 |
| Olink plasma proteomics | 498 | 510 | 494 | 426 | 470 | 21 |
| Bulk tumour RNA-seq, GTEx v11 pipeline | 496 | 491 | 1000 | 431 | 514 | 60 |
| T-cell receptor beta clones | 491 | 527 | 464 | 1000 | 495 | 64 |
| Multiplex immunofluorescence of the T0 tumour, one row per segmented cell | 487 | 469 | 467 | 464 | 481 | 18 |
| DIA mass-spec proteomics of laser-captured 112 um regions of the T0 tumour | 486 | 491 | 431 | 488 | 493 | 31 |
| Xenium in-situ transcriptomics (428-gene panel) | 128 | 125 | 108 | 108 | 252 | 33 |
| Somatic tumour/normal WGS calls (DRAGEN) | 54 | 59 | 67 | 64 | 51 | 4 |
| Shotgun metagenomics (MetaPhlAn) | 38 | 47 | 42 | 45 | 32 | 3 |
| Spectral flow cytometry of PBMCs | 25 | 54 | 18 | 23 | 29 | 2 |
| Germline HLA typing | 6 | 6 | 8 | 8 | 8 | 0 |
| Gene fusions from long-read single-cell RNA (CTAT-LR) | 1 | 1 | 2 | 3 | 1 | 0 |
