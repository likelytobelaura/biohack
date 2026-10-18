# Cross-type timepoint snapshots ("Sid at timepoint X, some values masked")

Every earlier training row and eval prompt held one data type for one entity, so the fine-tunes never saw two
assays side by side. This experiment continues the `lomo740_all` LoRA (740 steps, 1st half of `sid_full`) for one
more pass over the **other half** of the training rows (the 5,904 rows `lomo740_all` never saw,
`06_heldout_experiments/osteo_results_lomo/splits/lomo740_all_unseen_rows.jsonl`), rebuilt as cross-assay snapshots.

## Training data — `data/train_snap.jsonl` (5,904 prompts, 160,918 targets)
One snapshot per unseen row. The row's own known values and targets are kept; added as context:
- **same timepoint**: blood visits = collection dates within 7 days (flow, Olink, TCR month, MetaPhlAn);
  tumour timepoints T0–T3 (bulk RNA, somatic, fusions, DIA);
- **same cell**: CITE-seq rows get that cell's RNA, scRNA rows its antibody tags (same 10x barcode, from the h5).

~20% of the added values are masked as extra targets. Context comes only from the unseen half and the h5 files.
"Elsewhere in the dataset" hint lines are dropped. **No eval (gold) value appears** as a known value (0 / 588,000)
or a target (0 / 160,918), matching gold under both label and prompt item names.

## Eval — `data/eval_snap.jsonl` (4,240 gold prompts × 3 conditions)
Gold eval prompts (all of each type, capped at 300; 1,000 for CITE/scRNA/mIF, Xenium 600) in snapshot format,
scored only on the gold targets:
- `full`: own values + same-timepoint / same-cell context (context = training values, gold removed)
- `own`: same format, other assays removed
- `swap`: other assays from a different timepoint / a cell from another visit

full − own = what other assays add; full − swap = whether it is the *matching* timepoint that helps.
Gold labels for these prompts (`eval_snap_labels.csv`) are kept local, like `02_golden_labels`.

## Run
- `scripts/build_snapshots.py` → data files; `scripts/xtype_modal.py` (Modal app `osteo-xtype`): continue from
  `/data/adapters/lomo/lomo740_all`, 738 steps, bs 1 × 8 H200, LR 1e-4 cosine, same LoRA (r 32) →
  `/data/adapters/xtype/lomo740_all_snap`; then the standard 15,964-prompt eval and the snapshot eval
  (new adapter, `lomo740_all`, base).
- `scripts/score_snap.py <eval dirs>` → per type × condition pooled Spearman ρ, median asinh error, paired
  bootstrap CIs for full − own and full − swap.
- Weights: HF `likelytobelaura/osteo-glimmer-lora`, folder `xtype/lomo740_all_snap/`.

Results: see `RESULTS.md` (added when the evals finish).
