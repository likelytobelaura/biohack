# Classical hold-out experiments (17 Oct 2026)

Classical counterparts of the Glimmer hold-out runs: can a data type be predicted with less, or none, of its own data?
Same 24,634 masked targets and scorer (`osteo-results/scripts/score_compare.py`) as every other run; values modelled in
asinh space; no eval target is in the visible pool (checked: 0). Script: `scripts/holdout.py` (runs in ~1 min on a Mac);
predictions: `preds_holdout/<setting>__<method>.jsonl`; all numbers: `holdout_results.json` (per type and method:
Spearman ρ overall, on targets whose feature is seen elsewhere in the type, on never-seen ones, and on gene-covered ones).

Like every experiment so far, the eval is the **20% random mask, not a time split**: "carry forward" uses the latest
earlier dated sample of the same kind, but the comparison set still contains samples from after the target's date.

## The three settings

| Setting | What the predictor may use | Matches | Methods |
|---|---|---|---|
| **S1 type absent** | nothing of the type, anywhere; only other data types | context-free Glimmer (untrained, no values of the type in its prompt) | `zero` (what Glimmer did), `gene_score` (mean z-scored level of the same gene in other types; rank only), `transfer_gbm` (GBM trained on the other gene-linked types to map other types' gene levels to a value, applied to this type) |
| **S2 new sample** | the type's other samples, none of this entity's own values | a new blood draw / new cell / new tumour region | `feat_mean`, `feat_median` (leave-entity-out), `carry_forward` (latest earlier draw), `nearest_date` (closest draw either side), `gbm_link` (GBM on the type's other entities: leave-entity-out feature stats + carry-forward + other types' gene levels) |
| **S3 type unseen in training** | only this entity's own visible values of the type; models trained on other types | the leave-one-type-out LoRA (never trained on the type, sees the entity's values in its prompt) | `entity_mean`, `entity_median`, `transfer_gbm` (GBM trained on every other type with type-agnostic features: the entity's own value stats + gene-link levels) |

Which links exist: bulk RNA, scRNA, Xenium, CITE-seq, DIA and Olink share gene/protein names. Flow (cell-population
percentages), TCR (clonotypes), metagenomics, somatic calls, multiplex IF and fusions share none, so in S1 they can
only get a constant guess. Carry-forward/nearest-date only exist for dated sample series (flow, Olink, TCR; metagenomics
masks only never-seen species, so nothing to carry).

## The four types with Glimmer hold-outs (Spearman ρ)

| | Flow | Bulk RNA | TCR | Xenium |
|---|---:|---:|---:|---:|
| **S1** best classical, nothing of the type | none (no link) | **0.534** gene score (0.561 on the 27% gene-covered) | none (no link) | **0.450** gene score |
| Glimmer context-free (NVIDIA, base) | 0.669 | constant 0 | constant 0 | constant 0 |
| **S2** feature mean / median | 0.947 / 0.949 | 0.737 / 0.749 | −0.207 / 0.116 | 0.618 / 0.583 |
| **S2** carry forward / nearest draw | 0.913 / 0.949 | – (no dates) | −0.037 / 0.001 | – (cells) |
| **S2** GBM + links | **0.951** | 0.742 | **0.519** | 0.603 |
| **S3** entity mean / median | −0.127 / −0.091 | 0.153 / 0.161 | 0.010 / 0.188 | 0.345 / 0.307 |
| **S3** transfer GBM | −0.124 | 0.082 | −0.107 | 0.265 |
| Glimmer LoRA, type held out of training | 0.949 | 0.664 | 0.306 | running |
| Glimmer zero-shot with own values | 0.970 | 0.487 | 0.130 | 0.472 |
| Glimmer LoRA all types, same 740-step schedule (matched reference) | 0.973 | 0.694 | 0.526 | 0.461 |
| Glimmer LoRA sid_full (all types, stopped halfway) | 0.947 | 0.678 | 0.478 | 0.295 |
| Classical, everything visible (val pick) | 0.971 | 0.749 | 0.416 | 0.680 |

## All types, best method per setting (Spearman ρ; accuracy for multiplex IF)

| Type | S1 best | S2 best | S3 best | Classical, all visible |
|---|---|---|---|---|
| scRNA | 0.417 gene score (0.33 covered) | **0.666** GBM + links | 0.132 entity mean | 0.401 |
| CITE-seq | 0.089 | 0.607 feat median | 0.421 transfer GBM | 0.770 |
| Bulk RNA | 0.534 gene score | 0.749 feat median | 0.161 | 0.749 |
| TCR | – | 0.519 GBM | 0.188 entity median | 0.416 |
| DIA | 0.120 | **0.891** GBM + links | 0.116 | 0.868 |
| Multiplex IF | 0.892 (all 0) | 0.916 | 0.889 | 0.916 |
| Olink | 0.045 | **0.698** GBM + links | 0.199 | 0.800 |
| Xenium | 0.450 | 0.618 feat mean | 0.345 | 0.680 |
| Somatic | – | 0.660 feat median | 0.225 | 0.649 |
| Metagenomics | – | – (every masked species never seen) | 0.296 entity mean | 0.192 |
| Flow | – | 0.951 GBM | −0.091 | 0.971 |
| Fusions (21 targets) | – | 0.469 | 0.571 | 0.698 |

## What it says

1. **For classical methods, other samples of the same type carry almost everything.** S2 (no values from the sample
   itself) is within ~0.06 of using everything for flow, bulk RNA, DIA, Xenium and somatic (Olink loses 0.10, CITE-seq
   0.16: the cell's or draw's own values matter more there), and *beats* the original
   within-type pick on scRNA (0.67 vs 0.40), TCR (0.52 vs 0.42) and DIA (0.89 vs 0.87). For scRNA the gain comes from
   pricing never-seen genes with other types' level of the gene (the old stack mis-scaled them; seen genes alone
   reach 0.58 with a plain feature mean).
2. **A sample's own values, without the type's other samples, are weak for classical methods** (S3: ρ 0.1–0.4, flow
   negative). The Glimmer LoRA trained without bulk RNA reaches 0.66 from the same information (the entity's own
   values plus general training; 86% of the all-data LoRA's gain over zero-shot), and without flow reaches 0.95.
   TCR is different: without TCR training rows the LoRA reaches 0.31 against 0.53 with them (44% of the gain), so
   clonotype tables need their own training data; classical S2 (0.52) matches the all-data LoRA there.
   Off-target effects on scRNA and Xenium are **not** interpretable yet: against the matched all-data LoRA, removing
   bulk RNA moves scRNA −0.17, removing TCR (which shares no genes) −0.15, removing flow +0.09, and Xenium moves
   ±0.12 the same way. That pattern looks like training run-to-run variance; the bootstrap intervals cover eval
   sampling only. Repeat seeds are needed before reading any LoRA off-target effect as data-type transfer. So the LLM gets far more out of the entity's own
   values and the feature names than any classical model can.
3. **With no data of the type at all, only gene-linked types can be predicted, and only as rankings:** bulk RNA from
   other types' gene levels ρ 0.53, Xenium 0.45, scRNA 0.42; Olink, DIA and CITE-seq barely (≤ 0.12). Untrained Glimmer
   with no values answers a constant 0 for bulk RNA, TCR and Xenium, so classical gene links beat it there; Glimmer
   still wins on flow (0.67 vs no classical option).
4. **Learning "how to impute" on other types and applying it to a new type fails classically** (S1 and S3 transfer
   GBMs ≈ 0 or negative except CITE-seq/Xenium/metagenomics in S3): value scales and meanings differ too much
   between assays. This is exactly what the hold-out LoRA does manage.
5. **Time ordering adds nothing beyond the population average:** carry-forward from the previous draw is worse than
   the mean of all other draws (flow 0.913 vs 0.947, Olink 0.542 vs 0.595, TCR −0.04 vs 0.12 median), and the nearest
   draw either side only matches the mean. Forecasting the next draw from the last one alone looks harder than
   imputing a random gap; a proper time-split eval is still to do.

Caveats: within-10% rates in S1 are near 0 because gene scores are uncalibrated (ranks only); TCR "constant" cells
are a 73%-zero clonotype table where Spearman rewards getting the non-zero clones right; fusions have 21 targets.
