# Xenium hold-out result: why it comes from the metric, not transfer (2026-10-17)

In the leave-one-data-type-out table (`osteo_results_lomo/lomo_matrix.json`) the run that **never saw Xenium** scores best on Xenium:

| model | Xenium Spearman ρ |
|---|---|
| base (zero-shot) | 0.472 |
| sid_full fine-tune | 0.295 |
| all-data reference (`lomo740h100_all`) | 0.461 |
| **no_xenium** (`lomo740_xenium_full`) | **0.620** |
| no_rna / no_tcr | 0.342 / 0.338 |

This doesn't show that held-out training helps. It comes from how the task and the metric are set up. Re-run with
`python 06_heldout_experiments/xenium_metric_check/xenium_check.py` (stdlib only; it reuses `scripts/score_compare.py`, so the ρ values match `score_lomo.py` exactly). Output: `output.txt`.

## What a Xenium row is

One segmented cell from the T0 tumour section (block B3), 428-gene in-situ panel. Value = integer transcript count. The prompt gives:

- the cell's nonzero genes (e.g. `COL1A1 = 109`, `SPARC = 105`, …) and its total counts / detected features;
- for each gene to predict, a **dataset-wide hint**: `LUM: library mean 16.02, detected in 54% of cells`.

The eval slice (`lomo740_xenium/baselines/`) has 600 prompts with 1 target each. Truth: 61% are 0, 80% are 0 or 1, and there are only 20 distinct values.

## Findings

**1. Copying the hint beats every model.** Pooled Spearman ρ against the truth:

| predictor | ρ | MAE | exact |
|---|---|---|---|
| hint: library mean | 0.688 | 1.084 | 0.000 |
| hint: library mean × cell total counts | **0.742** | (unscaled) | — |
| hint: detected % | 0.690 | — | — |
| hint: round(library mean) | 0.552 | 1.083 | 0.615 |
| always 0 | n/a | 1.447 | 0.607 |
| best model (no_xenium) | 0.620 | 1.242 | 0.653 |

Most of the signal is "which gene is this", and the prompt already states it. Very little comes from reading the cell.

**2. One pooled ρ over tied predictions mostly measures how often a model answers something other than 0.** Every model answers almost only 0s and 1s, and ρ falls as the share of 0 answers rises:

| model | answers that are 0 | distinct answers | ρ(answer, hint mean) | ρ vs truth |
|---|---|---|---|---|
| no_xenium | 74% | 3 | 0.679 | 0.620 |
| base | 90% | 16 | 0.480 | 0.472 |
| all | 92% | 8 | 0.457 | 0.461 |
| no_tcr | 95% | 9 | 0.342 | 0.338 |
| no_rna | 96% | 5 | 0.345 | 0.342 |
| sid_full | 97% | 7 | 0.280 | 0.295 |

`no_xenium` answers 1 for 153 of 600 targets (all-data: 43), and mostly for genes with a high library mean. That splits the block of tied zeros into two groups in roughly the right order, which is all Spearman rewards. The models trained on Xenium learn the true prior that "most counts are 0", so they put more answers in the tied block and lose ranking.

**3. On absolute error there is no win.** MAE: base 1.000, no_tcr 1.142, no_xenium 1.242, all 1.297, round(hint) 1.083. Exact match is 61–65% for every model, against 60.7% for always answering 0.

## Implications for future work on this data

- Don't read the Xenium column of the hold-out matrix as transfer, and treat "training on Xenium hurts Xenium" with the same suspicion. The drops for no_rna / no_tcr / sid_full are probably the same zero-answer effect.
- Score count data with a sparse-count-aware measure: ρ within each gene or each cell, a zero-inflated / Poisson likelihood, or separate zero-vs-nonzero and nonzero-magnitude scores.
- Always report the **hint baseline** (library mean × cell total, ρ 0.742) next to model scores, and score models as their gain over it. Only Xenium prompts carry the "library mean / detected %" hint. Bulk RNA, TCR and flow prompts instead list the same feature's values in the other entities (other libraries or dates), so build the matching copy-the-hint baseline for those too.
- With 1 target per prompt, a cell's own profile barely matters. Several targets per cell (as in training rows, about 12) would test whether the model reads the cell.
