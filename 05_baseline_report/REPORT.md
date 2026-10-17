# Baseline report — Muse-Glimmer-30B zero-shot on the 20% mask

Source: `scripts/score.py` over `04_baseline_responses/baseline_zeroshot_merged.jsonl`
against `02_golden_labels/truth.jsonl`. Raw output: `score_py_output.txt`; per-type numbers:
`baseline_scores.json` (re-run 2026-10-17, byte-identical to the original).

**Coverage:** 24,634 masked targets in 15,964 prompts; 24,534 answered (99.6%).
The 100 missing are Olink (80) and somatic (20): the model replied without a parsable value.

| data type | targets | answered | Spearman ρ | median abs err (asinh) | within 10% | acc / exact | train-median ρ | train-median err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scRNA-seq | 6,800 | 100% | 0.453 | 0.000 | 0.65 | – | 0.580 | 0.375 |
| CITE-seq | 6,000 | 100% | 0.563 | 0.881 | 0.21 | – | 0.630 | 0.802 |
| Bulk RNA | 2,000 | 100% | 0.487 | 0.017 | 0.50 | – | 0.744 | 0.061 |
| TCR | 2,000 | 100% | 0.130 | 0.000 | 0.73 | – | 0.297 | 0.000 |
| DIA proteomics | 2,000 | 100% | **0.944** | 0.158 | 0.34 | – | 0.916 | 0.189 |
| Multiplex IF | 2,000 | 100% | **0.602** | 0.000 | 0.92 | 0.92 | 0.555 | 0.000 |
| Olink | 2,000 | 96% | 0.537 | 0.394 | 0.08 | – | 0.573 | 0.379 |
| Xenium | 600 | 100% | 0.472 | 0.000 | 0.61 | – | 0.614 | 0.562 |
| Somatic WGS | 501 | 96% | 0.813 | 0.000 | 0.70 | – | 0.840 | 0.000 |
| Metagenomics | 364 | 100% | 0.146 | 6.195 | 0.02 | – | 0.661 | 0.995 |
| Flow cytometry | 221 | 100% | **0.970** | 0.022 | 0.63 | – | 0.949 | 0.196 |
| HLA typing | 127 | 100% | – | – | – | 0.54 | – | – |
| Gene fusions | 21 | 100% | 0.502 | 0.600 | 0.19 | – | 0.541 | 0.510 |

"Train-median" is a no-model reference: predict the median training value of the same item.
Its ρ/err only covers targets that item has training values for (`ref_coverage` in the JSON).

## Reading it

- Zero-shot beats the train-median reference on rank correlation for only three types:
  DIA proteomics, multiplex IF and flow cytometry. Everywhere else the base model is below
  a lookup of training medians, so fine-tuning has an easy bar to clear.
- The largest gaps are metagenomics (ρ 0.15 vs 0.66, error 6.2 vs 1.0), bulk RNA (0.49 vs 0.74)
  and TCR (0.13 vs 0.30).
- A median error of 0.000 with low ρ (scRNA, TCR, Xenium) means most targets are true zeros
  and the model guesses 0; the non-zero values are where it loses.
- 54 prompts were re-run after truncation (`baseline_zeroshot_retry.jsonl`); `_merged` is the
  original with those 54 replaced and is the file every score uses.
