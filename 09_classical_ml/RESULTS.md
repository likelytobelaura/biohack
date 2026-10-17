# Classical ML imputation vs Muse-Glimmer-30B on Sid's osteosarcoma eval (17 Oct 2026)

Same 24,634 masked targets and scorer (`osteo-results/scripts/score_compare.py`) as the Glimmer runs.
Classical methods see only what Glimmer saw: training rows (with answers) plus visible values in eval prompts;
0 eval targets are in that pool (checked in `scripts/build_tables.py`). Values are modelled in asinh space.

Methods (`scripts/run_classical.py`): feature mean/median, entity mean, additive entity+feature effects,
entity-kNN and feature-kNN (k = 5/10/25/50), biased low-rank ALS (rank 2/8/32, λ 1/10), a gradient-boosted stack of
all of them; HLA: modal allele / nearest-caller vote. The per-type pick is chosen on a 10% split of visible values,
never on eval targets. Full table: `compare_classical.json` (all 22 methods x 13 types).

| Data type | Glimmer base | Glimmer FT | Classical (val pick) | Pick | Targets whose feature was never observed |
|---|---|---|---|---|---|
| CITE-seq | 0.563 | 0.551 | **0.770** | ALS r32 | 0% |
| Xenium | 0.472 | 0.295 | **0.680** | feature-kNN 25 | 0.5% |
| Gene fusions | 0.502 | 0.358 | **0.698** | ALS r2 | 67% |
| Bulk RNA TPM | 0.487 | 0.678 | **0.749** | feature median | 0% |
| Olink NPX | 0.537 | 0.748 | **0.800** | ALS r2 | 0% |
| Flow cytometry | 0.970 | 0.947 | 0.971 | ALS r32 | 0% |
| Multiplex IF (acc.) | 0.923 | 0.892 | 0.916 | feature median | 0% |
| DIA proteomics | **0.944** | 0.933 | 0.868 | feature-kNN 25 | 6.8% |
| TCR clones | 0.130 | **0.478** | 0.416 | GBM stack | 3.5% |
| scRNA-seq | 0.453 | **0.534** | 0.401 | GBM stack | 37% |
| Somatic WGS | 0.813 | **0.915** | 0.649 | ALS r32 | 35% |
| Metagenomics | 0.146 | **0.327** | 0.192 | entity-kNN 5 | 100% |
| HLA (exact) | **0.535** | 0.441 | no prediction (abstains) | — | 100% (whole genes masked) |

Takeaways: on dense tables where every masked feature was seen in other samples (CITE, Xenium, bulk RNA, Olink),
simple matrix methods beat Glimmer, base and fine-tuned, by 0.05–0.21. Glimmer wins where masked features
were never observed anywhere (HLA, metagenomics, much of scRNA/somatic): there it uses prior knowledge of gene/species
names, which no data-only method can.

## ET-SoC-1 card run (aifoundry3 card 0)
CITE-seq entity-kNN's similarity matrix S = A·Aᵀ (A = 8656 x 400, 60 GFLOP) computed on the card: 373 ms per launch,
160 GFLOP/s, max abs error 4.2e-7 vs a float64 reference on the Mac. kNN predictions from the card's S are identical to the
CPU's (ρ = 0.772 at k = 10, 0.773 at k = 25). Code, notes and output: `card/from_aifoundry3/` (NOTES.md, aat/), driver
`scripts/card_knn.py`.

## Cross-data-type prediction (`scripts/cross.py`, `cross_results.json`)

**Gene link:** other types' level of the same gene (mean asinh + detection rate) added to the gradient-boosted stack. The validation
set holds out random entries plus whole features, to mimic never-observed targets.

| Target | Within only | Within + other types | Notes |
|---|---|---|---|
| scRNA, genes observed elsewhere in scRNA (4,311) | 0.40* | **0.62** | driven by DIA proteomics (drop-one −0.17) and bulk RNA (−0.07) |
| scRNA, genes never observed in scRNA (2,489) | −0.05* | 0.05 | but bulk RNA TPM alone on its 611 covered targets: ρ 0.29, AUROC 0.82 for detected-vs-not |
| scRNA, all pooled | **0.40*** | 0.21 | pooled ρ falls: the never-observed predictions sit at a miscalibrated level |
| Bulk RNA | 0.765 | 0.779 | small gain, mostly from scRNA |
| DIA | 0.865 | 0.869 | never-observed proteins (136): 0.16 → 0.23 |
| Xenium | 0.681 | 0.685 | none |
| CITE-seq | 0.788 | 0.790 | none |
| Olink | 0.811 | 0.798 | slightly worse: plasma protein levels don't track tumour/cell RNA |

\* scRNA "within only" here is the earlier `gbm_stack` (validated on random entries); this script's own within-only stack scored −0.06 pooled.

**Date link** (cross-modal kNN between samples drawn within 21 days): no meaningful gain at 5–11 matched samples.
Olink→flow ρ 0.968 vs 0.959 (plain mean of other draws), MAE(asinh) 0.26 vs 0.30; flow→Olink, TCR→flow and flow→TCR
show no change; Olink/MetaPhlAn→CITE about +0.02, MetaPhlAn/Olink/flow→scRNA about 0.

**Takeaway:** other data types carry real information through shared genes, mainly for scRNA: bulk RNA and
proteomics tell you a gene's typical level, and bulk RNA even flags which never-observed scRNA genes a cell expresses.
Blood-draw-level links between flow, Olink, TCR and the microbiome add nothing detectable at this sample size.

## Hold-out settings (`scripts/holdout.py`, `HOLDOUT.md`, `holdout_results.json`, `preds_holdout/`)

Classical counterparts of the Glimmer hold-outs, on all 12 numeric types: **S1** nothing of the type anywhere (other types
only; matches context-free Glimmer), **S2** a new sample (the type's other samples, none of the entity's own values),
**S3** the type unseen in training, only the entity's own values (matches the leave-one-type-out LoRA).
Headline: other samples of the same type carry almost everything (S2 ≈ full data; scRNA 0.67, TCR 0.52, DIA 0.89 beat the
original pick); the entity's own values alone are weak classically (S3 0.1–0.4) where the hold-out LoRA reaches 0.66 on
bulk RNA and 0.95 on flow; with no data of the type, only gene-linked types rank at all (bulk RNA 0.53, Xenium 0.45,
scRNA 0.42); carry-forward from the previous draw is worse than the mean of all draws. Full tables in `HOLDOUT.md`.
