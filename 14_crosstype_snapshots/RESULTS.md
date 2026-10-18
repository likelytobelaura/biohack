# Results (in progress)

Snapshot eval: pooled Spearman ρ on the gold targets; full = own values + same-timepoint/same-cell context,
own = context removed, swap = context from another timepoint/cell. 95% CIs = paired bootstrap over prompts.
Full table: `snap_scores.json`. `score_snap.py` in `scripts/`.

## Before: does lomo740_all already use cross-type context? — No.

| type | n | full | own | swap | full − own [95% CI] | full − swap [95% CI] |
|---|---:|---:|---:|---:|---|---|
| CITE | 1004 | .297 | .358 | .269 | **−.061** [−.11, −.01] | +.028 [−.03, +.08] |
| scRNA | 1005 | .408 | .432 | .398 | −.024 [−.07, +.02] | +.010 [−.03, +.05] |
| Olink | 2000 | .483 | .470 | .467 | +.013 [−.01, +.04] | +.016 [−.01, +.04] |
| flow | 221 | .825 | .810 | .834 | +.014 [−.01, +.04] | −.009 [−.06, +.05] |
| TCR | 2000 | .198 | .160 | .161 | +.038 [−.01, +.09] | +.038 [+.00, +.07] |
| MetaPhlAn | 364 | .206 | .130 | .137 | +.076 [−.03, +.17] | +.069 [−.05, +.15] |
| bulk RNA | 2000 | .574 | .584 | .546 | −.010 [−.03, +.01] | +.027 [−.00, +.06] |
| somatic | 501 | .723 | .729 | .753 | −.006 [−.03, +.01] | −.030 [−.07, −.00] |
| mIF | 1003 | .396 | .314 | .296 | +.082 [−.03, +.20] | +.101 [−.02, +.21] |
| Xenium | 600 | .516 | .535 | .510 | −.019 [−.08, +.03] | +.005 [−.07, +.06] |
| DIA | 2000 | .034 | .054 | .041 | −.020 [−.08, +.04] | −.006 [−.06, +.05] |

- No type gains significantly from same-timepoint context; the same cell's RNA makes CITE predictions *worse*
  (−.06). TCR's full − swap is at the edge (+.038, CI touching 0). mIF/MetaPhlAn point up but CIs cross 0.
- Base model: no context effect either; it answers 0 for every bulk RNA / TCR / MetaPhlAn target in this format
  (ρ undefined), and extra assays hurt it on somatic (−.20) and Xenium (−.24).
- Absolute ρ is below the standard eval because the snapshot format drops the "elsewhere in the dataset" hint
  lines that the models mostly copy.

## After: lomo740_all_snap (continued on the snapshot format)
Pending — evals running.
