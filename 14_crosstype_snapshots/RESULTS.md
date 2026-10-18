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

### Standard eval (15,964 original prompts, with hint lines) — before = lomo740_all, after = lomo740_all_snap
Paired bootstrap over prompts (500); `standard_scores.json`, `scripts/score_standard.py`.

| type | targets | base | before | after | after − before [95% CI] |
|---|---:|---:|---:|---:|---|
| bulk RNA | 2000 | .487 | .694 | .761 | **+.068** [+.05, +.09] |
| CITE | 6000 | .563 | .523 | .562 | **+.039** [+.02, +.06] |
| scRNA | 6800 | .453 | .598 | .634 | **+.036** [+.02, +.05] |
| Xenium | 600 | .472 | .461 | .592 | +.131 [+.07, +.20] (pooled-ρ zeros artifact, see PR #4) |
| MetaPhlAn | 364 | .146 | .428 | .525 | +.097 [−.05, +.26] |
| mIF (acc) | 2000 | .923 | .892 | .898 | +.006 [+.00, +.01] |
| HLA (exact) | 127 | .535 | .518 | .536 | +.018 [.00, +.04] |
| flow | 221 | .970 | .973 | .973 | .000 |
| somatic | 501 | .813 | .933 | .930 | −.003 [−.02, +.02] |
| Olink | 2000 | .537 | .776 | .757 | **−.019** [−.04, −.01] |
| DIA | 2000 | .944 | .914 | .824 | **−.090** [−.11, −.07] |
| fusion | 21 | .502 | .281 | .179 | −.102 (n = 21) |
| TCR | 2000 | .130 | .526 | .276 | **−.250** [−.38, −.16] |

- TCR loses most of its fine-tune gain: that gain was the "clone is zero in the other months → present" rule, which
  is read from the hint lines; the snapshot format drops those lines, so the rule was not reinforced.
- Gains on bulk RNA / CITE / scRNA come from a second pass over new training rows; on their own they are not
  evidence of cross-type learning (no data-only control was run). The snapshot eval below is the test for that.

### Snapshot eval
Pending — evals running.
