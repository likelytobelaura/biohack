# Results

**Bottom line.** Continuing `lomo740_all` on cross-type snapshots made the model much better at the snapshot
format (own-only ρ up .10–.46 on most types) but it still does **not use the other assays**: with the same
timepoint's / same cell's other data added, no type improves meaningfully over the same prompt without it
(largest: DIA +.017, TCR +.016, Olink +.010, all CIs touching 0), and the right timepoint is no better than a
wrong one. What changed: other assays no longer *hurt* CITE (−.061 before → +.008 after). On the original
prompts it gained on bulk RNA / CITE / scRNA and lost most of its TCR gain (−.25). Single seed, one pass.

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
| mIF (accuracy) | 1003 | .898 | .895 | .893 | +.003 [−.01, +.02] | +.005 [−.01, +.02] |
| Xenium | 600 | .516 | .535 | .510 | −.019 [−.08, +.03] | +.005 [−.07, +.06] |
| DIA | 2000 | .034 | .054 | .041 | −.020 [−.08, +.04] | −.006 [−.06, +.05] |

- No type gains significantly from same-timepoint context; the same cell's RNA makes CITE predictions *worse*
  (−.06). TCR's full − swap is at the edge (+.038, CI touching 0). MetaPhlAn points up but its CI crosses 0.
  (mIF targets are 0/1 positivity, scored by accuracy as in the standard eval.)
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

### Snapshot eval — lomo740_all_snap

| type | n | full | own | swap | full − own [95% CI] | full − swap [95% CI] | own: before → after |
|---|---:|---:|---:|---:|---|---|---|
| CITE | 1004 | .590 | .582 | .570 | +.008 [−.02, +.03] | +.019 [−.01, +.05] | .358 → .582 |
| scRNA | 1005 | .634 | .639 | .643 | −.005 [−.03, +.02] | −.009 [−.03, +.01] | .432 → .639 |
| Olink | 2000 | .635 | .625 | .636 | +.010 [−.00, +.02] | −.001 [−.01, +.01] | .470 → .625 |
| flow | 221 | .982 | .981 | .937 | +.001 [−.00, +.00] | +.045 [−.00, +.12] | .810 → .981 |
| TCR | 2000 | .277 | .261 | .261 | +.016 [−.00, +.04] | +.016 [+.00, +.03] | .160 → .261 |
| MetaPhlAn | 364 | .472 | .471 | .464 | +.000 [−.05, +.04] | +.007 [−.06, +.07] | .130 → .471 |
| bulk RNA | 2000 | .642 | .657 | .630 | −.015 [−.03, −.00] | +.011 [−.00, +.03] | .584 → .657 |
| somatic | 501 | .905 | .909 | .907 | −.005 [−.02, +.01] | −.002 [−.01, +.01] | .729 → .909 |
| mIF (accuracy) | 1003 | .896 | .926 | .904 | **−.030** [−.05, −.01] | −.008 [−.02, +.00] | .895 → .926 |
| Xenium | 600 | .590 | .604 | .597 | −.014 [−.05, +.02] | −.007 [−.03, +.02] | .535 → .604 |
| DIA | 2000 | .535 | .518 | .510 | +.017 [−.01, +.05] | +.025 [−.00, +.05] | .054 → .518 |

- **Format learned, connections not.** Own-only scores jump (DIA .05 → .52, CITE .36 → .58, MetaPhlAn .13 → .47),
  but adding the matching other assays adds ≤ .02 anywhere, and swapping in a wrong timepoint costs about the same
  (flow −.045 is the largest, CI touching 0).
- Same-cell RNA ↔ antibody (the link with the most data, ~3,200 training cells) gives nothing either way: CITE
  +.008, scRNA −.005.
- mIF: tumour-wide context makes it call fewer cells positive (accuracy .926 → .896).
- 73 / 12,720 answers hit the 8k-token retry limit.

### Reading
One pass over 5,904 snapshots teaches the format; it does not make the model read across assays. Within one
patient there are only ~19 timepoints, so blood-level links have little to learn from; the same-cell link has
thousands of examples and still shows nothing, which suggests the objective (mostly own-type targets, context
values copied from the prompt) gives no pressure to use the other modality. Next options: targets that can
*only* be predicted from the other assay (e.g. mask a whole assay at a timepoint / all ADTs of a cell given its
RNA), a data-only control (same rows, single-type format) to separate format from data, and a second seed.

## Full report
In-depth write-up with charts (standard eval vs base / sid_full / lomo740_all / Spark / classical / LOMO, snapshot
eval, why TCR / DIA / Olink got worse): https://claude.ai/code/artifact/b5fab546-41aa-4c46-9642-dfbd8340cad7
Numbers: `report_standard.json` (`scripts/report_standard.py`), `hint_copy.json` (`scripts/hint_copy.py`: how closely
each model copies the prompt's "elsewhere" hint; TCR copy ρ −.113 → +.253, DIA .964 → .860).
