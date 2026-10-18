# MetaPhlAn log-abundance linear probe (2026-10-17)

Write-up page: `microbiome-neuron-hunt.html`, published at https://claude.ai/artifact/LsWtBTqzj641ws2Jcfvzdf (private).

Does Muse-Glimmer-30B (base, and the `lomo740_all` LoRA) linearly encode the log10 read count of a microbiome taxon,
and is there a "microbiome neuron"? Run `v1`, one H200 on Modal (app `osteo-mpa-probe`).

## Data

`scripts/build_mpa_probe.py` builds 61 prompts from 14 samples (stool/saliva by date):
38 MetaPhlAn training rows `lomo740_all` never trained on (`06_heldout_experiments/osteo_results_lomo/splits`)
+ the 23 MetaPhlAn eval prompts with gold answers (`02_golden_labels`). Answers are teacher-forced. Label y = log10(count + 1).
The examples file is not committed, because it contains the gold labels; rebuild it with the script.

| position | n | what the model has seen |
|---|---|---|
| `answer` | 1,074 | everything up to the missing item's value (the decision point) |
| `known_name` | 3,438 | a known line's taxon name, before its value |
| `known_value` | 3,438 | the known value just read (sanity check) |

## Method (`scripts/mpa_probe.py`)

Residual stream (embeddings + 52 layers, d = 6656) and MLP neuron activations (19,968 per layer) are taken at each
position, for the base model (adapter disabled) and `lomo740_all`. The probes are kernel ridge on z-scored activations,
scored by 7-fold CV with whole samples held out. The baseline is a linear fit on the taxon's out-of-fold mean
plus its values in other samples listed in the prompt. `resid_r2` is the probe's gain over that baseline.
`scripts/test_probe_synthetic.py` plants a signal and a neuron to check the pipeline end to end.
The activations (~20 GB) and per-neuron correlations stay on the Modal volume `osteo-data` at `/outputs/mpa_probe/v1/`.

## Results (`results/results.json`)

| | R² | Spearman |
|---|---|---|
| baseline (taxon prior + elsewhere values) | 0.29 | 0.58 |
| probe, base, layer 39 | 0.55 | 0.74 |
| probe, `lomo740_all`, layer 39 | 0.57 | 0.76 |
| gain over baseline (best layer, both) | ~0.06 | |
| shuffled-label control | ≈ 0 | |

For comparison, the models' own text answers on the eval MetaPhlAn targets reach Spearman 0.15 (base) and
0.33 (`sid_full`) (`07_finetune_sid_full`, compare_microbiome). That is a different, smaller subset, so the
comparison is indicative only.

- **The abundance is encoded from about layer 20 onward (R² 0.3) and peaks at layers 36–44.** The fine-tune barely moves it.
  The base model already "knows" roughly as much as the fine-tune but fails to write it out (it answers 0 on 52% of targets).
- **The strongest single neurons are shared with base, so they are not microbiome-specific.** Layer 37 #5605 and
  layer 40 #13961 correlate r ≈ 0.71 with y, and still r ≈ 0.6 in the base model.
  They look like generic magnitude units.
- **The best fine-tune-specific candidate is layer 32 #16003.** It correlates r = 0.29 with what the baseline misses
  (0.06 in base). As a single-neuron probe it explains ~8% of that leftover variance.
- **At `known_name`, the probe (0.33) is worse than the taxon-prior baseline (0.44).** `known_value` gives R² 0.998: numbers are encoded.

Known issues, to fix with `--probe-only` on the saved activations:

- **Layer 0 at `answer` is NaN.** The token there is always the same, so the layer-0 features never vary.
- **The `resid_r2` and full-MLP-layer fits overfit (negative R²).** The ridge alpha is chosen by leave-one-out
  inside the train fold, which mixes rows from the same sample. It should use sample-grouped inner CV.

## How the probe can be used

1. **Diagnose what fine-tuning changed.** The base model encodes abundance about as well as the fine-tune
   (R² 0.55 vs 0.57) but writes poor answers. So the fine-tune mostly fixed the *readout* (no zeros, the right
   scale), not knowledge. The same check applies to any data type before deciding to fine-tune.
2. **Use it as a cheap imputer.** One forward pass to layer ~39 plus a dot product gives a log-abundance estimate,
   with no generation or JSON parsing. It ranks better than the models' own answers (ρ 0.74–0.76 vs 0.15–0.33).
   This needs a fair head-to-head on the same 364 eval targets before it can be claimed.
3. **Steer.** Adding α·w (the probe direction) at layer ~39 should shift the written count by a predictable amount
   if the model *uses* the direction. If it does, steering could correct the base model's zero bias without a LoRA.
   It is also the causal test the probe itself cannot give.
4. **Hunt neurons through the probe direction.** Project w onto each layer's `down_proj` columns to find which neurons
   write it. Compare w between base and fine-tune (cosine) to see whether the fine-tune rotated the direction or only
   rescaled its readout. Then ablate the candidates (#16003, #5605, #13961) and re-score.
5. **Check whether it is really microbiome.** Apply the same probe, without retraining, to bulk RNA, Olink or CITE
   answer positions. If it decodes those magnitudes too, it is a generic "quantity" axis, which is what the
   shared neurons suggest. If it does not, it is microbiome-specific.
6. **Flag uncertainty and odd data.** Large gaps between the probe and the written answer, or between the probe and
   the truth, flag answers to distrust and possibly odd taxa (contaminants, near-zero bins).

## Neuron hunt through the probe (`scripts/mpa_neuron.py`, `results/neuron_results.json`)

The probe was refit at residual index 39, tuned by holding out whole samples: CV R² 0.55 for base, 0.57 for the fine-tune.
Each MLP neuron in layers 0–38 writes a_i · postnorm(W_down[:, i]) into the residual stream. The post-FFN RMSNorm is
linear once the sample's rms is known (reconstruction error 0.2%), so each neuron's share of the probe readout's
variance is exact along the direct path. Controls are answer positions from bulk RNA, DIA proteomics, Xenium, CITE
and flow (959 values, unseen rows). The causal test greedy-decodes all 364 eval MetaPhlAn values with neurons
mean-ablated, or with the residual steered by ±1 log10 unit along w.

**Attribution.** All MLP neurons together carry 33% (base) and 29% (fine-tune) of the readout variance directly;
the rest comes through attention.

| | top 1 | top 5 | top 20 |
|---|---|---|---|
| base | L32 #8787 2.0% | 9% | 21% |
| fine-tune | L37 #18399 5.1% | 17% | 34% |

**Specificity.** The probe is not microbiome-specific. Trained only on MetaPhlAn, it still decodes log values
elsewhere: r = 0.76 on DIA, 0.73 on bulk RNA and 0.43–0.56 on Xenium, CITE and flow (fine-tune). It is mostly a
generic quantity axis.

Mean activation at answer positions (sd ~0.1–1.6):

| neuron (model) | MetaPhlAn | DIA | RNA | Xenium | CITE | flow | share of readout |
|---|---|---|---|---|---|---|---|
| **L36 #15557 (fine-tune)** | **2.09** | 0.37 | 0.20 | −0.52 | −0.29 | −0.30 | 3.4% (base 1.2%) |
| L36 #15557 (base) | 0.98 | −0.04 | 0.17 | −0.28 | 0.03 | −0.47 | |
| L37 #18399 (fine-tune) | 3.37 | 2.34 | 0.12 | 0.09 | 0.16 | 0.20 | 5.1% (base ~2%) |
| L37 #18399 (base) | 2.29 | 1.68 | 0.07 | 0.16 | 0.14 | 0.16 | |

- **L36 #15557 is the best "microbiome neuron" candidate.** It is the only top-attributed neuron that fires on MetaPhlAn
  answers and stays near zero on all five other types, including DIA, whose values are just as large.
  It exists in the base model, and fine-tuning roughly doubled its MetaPhlAn activation (0.98 → 2.09) and tripled its
  readout share (1.2% → 3.4%). Within MetaPhlAn it is anti-correlated with abundance (r = −0.60), so it pushes
  estimates down for low-abundance taxa.
- **L37 #18399 is a "large raw count" neuron.** It fires on MetaPhlAn and DIA (both values around 10³–10⁶) and tracks the
  value in both (r 0.58 / 0.75). The fine-tune amplified it, but it is not microbiome-specific.
- **L32 #8787 tracks value with an opposite sign per data type** (+0.69 MetaPhlAn and DIA, −0.4 to −0.8 RNA, Xenium and CITE).

**Causal test** (364 eval values, teacher-forced earlier answers; shift = mean change in written log10 value):

| | base: Spearman / shift / zeros | fine-tune: Spearman / shift / zeros |
|---|---|---|
| clean | 0.49 / – / 33% | 0.63 / – / 15% |
| ablate top 5 | 0.46 / −0.16 / 35% | 0.69 / −0.05 / 15% |
| ablate top 20 | 0.46 / −0.08 / 34% | 0.67 / −0.17 / 15% |
| ablate 20 random | 0.48 / 0.00 / 33% | 0.64 / 0.00 / 15% |
| steer +1 | 0.56 / +0.26 / 27% | 0.67 / +0.04 / 15% |
| steer −1 | 0.45 / −0.43 / 45% | 0.67 / −0.05 / 15% |

- **The base model uses the probe direction.** Steering moves its written values and pushing up cuts zero answers.
- **The fine-tune hardly responds to steering at layer 39.** Its output is driven elsewhere, most likely by copying the
  other samples' values through attention.
- **Ablations of the top neurons have small effects, with direction consistent with the attribution.** Random ablations do nothing.
- **There is no single neuron whose removal breaks microbiome prediction.** The representation is distributed.

Next: ablate L36 #15557 alone and measure the effect per data type. Patch fine-tune activations of #15557 and #18399
into the base model. Check whether #15557 also fires on MetaPhlAn *prompt* tokens (taxon names), not just answers.

## Re-run

```
python scripts/build_mpa_probe.py examples.jsonl
modal volume put osteo-data examples.jsonl /mpa_probe/<tag>/examples.jsonl
modal run --detach scripts/mpa_probe.py --tag <tag> [--probe-only]
python scripts/build_control_examples.py <lomo740_all_unseen_rows.jsonl> control_examples.jsonl
modal volume put osteo-data control_examples.jsonl /mpa_probe/<tag>/control_examples.jsonl
modal run --detach scripts/mpa_neuron.py --tag <tag>     # ~40 min on one H200
```
