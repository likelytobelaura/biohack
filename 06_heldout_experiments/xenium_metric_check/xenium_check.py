"""Why the Xenium hold-out scores best on Xenium: python 06_heldout_experiments/xenium_metric_check/xenium_check.py
Compares each model with copying the prompt's own library-mean hint, and shows each model's answer distribution."""
import json, glob, re, os, sys, collections, statistics
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(H), 'scripts'))
from score_compare import parse, num, spear

B = f'{H}/lomo740_xenium/baselines'
truth = [json.loads(l) for l in open(f'{B}/truth_xenium.jsonl')]
hint = {}
for l in open(f'{B}/prompts_xenium.jsonl'):
    r = json.loads(l); u = r['messages'][-1]['content']
    tot = float(re.search(r'total visible counts / detected features\) = ([\d.]+)', u).group(1))
    for m in re.finditer(r'- (\S+): library mean ([\d.e+-]+), detected in (\d+)%', u):
        hint[(r['prompt_id'], m.group(1))] = (float(m.group(2)), float(m.group(3)), tot)


def preds(files):
    p = {}
    for f in files:
        for l in open(f):
            r = json.loads(l); p[r['prompt_id']] = parse(r['text']) or {}
    return p


models = {'base': preds([f'{B}/base_zeroshot_xenium.jsonl']),
          'sid_full': preds([f'{B}/sid_full_finetune_xenium.jsonl']),
          'all': preds(glob.glob(f'{H}/lomo740h100_all/outputs/shard*.jsonl')),
          'no_xenium': preds(glob.glob(f'{H}/lomo740_xenium_full/outputs/shard*.jsonl')),
          'no_rna': preds(glob.glob(f'{H}/lomo740_rna_full/outputs/shard*.jsonl')),
          'no_tcr': preds(glob.glob(f'{H}/lomo740_tcr_full/outputs/shard*.jsonl'))}
tv = [float(t['truth']) for t in truth]
hs = [hint[(t['prompt_id'], t['item'])] for t in truth]
print(f'{len(truth)} targets; truth zero {sum(v == 0 for v in tv) / len(tv):.2f}, <=1 {sum(v <= 1 for v in tv) / len(tv):.2f}, {len(set(tv))} distinct')
for name, f in [('library mean', lambda h: h[0]), ('library mean x cell total', lambda h: h[0] * h[2]),
                ('detected %', lambda h: h[1]), ('round(library mean)', lambda h: round(h[0])), ('all zero', lambda h: 0.0)]:
    a = [f(h) for h in hs]
    r = None if len(set(a)) == 1 else spear(a, tv)
    print(f"hint {name:26s} rho {'n/a' if r is None else f'{r:.3f}'}  MAE {statistics.mean(abs(x - y) for x, y in zip(a, tv)):.3f}  "
          f"exact {sum(x == y for x, y in zip(a, tv)) / len(tv):.3f}")
for k, p in models.items():
    pr = [(num(p.get(t['prompt_id'], {}).get(str(t['i']))), float(t['truth']), h) for t, h in zip(truth, hs)]
    pr = [x for x in pr if x[0] is not None]
    a = [x[0] for x in pr]
    print(f"{k:10s} n={len(pr)} rho {spear(a, [x[1] for x in pr]):.3f}  MAE {statistics.mean(abs(x[0] - x[1]) for x in pr):.3f}  "
          f"exact {sum(x[0] == x[1] for x in pr) / len(pr):.3f}  zeros {sum(v == 0 for v in a) / len(a):.2f}  distinct {len(set(a))}  "
          f"rho(answer, hint mean) {spear(a, [x[2][0] for x in pr]):.3f}  top {collections.Counter(a).most_common(3)}")
