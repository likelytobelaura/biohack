"""Standard 15,964-prompt eval: lomo740_all (before) vs lomo740_all_snap (after), per type, paired bootstrap of the difference.

python score_standard.py evals/xtype/lomo740_all_snap_standard  -> out/standard_scores.json
"""
import collections, glob, json, os, random, sys
sys.path.insert(0, os.path.expanduser('~/osteo-results/scripts'))
from score_compare import parse, num, spear, LABELS, CAT, BIN

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.expanduser('~/osteo-results/lomo/evals/lomo740h100_all')   # eval of /data/adapters/lomo/lomo740_all
BASE = os.path.expanduser('~/biohack/04_baseline_responses/baseline_zeroshot_merged.jsonl')


def preds(files):
    p = {}
    for f in files:
        for l in open(f):
            r = json.loads(l); p[r['prompt_id']] = parse(r['text']) or {}
    return p


def metric(t, pairs):
    if t in CAT:
        return sum(str(a).strip() == str(b).strip() for a, b in pairs) / max(len(pairs), 1)
    xy = [(num(a), num(b)) for a, b in pairs]
    xy = [(a, b) for a, b in xy if a is not None and b is not None]
    if t in BIN:
        return sum(round(a) == round(b) for a, b in xy) / max(len(xy), 1)
    return spear([a for a, b in xy], [b for a, b in xy])


models = {'base': preds([BASE]), 'before': preds(glob.glob(f'{REF}/shard*.jsonl')), 'after': preds(glob.glob(f'{sys.argv[1]}/shard*.jsonl'))}
assert len(models['after']) == 15964, len(models['after'])
truth = collections.defaultdict(lambda: collections.defaultdict(list))
for l in open(os.path.expanduser('~/biohack/02_golden_labels/truth.jsonl')):
    t = json.loads(l); truth[t['mod']][t['prompt_id']].append(t)
random.seed(0)
rows = []
for t in sorted(truth):
    bp = truth[t]; pids = list(bp)

    def score(pl):
        return {m: metric(t, [(p.get(x['prompt_id'], {}).get(str(x['i'])), x['truth']) for pid in pl for x in bp[pid]
                              if p.get(x['prompt_id'], {}).get(str(x['i'])) is not None]) for m, p in models.items()}
    pt = score(pids); ds = []
    for _ in range(500):
        s = score([random.choice(pids) for _ in pids])
        if s['after'] is not None and s['before'] is not None: ds.append(s['after'] - s['before'])
    ds.sort()
    row = {'type': t, 'label': LABELS.get(t, t), 'targets': sum(map(len, bp.values())),
           'metric': 'exact' if t in CAT else 'accuracy' if t in BIN else 'rho', **{m: pt[m] for m in models},
           'diff': None if None in (pt['after'], pt['before']) else pt['after'] - pt['before'],
           'diff_ci': [ds[int(.025 * len(ds))], ds[int(.975 * len(ds)) - 1]] if ds else None}
    rows.append(row)
    f = lambda v: '  –  ' if v is None else f'{v:.3f}'
    print(f"{t:8s} {row['targets']:6d} base {f(pt['base'])}  before {f(pt['before'])}  after {f(pt['after'])}  "
          f"diff {'' if row['diff'] is None else format(row['diff'], '+.3f')} {row['diff_ci'] and [round(x, 3) for x in row['diff_ci']]}")
json.dump(rows, open(f'{HERE}/out/standard_scores.json', 'w'), indent=1)
