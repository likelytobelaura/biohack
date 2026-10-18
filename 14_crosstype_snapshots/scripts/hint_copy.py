"""How closely each model copies the prompt's "elsewhere in the dataset" hint (standard eval) -> out/hint_copy.json
copy rho = Spearman(prediction, hint median); hint rho = Spearman(hint median, truth)."""
import collections, glob, json, os, re, statistics as st, sys
sys.path.insert(0, os.path.expanduser('~/osteo-results/scripts'))
from score_compare import parse, num, spear
H = os.path.expanduser('~')
P = {}
for m, g in [('base', f'{H}/biohack/04_baseline_responses/baseline_zeroshot_merged.jsonl'),
             ('lomo740_all', f'{H}/osteo-results/lomo/evals/lomo740h100_all/shard*.jsonl'),
             ('snap', f'{H}/osteo-results/crosstype/evals/xtype/lomo740_all_snap_standard/shard*.jsonl')]:
    P[m] = {}
    for f in glob.glob(g):
        for l in open(f):
            r = json.loads(l); P[m][r['prompt_id']] = parse(r['text']) or {}


def hints(u):
    out, sec = {}, False
    for l in u.split('\n'):
        if l.startswith('The missing features elsewhere'): sec = True; continue
        if l.startswith('Missing values'): break
        if not (sec and l.startswith('- ') and ': ' in l): continue
        name, rest = l[2:].split(': ', 1)
        m = re.match(r'median ([^,]+),', rest)          # DIA-style summary line
        vals = [num(m.group(1))] if m else [num(x.rsplit('=', 1)[1]) for x in rest.split(', ') if '=' in x]
        vals = [v for v in vals if v is not None]
        if vals: out[name] = st.median(vals)
    return out


rows = collections.defaultdict(lambda: collections.defaultdict(list))
for l in open(f'{H}/biohack/02_golden_labels/golden.jsonl'):
    g = json.loads(l); h = hints(g['messages'][-1]['content'])
    for i, item in g['items'].items():
        tr = num(g['gold'][i])
        if item not in h or tr is None: continue
        for m in P:
            p = num(P[m].get(g['prompt_id'], {}).get(i))
            if p is not None: rows[g['data_type']][m].append((p, h[item], tr))
out = {}
for t, d in rows.items():
    r = d['snap']
    out[t] = {'n': len(r), 'hint_rho': spear([b for a, b, c in r], [c for a, b, c in r]),
              **{f'{m}_copy_rho': spear([a for a, b, c in d[m]], [b for a, b, c in d[m]]) for m in P},
              **{f'{m}_rho': spear([a for a, b, c in d[m]], [c for a, b, c in d[m]]) for m in P}}
    print(t, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[t].items()})
json.dump(out, open('out/hint_copy.json', 'w'), indent=1)
