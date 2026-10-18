"""All-model comparison on the standard eval for the report -> out/report_standard.json"""
import collections, glob, json, math, os, random, sys
sys.path.insert(0, os.path.expanduser('~/osteo-results/scripts'))
from score_compare import parse, num, spear, CAT, BIN
H = os.path.expanduser('~')
files = {'base': [f'{H}/biohack/04_baseline_responses/baseline_zeroshot_merged.jsonl'],
         'sid_full': [f'{H}/osteo-results/finetuned_sid_full_eval.jsonl'],
         'lomo740_all': glob.glob(f'{H}/osteo-results/lomo/evals/lomo740h100_all/shard*.jsonl'),
         'snap': glob.glob(f'{H}/osteo-results/crosstype/evals/xtype/lomo740_all_snap_standard/shard*.jsonl'),
         'spark12': [f'{H}/osteo-results/spark/muse-spark-1.2-contributor.jsonl']}
P = {}
for m, fs in files.items():
    p = {}
    for f in fs:
        for l in open(f):
            r = json.loads(l); p[r['prompt_id']] = parse(r.get('text') or r.get('response') or '') or {}
    P[m] = p
truth = collections.defaultdict(lambda: collections.defaultdict(list))
for l in open(f'{H}/biohack/02_golden_labels/truth.jsonl'):
    t = json.loads(l); truth[t['mod']][t['prompt_id']].append(t)

def pairs(m, t, pids):
    return [(P[m].get(x['prompt_id'], {}).get(str(x['i'])), x['truth']) for pid in pids for x in truth[t][pid]]

def prim(t, pr):
    pr = [(a, b) for a, b in pr if a is not None]
    if t in CAT: return sum(str(a).strip() == str(b).strip() for a, b in pr) / max(len(pr), 1)
    xy = [(num(a), num(b)) for a, b in pr]; xy = [(a, b) for a, b in xy if a is not None and b is not None]
    if t in BIN: return sum(round(a) == round(b) for a, b in xy) / max(len(xy), 1)
    return spear([a for a, b in xy], [b for a, b in xy])

def extra(t, pr):
    n = len(pr); ans = [(a, b) for a, b in pr if a is not None]
    xy = [(num(a), num(b)) for a, b in ans]; xy = [(a, b) for a, b in xy if a is not None and b is not None]
    err = sorted(abs(math.asinh(a) - math.asinh(b)) for a, b in xy)
    return {'answered': len(ans) / max(n, 1), 'mae_asinh': err[len(err) // 2] if err else None,
            'pred_zero': sum(a == 0 for a, b in xy) / max(len(xy), 1), 'truth_zero': sum(b == 0 for a, b in xy) / max(len(xy), 1)}

random.seed(0)
out = {}
for t in sorted(truth):
    pids = list(truth[t]); row = {'targets': sum(map(len, truth[t].values())), 'prompts': len(pids)}
    for m in P:
        pr = pairs(m, t, pids); row[m] = prim(t, pr); row[m + '_x'] = extra(t, pr)
    for a, b in [('snap', 'lomo740_all'), ('snap', 'sid_full'), ('lomo740_all', 'sid_full')]:
        ds = []
        for _ in range(400):
            s = [random.choice(pids) for _ in pids]
            x, y = prim(t, pairs(a, t, s)), prim(t, pairs(b, t, s))
            if x is not None and y is not None: ds.append(x - y)
        ds.sort(); row[f'{a}-{b}'] = [None if row[a] is None or row[b] is None else row[a] - row[b], ds[int(.025 * len(ds))], ds[int(.975 * len(ds)) - 1]]
    out[t] = row
cl = {r['key']: r for r in json.load(open(f'{H}/osteo-classical/compare_classical.json'))}
bs = json.load(open(f'{H}/biohack/05_baseline_report/baseline_scores.json'))
bs = {r['data_type']: r for r in (bs if isinstance(bs, list) else bs.values())} if not isinstance(bs, dict) or 'scrna' not in bs else bs
lomo = {r['key']: r['holdout'] for r in json.load(open(f'{H}/osteo-results/lomo/lomo_scores.json'))}
for t in out:
    out[t]['classical_best_by_val'] = cl.get(t, {}).get('best_by_val')
    out[t]['classical_pick'] = cl.get(t, {}).get('val_pick')
    out[t]['lomo_holdout'] = lomo.get(t)
    b = bs.get(t) or {}
    out[t]['train_median_ref'] = b.get('train_median_ref_spearman')
json.dump(out, open('out/report_standard.json', 'w'), indent=1)
f = lambda v: '  –  ' if v is None else f'{v:.3f}'
for t, r in out.items():
    print(f"{t:8s} base {f(r['base'])} sid {f(r['sid_full'])} l740 {f(r['lomo740_all'])} snap {f(r['snap'])} spark {f(r['spark12'])} "
          f"cls {f(r['classical_best_by_val'])}({r['classical_pick']}) med {f(r['train_median_ref'])} lomoHO {f(r['lomo_holdout'])} | "
          f"snap-l740 {f(r['snap-lomo740_all'][0])} [{f(r['snap-lomo740_all'][1])},{f(r['snap-lomo740_all'][2])}] snap-sid {f(r['snap-sid_full'][0])} [{f(r['snap-sid_full'][1])},{f(r['snap-sid_full'][2])}]")
