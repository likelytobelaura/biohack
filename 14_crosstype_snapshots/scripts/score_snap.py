"""Score the snapshot eval: per data type x condition x model, plus paired bootstrap of the context effects.

python score_snap.py evals/snapeval_base evals/snapeval_lomo740_all evals/snapeval_lomo740_all_snap  -> out/snap_scores.json
"""
import collections, csv, glob, json, math, os, random, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
labels = collections.defaultdict(dict)
for r in csv.DictReader(open(f'{HERE}/out/eval_snap_labels.csv')):
    labels[r['prompt_id'], r['cond']][int(r['i'])] = (r['data_type'], r['truth'])


def parse_ans(t):
    t = re.sub(r'^.*?</think>', '', t, flags=re.S)
    m = re.search(r'\{.*\}', t, re.S)
    if not m: return {}
    try:
        return {int(k): v for k, v in json.loads(m.group(0)).items() if str(k).isdigit()}
    except json.JSONDecodeError:
        return {int(k): v for k, v in re.findall(r'"(\d+)"\s*:\s*("?[^,"}]*"?)', m.group(0))}


def num(x):
    try:
        f = float(str(x).strip('"'))
        return f if math.isfinite(f) else None
    except ValueError:
        return None


def rank(a):
    o = sorted(range(len(a)), key=lambda i: a[i]); r = [0.0] * len(a); i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and a[o[j + 1]] == a[o[i]]: j += 1
        for k in range(i, j + 1): r[o[k]] = (i + j) / 2
        i = j + 1
    return r


def spear(x, y):
    if len(x) < 3: return None
    rx, ry = rank(x), rank(y); mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx)); sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return None if sx == 0 or sy == 0 else sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / (sx * sy)


def load(d):
    out = {}
    for f in glob.glob(f'{d}/shard*.jsonl'):
        for l in open(f):
            x = json.loads(l); out[x['prompt_id'], x['cond']] = parse_ans(x['text'])
    return out


def pairs(ans, cond, t, pids=None):
    """(pid, pred, truth) numeric pairs; HLA scored as exact match separately."""
    out = []
    for (pid, c), lab in labels.items():
        if c != cond or (pids is not None and pid not in pids): continue
        a = ans.get((pid, c), {})
        for i, (dt, tr) in lab.items():
            if dt != t: continue
            out.append((pid, a.get(i), tr))
    return out


def metrics(pr, t):
    if t == 'hla':
        return {'n': len(pr), 'exact': sum(str(p).strip('"') == tr for _, p, tr in pr) / max(len(pr), 1)}
    xs, ys, err = [], [], []
    for _, p, tr in pr:
        p, tr = num(p), num(tr)
        if p is None or tr is None: continue
        xs.append(p); ys.append(tr); err.append(abs(math.asinh(p) - math.asinh(tr)))
    err.sort()
    return {'n': len(pr), 'answered': len(xs), 'rho': spear(xs, ys), 'mae_asinh': err[len(err) // 2] if err else None}


def boot_diff(ans, t, c1, c2, B=500):
    pids = sorted({pid for (pid, c) in labels if labels[pid, c] and any(dt == t for dt, _ in labels[pid, c].values())})
    if t == 'hla' or len(pids) < 10: return None
    r = random.Random(0); ds = []
    by = {c: collections.defaultdict(list) for c in (c1, c2)}
    for c in (c1, c2):
        for pid, p, tr in pairs(ans, c, t):
            by[c][pid].append((pid, p, tr))
    for _ in range(B):
        s = [r.choice(pids) for _ in pids]
        m1 = metrics([x for pid in s for x in by[c1][pid]], t)['rho']; m2 = metrics([x for pid in s for x in by[c2][pid]], t)['rho']
        if m1 is not None and m2 is not None: ds.append(m1 - m2)
    ds.sort()
    return [ds[int(.025 * len(ds))], ds[int(.975 * len(ds))]] if ds else None


res = {}
types = sorted({dt for lab in labels.values() for dt, _ in lab.values()})
for d in sys.argv[1:]:
    name = os.path.basename(d.rstrip('/')).replace('snapeval_', '')
    ans = load(d)
    res[name] = {}
    for t in types:
        row = {c: metrics(pairs(ans, c, t), t) for c in ('full', 'own', 'swap')}
        if t != 'hla':
            row['full_minus_own'] = (row['full']['rho'] or 0) - (row['own']['rho'] or 0)
            row['full_minus_swap'] = (row['full']['rho'] or 0) - (row['swap']['rho'] or 0)
            row['ci_full_minus_own'] = boot_diff(ans, t, 'full', 'own')
            row['ci_full_minus_swap'] = boot_diff(ans, t, 'full', 'swap')
        res[name][t] = row
json.dump(res, open(f'{HERE}/out/snap_scores.json', 'w'), indent=1)

f3 = lambda v: '   –  ' if v is None else f'{v:6.3f}'
for name, rs in res.items():
    print(f'\n== {name}')
    print(f"{'type':8s} {'n':>5s} {'full':>6s} {'own':>6s} {'swap':>6s} | {'f-own':>6s} {'95% CI':>15s} | {'f-swap':>6s} {'95% CI':>15s}")
    for t, r in rs.items():
        if t == 'hla':
            print(f"{t:8s} {r['full']['n']:5d} exact {r['full']['exact']:.2f} / {r['own']['exact']:.2f} / {r['swap']['exact']:.2f}"); continue
        ci = lambda c: '' if not c else f'[{c[0]:+.2f},{c[1]:+.2f}]'
        print(f"{t:8s} {r['full']['n']:5d} {f3(r['full']['rho'])} {f3(r['own']['rho'])} {f3(r['swap']['rho'])} | "
              f"{r['full_minus_own']:+6.3f} {ci(r['ci_full_minus_own']):>15s} | {r['full_minus_swap']:+6.3f} {ci(r['ci_full_minus_swap']):>15s}")
