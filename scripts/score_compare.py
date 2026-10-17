import json, re, sys, math, random, collections, statistics as st
from common import D, load

LABELS = {'scrna': 'scRNA-seq (10x)', 'cite': 'CITE-seq tags', 'rna': 'Bulk RNA-seq TPM', 'tcr': 'TCR beta clones',
          'dia': 'DIA proteomics', 'mif': 'Multiplex IF', 'olink': 'Olink plasma NPX', 'xenium': 'Xenium',
          'somatic': 'Somatic WGS calls', 'mpa': 'Metagenomics', 'flow': 'Flow cytometry', 'hla': 'HLA typing',
          'fusion': 'Gene fusions'}
CAT, BIN = {'hla'}, {'mif'}


def parse(t):
    for m in reversed(list(re.finditer(r'\{[^{}]*\}', t))):
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return None


def num(x):
    try:
        return float(x)
    except Exception:
        return None


def spear(x, y):
    if len(x) < 3:
        return None
    def rank(a):
        o = sorted(range(len(a)), key=lambda i: a[i]); r = [0.0] * len(a); i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and a[o[j + 1]] == a[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = rank(x), rank(y); mx, my = st.mean(rx), st.mean(ry)
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / den if den else None


def score(pred_file, truth):
    pred = {}
    for l in open(pred_file):
        r = json.loads(l); pred[r['prompt_id']] = parse(r['text']) or {}
    out, pts = {}, collections.defaultdict(list)
    by = collections.defaultdict(list)
    for t in truth:
        by[t['mod']].append(t)
    for lab, ts in by.items():
        ans = 0; xs = []; ys = []; w10 = 0; hit = 0
        for t in ts:
            p = pred.get(t['prompt_id'], {}).get(str(t['i']))
            if p is None:
                continue
            ans += 1
            if lab in CAT:
                hit += str(p).strip() == str(t['truth']).strip(); continue
            pv, tv = num(p), num(t['truth'])
            if pv is None or tv is None:
                continue
            xs.append(pv); ys.append(tv); pts[lab].append((tv, pv))
            w10 += abs(pv - tv) <= 0.1 * abs(tv) + 1e-9
            if lab in BIN:
                hit += round(pv) == round(tv)
        r = {'targets': len(ts), 'answered': ans / len(ts)}
        if lab in CAT:
            r['primary'] = hit / max(ans, 1); r['metric'] = 'exact match'
        elif lab in BIN:
            r['primary'] = hit / max(len(xs), 1); r['metric'] = 'accuracy'
        else:
            r['primary'] = spear(xs, ys); r['metric'] = 'Spearman ρ'
        r['within_10pct'] = None if lab in CAT else w10 / max(len(xs), 1)
        r['median_abs_err_asinh'] = None if lab in CAT or not xs else st.median(abs(math.asinh(a) - math.asinh(b)) for a, b in zip(xs, ys))
        out[lab] = r
    return out, pts


if __name__ == '__main__':
    files = dict(a.split('=', 1) for a in sys.argv[1:])
    truth = [json.loads(l) for l in open(D + '/truth.jsonl')]
    res = {m: score(f, truth) for m, f in files.items()}
    names = list(files)
    rows = []
    for lab in LABELS:
        row = {'data_type': LABELS[lab], 'key': lab, 'metric': res[names[0]][0][lab]['metric'],
               'targets': res[names[0]][0][lab]['targets']}
        for m in names:
            r = res[m][0][lab]
            for k in ('primary', 'answered', 'within_10pct', 'median_abs_err_asinh'):
                row[f'{m}_{k}'] = None if r[k] is None else round(r[k], 4)
        if row.get(f'{names[0]}_primary') is not None and row.get(f'{names[-1]}_primary') is not None:
            row['delta'] = round(row[f'{names[-1]}_primary'] - row[f'{names[0]}_primary'], 4)
        rows.append(row)
    random.seed(0); pts = []
    for m in names:
        for lab, p in res[m][1].items():
            for tv, pv in random.sample(p, min(400, len(p))):
                pts.append({'model': m, 'data_type': LABELS[lab], 'truth': tv, 'pred': pv})
    json.dump(rows, open('/Users/laurabocek/osteo-results/compare_scores.json', 'w'), indent=1)
    json.dump(pts, open('/Users/laurabocek/osteo-results/compare_points.json', 'w'))
    for r in rows:
        print(f"{r['data_type']:20s} {r['metric']:11s} " + '  '.join(f"{m}={r[f'{m}_primary']}" for m in names) + f"  Δ={r.get('delta')}")
