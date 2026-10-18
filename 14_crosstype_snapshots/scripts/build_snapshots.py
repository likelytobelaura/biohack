"""Build "Sid at timepoint X, some values masked" prompts.

train: one snapshot per lomo740_all-UNSEEN training row (5,904). The row's own known values + its own targets
       are kept; other assays from the same timepoint (and, for single cells, the same cell's other modality from
       the 10x h5) are added as context, with ~20% of the added values masked as extra targets.
       Context comes only from the unseen half (+ h5 for the cell partner); no gold (eval) value is ever used.
eval:  gold eval prompts (sampled per type) in 3 conditions with gold targets only:
       full  = own values + same-timepoint context
       own   = same snapshot format, other assays removed
       swap  = other assays taken from a different timepoint / a cell from another visit
       Context for eval comes from all training rows (both halves) minus gold.

python build_snapshots.py            -> out/train_snap.jsonl, out/eval_snap.jsonl, out/eval_snap_labels.csv, out/stats.json
"""
import collections, csv, glob, json, os, random, re, sys
import h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_x import *

rng = random.Random(7)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
os.makedirs(OUT, exist_ok=True)
SYS = ("Reasoning strength: low\nYou fill in missing values in a multi-omic dataset from one osteosarcoma patient. "
       "Use the known values given. Reply with only a JSON object mapping each missing item's number to your predicted value.")
VISIT_TYPES = ('olink', 'flow', 'tcr', 'mpa')
TUMOUR_TYPES = ('rna', 'somatic', 'fusion', 'dia')
N_CTX = {'olink': 30, 'flow': 30, 'tcr': 25, 'mpa': 20, 'rna': 30, 'somatic': 20, 'fusion': 10, 'dia': 30}
MASK_FRAC = 0.2
DESC = {k: p for k, p in TYPE}

gold = gold_set()


def load_rows(path):
    out = []
    for i, l in enumerate(open(path)):
        r = json.loads(l)
        p = parse(r['prompt'][-1]['content'])
        ans = json.loads(r['completion'][0]['content'])
        p['answers'] = [ans[str(k + 1)] for k in range(len(p['targets']))]
        p['row_id'] = r.get('row_id', i)
        out.append(p)
    return out


unseen = load_rows(UNSEEN)
full = load_rows(FULL)
descs = {p['type']: p['desc'] for p in full}


def fmt(v):
    return v if isinstance(v, str) else (str(int(v)) if float(v).is_integer() else f'{v:.4g}')


def make_pool(rows):
    pool = collections.defaultdict(dict)
    for p in rows:
        d = pool[p['type'], p['entity']]
        for n, v in p['known']:
            d[n] = v
        for n, v in zip(p['targets'], p['answers']):
            d[n] = fmt(v)
    for (t, e), d in pool.items():
        for n in [n for n in d if (t, e, n) in gold]:
            del d[n]
    return pool


pool_unseen, pool_all = make_pool(unseen), make_pool(full)

# ---------------- timepoints ----------------
# visits: blood/saliva/stool/cell-capture dates within 7 days are one visit
dates = sorted({date_of(p['type'], p['entity']) for p in unseen + full
                                          if date_of(p['type'], p['entity']) and len(date_of(p['type'], p['entity'])) == 10}
               | {date_of(r['data_type'], r['entity']) for r in csv.DictReader(open(LABELS))
                  if date_of(r['data_type'], r['entity']) and len(date_of(r['data_type'], r['entity'])) == 10})
import datetime as dt
D = lambda s: dt.date.fromisoformat(s)
visit_of_date, cur = {}, None
for d in dates:
    if cur is None or (D(d) - D(cur[-1])).days > 7:
        cur = [d]
    else:
        cur.append(d)
    visit_of_date[d] = cur
visit_of_date = {d: v[0] for d, v in visit_of_date.items()}
visit_label = collections.defaultdict(set)
for d, v in visit_of_date.items():
    visit_label[v].add(d)

def tp_of(t, e):
    d = date_of(t, e)
    if t == 'tcr':
        m = re.match(r'tcr_stim_(May|Aug)', e)
        if m: d = '2025-05' if m.group(1) == 'May' else '2025-08'
    if d and len(d) == 7:   # month -> the visit in that month with most assays
        cands = [v for v in visit_label if v.startswith(d)]
        if not cands: return None
        return 'V' + max(cands, key=lambda v: sum(1 for (tt, ee) in pool_all if tt in ('olink', 'flow', 'mpa')
                                                   and date_of(tt, ee) and visit_of_date.get(date_of(tt, ee)) == v))
    if d: return 'V' + visit_of_date[d]
    tm = tumour_of(t, e)
    return tm


_tp = {}
def tp(t, e):
    if (t, e) not in _tp: _tp[t, e] = tp_of(t, e)
    return _tp[t, e]


_ctx_index = {}
def ctx_entities(pool, key, own):
    """visit-/tumour-level context entities at timepoint `key` (not the row's own type+entity)."""
    if id(pool) not in _ctx_index:
        ix = collections.defaultdict(lambda: collections.defaultdict(list))
        for (t, e), d in pool.items():
            if t in VISIT_TYPES + TUMOUR_TYPES and d and tp(t, e):
                ix[tp(t, e)][t].append(e)
        _ctx_index[id(pool)] = ix
    out = collections.defaultdict(list)
    for t, es in _ctx_index[id(pool)].get(key, {}).items():
        if t in (VISIT_TYPES if key.startswith('V') else TUMOUR_TYPES):
            es = [e for e in es if (t, e) != own]
            if es: out[t] = es
    return out


def pick(d, n, r):
    """half nonzero-biased, half random features."""
    names = list(d)
    nz = [k for k in names if d[k] not in ('0', '0.0')]
    a = r.sample(nz, min(len(nz), n // 2 + n % 2))
    rest = [k for k in names if k not in set(a)]
    return a + r.sample(rest, min(len(rest), n - len(a)))


# ---------------- single-cell partner (same cell, other modality) ----------------
val = json.load(open(VALID))
h5cache = {}


def h5_for(sample):
    f = val.get(sample)
    if not f: return None
    f = H5DIR + os.path.basename(f)
    if not os.path.exists(f): return None
    if f not in h5cache:
        g = h5py.File(f, 'r')['matrix']
        bc = {b.decode(): i for i, b in enumerate(g['barcodes'][:])}
        ids = [x.decode() for x in g['features/id'][:]]
        nm = [x.decode() for x in g['features/name'][:]]
        ft = [x.decode() for x in g['features/feature_type'][:]]
        h5cache[f] = (g, bc, ids, nm, ft, g['indptr'][:])
    return h5cache[f]


ANTIGEN = {'CD3': 'CD3E', 'CD4': 'CD4', 'CD8': 'CD8A', 'CD19': 'CD19', 'CD14': 'CD14', 'CD16': 'FCGR3A', 'CD56': 'NCAM1',
           'CD45RA': 'PTPRC', 'CD45RO': 'PTPRC', 'CD45': 'PTPRC', 'CD11c': 'ITGAX', 'CD11b': 'ITGAM', 'CD25': 'IL2RA',
           'CD127': 'IL7R', 'CD279': 'PDCD1', 'TIGIT': 'TIGIT', 'CD20': 'MS4A1', 'CD27': 'CD27', 'CD28': 'CD28', 'CD38': 'CD38',
           'CD69': 'CD69', 'CD62L': 'SELL', 'CD197': 'CCR7', 'CD161': 'KLRB1', 'CD57': 'B3GAT1', 'HLA.DR': 'HLA-DRA',
           'CD1c': 'CD1C', 'CD123': 'IL3RA', 'CD141': 'THBD', 'CD303': 'CLEC4C', 'CD34': 'CD34', 'CD33': 'CD33', 'CD64': 'FCGR1A',
           'CD32': 'FCGR2A', 'CD335': 'NCR1', 'CD314': 'KLRK1', 'CD94': 'KLRD1', 'CD159a': 'KLRC1', 'CD159c': 'KLRC2', 'CD2': 'CD2',
           'CD5': 'CD5', 'CD7': 'CD7', 'CD26': 'DPP4', 'CD95': 'FAS', 'CD39': 'ENTPD1', 'CD73': 'NT5E', 'CD152': 'CTLA4',
           'CD223': 'LAG3', 'CD366': 'HAVCR2', 'KLRG1': 'KLRG1', 'CD183': 'CXCR3', 'CD194': 'CCR4', 'CD196': 'CCR6',
           'CD195': 'CCR5', 'CD185': 'CXCR5', 'CD184': 'CXCR4', 'CD86': 'CD86', 'CD80': 'CD80', 'CD274': 'CD274', 'CD40': 'CD40',
           'CD52': 'CD52', 'CD47': 'CD47', 'CD48': 'CD48', 'CD44': 'CD44', 'CD79b': 'CD79B', 'IgD': 'IGHD', 'IgM': 'IGHM',
           'CD21': 'CR2', 'CD22': 'CD22', 'CD24': 'CD24', 'CD10': 'MME', 'CD43': 'SPN', 'CD18': 'ITGB2', 'CD11a': 'ITGAL',
           'CD49d': 'ITGA4', 'CD54': 'ICAM1', 'CD58': 'CD58', 'CD99': 'CD99', 'CD9': 'CD9', 'CD81': 'CD81', 'CD82': 'CD82',
           'CD63': 'CD63', 'CD36': 'CD36', 'CD163': 'CD163', 'CD68': 'CD68', 'CD83': 'CD83', 'CD13': 'ANPEP', 'CD93': 'CD93',
           'CD35': 'CR1', 'TCR.AB': 'TRAC', 'TCR.Vd2': 'TRDV2', 'TCR.Va7.2': 'TRAV1-2', 'CD103': 'ITGAE', 'GPR56': 'ADGRG1',
           'CX3CR1': 'CX3CR1', 'CD244': 'CD244', 'CD226': 'CD226', 'CD96': 'CD96', 'CD134': 'TNFRSF4', 'CD137': 'TNFRSF9',
           'CD357': 'TNFRSF18', 'CD278': 'ICOS', 'CD154': 'CD40LG', 'CD122': 'IL2RB', 'CD124': 'IL4R', 'CD192': 'CCR2',
           'CD193': 'CCR3', 'CD186': 'CXCR6', 'CD181': 'CXCR1', 'CD1d': 'CD1D', 'CD158': 'KIR2DL1', 'CD85j': 'LILRB1',
           'FceRIa': 'FCER1A', 'CD304': 'NRP1', 'CD23': 'FCER2', 'CD268': 'TNFRSF13C', 'CD267': 'TNFRSF13B',
           'CD269': 'TNFRSF17', 'CD138': 'SDC1', 'CD41': 'ITGA2B', 'CD61': 'ITGB3', 'CD62P': 'SELP', 'CD235ab': 'GYPA',
           'CD66b': 'CEACAM8', 'CD117': 'KIT', 'CD135': 'FLT3', 'CD90': 'THY1', 'CD105': 'ENG', 'CD31': 'PECAM1',
           'CD326': 'EPCAM', 'EGFR': 'EGFR', 'Podoplanin': 'PDPN', 'CD206': 'MRC1', 'CD209': 'CD209', 'CD169': 'SIGLEC1',
           'Siglec.8': 'SIGLEC8', 'CD294': 'PTGDR2', 'CD70': 'CD70', 'CD30': 'TNFRSF8', 'CD155': 'PVR', 'CD273': 'PDCD1LG2',
           'CD275': 'ICOSLG', 'CD107a': 'LAMP1', 'CD272': 'BTLA', 'CD200': 'CD200', 'CD119': 'IFNGR1', 'CD106': 'VCAM1',
           'CD49f': 'ITGA6', 'CD49a': 'ITGA1', 'CD49e': 'ITGA5', 'CD336': 'NCR2', 'CD337': 'NCR3', 'NKp80': 'KLRF1',
           'CD85d': 'LILRB2', 'CD85k': 'LILRB4', 'CD227': 'MUC1', 'CD191': 'CCR1', 'CD312': 'ADGRE2', 'CLEC12A': 'CLEC12A',
           'CLEC1B': 'CLEC1B', 'CD151': 'CD151', 'CD84': 'CD84', 'CD200R': 'CD200R1'}
ANTIGEN_GENES = set(ANTIGEN.values())


def cell_partner(t, e, r, n_rna=40, n_adt=30):
    """same cell's other modality: CITE row -> RNA (antigen genes + top expressed); scRNA row -> ADTs."""
    sample, b = e.rsplit(':', 1)
    h = h5_for(sample)
    if not h or b not in h[1]: return None
    g, bc, ids, nm, ft, ip = h
    j = bc[b]
    idx, dat = g['indices'][ip[j]:ip[j + 1]], g['data'][ip[j]:ip[j + 1]]
    vals = dict(zip(idx.tolist(), dat.tolist()))
    if t == 'cite':
        pt = 'scrna'
        rna = [i for i in range(len(ids)) if ft[i] == 'Gene Expression']
        ag = [i for i in rna if nm[i] in ANTIGEN_GENES]
        top = sorted((i for i in vals if ft[i] == 'Gene Expression'), key=lambda i: -vals[i])[:20]
        chosen = list(dict.fromkeys(top + r.sample(ag, min(len(ag), n_rna - len(top)))))
    else:
        pt = 'cite'
        adt = [i for i in range(len(ids)) if ft[i] == 'Antibody Capture' and not nm[i].startswith('Isotype')]
        if not adt: return None
        nz = [i for i in adt if vals.get(i, 0) > 0]
        chosen = list(dict.fromkeys(sorted(nz, key=lambda i: -vals[i])[:n_adt // 2 + 5] + r.sample(adt, min(len(adt), n_adt))))[:n_adt]
    out = {}
    for i in chosen:
        name = f'{ids[i]}|{nm[i]}'
        if (pt, e, name) not in gold:
            out[name] = fmt(vals.get(i, 0))
    return pt, out


CELL_DESC = {'cite': "CITE-seq antibody tags of THIS cell (same 10x barcode). Value = antibody-derived tag count.",
             'scrna': "RNA of THIS cell (same 10x barcode). Value = UMI count."}


def tp_text(key):
    if key.startswith('V'):
        ds = sorted(visit_label[key[1:]])
        return f"blood/stool/saliva visit around {key[1:]}" + (f" (samples {ds[0]} to {ds[-1]})" if len(ds) > 1 else '')
    return f"tumour timepoint {key}"


def section(t, e, items):
    head = f"### {DESC[t]} — {descs[t].split('. ', 1)[1] if '. ' in descs[t] else descs[t]}" if t in descs else f"### {t}"
    return [head, f"Entity: {e}"] + [f"- {n} = {v}" for n, v in items]


def build(own, key, ctx_pool, r, mask=True, cond='full', swap_key=None, own_values=None):
    """own: parsed row. returns (user_text, targets[(type, entity, item, value_or_None)])."""
    t, e = own['type'], own['entity']
    sections, targets = [], []
    is_cell = t in ('cite', 'scrna') and ':' in e
    head = (f"Sid's data from one cell ({e}, {tp_text(key) if key else 'unknown timepoint'}) and other measurements from the "
            f"same timepoint." if is_cell else f"Sid's data at one timepoint: {tp_text(key)}.") if key else \
        "Sid's data for one sample."
    head += " Some values are masked. Predict every masked value."
    own_desc = descs[t]
    sections.append([f"### {own_desc}", f"Entity: {e}"] + [f"- {n} = {v}" for n, v in own['known']])
    targets += [(t, e, n, v) for n, v in zip(own['targets'], own_values or [None] * len(own['targets']))]
    if cond != 'own':
        ckey = swap_key if cond == 'swap' else key
        if is_cell:
            pe = e
            if cond == 'swap':   # a cell of the same type from another visit
                pe = swap_cell(t, e, r)
            cp = cell_partner(t, pe, r) if pe else None
            if cp and cp[1]:
                pt, vals = cp
                items = list(vals.items())
                m = set(r.sample(range(len(items)), int(len(items) * MASK_FRAC))) if mask else set()
                sections.append([f"### {CELL_DESC[pt]}"] + [f"- {n} = {v}" for k, (n, v) in enumerate(items) if k not in m])
                targets += [(pt, pe, items[k][0], items[k][1]) for k in sorted(m)]
        if ckey:
            ents = ctx_entities(ctx_pool, ckey, (t, e))
            for ct in sorted(ents):
                ce = r.choice(sorted(ents[ct]))
                d = ctx_pool[ct, ce]
                names = pick(d, N_CTX[ct], r)
                m = set(r.sample(range(len(names)), int(len(names) * MASK_FRAC))) if mask else set()
                sections.append([f"### {descs[ct]}", f"Entity: {ce}"] +
                                [f"- {n} = {d[n]}" for k, n in enumerate(names) if k not in m])
                targets += [(ct, ce, names[k], d[names[k]]) for k in sorted(m)]
    r.shuffle(sections)
    tgt_lines = [f"{k + 1}. [{tt} {te}] {tn}" for k, (tt, te, tn, _) in enumerate(targets)]
    user = '\n\n'.join([head] + ['\n'.join(s) for s in sections] +
                       ["Missing values to predict:\n" + '\n'.join(tgt_lines),
                        'Answer as JSON, e.g. {"1": <value>, "2": <value>}.'])
    return user, targets


cells_by_visit = collections.defaultdict(list)
for (t, e) in pool_all:
    if t in ('cite', 'scrna') and ':' in e and tp(t, e):
        cells_by_visit[t, tp(t, e)].append(e)


def swap_cell(t, e, r):
    k = tp(t, e)
    others = [v for (tt, v) in cells_by_visit if tt == t and v != k and v and k and v[0] == k[0]]
    if not others: return None
    return r.choice(cells_by_visit[t, r.choice(sorted(others))])


def swap_key_for(key, r):
    if not key: return None
    same = sorted({tp(t, e) for (t, e) in pool_all if tp(t, e) and tp(t, e) != key and tp(t, e)[0] == key[0]
                   and t in (VISIT_TYPES if key[0] == 'V' else TUMOUR_TYPES)})
    return r.choice(same) if same else None


def answer_json(targets):
    def num(v):
        try:
            f = float(v)
            return int(f) if f.is_integer() and 'e' not in str(v) and '.' not in str(v) else f
        except ValueError:
            return v
    return json.dumps({str(k + 1): num(v) for k, (_, _, _, v) in enumerate(targets)})


# ---------------- train ----------------
stats = collections.Counter()
with open(f'{OUT}/train_snap.jsonl', 'w') as f:
    for p in unseen:
        key = tp(p['type'], p['entity'])
        user, targets = build(p, key, pool_unseen, rng, mask=True, cond='full',
                              own_values=[fmt(v) for v in p['answers']])
        for (tt, te, tn, _) in targets:
            assert (tt, te, tn) not in gold, (tt, te, tn)
        n_ctx = len(targets) - len(p['targets'])
        stats['rows'] += 1; stats['ctx_targets'] += n_ctx; stats[f'with_ctx_{p["type"]}'] += user.count('### ') > 1
        stats['chars_max'] = max(stats['chars_max'], len(user))
        f.write(json.dumps({'row_id': p['row_id'], 'prompt': [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': user}],
                            'completion': [{'role': 'assistant', 'content': answer_json(targets)}]}) + '\n')

# ---------------- eval ----------------
N_EVAL = 300
N_EVAL_CELL = 1000
golden = [json.loads(l) for l in open(GOLD)]
lab = collections.defaultdict(list)
for r_ in csv.DictReader(open(LABELS)):
    lab[r_['prompt_id']].append(r_)
by_type = collections.defaultdict(list)
for g in golden:
    by_type[g['data_type']].append(g)
erng = random.Random(11)
with open(f'{OUT}/eval_snap.jsonl', 'w') as f, open(f'{OUT}/eval_snap_labels.csv', 'w', newline='') as fl:
    w = csv.writer(fl); w.writerow(['prompt_id', 'cond', 'i', 'data_type', 'entity', 'item', 'truth'])
    for t in sorted(by_type):
        gs = by_type[t][:]
        erng.shuffle(gs)
        for g in gs[:N_EVAL_CELL if t in ("cite", "scrna", "mif", "xenium") else N_EVAL]:
            p = parse(g['messages'][-1]['content'])
            key = tp(t, p['entity'])
            sk = swap_key_for(key, erng)
            assert [g['items'][str(k + 1)] for k in range(len(p['targets']))] == p['targets'], g['prompt_id']
            truth = {g['items'][k]: v for k, v in g['gold'].items()}
            for cond in ('full', 'own', 'swap'):
                r = random.Random(f"{g['prompt_id']}{cond}")
                user, targets = build(p, key, pool_all, r, mask=False, cond=cond, swap_key=sk)
                assert len(targets) == len(p['targets'])
                f.write(json.dumps({'prompt_id': g['prompt_id'], 'cond': cond, 'data_type': t, 'timepoint': key,
                                    'n_sections': user.count('### '),
                                    'messages': [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': user}]}) + '\n')
                for k, n in enumerate(p['targets']):
                    w.writerow([g['prompt_id'], cond, k + 1, t, p['entity'], n, truth[n]])
            stats[f'eval_{t}'] += 1
stats['visits'] = len(visit_label)
json.dump(dict(stats), open(f'{OUT}/stats.json', 'w'), indent=1)
print(json.dumps(dict(stats), indent=1))
