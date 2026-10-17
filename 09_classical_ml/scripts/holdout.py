"""Classical counterparts of the Glimmer hold-out experiments, on the same 24,634 masked targets and scorer.

For each numeric data type T (HLA is categorical and skipped), predict T's masked targets under three settings:

S1  type absent    No value of T anywhere (matches context-free Glimmer: untrained, no values of T in the prompt).
                   zero: predict 0 (what Glimmer did); gene_score: mean z-scored level of the same gene in other types
                   (rank only, uncalibrated); transfer_gbm: a GBM trained on the OTHER gene-linked types to predict a
                   value from other types' level of its gene, then applied to T.
S2  new sample     None of this entity's own T values, but T's other samples are known (a new draw / new cell).
                   feat_mean, feat_median: over other entities; carry_forward: same feature in the latest earlier dated
                   sample of the same kind (fallback feat_mean); nearest_date: closest dated sample either side
                   (fallback feat_mean); gbm_link: GBM trained on T's other entities with leave-entity-out feature stats,
                   carry-forward and other types' gene levels.
S3  type unseen    T removed from training, but the entity's own visible T values are known (matches the hold-out LoRA,
    in training    which sees the entity's values in its prompt). entity_mean, entity_median: the entity's own visible
                   values; transfer_gbm: a GBM trained on every OTHER type, with type-agnostic features (entity's own
                   value stats + gene-link levels), applied to T.

Outputs: preds_holdout/<setting>__<method>.jsonl (all types in one file, score_compare format),
holdout_results.json, and the table printed here (HOLDOUT.md holds the write-up).
"""
import collections, json, math, os, random, sys
from datetime import date
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/Users/laurabocek/osteo-results/scripts")
from cross import sym, gsum, numeric, date_of, truth, fnum, GENE_TYPES, OUT
from score_compare import score, LABELS

TYPES = [m for m in LABELS if m != "hla"]
NTRAIN = 20000
rng = random.Random(0)
A = math.asinh

# ---- gene-link features: other types' level of the same gene (excluding `skip` types) -------------------
gz = {}
for s in GENE_TYPES:
    lv = np.array([v[0] for v in gsum[s].values()]); mu, sd = lv.mean(), lv.std() or 1.0
    gz[s] = {g: ((v[0] - mu) / sd, v[1]) for g, v in gsum[s].items()}


def link(t, f, skip):
    """[mean z level, mean detection rate, max z level, n sources] over gene types not in skip."""
    hits = [gz[s][g] for s in GENE_TYPES if s not in skip for g in sym(t, f) if g in gz[s]]
    if not hits:
        return [np.nan, np.nan, np.nan, 0]
    z = [h[0] for h in hits]
    return [float(np.mean(z)), float(np.mean([h[1] for h in hits])), float(max(z)), len(hits)]


# ---- per-type indexes ------------------------------------------------------------------------------------
byf = {t: collections.defaultdict(dict) for t in TYPES}      # t -> feature -> {entity: asinh value}
bye = {t: collections.defaultdict(dict) for t in TYPES}      # t -> entity -> {feature: asinh value}
for t in TYPES:
    for (e, f), v in numeric.get(t, {}).items():
        byf[t][f][e] = A(v); bye[t][e][f] = A(v)
gmean = {t: float(np.mean([v for d in byf[t].values() for v in d.values()])) if byf[t] else 0.0 for t in TYPES}


def _summ(d):
    a = np.array(list(d.values()))
    return (a.sum(), (a * a).sum(), (a > 0).sum(), len(a), float(np.median(a)))


_cs, _es = {}, {}


def _loo(d, summ, excl):
    """mean, median, detection, std, n of the values in d, leaving out key excl (exact median when small)."""
    S, SS, P, n, med = summ
    if excl in d:
        v = d[excl]; S, SS, P, n = S - v, SS - v * v, P - (v > 0), n - 1
        if n <= 200:
            med = float(np.median([x for k, x in d.items() if k != excl])) if n else np.nan
    if n <= 0:
        return [np.nan, np.nan, np.nan, np.nan, 0]
    mu = S / n
    return [mu, med, P / n, math.sqrt(max(SS / n - mu * mu, 0)), n]


def col_stats(t, f, excl):
    """Leave-entity-out mean, median, detection rate, std, count of feature f in type t."""
    d = byf[t].get(f)
    if not d:
        return [np.nan, np.nan, np.nan, np.nan, 0]
    if (t, f) not in _cs:
        _cs[(t, f)] = _summ(d)
    return _loo(d, _cs[(t, f)], excl)


def ent_stats(t, e, excl_f):
    """The entity's own visible values of type t, leaving out feature excl_f: mean, median, std, detection, n."""
    d = bye[t].get(e)
    if not d:
        return [np.nan, np.nan, np.nan, np.nan, 0]
    if (t, e) not in _es:
        _es[(t, e)] = _summ(d)
    mu, med, det, sd, n = _loo(d, _es[(t, e)], excl_f)
    return [mu, med, sd, det, n]


def kind(e):
    return e.split("_")[0]


def dated(t):
    return {e: date_of(e) for e in bye[t] if date_of(e) and ":" not in e}


def carry(t, e, f, nearest=False):
    if t not in _dcache:
        _dcache[t] = dated(t)
    ds = _dcache[t]
    if e not in ds:
        return np.nan
    d0 = date.fromisoformat(ds[e])
    cand = [(date.fromisoformat(ds[o]), v) for o, v in byf[t].get(f, {}).items()
            if o != e and o in ds and kind(o) == kind(e)]
    if nearest:
        cand = sorted(cand, key=lambda x: abs((x[0] - d0).days))
    else:
        cand = sorted([c for c in cand if c[0] < d0], key=lambda x: x[0], reverse=True)
    return cand[0][1] if cand else np.nan


_dcache = {}
class gb:
    """GBM that drops feature columns with no values in training (e.g. carry-forward for undated cells)."""
    def __init__(self, X, y):
        self.keep = ~np.all(np.isnan(X), axis=0)
        self.m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(X[:, self.keep], y)

    def predict(self, X):
        return self.m.predict(X[:, self.keep])


def sample_rows(t, n):
    keys = list(numeric.get(t, {}))
    rng.shuffle(keys)
    return keys[:n]


# ---- S1 transfer GBM: learn value ~ other types' gene level on gene-linked types != T -------------------
def s1_transfer(t, q):
    if t not in GENE_TYPES:
        return None
    X, y = [], []
    for u in GENE_TYPES:
        if u == t:
            continue
        for e, f in sample_rows(u, NTRAIN // 5):
            X.append(link(u, f, {u, t})); y.append(A(numeric[u][(e, f)]))
    m = gb(np.array(X), np.array(y))
    return m.predict(np.array([link(t, f, {t}) for _, f in q]))


# ---- S2 GBM: within T, leave-entity-out feature stats + carry-forward + gene link -----------------------
def s2_feats(t, e, f):
    return col_stats(t, f, e) + [carry(t, e, f), carry(t, e, f, nearest=True)] + link(t, f, {t})


def s2_gbm(t, q):
    rows = sample_rows(t, NTRAIN)
    X = np.array([s2_feats(t, e, f) for e, f in rows]); y = np.array([A(numeric[t][k]) for k in rows])
    return gb(X, y).predict(np.array([s2_feats(t, e, f) for e, f in q]))


# ---- S3 transfer GBM: trained on every other type, type-agnostic features -------------------------------
def s3_feats(t, e, f, skip):
    return ent_stats(t, e, f) + link(t, f, skip)


def s3_transfer(t, q):
    X, y = [], []
    for u in TYPES:
        if u == t or not numeric.get(u):
            continue
        for e, f in sample_rows(u, NTRAIN // 8):
            X.append(s3_feats(u, e, f, {u, t})); y.append(A(numeric[u][(e, f)]))
    m = gb(np.array(X), np.array(y))
    return m.predict(np.array([s3_feats(t, e, f, {t}) for e, f in q]))


def fill(p, fb):
    p = np.array(p, dtype=float)
    return np.where(np.isnan(p), fb, p)


METHODS = [("S1", "zero"), ("S1", "gene_score"), ("S1", "transfer_gbm"),
           ("S2", "feat_mean"), ("S2", "feat_median"), ("S2", "carry_forward"), ("S2", "nearest_date"), ("S2", "gbm_link"),
           ("S3", "entity_mean"), ("S3", "entity_median"), ("S3", "transfer_gbm")]

if __name__ == "__main__":
    preds = {m: collections.defaultdict(dict) for m in METHODS}
    cover = collections.defaultdict(dict)
    for t in TYPES:
        tg = [x for x in truth if x["mod"] == t]
        q = [(x["entity"], x["item"]) for x in tg]
        fm = np.array([col_stats(t, f, e)[0] for e, f in q])
        fm_fb = fill(fm, gmean[t])
        out = {}
        out[("S1", "zero")] = np.zeros(len(q))
        if t in GENE_TYPES:
            gs = np.array([link(t, f, {t})[0] for _, f in q])
            cover[t]["gene_score"] = float(np.mean(~np.isnan(gs)))
            out[("S1", "gene_score")] = fill(gs, np.nanmin(gs) - 1 if np.any(~np.isnan(gs)) else 0.0)
            out[("S1", "transfer_gbm")] = s1_transfer(t, q)
        out[("S2", "feat_mean")] = fm_fb
        out[("S2", "feat_median")] = fill([col_stats(t, f, e)[1] for e, f in q], gmean[t])
        cf = np.array([carry(t, e, f) for e, f in q]); nd = np.array([carry(t, e, f, nearest=True) for e, f in q])
        cover[t]["carry_forward"] = float(np.mean(~np.isnan(cf))); cover[t]["nearest_date"] = float(np.mean(~np.isnan(nd)))
        if cover[t]["nearest_date"] > 0:
            out[("S2", "carry_forward")] = fill(cf, fm_fb); out[("S2", "nearest_date")] = fill(nd, fm_fb)
        out[("S2", "gbm_link")] = s2_gbm(t, q)
        es = np.array([ent_stats(t, e, f)[:2] for e, f in q])
        cover[t]["entity_own"] = float(np.mean(~np.isnan(es[:, 0])))
        out[("S3", "entity_mean")] = fill(es[:, 0], gmean[t]); out[("S3", "entity_median")] = fill(es[:, 1], gmean[t])
        out[("S3", "transfer_gbm")] = s3_transfer(t, q)
        for m, p in out.items():
            for x, v in zip(tg, p):
                preds[m][x["prompt_id"]][str(x["i"])] = round(float(np.sinh(v)), 6)
        print(f"{t}: done ({len(q)} targets) coverage {cover[t]}", flush=True)

    os.makedirs(f"{OUT}/preds_holdout", exist_ok=True)
    from scipy.stats import spearmanr
    seen = {t: {f for _, f in numeric.get(t, {})} for t in TYPES}

    def split_rho(m, t):
        """Spearman on gene-covered targets, on targets whose feature is seen elsewhere in T, and on never-seen ones."""
        ts = [x for x in truth if x["mod"] == t and fnum(x["truth"]) is not None and str(x["i"]) in preds[m].get(x["prompt_id"], {})]
        if not ts:
            return {}
        y = np.array([A(fnum(x["truth"])) for x in ts]); p = np.array([A(preds[m][x["prompt_id"]][str(x["i"])]) for x in ts])
        masks = {"seen": np.array([x["item"] in seen[t] for x in ts])}
        masks["never_seen"] = ~masks["seen"]
        if t in GENE_TYPES:
            masks["gene_covered"] = np.array([not np.isnan(link(t, x["item"], {t})[0]) for x in ts])
        out = {}
        for k, mk in masks.items():
            ok = mk.sum() >= 20 and np.std(p[mk]) > 0 and np.std(y[mk]) > 0
            out[f"rho_{k}"] = float(spearmanr(p[mk], y[mk]).statistic) if ok else None
            out[f"n_{k}"] = int(mk.sum())
        return out
    res = []
    for m in METHODS:
        f = f"{OUT}/preds_holdout/{m[0]}__{m[1]}.jsonl"
        with open(f, "w") as fh:
            for pid, d in preds[m].items():
                fh.write(json.dumps({"prompt_id": pid, "text": json.dumps(d)}) + "\n")
        sc = score(f, truth)[0]
        for t in TYPES:
            r = sc.get(t)
            if not r or not any(str(x["i"]) in preds[m].get(x["prompt_id"], {}) for x in truth if x["mod"] == t):
                continue
            res.append({"setting": m[0], "method": m[1], "key": t, "data_type": LABELS[t], "targets": r["targets"],
                        "primary": r["primary"], "answered": r["answered"], "within_10pct": r["within_10pct"],
                        "median_abs_err_asinh": r["median_abs_err_asinh"], **split_rho(m, t)})
    json.dump({"results": res, "coverage": cover}, open(f"{OUT}/holdout_results.json", "w"), indent=1, default=float)
    f3 = lambda v: "    –" if v is None else f"{v:5.3f}"
    print(f"\n{'type':10s} " + " ".join(f"{s}.{m}"[:14].rjust(14) for s, m in METHODS))
    for t in TYPES:
        row = {(r["setting"], r["method"]): r["primary"] for r in res if r["key"] == t}
        print(f"{t:10s} " + " ".join(f3(row.get(m)).rjust(14) for m in METHODS))
