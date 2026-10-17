"""Classical imputation baselines for every data type, scored like Glimmer (score_compare.score).
Hyperparameters and the per-type 'best' are chosen on a 10% split of the visible values, never on eval targets."""
import collections, json, math, os, pickle, random, sys, time
import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0, "/Users/laurabocek/osteo-results/scripts")
from score_compare import score, LABELS, CAT

OUT = "/Users/laurabocek/osteo-classical"
obs_all, truth = pickle.load(open(f"{OUT}/tables.pkl", "rb"))
rng = random.Random(0)
NVAL = 5000


def fnum(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


class Mat:
    """Observed numeric entries of one type in asinh space, as sparse matrices plus per-row/col stats."""
    def __init__(self, entries):
        self.E = sorted({e for e, _ in entries}); self.F = sorted({f for _, f in entries})
        self.ei = {e: i for i, e in enumerate(self.E)}; self.fi = {f: j for j, f in enumerate(self.F)}
        r = np.array([self.ei[e] for e, _ in entries]); c = np.array([self.fi[f] for _, f in entries])
        y = np.arcsinh(np.array([v for v in entries.values()], dtype=float))
        self.r, self.c, self.y = r, c, y
        self.mu = y.mean()
        n_f = np.bincount(c, minlength=len(self.F)); n_e = np.bincount(r, minlength=len(self.E))
        self.colmean = np.where(n_f > 0, np.bincount(c, y, len(self.F)) / np.maximum(n_f, 1), self.mu)
        self.rowmean = np.where(n_e > 0, np.bincount(r, y, len(self.E)) / np.maximum(n_e, 1), self.mu)
        self.colmed = np.full(len(self.F), self.mu)
        order = np.argsort(c, kind="stable"); cs, ys = c[order], y[order]
        bounds = np.searchsorted(cs, np.arange(len(self.F) + 1))
        for j in range(len(self.F)):
            if bounds[j + 1] > bounds[j]: self.colmed[j] = np.median(ys[bounds[j]:bounds[j + 1]])
        self.n_f, self.n_e = n_f, n_e
        self.colstd = np.sqrt(np.maximum(np.bincount(c, y * y, len(self.F)) / np.maximum(n_f, 1) - self.colmean ** 2, 0))
        # additive entity + feature effects (a few rounds of backfitting)
        be, bf = np.zeros(len(self.E)), self.colmean - self.mu
        for _ in range(10):
            be = np.bincount(r, y - self.mu - bf[c], len(self.E)) / (n_e + 1.0)
            bf = np.bincount(c, y - self.mu - be[r], len(self.F)) / (n_f + 1.0)
        self.be, self.bf = be, bf
        z = y - self.colmean[c]                       # column-centred residuals for similarity
        X = sp.csr_matrix((z, (r, c)), shape=(len(self.E), len(self.F)))
        nr = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel()); nr[nr == 0] = 1
        self.Xr = sp.csr_matrix(sp.diags(1 / nr) @ X)                 # row-normalised (entity cosine)
        nc = np.sqrt(np.asarray(X.multiply(X).sum(0)).ravel()); nc[nc == 0] = 1
        self.Xc = sp.csc_matrix(X @ sp.diags(1 / nc))                 # col-normalised (feature cosine)
        self.Z = sp.csc_matrix(X); self.Zr = sp.csr_matrix(X)

    def base(self, e, f):
        i, j = self.ei.get(e), self.fi.get(f)
        return i, j, self.fallback(i, j)

    def fallback(self, i, j):
        """Column mean; for a never-seen feature, the entity's own level (mu + entity effect)."""
        if j is not None: return self.colmean[j]
        return self.mu + (self.be[i] if i is not None else 0.0)


def knn_entity(M, q, k):
    out = []
    for e, f in q:
        i, j, cm = M.base(e, f)
        if i is None or j is None: out.append(cm); continue
        col = M.Z[:, j]; rows = col.indices
        rows_mask = rows != i; rows = rows[rows_mask]; vals = col.data[rows_mask]
        if len(rows) == 0: out.append(cm); continue
        s = (M.Xr[rows] @ M.Xr[i].T).toarray().ravel()
        top = np.argsort(-s)[:k]; w = np.maximum(s[top], 0)
        out.append(cm + (w @ vals[top] / w.sum() if w.sum() > 1e-9 else 0.0))
    return np.array(out)


def knn_feature(M, q, k):
    out = []
    for e, f in q:
        i, j, cm = M.base(e, f)
        if i is None or j is None: out.append(cm); continue
        row = M.Zr[i]; cols = row.indices; vals = row.data
        m = cols != j; cols, vals = cols[m], vals[m]
        if len(cols) == 0: out.append(cm); continue
        s = (M.Xc[:, cols].T @ M.Xc[:, j]).toarray().ravel()
        top = np.argsort(-np.abs(s))[:k]; w = s[top]
        # residual of this entity on similar features, scaled to the target feature's spread
        resid = vals[top] / np.maximum(M.colstd[cols[top]], 1e-6)
        out.append(cm + (w @ resid / np.abs(w).sum() * M.colstd[j] if np.abs(w).sum() > 1e-9 else 0.0))
    return np.array(out)


def als(M, rank, lam, iters=12, seed=0):
    g = np.random.default_rng(seed)
    U = 0.1 * g.standard_normal((len(M.E), rank)); V = 0.1 * g.standard_normal((len(M.F), rank))
    res = M.y - M.mu - M.be[M.r] - M.bf[M.c]
    by_r = collections.defaultdict(list); by_c = collections.defaultdict(list)
    for t, (a, b) in enumerate(zip(M.r, M.c)): by_r[a].append(t); by_c[b].append(t)
    by_r = {a: np.array(t) for a, t in by_r.items()}; by_c = {b: np.array(t) for b, t in by_c.items()}
    I = lam * np.eye(rank)
    for _ in range(iters):
        for a, t in by_r.items():
            Vb = V[M.c[t]]; U[a] = np.linalg.solve(Vb.T @ Vb + I, Vb.T @ res[t])
        for b, t in by_c.items():
            Ua = U[M.r[t]]; V[b] = np.linalg.solve(Ua.T @ Ua + I, Ua.T @ res[t])
    return U, V


def predict(M, q, method, cache):
    name, *p = method
    if name == "col_mean":   return np.array([M.base(e, f)[2] for e, f in q])
    if name == "col_median": return np.array([M.colmed[M.fi[f]] if f in M.fi else M.fallback(M.ei.get(e), None) for e, f in q])
    if name == "row_mean":   return np.array([M.rowmean[M.ei[e]] if e in M.ei else M.mu for e, f in q])
    if name == "additive":
        return np.array([M.mu + (M.be[M.ei[e]] if e in M.ei else 0) + (M.bf[M.fi[f]] if f in M.fi else 0) for e, f in q])
    if name == "knn_entity": return knn_entity(M, q, p[0])
    if name == "knn_feature": return knn_feature(M, q, p[0])
    if name == "als":
        key = ("als", p[0], p[1])
        if key not in cache: cache[key] = als(M, p[0], p[1])
        U, V = cache[key]
        return np.array([M.mu + (M.be[M.ei[e]] if e in M.ei else 0) + (M.bf[M.fi[f]] if f in M.fi else 0)
                         + (U[M.ei[e]] @ V[M.fi[f]] if e in M.ei and f in M.fi else 0) for e, f in q])
    raise ValueError(name)


METHODS = [("col_mean",), ("col_median",), ("row_mean",), ("additive",)] + \
          [("knn_entity", k) for k in (5, 10, 25, 50)] + [("knn_feature", k) for k in (5, 10, 25, 50)] + \
          [("als", r, l) for r in (2, 8, 32) for l in (1.0, 10.0)]
mname = lambda m: "_".join(str(x) for x in m)


def spearman(a, b):
    from scipy.stats import spearmanr
    if np.std(a) == 0 or np.std(b) == 0: return float("nan")
    return spearmanr(a, b).statistic


def numeric_type(mod, tgts):
    entries = {k: fnum(v) for k, v in obs_all[mod].items()}
    entries = {k: v for k, v in entries.items() if v is not None}
    keys = list(entries); rng.shuffle(keys)
    val = keys[:max(1, min(NVAL, len(keys) // 10))]; vs = set(val)
    Mv = Mat({k: v for k, v in entries.items() if k not in vs}); Mt = Mat(entries)
    yv = np.arcsinh([entries[k] for k in val]); q = [(t["entity"], t["item"]) for t in tgts]
    cv, ct, preds_v, preds_t, vscore = {}, {}, {}, {}, {}
    for m in METHODS:
        t0 = time.time()
        preds_v[mname(m)] = predict(Mv, val, m, cv); preds_t[mname(m)] = predict(Mt, q, m, ct)
        vscore[mname(m)] = spearman(preds_v[mname(m)], yv)
        print(f"  {mod:8s} {mname(m):18s} val ρ={vscore[mname(m)]:.3f}  ({time.time()-t0:.1f}s)", flush=True)
    # stacked gradient boosting over all base predictions + simple stats; trained on the validation split
    def feats(M, qq, P):
        cols = [P[mname(m)] for m in METHODS]
        extra = [[M.n_f[M.fi[f]] if f in M.fi else 0, M.n_e[M.ei[e]] if e in M.ei else 0,
                  M.colstd[M.fi[f]] if f in M.fi else 0] for e, f in qq]
        return np.column_stack(cols + [np.array(extra)])
    Xv = feats(Mv, val, preds_v); Xt = feats(Mt, q, preds_t)
    half = len(val) // 2                                   # fit on half the val split, score on the other half
    gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(Xv[:half], yv[:half])
    vscore["gbm_stack"] = spearman(gb.predict(Xv[half:]), yv[half:])
    for m in METHODS: vscore[mname(m) + "_h2"] = spearman(preds_v[mname(m)][half:], yv[half:])
    gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(Xv, yv)
    preds_t["gbm_stack"] = gb.predict(Xt)
    print(f"  {mod:8s} gbm_stack          val ρ={vscore['gbm_stack']:.3f} (held-out half)", flush=True)
    return {k: np.sinh(v) for k, v in preds_t.items()}, {k: v for k, v in vscore.items() if not k.endswith("_h2")}


def categorical_type(mod, tgts):
    d = obs_all[mod]; byf = collections.defaultdict(list); bye = collections.defaultdict(dict)
    for (e, f), v in d.items(): byf[f].append(v); bye[e][f] = v
    preds = {"col_mode": [], "knn_entity_vote": []}
    for t in tgts:
        vals = byf.get(t["item"], [])
        if not vals:                                    # feature never observed: use the gene's other copy
            gene, _, copy = t["item"].rpartition("_")
            sib = f"{gene}_{'2' if copy == '1' else '1'}"
            own = bye.get(t["entity"], {}).get(sib)
            vals = [own] if own else byf.get(sib, [])
        preds["col_mode"].append(collections.Counter(vals).most_common(1)[0][0] if vals else "")
        # vote of the entities that agree most with this one on shared features
        me = bye.get(t["entity"], {}); best = []
        for e2, fv in bye.items():
            if e2 == t["entity"] or t["item"] not in fv: continue
            sh = [f for f in me if f in fv]; agree = sum(me[f] == fv[f] for f in sh) / max(len(sh), 1)
            best.append((agree, fv[t["item"]]))
        best.sort(key=lambda x: -x[0])
        preds["knn_entity_vote"].append(collections.Counter(v for _, v in best[:3]).most_common(1)[0][0] if best else preds["col_mode"][-1])
    return preds, {}


def main():
    by_mod = collections.defaultdict(list)
    for t in truth: by_mod[t["mod"]].append(t)
    all_preds = collections.defaultdict(dict); vsel = {}
    for mod in sorted(by_mod):
        print(f"== {mod}: {len(by_mod[mod])} targets", flush=True)
        preds, vs = (categorical_type if mod in CAT else numeric_type)(mod, by_mod[mod])
        for name, arr in preds.items():
            for t, p in zip(by_mod[mod], arr): all_preds[name][(t["prompt_id"], t["i"])] = p
        vsel[mod] = max(vs, key=lambda k: -1 if np.isnan(vs[k]) else vs[k]) if vs else "knn_entity_vote"
        print(f"  -> validation pick for {mod}: {vsel[mod]}", flush=True)
        all_preds["best_by_val"].update({(t["prompt_id"], t["i"]): all_preds[vsel[mod]][(t["prompt_id"], t["i"])] for t in by_mod[mod]})

    os.makedirs(f"{OUT}/preds", exist_ok=True)
    def as_jsonable(v):
        return v if isinstance(v, str) else float(v)
    for name, pm in all_preds.items():
        if all(v == "" for v in pm.values()):          # abstained everywhere: no file, so it can't pass for a method
            print(f"skipped {name}: no predictions (abstained on every target)")
            continue
        byp = collections.defaultdict(dict)
        for (pid, i), v in pm.items(): byp[pid][str(i)] = as_jsonable(v)
        with open(f"{OUT}/preds/{name}.jsonl", "w") as f:
            for pid, d in byp.items(): f.write(json.dumps({"prompt_id": pid, "text": json.dumps(d)}) + "\n")
    json.dump(vsel, open(f"{OUT}/val_picks.json", "w"), indent=1)
    print("wrote", len(all_preds), "prediction files")


if __name__ == "__main__":
    main()
