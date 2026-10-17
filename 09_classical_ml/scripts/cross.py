"""Cross-data-type prediction: how much do OTHER data types add?

Part A (gene link): for scRNA, Xenium, DIA, Olink, CITE and bulk RNA targets, add other types' measurement of the same
gene (mean asinh level and detection rate over all their samples; for scRNA<->CITE also the same sample's level) to the
gradient-boosted stack, and compare with the same stack on within-type features only. Both are trained on one
validation set of visible values: random entries plus whole held-out features (mimicking never-observed targets).
Part B (date link): flow / Olink / MetaPhlAn / TCR. Cross-modal kNN: rank the target type's other samples by how similar
their date-matched samples are in a source type, and average their values.
"""
import collections, json, math, os, pickle, random, re, sys
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_classical import Mat, predict, METHODS, mname, fnum

OUT = "/Users/laurabocek/osteo-classical"
obs_all, truth = pickle.load(open(f"{OUT}/tables.pkl", "rb"))
GENE_TYPES = ["rna", "scrna", "xenium", "olink", "dia", "cite"]


def sym(m, f):
    if m in ("rna", "scrna"): return {f.split("|")[-1].upper()}
    if m == "xenium": return {f.split("|")[-1].upper()} if f.startswith("ENSG") else set()
    if m == "olink": return {f.upper()}
    if m == "dia": return {g.upper() for g in f.split("|")[-1].split(";")}
    if m == "cite": return {re.sub(r"^(HU|MU)\.", "", f.split("|")[-1].upper())}
    return set()


sample = lambda e: e.split(":")[0]          # 10x capture / tumour sample of a cell
numeric = {m: {k: fnum(v) for k, v in d.items() if fnum(v) is not None} for m, d in obs_all.items()}

# gene-level summaries per source type: mean asinh, detection rate; and per (sample, gene) for scRNA/CITE
gsum, gsamp = {}, {}
for s in GENE_TYPES:
    acc = collections.defaultdict(list); accs = collections.defaultdict(list)
    for (e, f), v in numeric[s].items():
        for g in sym(s, f):
            acc[g].append(math.asinh(v))
            if s in ("scrna", "cite"): accs[(sample(e), g)].append(math.asinh(v))
    gsum[s] = {g: (np.mean(v), np.mean(np.array(v) > 0)) for g, v in acc.items()}
    gsamp[s] = {k: np.mean(v) for k, v in accs.items()}


def cross_feats(t, q):
    cols = []
    for s in GENE_TYPES:
        if s == t: continue
        lv, dr = [], []
        for e, f in q:
            hits = [gsum[s][g] for g in sym(t, f) if g in gsum[s]]
            lv.append(np.mean([h[0] for h in hits]) if hits else np.nan)
            dr.append(np.mean([h[1] for h in hits]) if hits else np.nan)
        cols += [lv, dr]
    if t in ("scrna", "cite"):
        s = "cite" if t == "scrna" else "scrna"
        cols.append([np.mean([gsamp[s][(sample(e), g)] for g in sym(t, f) if (sample(e), g) in gsamp[s]])
                     if any((sample(e), g) in gsamp[s] for g in sym(t, f)) else np.nan for e, f in q])
    return np.column_stack(cols)


def within_feats(M, q, cache):
    P = [predict(M, q, m, cache) for m in METHODS]
    extra = [[M.n_f[M.fi[f]] if f in M.fi else 0, M.n_e[M.ei[e]] if e in M.ei else 0,
              M.colstd[M.fi[f]] if f in M.fi else 0] for e, f in q]
    return np.column_stack(P + [np.array(extra)])


def rho(a, b):
    return float("nan") if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0 else spearmanr(a, b).statistic


def part_a(t, rng):
    ent = numeric[t]; keys = list(ent)
    feats = sorted({f for _, f in keys}); rng.shuffle(feats)
    hold_f = set(feats[:max(1, len(feats) // 10)])                      # whole features held out
    fe = [k for k in keys if k[1] in hold_f]; rng.shuffle(fe); fe = fe[:4000]
    re_ = [k for k in keys if k[1] not in hold_f]; rng.shuffle(re_); re_ = re_[:4000]
    val = fe + re_; drop = set(val) | {k for k in keys if k[1] in hold_f}
    Mv = Mat({k: v for k, v in ent.items() if k not in drop}); Mt = Mat(ent)
    yv = np.arcsinh([ent[k] for k in val])
    tg = [x for x in truth if x["mod"] == t]; q = [(x["entity"], x["item"]) for x in tg]
    yt = np.arcsinh([fnum(x["truth"]) if fnum(x["truth"]) is not None else np.nan for x in tg])
    Wv, Wt = within_feats(Mv, val, {}), within_feats(Mt, q, {})
    Cv, Ct = cross_feats(t, val), cross_feats(t, q)
    gb = lambda X, y: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(X, y)
    pw = gb(Wv, yv).predict(Wt); pc = gb(np.hstack([Wv, Cv]), yv).predict(np.hstack([Wt, Ct]))
    ok = ~np.isnan(yt)
    unseen = np.array([f not in Mt.fi for _, f in q]); covered = ~np.all(np.isnan(Ct), axis=1)
    res = {"type": t, "targets": int(ok.sum()), "within": rho(pw[ok], yt[ok]), "within+cross": rho(pc[ok], yt[ok])}
    for name, m in (("never_observed", unseen), ("gene_covered", covered)):
        mm = ok & m
        res[f"n_{name}"] = int(mm.sum())
        res[f"within_{name}"] = rho(pw[mm], yt[mm]); res[f"cross_{name}"] = rho(pc[mm], yt[mm])
    # each source on its own: its gene level vs truth on targets it covers
    srcs = [s for s in GENE_TYPES if s != t]
    res["source_alone"] = {}
    for k, s in enumerate(srcs):
        lv = Ct[:, 2 * k]; mm = ok & ~np.isnan(lv)
        if mm.sum() >= 20:
            res["source_alone"][s] = {"n": int(mm.sum()), "rho": rho(lv[mm], yt[mm]),
                                      "within_col_mean_same_targets": rho(Wt[mm, 0], yt[mm])}
    # does each source add on top of within + the others? (drop-one ablation on test)
    res["drop_one"] = {}
    for k, s in enumerate(srcs):
        keep = [c for c in range(Cv.shape[1]) if c not in (2 * k, 2 * k + 1)]
        p = gb(np.hstack([Wv, Cv[:, keep]]), yv).predict(np.hstack([Wt, Ct[:, keep]]))
        res["drop_one"][s] = res["within+cross"] - rho(p[ok], yt[ok])
    return res, pw, pc, tg


DATE = re.compile(r"(\d{4}-\d{2})(-\d{2})?")
def date_of(e):
    m = DATE.search(e)
    if not m: return None
    return m.group(1) + (m.group(2) or "-15")


def part_b(t, s, max_days=21):
    """Cross-modal kNN over dated samples: weight target type's other samples by source-type similarity."""
    from datetime import date
    D = lambda x: date.fromisoformat(x)
    te = {e for e, _ in numeric[t]}; se = {e for e, _ in numeric[s]}
    sdates = {e: date_of(e) for e in se if date_of(e)}
    match = {}
    for e in te:
        if not date_of(e): continue
        best = min(sdates, key=lambda x: abs((D(sdates[x]) - D(date_of(e))).days), default=None)
        if best and abs((D(sdates[best]) - D(date_of(e))).days) <= max_days: match[e] = best
    if len(match) < 4: return None
    sv = collections.defaultdict(dict)
    for (e, f), v in numeric[s].items(): sv[e][f] = math.asinh(v)
    def sim(a, b):
        sh = [f for f in sv[a] if f in sv[b]]
        if len(sh) < 5: return 0.0
        r = rho([sv[a][f] for f in sh], [sv[b][f] for f in sh])
        return 0.0 if math.isnan(r) else r
    tv = collections.defaultdict(dict)
    for (e, f), v in numeric[t].items(): tv[f][e] = math.asinh(v)
    tg = [x for x in truth if x["mod"] == t and x["entity"] in match and fnum(x["truth"]) is not None]
    p_cross, p_mean, y = [], [], []
    for x in tg:
        others = {e: v for e, v in tv.get(x["item"], {}).items() if e != x["entity"] and e in match}
        if not others: continue
        w = np.array([max(sim(match[x["entity"]], match[e]), 0) for e in others]); v = np.array(list(others.values()))
        p_cross.append(w @ v / w.sum() if w.sum() > 0 else v.mean()); p_mean.append(v.mean()); y.append(math.asinh(fnum(x["truth"])))
    if len(y) < 20: return None
    return {"target": t, "source": s, "matched_samples": len(match), "targets": len(y),
            "cross_knn": rho(np.array(p_cross), np.array(y)), "mean_of_same_samples": rho(np.array(p_mean), np.array(y)),
            "mae_asinh_cross": float(np.mean(np.abs(np.array(p_cross) - y))), "mae_asinh_mean": float(np.mean(np.abs(np.array(p_mean) - y)))}


if __name__ == "__main__":
    rng = random.Random(0); out = {"gene_link": [], "date_link": []}
    for t in GENE_TYPES:
        r, pw, pc, tg = part_a(t, rng); out["gene_link"].append(r)
        print(f"[gene] {t:6s} within={r['within']:.3f} within+cross={r['within+cross']:.3f} | never-observed n={r['n_never_observed']}: "
              f"{r['within_never_observed']:.3f} -> {r['cross_never_observed']:.3f} | gene-covered n={r['n_gene_covered']}: "
              f"{r['within_gene_covered']:.3f} -> {r['cross_gene_covered']:.3f}", flush=True)
        for s, d in r["source_alone"].items():
            print(f"         {s:6s} alone on its {d['n']} targets: rho={d['rho']:.3f} (within feature mean on same: {d['within_col_mean_same_targets']:.3f}); "
                  f"drop-one Δ={r['drop_one'][s]:+.3f}", flush=True)
        os.makedirs(f"{OUT}/preds_cross", exist_ok=True)
        byp = collections.defaultdict(dict)
        for x, v in zip(tg, pc): byp[x["prompt_id"]][str(x["i"])] = float(np.sinh(v))
        with open(f"{OUT}/preds_cross/{t}_within_plus_cross.jsonl", "w") as f:
            for pid, d in byp.items(): f.write(json.dumps({"prompt_id": pid, "text": json.dumps(d)}) + "\n")
    for t in ["flow", "olink", "mpa", "tcr", "cite", "scrna"]:
        for s in ["flow", "olink", "mpa", "tcr"]:
            if s == t: continue
            r = part_b(t, s)
            if r:
                out["date_link"].append(r)
                print(f"[date] {s:5s} -> {t:5s}: {r['matched_samples']} matched samples, {r['targets']} targets: "
                      f"cross-kNN rho={r['cross_knn']:.3f} vs plain mean {r['mean_of_same_samples']:.3f}; "
                      f"MAE(asinh) {r['mae_asinh_cross']:.3f} vs {r['mae_asinh_mean']:.3f}", flush=True)
    json.dump(out, open(f"{OUT}/cross_results.json", "w"), indent=1, default=float)
    print("saved cross_results.json")
