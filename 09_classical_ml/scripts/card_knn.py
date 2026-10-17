"""CITE-seq entity-kNN imputation driven by the similarity matrix computed on the ET-SoC-1 card (aifoundry3 card 0)."""
import json, math, pickle, sys
import numpy as np
sys.path.insert(0, "/Users/laurabocek/osteo-results/scripts")
from score_compare import score

C = "/Users/laurabocek/osteo-classical"
S = np.fromfile(f"{C}/card/from_aifoundry3/cite_S_card.f32", dtype=np.float32).reshape(8656, 8656)
R = np.fromfile(f"{C}/card/cite_S_ref.f32", dtype=np.float32).reshape(8656, 8656)
print(f"card vs Mac float64 reference: max abs diff {np.abs(S - R).max():.2e}, entries off >1e-4: {(np.abs(S - R) > 1e-4).sum()}")
del R
obs, truth = pickle.load(open(f"{C}/tables.pkl", "rb"))
ent = {}
for k, v in obs["cite"].items():
    try:
        x = float(v)
        if math.isfinite(x): ent[k] = x
    except Exception: pass
E = sorted({e for e, _ in ent}); F = sorted({f for _, f in ent})
ei = {e: i for i, e in enumerate(E)}; fi = {f: j for j, f in enumerate(F)}
r = np.array([ei[e] for e, _ in ent]); c = np.array([fi[f] for _, f in ent]); y = np.arcsinh(np.array(list(ent.values())))
n_f = np.bincount(c, minlength=len(F)); cm = np.bincount(c, y, len(F)) / np.maximum(n_f, 1)
z = y - cm[c]
order = np.argsort(c, kind="stable"); bounds = np.searchsorted(c[order], np.arange(len(F) + 1))
tg = [t for t in truth if t["mod"] == "cite"]
for k in (10, 25):
    preds = {}
    for t in tg:
        i, j = ei.get(t["entity"]), fi.get(t["item"])
        p = cm[j] if j is not None else y.mean()
        if i is not None and j is not None:
            idx = order[bounds[j]:bounds[j + 1]]; idx = idx[r[idx] != i]
            if len(idx):
                s = S[i, r[idx]].astype(float); top = np.argsort(-s)[:k]; w = np.maximum(s[top], 0)
                if w.sum() > 1e-9: p = cm[j] + w @ z[idx][top] / w.sum()
        preds.setdefault(t["prompt_id"], {})[str(t["i"])] = float(np.sinh(p))
    out = f"{C}/preds/card_knn_entity_{k}.jsonl"
    with open(out, "w") as f:
        for pid, d in preds.items(): f.write(json.dumps({"prompt_id": pid, "text": json.dumps(d)}) + "\n")
    rc = score(out, tg)[0]["cite"]["primary"]
    cpu = f"{C}/preds/knn_entity_{k}.jsonl"
    rcpu = score(cpu, tg)[0]["cite"]["primary"]
    a = {(json.loads(l)["prompt_id"]): json.loads(json.loads(l)["text"]) for l in open(out)}
    b = {(json.loads(l)["prompt_id"]): json.loads(json.loads(l)["text"]) for l in open(cpu)}
    diffs = [abs(a[p][i] - b[p][i]) / (abs(b[p][i]) + 1e-9) for p in a for i in a[p] if p in b and i in b[p]]
    print(f"k={k}: card-kNN Spearman {rc:.4f} | CPU-kNN {rcpu:.4f} | predictions within 0.1% of CPU: {np.mean(np.array(diffs) < 1e-3):.1%}")
