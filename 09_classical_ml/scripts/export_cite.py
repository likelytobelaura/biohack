"""Export CITE-seq's centred, row-normalised matrix (same as run_classical.Mat.Xr on all visible values) for the card."""
import math, pickle, numpy as np
obs, _ = pickle.load(open("/Users/laurabocek/osteo-classical/tables.pkl", "rb"))
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
X = np.zeros((len(E), len(F))); X[r, c] = y - cm[c]
nr = np.linalg.norm(X, axis=1); nr[nr == 0] = 1; X /= nr[:, None]
pad = lambda n: -(-n // 16) * 16
A = np.zeros((pad(len(E)), pad(len(F))), dtype=np.float32); A[:len(E), :len(F)] = X
A.tofile("/Users/laurabocek/osteo-classical/card/cite_A.f32")
S = (A.astype(np.float64) @ A.T.astype(np.float64)).astype(np.float32)
S.tofile("/Users/laurabocek/osteo-classical/card/cite_S_ref.f32")
open("/Users/laurabocek/osteo-classical/card/README.txt", "w").write(
    f"cite_A.f32: row-major fp32, {A.shape[0]} x {A.shape[1]} (real {len(E)} cells x {len(F)} features, zero padded to multiples of 16)\n"
    f"cite_S_ref.f32: S = A @ A.T computed on the host in float64, stored fp32, {S.shape[0]} x {S.shape[1]}\n")
print(A.shape, S.shape, f"{2*A.shape[0]**2*A.shape[1]/1e9:.1f} GFLOP")
