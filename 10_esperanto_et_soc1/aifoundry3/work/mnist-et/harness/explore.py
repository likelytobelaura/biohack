"""Quick NumPy experiments to choose the method for the device (not the device oracle).

    python3 explore.py RUN_DIR METHOD [key=value ...]
"""

import sys
import time
from pathlib import Path

import numpy as np

import release


def rownorm(x):
    return x * np.sqrt(x.shape[1] / np.maximum((x * x).sum(1, keepdims=True), 1e-20))


def knn_vote(ftr, ytr, fte, k):
    ftr = ftr / np.linalg.norm(ftr, axis=1, keepdims=True)
    fte = fte / np.linalg.norm(fte, axis=1, keepdims=True)
    s = fte @ ftr.T
    idx = np.argpartition(-s, k, axis=1)[:, :k]
    votes = np.zeros((len(fte), 10))
    for j in range(k):
        np.add.at(votes, (np.arange(len(fte)), ytr[idx[:, j]]), 1 + 1e-3 * s[np.arange(len(fte)), idx[:, j]])
    return votes.argmax(1)


def mlp(x, y, tx, hidden=256, layers=2, steps=30, lr=0.4, mu=0.95, batch=1024, center=1, wd=0.001, knn=0, seed=0,
        norm=1, clip=0.03):
    clip = float(clip)
    hidden, layers, steps, batch, center, knn, norm = int(hidden), int(layers), int(steps), int(batch), int(center), int(knn), int(norm)
    if norm:
        x, tx = rownorm(x), rownorm(tx)
    rng = np.random.default_rng(seed)
    sizes = [x.shape[1]] + [hidden] * layers + [10]
    ws = [(rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32) for a, b in zip(sizes[:-1], sizes[1:])]
    ws[-1] *= 0
    vs = [np.zeros_like(w) for w in ws]
    onehot = np.eye(10, dtype=np.float32)[y]

    def fwd(inp):
        acts = [inp]
        for i, w in enumerate(ws):
            z = acts[-1] @ w
            if i < len(ws) - 1:
                if center:
                    z = z - z.mean(1, keepdims=True)
                z = np.maximum(z, 0)
            acts.append(z)
        return acts

    n = len(x)
    for t in range(steps):
        off = (t * batch) % n
        xb, ob = x[off:off + batch], onehot[off:off + batch]
        acts = fwd(xb)
        lg = acts[-1]
        e = np.exp(lg - lg.max(1, keepdims=True))
        g = (e / e.sum(1, keepdims=True) - ob)
        for i in range(len(ws) - 1, -1, -1):
            gw = acts[i].T @ g / len(xb) + wd * ws[i]
            if clip > 0:
                gw = np.clip(gw, -clip, clip)
            if i:
                g = (g @ ws[i].T) * (acts[i] > 0)
                if center:
                    g = g - g.mean(1, keepdims=True)
            vs[i] = mu * vs[i] + gw
            ws[i] -= lr * (gw + mu * vs[i])  # Nesterov
    if knn:
        return knn_vote(fwd(x)[-2], y, fwd(tx)[-2], knn)
    return fwd(tx)[-1].argmax(1)


def rf_ridge(x, y, tx, features=2048, lam=1e-2, seed=0, scale=1.0, knn=0):
    """One random ReLU layer, then the output layer in closed form (ridge regression on one-hot targets)."""
    features = int(features)
    x, tx = rownorm(x), rownorm(tx)
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((x.shape[1], features)) * float(scale) / np.sqrt(x.shape[1])).astype(np.float32)
    f, tf = np.maximum(x @ w, 0), np.maximum(tx @ w, 0)
    onehot = np.eye(10, dtype=np.float32)[y]
    a = f.T @ f + float(lam) * len(f) * np.eye(features, dtype=np.float32)
    out = np.linalg.solve(a, f.T @ onehot)
    return (tf @ out).argmax(1)


def knn(x, y, tx, k=3, norm=1):
    if int(norm):
        x, tx = rownorm(x), rownorm(tx)
    return knn_vote(x, y, tx, int(k))


METHODS = {"mlp": mlp, "rf": rf_ridge, "knn": knn}

if __name__ == "__main__":
    run = Path(sys.argv[1])
    method = METHODS[sys.argv[2]]
    kw = dict(a.split("=") for a in sys.argv[3:])
    kw = {k: float(v) for k, v in kw.items()}
    answers = np.load(run / "answers.npz")
    errs = {}
    t0 = time.time()
    limit = int(kw.pop("calls", 99))
    for i, name in enumerate(answers["order"][:limit]):
        x, y, tx = release.read_call(run / f"call_{i:02d}.bin")
        pred = method(x.astype(np.float32), y.astype(np.int64), tx.astype(np.float32), **kw)
        errs.setdefault(str(name), []).append(float((pred != answers[f"y_{i:02d}"]).mean()))
    print(f"{sys.argv[2]} {' '.join(sys.argv[3:])}: "
          + ", ".join(f"{k} {100 * np.mean(v):.2f}% (worst {100 * max(v):.2f}%)" for k, v in errs.items())
          + f"  [{time.time() - t0:.1f} s]")
