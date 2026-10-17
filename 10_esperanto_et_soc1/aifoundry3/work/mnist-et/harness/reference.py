"""NumPy reference for the device method: an MLP trained full batch, written with only the operations the
device kernel will use (matmul, ReLU, exp, reciprocal, multiply-add; no division or sqrt beyond what Newton
steps on frcp give). Used to choose the architecture and optimizer, and as the bit-for-bit-ish oracle for
the device kernel.

    python3 -I reference.py RUN_DIR [--hidden 256,256] [--steps 30] [--lr 0.05] [--opt adam|sgdm] [--calls mnist]
"""

import argparse
import time
from pathlib import Path

import numpy as np

import release


def init(sizes, seed):
    """He-uniform weights from a fixed integer hash, so the device can make the same ones."""
    rng = np.random.default_rng(seed)
    ws, bs = [], []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        bound = np.sqrt(6.0 / fan_in)
        ws.append(rng.uniform(-bound, bound, (fan_in, fan_out)).astype(np.float32))
        bs.append(np.zeros(fan_out, np.float32))
    return ws, bs


def forward(ws, bs, x):
    acts = [x]
    for i, (w, b) in enumerate(zip(ws, bs)):
        z = acts[-1] @ w + b
        acts.append(np.maximum(z, 0) if i < len(ws) - 1 else z)
    return acts


def train(x, y, hidden, steps, lr, opt, seed=0, beta1=0.9, beta2=0.99, eps=1e-8, wd=0.0, batch=0):
    sizes = [x.shape[1], *hidden, 10]
    ws, bs = init(sizes, seed)
    params = ws + bs
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    onehot_all = np.eye(10, dtype=np.float32)[y]
    x_all = x
    batch = batch or len(x_all)
    per_epoch = len(x_all) // batch
    for t in range(1, steps + 1):
        j = (t - 1) % per_epoch  # fixed contiguous batches: the draw is already in random order
        x = x_all[j * batch:(j + 1) * batch]
        onehot = onehot_all[j * batch:(j + 1) * batch]
        n = len(x)
        acts = forward(ws, bs, x)
        logits = acts[-1]
        e = np.exp(logits - logits.max(1, keepdims=True))
        p = e / e.sum(1, keepdims=True)
        g = (p - onehot) / n
        gws, gbs = [None] * len(ws), [None] * len(ws)
        for i in range(len(ws) - 1, -1, -1):
            gws[i] = acts[i].T @ g
            gbs[i] = g.sum(0)
            if i:
                g = (g @ ws[i].T) * (acts[i] > 0)
        grads = gws + gbs
        for k, (pk, gk) in enumerate(zip(params, grads)):
            if opt == "adam":
                m[k] = beta1 * m[k] + (1 - beta1) * gk
                v[k] = beta2 * v[k] + (1 - beta2) * gk * gk
                mh = m[k] / (1 - beta1 ** t)
                vh = v[k] / (1 - beta2 ** t)
                pk -= lr * (mh / (np.sqrt(vh) + eps) + wd * pk)
            else:  # SGD with Nesterov-free momentum
                m[k] = beta1 * m[k] + gk
                pk -= lr * m[k]
    return ws, bs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--hidden", default="256,256")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--opt", default="adam")
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--write", action="store_true", help="write call_NN.labels for release.py score")
    a = ap.parse_args()
    hidden = [int(h) for h in a.hidden.split(",") if h]
    run = Path(a.run_dir)
    answers = np.load(run / "answers.npz")
    errs = {}
    t0 = time.time()
    for i, name in enumerate(answers["order"]):
        x, y, tx = release.read_call(run / f"call_{i:02d}.bin")
        ws, bs = train(x, y, hidden, a.steps, a.lr, a.opt, wd=a.wd, batch=a.batch)
        pred = forward(ws, bs, tx)[-1].argmax(1).astype(np.int32)
        if a.write:
            pred.tofile(run / f"call_{i:02d}.labels")
        errs.setdefault(str(name), []).append(float((pred != answers[f"y_{i:02d}"]).mean()))
    print(f"hidden={hidden} steps={a.steps} lr={a.lr} opt={a.opt} wd={a.wd} batch={a.batch}: "
          + ", ".join(f"{k} {100 * np.mean(v):.2f}% (worst {100 * max(v):.2f}%)" for k, v in errs.items())
          + f"  [{time.time() - t0:.1f} s]")


if __name__ == "__main__":
    main()
