"""The method, exactly as the device kernel runs it, in NumPy float32: the oracle for the device.

MLP 60(+4 pad)-256-256-10(+6 pad), no biases. Inputs scaled per row to norm sqrt(60). Hidden layers use a
mean-centred ReLU (relu(z - mean(z))), as in the A100 record native-mlp-d1. Softmax cross-entropy.
SGD with Nesterov momentum: g = clip(grad + wd*w, +-clip); v = mu*v + g; w -= lr*(g + mu*v).
Batches of 1024 consecutive rows starting at (t*1024) mod 10000 (the last one of an epoch is shorter).
Initial weights: uniform(-b, b), b = sqrt(6/fan_in), from an integer hash both sides compute; W3 starts at 0.

    python3 oracle.py RUN_DIR [--steps 100] [--calls N] [--write]
"""

import argparse
import time
from pathlib import Path

import numpy as np

import release

D_IN, D_PAD, H, C, C_PAD = 60, 64, 256, 10, 16
HP = dict(steps=100, batch=1024, lr=0.4, mu=0.95, wd=0.001, clip=0.03)
SEED_W1, SEED_W2 = 0x1234567, 0x89ABCDE


def hash_uniform(seed, n):
    """u in [0,1) for idx 0..n-1; the device's hash_u32 (mnist.c), in uint32 arithmetic."""
    h = (np.arange(n, dtype=np.uint64) * np.uint64(0x9E3779B1) + np.uint64(seed)) & np.uint64(0xFFFFFFFF)
    h ^= h >> np.uint64(16)
    h = (h * np.uint64(0x7FEB352D)) & np.uint64(0xFFFFFFFF)
    h ^= h >> np.uint64(15)
    h = (h * np.uint64(0x846CA68B)) & np.uint64(0xFFFFFFFF)
    h ^= h >> np.uint64(16)
    return ((h >> np.uint64(8)).astype(np.float32) * np.float32(1.0 / 16777216.0))


def bound(fan_in):
    return np.float32(np.sqrt(6.0 / fan_in))


def init_weights():
    w1 = np.zeros((D_PAD, H), np.float32)
    u = hash_uniform(SEED_W1, D_PAD * H).reshape(D_PAD, H)
    w1[:D_IN] = ((np.float32(2) * u - np.float32(1)) * bound(D_IN))[:D_IN]
    u = hash_uniform(SEED_W2, H * H).reshape(H, H)
    w2 = (np.float32(2) * u - np.float32(1)) * bound(H)
    w3 = np.zeros((H, C_PAD), np.float32)
    return [w1, w2, w3]


def prep(x):
    """Pad to 64 columns and scale each row to norm sqrt(60)."""
    xp = np.zeros((len(x), D_PAD), np.float32)
    xp[:, :D_IN] = x
    ss = (xp * xp).sum(1, keepdims=True)
    return xp * np.sqrt(np.float32(D_IN) / np.maximum(ss, np.float32(1e-20))).astype(np.float32)


def act(z):
    return np.maximum(z - z.mean(1, keepdims=True), 0)


def forward(ws, x):
    a1 = act(x @ ws[0])
    a2 = act(a1 @ ws[1])
    return a1, a2, a2 @ ws[2]


def train(x, y, steps=HP["steps"], batch=HP["batch"], lr=HP["lr"], mu=HP["mu"], wd=HP["wd"], clip=HP["clip"]):
    x = prep(x)
    ws = init_weights()
    vs = [np.zeros_like(w) for w in ws]
    n = len(x)
    onehot = np.zeros((n, C_PAD), np.float32)
    onehot[np.arange(n), y] = 1
    lr, mu, wd, clip = (np.float32(v) for v in (lr, mu, wd, clip))
    for t in range(steps):
        off = (t * batch) % n
        xb, ob = x[off:off + batch], onehot[off:off + batch]
        a1, a2, lg = forward(ws, xb)
        lg = lg[:, :C]
        e = np.exp(lg - lg.max(1, keepdims=True))
        g3 = np.zeros((len(xb), C_PAD), np.float32)
        g3[:, :C] = e / e.sum(1, keepdims=True)
        g3 = (g3 - ob) * np.float32(1.0 / len(xb))  # the 1/B folded into the output delta
        d2 = g3 @ ws[2].T * (a2 > 0)
        d2 -= d2.mean(1, keepdims=True)
        d1 = d2 @ ws[1].T * (a1 > 0)
        d1 -= d1.mean(1, keepdims=True)
        grads = [xb.T @ d1, a1.T @ d2, a2.T @ g3]
        for i in range(3):
            g = np.clip(grads[i] + wd * ws[i], -clip, clip)
            vs[i] = mu * vs[i] + g
            ws[i] = ws[i] - lr * (g + mu * vs[i])
    return ws


def predict(ws, tx):
    return forward(ws, prep(tx))[2][:, :C].argmax(1).astype(np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--steps", type=int, default=HP["steps"])
    ap.add_argument("--calls", type=int, default=99)
    ap.add_argument("--write", action="store_true", help="write call_NN.oracle.labels and call_NN.oracle.w.npz")
    a = ap.parse_args()
    run = Path(a.run_dir)
    answers = np.load(run / "answers.npz")
    errs = {}
    t0 = time.time()
    for i, name in enumerate(answers["order"][:a.calls]):
        x, y, tx = release.read_call(run / f"call_{i:02d}.bin")
        ws = train(x, y, steps=a.steps)
        pred = predict(ws, tx)
        if a.write:
            pred.tofile(run / f"call_{i:02d}.oracle.labels")
            np.savez(run / f"call_{i:02d}.oracle.w.npz", w1=ws[0], w2=ws[1], w3=ws[2])
        errs.setdefault(str(name), []).append(float((pred != answers[f"y_{i:02d}"]).mean()))
    print(f"oracle steps={a.steps}: "
          + ", ".join(f"{k} {100 * np.mean(v):.2f}% (worst {100 * max(v):.2f}%)" for k, v in errs.items())
          + f"  [{time.time() - t0:.1f} s]")


if __name__ == "__main__":
    main()
