"""Compare a device run with the oracle trained on the same draw, batch and steps.

    python3 compare.py RUN_DIR OUT_DIR --call 00 --steps S --batch B [--n-test N]

Prints the largest weight differences per layer (if OUT_DIR/wfinal.f32 exists) and the label agreement.
"""

import argparse
from pathlib import Path

import numpy as np

import oracle
import release


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--call", default="00")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--n-test", type=int, default=10000)
    ap.add_argument("--rowmajor", action="store_true", help="wfinal from the first (row-major) kernel")
    a = ap.parse_args()
    run, out = Path(a.run_dir), Path(a.out_dir)
    x, y, tx = release.read_call(run / f"call_{a.call}.bin")
    ws = oracle.train(x, y, steps=a.steps, batch=a.batch)
    ok = True
    wf = out / "wfinal.f32"
    if wf.exists():
        w = np.fromfile(wf, dtype="<f4")
        # the device stores each matrix as column panels (16 columns, all rows, contiguous)
        dev = [w[:64 * 256].reshape(16, 64, 16).transpose(1, 0, 2).reshape(64, 256),
               w[64 * 256:64 * 256 + 256 * 256].reshape(16, 256, 16).transpose(1, 0, 2).reshape(256, 256),
               w[64 * 256 + 256 * 256:].reshape(256, 16)]
        if a.rowmajor:
            dev = [w[:64 * 256].reshape(64, 256), w[64 * 256:64 * 256 + 256 * 256].reshape(256, 256),
                   w[64 * 256 + 256 * 256:].reshape(256, 16)]
        for name, d, o in zip(("W1", "W2", "W3"), dev, ws):
            diff = np.abs(d - o)
            scale = np.abs(o).max()
            rel = diff.max() / max(scale, 1e-30)
            print(f"{name}: max |dev - oracle| {diff.max():.3e} (largest |w| {scale:.3e}, relative {rel:.2e}); "
                  f"worst at {np.unravel_index(diff.argmax(), diff.shape)}")
            ok &= bool(rel < 1e-3)
    lab = np.fromfile(out / f"call_{a.call}.labels", dtype="<i4")[:a.n_test]
    pred = oracle.predict(ws, tx[:a.n_test])
    answers = np.load(run / "answers.npz")
    truth = answers[f"y_{a.call}"][:a.n_test]
    agree = float((lab == pred).mean())
    print(f"labels: agree with the oracle on {100 * agree:.2f}% of {a.n_test}; error device "
          f"{100 * (lab != truth).mean():.2f}%, oracle {100 * (pred != truth).mean():.2f}%")
    print("MATCH" if ok and agree > 0.99 else "DIFFER")


if __name__ == "__main__":
    main()
