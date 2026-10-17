"""The mnist-a100 data pipeline, for the ET-SoC-1 port.

The data functions below are copied from cybertronai/sutro-problems mnist-a100/mnist.py
(VERSION mnist-a100/1.2.0; local copy ~/work/ref/sutro-mnist-a100/mnist.py), so a draw here is made
exactly as the A100 scorer makes it: 10,000 training and 10,000 test images from opposite halves of the
60,000-image pool, box-averaged 28x28 -> 9x9, PCA-whitened onto the draw's top 60 directions, turned by a
fresh Haar rotation, with the ten labels secretly permuted.

    python3 -I release.py make RUN_DIR [--seed S]   # 1 warm-up draw + 15 calls (11 MNIST, 4 hold-out) in secret order
    python3 -I release.py score RUN_DIR --difficulty D [--times times.json]

Each call is written as call_NN.bin: train_x (10000x60 f32), train_y (10000 i32), test_x (10000x60 f32),
little-endian, nothing else. The answers (test_y and the dataset) go to answers.npz, which the method never
reads. The method writes call_NN.labels: 10000 int32.
"""

import argparse
import gzip
import hashlib
import json
import os
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np

TRAIN, TEST, DIMS, SIDE = 10000, 10000, 60, 9
MNIST_CALLS, HOLDOUT_CALLS = 11, 4
DIFFICULTY = {1: (540, 200), 2: (340, 532), 3: (270, 1414), 4: (230, 3761), 5: (190, 10000)}
DRAW_SLACK_BP = 150
HOLDOUT_ERROR_BP = 8500
DISPERSION = 2.0
FOREIGN = ("fashion", "kmnist")
POOL_SIZE = 60000
CACHE = Path(os.environ.get("MNIST_CACHE_DIR", Path.home() / "data" / "sutro-mnist"))

SOURCES = {
    "mnist": {
        "mirrors": ("https://ossci-datasets.s3.amazonaws.com/mnist/", "https://storage.googleapis.com/cvdf-datasets/mnist/"),
        "images": ("train-images-idx3-ubyte.gz", "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609"),
        "labels": ("train-labels-idx1-ubyte.gz", "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c"),
    },
    "fashion": {
        "mirrors": ("https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
                    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"),
        "images": ("train-images-idx3-ubyte.gz", "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84"),
        "labels": ("train-labels-idx1-ubyte.gz", "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845"),
    },
    "kmnist": {
        "mirrors": ("https://codh.rois.ac.jp/kmnist/dataset/kmnist/",),
        "images": ("train-images-idx3-ubyte.gz", "51467d22d8cc72929e2a028a0428f2086b092bb31cfb79c69cc0a90ce135fde4"),
        "labels": ("train-labels-idx1-ubyte.gz", "e38f9ebcd0f3ebcdec7fc8eabdcdaef93bb0df8ea12bee65224341c8183d8e17"),
    },
}


def fetch(dataset, kind):
    """The verified gzip bytes of one file, from the cache or a mirror (as mnist.py's fetch)."""
    filename, digest = SOURCES[dataset][kind]
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / f"{dataset}-{filename}"
    if cached.exists():
        data = cached.read_bytes()
        if hashlib.sha256(data).hexdigest() == digest:
            return data
    errors = []
    for mirror in SOURCES[dataset]["mirrors"]:
        try:
            request = urllib.request.Request(mirror + filename, headers={"User-Agent": "mnist-a100/1"})
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
        except Exception as error:  # noqa: BLE001 - try the next mirror
            errors.append(f"{mirror}: {error}")
            continue
        if hashlib.sha256(data).hexdigest() != digest:
            errors.append(f"{mirror}: sha256 mismatch")
            continue
        partial = cached.with_suffix(".part")
        partial.write_bytes(data)
        partial.replace(cached)
        return data
    raise RuntimeError(f"could not load {dataset} {filename}: " + "; ".join(errors))


def read_idx(payload, images):
    header = 16 if images else 8
    fields = struct.unpack(">IIII" if images else ">II", payload[:header])
    if fields != ((2051, POOL_SIZE, 28, 28) if images else (2049, POOL_SIZE)):
        raise ValueError(f"unexpected IDX header {fields}")
    values = np.frombuffer(payload, dtype=np.uint8, offset=header)
    return values.reshape(POOL_SIZE, 28, 28) if images else values.copy()


def area_weights(input_size, output_size):
    left = np.arange(output_size, dtype=np.int64)[:, None] * input_size
    right = left + input_size
    pixel_left = np.arange(input_size, dtype=np.int64)[None, :] * output_size
    pixel_right = pixel_left + output_size
    overlap = np.maximum(0, np.minimum(right, pixel_right) - np.maximum(left, pixel_left))
    return (overlap / input_size).astype(np.float32)


def load_pool(dataset):
    images = read_idx(gzip.decompress(fetch(dataset, "images")), images=True)
    labels = read_idx(gzip.decompress(fetch(dataset, "labels")), images=False)
    weights = area_weights(28, SIDE)
    pixels = np.matmul(np.matmul(weights, images.astype(np.float32) / np.float32(255)), weights.T)
    np.clip(pixels, 0.0, 1.0, out=pixels)
    return np.ascontiguousarray(pixels, dtype=np.float32), labels


def haar_rotation(dim, rng):
    q, r = np.linalg.qr(rng.standard_normal((dim, dim)))
    return q * np.sign(np.diag(r))[None, :]


def fit_whitener(train_rows, dims=DIMS):
    flat = np.asarray(train_rows, dtype=np.float64).reshape(len(train_rows), -1)
    dim = flat.shape[1]
    mu = flat.mean(0)
    eigenvalues, vectors = np.linalg.eigh(np.cov(flat - mu, rowvar=False))
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    signs = np.sign(vectors[np.argmax(np.abs(vectors), axis=0), np.arange(dim)])
    signs[signs == 0] = 1.0
    vectors = vectors * signs[None, :]
    top = np.arange(dim - dims, dim)
    eigenvalues, vectors = eigenvalues[top], vectors[:, top]
    if eigenvalues.min() <= 1e-10:
        raise ValueError(f"the training rows are rank deficient: direction {dims} has variance {eigenvalues.min():.3e}")
    return mu, (vectors / np.sqrt(eigenvalues)[None, :]).T


def apply_release(images, mu, transform):
    flat = np.asarray(images, dtype=np.float64).reshape(len(images), -1)
    return np.ascontiguousarray((flat - mu) @ transform.T, dtype=np.float32)


class Pool:
    def __init__(self, name, pixels, labels, rng):
        self.name, self.pixels, self.labels = name, pixels, labels
        order = rng.permutation(len(labels))
        self.train_half, self.test_half = order[: len(order) // 2], order[len(order) // 2:]

    def draw(self, rng, n_train=TRAIN, n_test=TEST):
        train = rng.choice(self.train_half, n_train, replace=False)
        test = rng.choice(self.test_half, n_test, replace=False)
        relabel = rng.permutation(10).astype(np.int64)
        mu, whitener = fit_whitener(self.pixels[train])
        transform = haar_rotation(DIMS, rng) @ whitener
        return {"dataset": self.name,
                "train_x": apply_release(self.pixels[train], mu, transform), "train_y": relabel[self.labels[train]],
                "test_x": apply_release(self.pixels[test], mu, transform), "test_y": relabel[self.labels[test]]}


# ---------------------------------------------------------------------------- the port's own files


def write_call(path, d):
    with open(path, "wb") as f:
        f.write(d["train_x"].astype("<f4").tobytes())
        f.write(d["train_y"].astype("<i4").tobytes())
        f.write(d["test_x"].astype("<f4").tobytes())


def read_call(path):
    raw = np.fromfile(path, dtype="<u4")
    n = TRAIN * DIMS
    train_x = raw[:n].view("<f4").reshape(TRAIN, DIMS)
    train_y = raw[n:n + TRAIN].view("<i4")
    test_x = raw[n + TRAIN:].view("<f4").reshape(TEST, DIMS)
    return train_x, train_y, test_x


def make(run_dir, seed):
    """As mnist.py's evaluate: a hold-out dataset and a warm-up dataset (the other foreign one),
    11 MNIST and 4 hold-out calls in a secret order, each a fresh draw."""
    rng = np.random.default_rng(seed)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    holdout = FOREIGN[int(rng.integers(2))]
    warm = FOREIGN[1 - FOREIGN.index(holdout)]
    pools = {name: Pool(name, *load_pool(name), rng) for name in ("mnist", holdout, warm)}
    order = ["mnist"] * MNIST_CALLS + [holdout] * HOLDOUT_CALLS
    rng.shuffle(order)
    write_call(run_dir / "warmup.bin", pools[warm].draw(rng))
    answers = {}
    for i, name in enumerate(order):
        d = pools[name].draw(rng)
        write_call(run_dir / f"call_{i:02d}.bin", d)
        answers[f"y_{i:02d}"] = d["test_y"].astype(np.int32)
    np.savez(run_dir / "answers.npz", order=np.array(order), **answers)
    print(f"{run_dir}: warm-up on {warm}, hold-out {holdout}, {len(order)} calls, seed {seed}")


def required_correct(total, error_bp):
    return -(-(total * (10000 - error_bp)) // 10000)


def median(values):
    s = sorted(values)
    m = len(s) // 2
    return s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m])


def score(run_dir, difficulty, times_path=None):
    """mnist.py's judge, without the parts that need its worker process: accuracy bands, the per-draw floor,
    the hold-out floor and the dispersion rule; the time is the slower dataset's mean."""
    run_dir = Path(run_dir)
    band = DIFFICULTY[difficulty][0]
    answers = np.load(run_dir / "answers.npz")
    order = list(answers["order"])
    times = json.loads(Path(times_path).read_text()) if times_path else {}
    rows = []
    for i, name in enumerate(order):
        path = run_dir / f"call_{i:02d}.labels"
        labels = np.fromfile(path, dtype="<i4") if path.exists() else np.full(TEST, -1, np.int32)
        correct = int((labels == answers[f"y_{i:02d}"]).sum())
        rows.append((name, correct, times.get(f"{i:02d}")))
    problems = []
    mnist = [r for r in rows if r[0] == "mnist"]
    hold = [r for r in rows if r[0] != "mnist"]
    correct, total = sum(r[1] for r in mnist), TEST * len(mnist)
    if correct < required_correct(total, band):
        problems.append(f"MNIST accuracy {correct / total:.3%}; the band needs {required_correct(total, band):,} correct")
    floor = required_correct(TEST, band + DRAW_SLACK_BP)
    if min(r[1] for r in mnist) < floor:
        problems.append(f"one MNIST draw scored {min(r[1] for r in mnist):,}/{TEST:,}; no draw may fall below {floor:,}")
    hc, ht = sum(r[1] for r in hold), TEST * len(hold)
    if hold and hc < required_correct(ht, HOLDOUT_ERROR_BP):
        problems.append(f"hold-out accuracy {hc / ht:.2%}: below 15%")
    ms = {}
    for group, label in ((mnist, "MNIST"), (hold, "hold-out")):
        t = [r[2] for r in group if r[2] is not None]
        if len(t) == len(group) and t:
            m = median(t)
            if max(t) > DISPERSION * m + 2.0 or min(t) < m / DISPERSION - 2.0:
                problems.append(f"{label} call times {min(t):.3f}-{max(t):.3f} ms break the 2x dispersion rule")
            ms[label] = sum(t) / len(t)
    for i, (name, c, t) in enumerate(rows):
        print(f"call {i:02d} {name:8s} {c:5d}/{TEST} correct ({100 * (1 - c / TEST):5.2f}% error)"
              + (f"  {t:8.3f} ms" if t is not None else ""))
    summary = f"MNIST {correct / total:.2%} ({correct:,}/{total:,}); hold-out ({hold[0][0]}) {hc / ht:.2%}"
    if ms:
        summary += f"; ms/call MNIST {ms.get('MNIST', 0):.3f}, hold-out {ms.get('hold-out', 0):.3f}; ranked {max(ms.values()):.3f} ms"
    print(summary)
    print(f"difficulty {difficulty} (error <= {band / 100:.2f}%): " + ("PASS" if not problems else "FAIL"))
    for p in problems:
        print("  - " + p)
    return not problems


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("make", "score", "fetch"))
    p.add_argument("run_dir", nargs="?")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--difficulty", type=int, default=1)
    p.add_argument("--times")
    a = p.parse_args()
    if a.action == "fetch":
        for name in SOURCES:
            for kind in ("images", "labels"):
                print(name, kind, len(fetch(name, kind)), "bytes")
    elif a.action == "make":
        make(a.run_dir, a.seed if a.seed is not None else int.from_bytes(os.urandom(4), "little"))
    else:
        sys.exit(0 if score(a.run_dir, a.difficulty, a.times) else 1)
