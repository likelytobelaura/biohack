"""MNIST difficulty 1 (band 5.40%): Nystrom kernel ridge regression with class-wise k-means landmarks and a
Fisher-metric RBF kernel. No backprop and no iterations beyond two k-means steps: everything is closed form.

Step by step (n = 10,000 train rows of dimension d = 60, labels one-hot Y):
1. Fisher metric. The inputs are whitened, so the within-class scatter is S_w = I - M'DM (M: class means, D: class
   frequencies). By Woodbury, S_w^-1 = I + V'V with V = L^-1 M and L L' = D^-1 - MM', so the map
   z -> p = [z, s V z] (70 dims) turns Euclidean distance into the LDA / Mahalanobis distance; s = 1.5 stretches
   the 9 discriminant directions further. Rows are rescaled to norm sqrt(60). Why: whitening erased the useful
   metric, and the labels are the cheapest source of a better one.
2. Landmarks. 48 centres per class (480 in all), seeded with each class's first rows and refined by 2 Lloyd
   (k-means) steps restricted to the class: a row only competes for centres of its own class, as in competitive
   learning. Why: the Nystrom landmarks should cover each class's own part of the space.
3. Features. phi(x) = [1 | RBF(p, c_j) for the 480 landmarks | 0.3 x the 60 coordinates z of p | 0.1 x the 45
   pairwise products (squares included) of the first 9 coordinates of s V z]. The linear and quadratic (LDA-like)
   terms cheaply add what the RBF misses. All come from ONE fp16 GEMM per side on augmented rows, plus an exp.
4. Readout. Nystrom kernel ridge: (Phi'Phi + lam K_mm + eps I) a = Phi'Y on centred train features. Y rides along
   in the feature GEMM, so one "bordered" Cholesky of [[Phi'Phi, Phi'Y], [Y'Phi, Y'Y]] also yields L^-1 Phi'Y, and
   ten triangular solves give a. Prediction: argmax over classes of phi(x_test)'a.
test_x is only featurised and scored; it is not used to fit anything (no transduction). Runs are not bit-exact:
the k-means sums use index_add, whose GPU float atomics add in varying order, so 0-2 of the 10,000 test
predictions can flip from run to run.

Speed: the whole fit is captured once in a CUDA graph and replayed on later calls; independent pieces (the
k-means layout, K_mm, the test features, the fallback factor, the ten solves) run on side streams.
Numerics: an fp16 GEMM accumulating all 10k rows in fp32 shrinks the Gram diagonal slightly (~3e-5 relative on an
A100, ~8e-6 on an RTX 4090). Near-singular Grams (some Fashion-MNIST hold-out draws) then became indefinite on the
A100, the Cholesky failed and every prediction was class 0. Two fixes: the Gram is summed over GCHUNK row chunks,
and a more regularised second factor is always computed and chosen by torch.where on the first factor's info flag
(no host branch, so it is safe inside the CUDA graph). No torch.compile: eager ops in a CUDA graph keep the
scorer's warm-up to 1-2 s, while a long compile warm-up made the sandboxed scorer's /tmp staging of the timed call
a few ms slower. Official A100 scorer: about 1.6 ms per call at about 4.9% error.

Credits (ideas): Nystrom method for kernel machines (Williams & Seeger, 2001); k-means centres as Nystrom
landmarks (Zhang, Tsang & Kwok, 2008); competitive learning (Rumelhart & Zipser, 1985); Fisher's discriminant
(Fisher, 1936). Plain PyTorch only (cuBLAS, cuSOLVER and ATen kernels).

    python mnist.py mnist_d1_nystrom.py:nys_d1 --difficulty 1
"""
import math

import torch

torch.backends.cuda.matmul.allow_tf32 = False

MC = 48           # landmarks per class
GAMMA = 0.018     # RBF width on rows of norm sqrt(60)
STRETCH = 1.5     # extra weight on the discriminant coordinates
LAM = 0.003       # RKHS penalty lam * K_mm
JITTER = 3e-7     # ridge floor, times n
LLOYD = 2         # class-restricted Lloyd steps
LIN = 0.3         # scale of the 60 linear features
QUAD = 0.1        # scale of the LDA product features
QDIM = 9          # discriminant coordinates used for the products
PAD = 16          # Gram width padding
CAP = 1280        # rows per class in the padded Lloyd layout (rows beyond it are left out of the k-means)
TPAR = 10         # parallel triangular solves
GCHUNK = 4        # K-chunks of the readout Gram (bounds the fp32 accumulation length), one stream each
FALLBACK = True   # second, more regularised Cholesky selected on failure (graph-safe)
COMPILE = False   # eager glue: a long torch.compile warm-up slows the sandboxed scorer's timed calls
DIAGFIX = False   # (off) replace the Gram diagonal by an fp32 sum of squares (removes the diagonal's accumulation bias)
FB_JIT = 10.0     # fallback ridge: FB_JIT x the jitter ...
FB_DG = 1e-4      # ... plus FB_DG x the Gram diagonal
ST = {}           # per-process cache: wrapped functions, streams, constant tensors and the captured CUDA graph


def _c(f):
    """f, or f compiled by torch.compile when COMPILE is set."""
    return torch.compile(f, fullgraph=True, dynamic=False) if COMPILE else f


def _prep1(tx, ty, ar10):
    """One-hot Y, class counts, class means M and B = D^-1 - M M' (its Cholesky factor gives the Fisher map)."""
    Y = (ty[:, None] == ar10).float()
    cnt = Y.sum(0)
    M = (Y.T @ tx) / cnt[:, None]
    B = torch.diag(tx.shape[0] / cnt) - M @ M.T
    return Y, cnt, M, B


def _rows(x, V, eye, Y, zpad):
    """Augmented rows [p (70), 1, 0, Y or 0, 0-pad]; p = fp16 GEMM of x with [I | s V'], normalised to norm sqrt(d).

    The constant column carries the RBF and bias terms of the feature GEMM; Y (train side only) rides along so
    that the readout Gram also holds Phi'Y and Y'Y."""
    d = x.shape[1]
    T = torch.cat([eye, V.T * STRETCH], 1)
    X = (x.half() @ T.half()).float()
    X = X * (d ** 0.5 / X.norm(dim=1, keepdim=True))
    one = torch.ones_like(X[:, :1])
    return torch.cat([X, one, one * 0, Y if Y is not None else torch.zeros_like(X[:, :10]), zpad], 1)


def _rows_tr(x, V, eye, Y, zpad):
    return _rows(x, V, eye, Y, zpad).half()


def _rows_te(x, V, eye, zpad):
    return _rows(x, V, eye, None, zpad).half()


def _perm(tx, ty, ar10, fillv, arn):
    """Class-padded layout for the k-means, from the labels only.

    Class k's rows go to slots k * CAP + (rank within the class); perm[slot] = row, and unused slots point at an
    appended zero row n. Returns x in that order and the mask of real rows."""
    Yt = (ar10[:, None] == ty).float()   # (10, n): the cumsum scans the contiguous dim (scanning dim 0 of an
    rank = (Yt.cumsum(1) * Yt).sum(0) - 1   # (n, 10) tensor is much slower)
    slot = torch.where(rank < CAP, ty * CAP + rank.long(), fillv)
    perm = torch.full((10 * CAP + 1,), tx.shape[0], device=tx.device, dtype=torch.long).scatter(0, slot, arn)
    perm = perm[:10 * CAP]
    xe = torch.cat([tx, torch.zeros_like(tx[:1])])
    return xe[perm], perm < tx.shape[0]


def _rows_pad(xp, valid, V, eye):
    """Padded-layout Lloyd rows: fp32 [p, 1] (zero rows at pad slots) and fp16 p.

    The trailing 1 counts the rows assigned to each centre when the rows are summed by index_add."""
    d = xp.shape[1]
    T = torch.cat([eye, V.T * STRETCH], 1)
    X = (xp.half() @ T.half()).float()
    X = X * (d ** 0.5 / X.norm(dim=1, keepdim=True).clamp_min(1e-6))
    one = torch.ones_like(X[:, :1])
    Q = torch.cat([X, one], 1) * valid[:, None]
    return Q, Q[:, :d + 10].half()


def _lloyd(Cc, Qh, Qf, off):
    """One class-restricted Lloyd step: Cc (10, MC, D) centres, Qh (10, CAP, D) fp16 rows, Qf the [p, 1] rows.

    Each row picks its nearest centre within its own class (max of p.c - |c|^2 / 2, one batched GEMM); each centre
    moves to the mean of its rows, and a centre with no rows stays where it is."""
    D = Cc.shape[-1]
    cb = (Cc * Cc).sum(-1).mul(-0.5)[:, None, :]
    sc = torch.baddbmm(cb.half(), Qh, Cc.transpose(1, 2).half())
    gi = (sc.argmax(-1) + off).reshape(-1)
    S = torch.zeros(10 * MC, D + 1, device=Cc.device, dtype=Cc.dtype).index_add(0, gi, Qf).view(10, MC, D + 1)
    return torch.where(S[..., D:] > 0, S[..., :D] / S[..., D:].clamp_min(1), Cc)


def _wmat(C, W0, d: int):
    """Feature-GEMM weights: W0 with the RBF block filled in (exponent 2g p.c - g|c|^2 - g d, |p|^2 = d).

    Because every row has |p|^2 = d, -g|p - c|^2 is linear in the augmented row [p, 1], so one GEMM yields all
    the RBF exponents. (The -g row multiplies the always-zero column of the augmented rows and has no effect.)"""
    m, D = C.shape
    cn = (C * C).sum(1)
    kb = torch.cat([C.T * (2 * GAMMA), (-GAMMA * cn - GAMMA * d)[None], torch.full_like(cn, -GAMMA)[None],
                    torch.zeros(W0.shape[0] - D - 2, m, device=C.device)], 0)
    return torch.cat([W0[:, :1], kb, W0[:, 1 + m:]], 1).half(), cn


def _kmm(C, cn):
    """K_mm: the RBF kernel among the landmarks (the Nystrom RKHS penalty)."""
    return torch.exp(2 * GAMMA * (C @ C.T) - GAMMA * cn[:, None] - GAMMA * cn[None, :])


def _dsq(G):
    """Exact fp32 squared norms of the feature rows (used only when DIAGFIX is set)."""
    return G.float().square().sum(1)


def _amat(Ab, dsq, Kmm, jn: float, me: int):
    """Sum the chunk Grams, add lam K_mm and the ridge; also the fallback matrix (FB_JIT ridge + FB_DG diag).

    The label block and the padding get +1 on the diagonal so the bordered matrix stays positive definite."""
    Aa = Ab[0]
    for t in Ab[1:]:
        Aa = Aa + t
    if DIAGFIX:
        Aa = Aa + torch.diag(dsq - Aa.diagonal())
    m = Kmm.shape[0]
    W = Aa.shape[0]
    Kp = torch.nn.functional.pad(Kmm * LAM, (1, W - 1 - m, 1, W - 1 - m))
    feat = torch.arange(W, device=Aa.device) < me
    A1 = Aa + Kp + torch.diag(torch.where(feat, jn, jn + 1.0))
    A2 = A1 + torch.diag(torch.where(feat, (FB_JIT - 1.0) * jn + FB_DG * Aa.diagonal(), 0.0))
    return A1, A2


def _sel(L1, info, L2):
    """The first factor if its Cholesky succeeded (info == 0), else the fallback; no host synchronisation."""
    return torch.where(info == 0, L1, L2)


def _coef(a, muh):
    """Fold the train centring into the constant column's coefficient; hi/lo fp16 split for the test GEMM.

    The coefficients are scaled to max |a| = 1 (argmax is unchanged) and split as hi + lo, both fp16, so the
    fp16 test GEMM keeps close to fp32 precision in the coefficients."""
    a0 = a[:1] - muh.float()[None] @ a
    a = torch.cat([a0, a[1:]])
    an = a / a.abs().max()
    hi = an.half()
    return torch.cat([hi, (an - hi.float()).half()], 1)


def _pick(o):
    """Add the hi and lo halves of the test scores and take the argmax over the 10 classes."""
    return (o[:, :10] + o[:, 10:]).argmax(1)


def _fn(name, f):
    """f (compiled when COMPILE is set), created once per process."""
    if name not in ST:
        ST[name] = _c(f)
    return ST[name]


def _stream(name, prio=0):
    """A named CUDA stream, created once per process (priority -1 is high)."""
    if name not in ST:
        ST[name] = torch.cuda.Stream(priority=prio)
    return ST[name]


def _feat(Wh, Ah, m, o_q, me, Wg, nq):
    """G' (rows = features) = W' A': exp on the RBF rows, products of the two scratch blocks into the quad rows."""
    GT = torch.mm(Wh.T, Ah.T)
    GT[1:1 + m].exp_()
    torch.mul(GT[Wg:Wg + nq], GT[Wg + nq:Wg + 2 * nq], out=GT[o_q:me])
    return GT


def _back(La, me):
    """a = L^-T w for the 10 columns w of the bordered factor: TPAR parallel single-RHS triangular solves."""
    U = La[:me, :me].T.contiguous()
    w = La[me:me + 10, :me].T.contiguous()
    out = torch.empty(me, 10, device=La.device)
    cur = torch.cuda.current_stream()
    per = 10 // TPAR
    sts = []
    for i in range(TPAR):
        st = _stream(f"t{i}", -1)
        st.wait_stream(cur)
        with torch.cuda.stream(st):
            out[:, i * per:(i + 1) * per] = torch.linalg.solve_triangular(U, w[:, i * per:(i + 1) * per], upper=True)
        sts.append(st)
    for st in sts:
        cur.wait_stream(st)
    return out


def _fit_predict(tx, ty, vx):
    """The whole fit on a high-priority stream (joined back to the caller's stream)."""
    hp = _stream("hp", -1)
    cur = torch.cuda.current_stream()
    hp.wait_stream(cur)
    with torch.cuda.stream(hp):
        out = _fit(tx, ty, vx)
    cur.wait_stream(hp)
    return out


def _fit(tx, ty, vx):
    """Fit on (tx, ty) and return the predicted labels of vx.

    Rows of GT (= columns of the feature matrix G): [0] constant, [1, 1 + m) RBF, [o_l, o_q) linear,
    [o_q, me) quadratic products, [me, me + 10) labels Y, zero padding up to Wg, then [Wg, Wg + 2 nq) the two
    scratch factors of the products, padded to W. Augmented input rows have width Wa: [p (D), 1, 0, Y, 0-pad]."""
    n, d = tx.shape
    dev = tx.device
    mc = MC
    m = 10 * mc
    D = d + 10
    nq = QDIM * (QDIM + 1) // 2
    o_l, o_q = 1 + m, 1 + m + d
    me = 1 + m + d + nq
    Wg = me + 10
    Wg += (-Wg) % PAD
    W = Wg + 2 * nq
    W += (-W) % 8
    Wa = D + 12
    Wa += (-Wa) % 8
    ck = ("c", n, d, str(dev))
    if ck not in ST:   # constant tensors, built once: the fixed (non-RBF) part W0 of the feature-GEMM weights
        iu = torch.triu_indices(QDIM, QDIM)
        W0 = torch.zeros(Wa, W, device=dev)
        W0[D, 0] = 1
        W0[:d, o_l:o_q] = torch.eye(d, device=dev) * LIN
        W0[d:d + QDIM, Wg:Wg + nq] = torch.eye(QDIM, device=dev)[:, iu[0]] * math.sqrt(QUAD)
        W0[d:d + QDIM, Wg + nq:Wg + 2 * nq] = torch.eye(QDIM, device=dev)[:, iu[1]] * math.sqrt(QUAD)
        W0[D + 2:D + 12, me:me + 10] = torch.eye(10, device=dev)
        ST[ck] = dict(eye=torch.eye(d, device=dev), ar10=torch.arange(10, device=dev), W0=W0,
                      off=torch.arange(10, device=dev)[:, None] * mc, zpad=torch.zeros(n, Wa - D - 12, device=dev),
                      fill=torch.full((n,), 10 * CAP, device=dev, dtype=torch.long), arn=torch.arange(n, device=dev))
    c = ST[ck]
    main = torch.cuda.current_stream()
    sg = _stream("gather")
    sg.wait_stream(main)
    with torch.cuda.stream(sg):          # class-padded layout from the labels alone
        xp, valid = _fn("perm", _perm)(tx, ty, c["ar10"], c["fill"], c["arn"])
    # Fisher map: B = D^-1 - M M' = L L', V = L^-1 M; p = [x, s V x]
    Y, cnt, M, B = _fn("prep1", _prep1)(tx, ty, c["ar10"])
    Lb, _ = torch.linalg.cholesky_ex(B)
    V = torch.linalg.solve_triangular(Lb, M, upper=False)
    sg.wait_stream(main)
    with torch.cuda.stream(sg):
        Qf, Qh = _fn("rows_pad", _rows_pad)(xp, valid, V, c["eye"])
    Atr = _fn("rows_tr", _rows_tr)(tx, V, c["eye"], Y, c["zpad"])
    main.wait_stream(sg)
    # landmarks: the first MC rows of each class, then LLOYD class-restricted k-means steps
    Qv = Qf.view(10, CAP, D + 1)
    Qh = Qh.view(10, CAP, D)
    Cc = Qv[:, :mc, :D]
    lf = _fn("lloyd", _lloyd)
    for _ in range(LLOYD):
        Cc = lf(Cc, Qh, Qf, c["off"])
    C = Cc.reshape(m, D)
    Wh, cn = _fn("wmat", _wmat)(C, c["W0"], d)
    sk = _stream("feat")
    sk.wait_stream(main)
    with torch.cuda.stream(sk):          # K_mm off the critical path
        Kmm = _fn("kmm", _kmm)(C, cn)
        ev = torch.cuda.Event()
        ev.record(sk)
    # train features; centre them (not the constant column); _coef folds the centring into the test side
    GT = _feat(Wh, Atr, m, o_q, me, Wg, nq)
    muh = GT[:me].mean(1, dtype=torch.float32).half()
    muh[:1].zero_()
    GT[:me] -= muh[:, None]
    G, kc, Ab = GT[:Wg], n // GCHUNK, []
    for i in range(GCHUNK):          # split-K by streams: short fp32 accumulations, and more CTAs in flight
        st = _stream(f"g{i}", -1)
        st.wait_stream(main)
        with torch.cuda.stream(st):
            g = G[:, i * kc:(i + 1) * kc]
            Ab.append(torch.mm(g, g.T, out_dtype=torch.float32))
    dsq = _fn("dsq", _dsq)(G) if DIAGFIX else None
    for i in range(GCHUNK):
        main.wait_stream(_stream(f"g{i}", -1))
    sk.wait_stream(main)
    with torch.cuda.stream(sk):          # the test side runs during the Cholesky
        Ate = _fn("rows_te", _rows_te)(vx, V, c["eye"], c["zpad"])
        GqT = _feat(Wh, Ate, m, o_q, me, Wg, nq)
    main.wait_event(ev)
    # bordered Cholesky of [[A, Phi'Y], [Y'Phi, Y'Y + I]]: its last 10 rows hold (L^-1 Phi'Y)'
    Aa, A2 = _fn("amat", _amat)(Ab, dsq, Kmm, JITTER * n, me)
    if FALLBACK:   # the more regularised factor runs concurrently on its own stream
        sf = _stream("fb", -1)
        sf.wait_stream(main)
        with torch.cuda.stream(sf):
            L2, _ = torch.linalg.cholesky_ex(A2)
    La, info = torch.linalg.cholesky_ex(Aa)
    if FALLBACK:
        main.wait_stream(sf)
        La = _fn("sel", _sel)(La, info, L2)
    hl = _fn("coef", _coef)(_back(La, me), muh)
    main.wait_stream(sk)
    o = torch.mm(GqT[:me].T, hl, out_dtype=torch.float32)
    return _fn("pick", _pick)(o)


def nys_d1(train_x, train_y, test_x):
    """Predict labels for test_x from (train_x, train_y); see the module docstring.

    The first call for a given shape runs the fit three times on a side stream (allocator and library warm-up),
    then captures one fit in a CUDA graph on the caller's own input tensors. Every call replays that graph,
    after copying its inputs into the captured tensors when they are different tensors."""
    torch.backends.cuda.matmul.allow_tf32 = False
    key = ("g", tuple(train_x.shape), tuple(test_x.shape), str(train_x.device))
    st = ST.get(key)
    if st is None:   # warm-up call: compile, then capture the whole fit on the caller's own input tensors
        if COMPILE:
            import torch._inductor.config as icfg
            icfg.compile_threads = 1   # compile in-process: an idle compile-worker pool costs wall-clock time per call
        args = [train_x, train_y, test_x]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _fit_predict(*args)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = _fit_predict(*args)
        st = ST[key] = (graph, args, out)
        import gc
        gc.collect()
        gc.freeze()   # move the warm-up's objects out of later garbage collections (less per-call wall clock)
    graph, args, out = st
    for buf, x in zip(args, (train_x, train_y, test_x)):
        if buf.data_ptr() != x.data_ptr():   # not the captured tensors: copy the inputs in
            buf.copy_(x)
    graph.replay()
    return out.clone()
