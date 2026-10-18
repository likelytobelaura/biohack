"""MetaPhlAn log-abundance probe: base Glimmer vs lomo740_all, every layer (examples from build_mpa_probe.py).

modal volume put osteo-data <examples.jsonl> /mpa_probe/<tag>/examples.jsonl
modal run --detach mpa_probe.py --tag <tag>                 # extract (GPU) then probe
modal run --detach mpa_probe.py --tag <tag> --probe-only    # re-run probes on saved activations
modal volume get osteo-data /outputs/mpa_probe/<tag>/results.json <local dir>

extract: one teacher-forced forward per prompt per model (adapter on / disabled). Saves, at every mark, the residual
  stream after the embeddings and after each of the 52 layers (resid_<model>_<kind>.npy, N x 53 x 6656 fp16) and the
  MLP neuron activations (down_proj input) for kinds answer + every 2nd known_name (mlp_<model>_<kind>.npy, N x 52 x 19968).
probe: entity-grouped 7-fold CV. Per layer, kernel ridge on z-scored residuals (alpha by LOO inside the train fold),
  scored on pooled out-of-fold predictions. Baselines: out-of-fold taxon prior (mean y of the taxon in train-fold
  entities) and, for answers, + the 'elsewhere' values; 'resid' probes predict y - baseline (gain over baseline).
  Neurons: per-neuron Pearson r with y and with y - baseline, every layer; sparse probes with the top-k neurons
  (chosen on the train fold) at the best layers; shuffled-label control at each probe's best layer.
"""
import os

import modal

app = modal.App("osteo-mpa-probe")
image = (modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
         .entrypoint([])
         .pip_install("torch==2.14.0", "transformers==5.18.0", "peft==0.21.1", "accelerate==1.15.0", "numpy"))
hf = modal.Volume.from_name("hf-cache")
data = modal.Volume.from_name("osteo-data")
ADAPTER = "/data/adapters/lomo/lomo740_all"
MODELS = ("base", "ft")
KINDS = ("answer", "known_name", "known_value")
MLP_KINDS = ("answer", "known_name")
ROOT = os.environ.get("MPA_ROOT", "/data")  # local test: MPA_ROOT=<dir> ... probe.local(tag)


def load(tag):
    import json
    return [json.loads(l) for l in open(f"{ROOT}/mpa_probe/{tag}/examples.jsonl")]


def flat_marks(prompts, kind):
    """marks of one kind in prompt order; known_name_mlp = every 2nd known_name within each prompt (the MLP subset)"""
    sub = kind == "known_name_mlp"
    kind = "known_name" if sub else kind
    out = []
    for p in prompts:
        mk = [dict(m, entity=p["entity"], src=p["src"], prompt=p["id"]) for m in p["marks"] if m["kind"] == kind]
        out += mk[::2] if sub else mk
    return out


@app.function(image=image, gpu=["H200", "H100"], volumes={"/hf": hf, "/data": data}, timeout=3 * 3600, memory=98304)
def extract(tag: str, limit: int = 0):
    import json, os, re, time
    import numpy as np, torch
    from transformers import AutoModelForImageTextToText, AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    prompts = load(tag)
    if limit:
        prompts = prompts[:limit]
    out_dir = f"/data/outputs/mpa_probe/{tag}"
    os.makedirs(out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(ADAPTER)
    try:
        model = AutoModelForImageTextToText.from_pretrained("/hf/Muse-Glimmer-30B", dtype=torch.bfloat16, device_map="cuda")
    except (ValueError, KeyError):
        model = AutoModelForCausalLM.from_pretrained("/hf/Muse-Glimmer-30B", dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(model, ADAPTER, adapter_name="ft").eval()

    layers, downs = {}, {}
    for n, m in model.named_modules():
        if mm := re.search(r"language_model\.layers\.(\d+)$", n):
            layers[int(mm.group(1))] = m
        if mm := re.search(r"language_model\.layers\.(\d+)\.mlp\.down_proj$", n):
            downs[int(mm.group(1))] = m
    NL = len(layers)
    assert sorted(layers) == list(range(NL)) == sorted(downs), (len(layers), len(downs))
    D, NI = model.get_input_embeddings().embedding_dim, downs[0].in_features
    print(f"{NL} layers, d={D}, {NI} MLP neurons", flush=True)

    rec = {"pos": None, "mlp_pos": None, "res": {}, "mlp": {}}
    take = lambda x, p: x[0, p].detach().to(torch.float16).cpu()
    model.get_input_embeddings().register_forward_hook(lambda m, i, o: rec["res"].__setitem__(0, take(o, rec["pos"])))
    for l, m in layers.items():
        m.register_forward_hook(lambda m, i, o, l=l: rec["res"].__setitem__(l + 1, take(o[0] if isinstance(o, tuple) else o, rec["pos"])))
    for l, m in downs.items():
        m.register_forward_pre_hook(lambda m, a, l=l: rec["mlp"].__setitem__(l, take(a[0], rec["mlp_pos"])))

    def render(p):
        msgs = p["messages"]
        head = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": p["comp"]}], tokenize=False)
        start = full.index("<|message|>", len(head)) + len("<|message|>")
        text = full[:start] + p["comp"]
        user = msgs[-1]["content"]
        assert text.count(user) == 1, p["id"]
        return text, {"user": text.index(user), "comp": start}

    def fwd(x):
        try:
            model(input_ids=x, use_cache=False, logits_to_keep=1)
        except TypeError:
            model(input_ids=x, use_cache=False)

    # per kind: list of per-prompt arrays, concatenated in prompt order (= flat_marks order)
    R = {(mo, k): [] for mo in MODELS for k in KINDS}
    M = {(mo, k): [] for mo in MODELS for k in MLP_KINDS}
    meta = {k: [] for k in KINDS}
    t0 = time.time()
    for i, p in enumerate(prompts):
        text, base = render(p)
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        pos, kind_of = [], []
        for m in p["marks"]:
            ch = base[m["where"]] + m["ch"]
            if m["rule"] == "before":
                j = max(t for t, (s, e) in enumerate(offs) if e <= ch)
            else:
                j = max(t for t, (s, e) in enumerate(offs) if s < ch)
            pos.append(j); kind_of.append(m["kind"])
            meta[m["kind"]].append(dict(prompt=p["id"], tok=tok.decode(ids[j]), next=tok.decode(ids[j + 1]) if j + 1 < len(ids) else ""))
        kn = [t for t, k in enumerate(kind_of) if k == "known_name"]
        mlp_sel = [t for t, k in enumerate(kind_of) if k == "answer"] + kn[::2]
        rec["pos"], rec["mlp_pos"] = pos, [pos[t] for t in mlp_sel]
        x = torch.tensor([ids], device="cuda")
        for mo in MODELS:
            rec["res"], rec["mlp"] = {}, {}
            with torch.no_grad():
                if mo == "base":
                    with model.disable_adapter():
                        fwd(x)
                else:
                    fwd(x)
            res = torch.stack([rec["res"][l] for l in range(NL + 1)], 1)  # marks x 53 x D
            mlp = torch.stack([rec["mlp"][l] for l in range(NL)], 1)  # mlp marks x 52 x NI
            for k in KINDS:
                R[(mo, k)].append(res[[t for t, kk in enumerate(kind_of) if kk == k]].numpy())
            na = sum(k == "answer" for k in kind_of)
            M[(mo, "answer")].append(mlp[:na].numpy()); M[(mo, "known_name")].append(mlp[na:].numpy())
        if i < 2 or (i + 1) % 10 == 0 or i + 1 == len(prompts):
            print(f"{i + 1}/{len(prompts)} prompts, {len(ids)} tok, {(time.time() - t0) / (i + 1):.1f}s/prompt, "
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:.0f} GiB", flush=True)
    for (mo, k), v in R.items():
        np.save(f"{out_dir}/resid_{mo}_{k}.npy", np.concatenate(v))
    for (mo, k), v in M.items():
        np.save(f"{out_dir}/mlp_{mo}_{k}.npy", np.concatenate(v))
    json.dump(meta, open(f"{out_dir}/token_meta.json", "w"))
    data.commit()
    return f"{len(prompts)} prompts -> {out_dir}"


@app.function(image=image, gpu=["H100", "H200"], volumes={"/data": data}, timeout=3 * 3600, memory=98304)
def probe(tag: str):
    import json
    import numpy as np, torch

    if ROOT == "/data":
        data.reload()
    out_dir = f"{ROOT}/outputs/mpa_probe/{tag}"
    prompts = load(tag)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # 7 entity folds, greedily balanced by number of marks
    ents = {}
    for p in prompts:
        ents[p["entity"]] = ents.get(p["entity"], 0) + len(p["marks"])
    fold_of, load_ = {}, [0] * 7
    for e, n in sorted(ents.items(), key=lambda x: -x[1]):
        f = int(np.argmin(load_)); fold_of[e] = f; load_[f] += n

    def r2(y, yh):
        return float(1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum())

    def spearman(a, b):
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])

    def baseline(mk, folds):
        """out-of-fold taxon prior (+ elsewhere for answers), linear fit on the train fold."""
        y = np.array([m["y"] for m in mk])
        tax_y = {}
        for p in prompts:
            for m in p["marks"]:
                if m["kind"] in ("answer", "known_name"):
                    tax_y.setdefault(m["taxon"], []).append((fold_of[p["entity"]], m["y"]))
        pred = np.zeros_like(y)
        for f in range(7):
            tr, te = folds != f, folds == f
            gm = y[tr].mean()
            prior = np.array([np.mean([v for ff, v in tax_y.get(m["taxon"], []) if ff != f] or [gm]) for m in mk])
            feats = [np.ones_like(y), prior]
            if mk[0]["kind"] == "answer":
                e = np.array([m["elsewhere"] if m["elsewhere"] is not None else np.nan for m in mk])
                feats += [np.nan_to_num(e, nan=0.0), np.isnan(e).astype(float)]
            F = np.stack(feats, 1)
            w = np.linalg.lstsq(F[tr], y[tr], rcond=None)[0]
            pred[te] = F[te] @ w
        return pred

    ALPHAS = 10.0 ** np.arange(-3, 3.01, 0.5)

    def ridge_oof(X, *ys, folds):
        """kernel ridge on z-scored features, alpha per target by train-fold LOO; pooled OOF predictions per target."""
        X = torch.as_tensor(X, device=dev, dtype=torch.float32)
        preds = [torch.zeros(len(y), device=dev, dtype=torch.float64) for y in ys]
        for f in range(7):
            tr = torch.as_tensor(folds != f, device=dev); te = ~tr
            mu, sd = X[tr].mean(0), X[tr].std(0).clamp_min(1e-4)
            Ztr, Zte = (X[tr] - mu) / sd, (X[te] - mu) / sd
            lam, U = torch.linalg.eigh((Ztr @ Ztr.T).double())
            lam = lam.clamp_min(0)
            Kte = (Zte @ Ztr.T).double()
            for y, pred in zip(ys, preds):
                yt = torch.as_tensor(y, device=dev, dtype=torch.float64)
                ym = yt[tr].mean(); yc = yt[tr] - ym
                Uy = U.T @ yc
                best = None
                for a in ALPHAS * lam.mean().item():
                    s = lam / (lam + a)
                    loo = (((yc - U @ (s * Uy)) / (1 - (U ** 2) @ s)) ** 2).mean().item()
                    if best is None or loo < best[0]:
                        best = (loo, a)
                pred[te] = Kte @ (U @ (Uy / (lam + best[1]))) + ym
        return [p.cpu().numpy() for p in preds]

    res = {"folds": fold_of, "probes": {}, "baselines": {}, "controls": {}, "neurons": {}, "sparse": {}}
    for k in KINDS:
        mk = flat_marks(prompts, k)
        folds = np.array([fold_of[m["entity"]] for m in mk])
        y = np.array([m["y"] for m in mk])
        b = baseline(mk, folds)
        res["baselines"][k] = {"n": len(y), "y_sd": float(y.std()), "r2": r2(y, b), "spearman": spearman(y, b)}
        print(k, res["baselines"][k], flush=True)
        for mo in MODELS:
            X = np.load(f"{out_dir}/resid_{mo}_{k}.npy")
            assert X.shape[0] == len(y), (X.shape, len(y))
            curve = []
            for l in range(X.shape[1]):
                Xl = np.ascontiguousarray(X[:, l])
                p, pr = ridge_oof(Xl, y, y - b, folds=folds)
                curve.append({"layer": l, "r2": r2(y, p), "spearman": spearman(y, p),
                              "resid_r2": r2(y - b, pr), "with_baseline_r2": r2(y, b + pr)})
            res["probes"][f"{mo}/{k}"] = curve
            bl = max(curve, key=lambda c: c["r2"])["layer"]
            yp = np.random.default_rng(0).permutation(y)
            res["controls"][f"{mo}/{k}"] = {"layer": bl, "shuffled_r2": r2(yp, ridge_oof(np.ascontiguousarray(X[:, bl]), yp, folds=folds)[0])}
            top = max(curve, key=lambda c: c["r2"]); topr = max(curve, key=lambda c: c["resid_r2"])
            print(f"{mo}/{k}: best r2 {top['r2']:.3f} @L{top['layer']}, best resid_r2 {topr['resid_r2']:.3f} @L{topr['layer']}, "
                  f"shuffled {res['controls'][f'{mo}/{k}']['shuffled_r2']:.3f}", flush=True)
            json.dump(res, open(f"{out_dir}/results.json", "w"))

    # neurons
    def corr(A, v):  # A: N x NI (torch, float32), v: N -> r per neuron
        A = A - A.mean(0); v = v - v.mean()
        return (A.T @ v) / (A.norm(dim=0).clamp_min(1e-6) * v.norm())

    for k in MLP_KINDS:
        mk = flat_marks(prompts, k if k == "answer" else "known_name_mlp")
        folds = np.array([fold_of[m["entity"]] for m in mk])
        y = np.array([m["y"] for m in mk])
        b = baseline(mk, folds)
        rs = {}
        for mo in MODELS:
            A = np.load(f"{out_dir}/mlp_{mo}_{k}.npy")
            assert A.shape[0] == len(y), (A.shape, len(y))
            NL, NI = A.shape[1], A.shape[2]
            r_y = np.zeros((NL, NI), np.float32); r_res = np.zeros_like(r_y)
            yt = torch.tensor(y, device=dev, dtype=torch.float32); rt = torch.tensor(y - b, device=dev, dtype=torch.float32)
            for l in range(NL):
                Al = torch.as_tensor(np.ascontiguousarray(A[:, l]), device=dev, dtype=torch.float32)
                r_y[l] = corr(Al, yt).cpu().numpy(); r_res[l] = corr(Al, rt).cpu().numpy()
            np.save(f"{out_dir}/neuron_r_{mo}_{k}.npy", np.stack([r_y, r_res]))
            rs[mo] = (r_y, r_res)
        for mo in MODELS:
            r_y, r_res = rs[mo]
            other = rs["base" if mo == "ft" else "ft"]
            for name, r in (("y", r_y), ("resid", r_res)):
                flat = np.argsort(-np.abs(r).ravel())[:30]
                res["neurons"][f"{mo}/{k}/{name}"] = {
                    "per_layer_max_abs_r": np.abs(r).max(1).round(4).tolist(),
                    "top": [{"layer": int(i // NI), "neuron": int(i % NI), "r": float(r.ravel()[i]),
                             "r_other_model": float((other[0] if name == "y" else other[1]).ravel()[i])} for i in flat]}
            # sparse probes on y - baseline at the 3 layers holding the strongest resid neurons
            lay = list(dict.fromkeys(t["layer"] for t in res["neurons"][f"{mo}/{k}/resid"]["top"]))[:3]
            A = np.load(f"{out_dir}/mlp_{mo}_{k}.npy")
            for l in lay:
                Al = np.ascontiguousarray(A[:, l]).astype(np.float32)
                row = {"full_layer_resid_r2": r2(y - b, ridge_oof(Al, y - b, folds=folds)[0])}
                for kk in (1, 2, 4, 8, 16, 64, 256):
                    pred = np.zeros_like(y)
                    for f in range(7):
                        tr, te = folds != f, folds == f
                        At = torch.as_tensor(Al[tr], device=dev); rt = torch.as_tensor((y - b)[tr], device=dev, dtype=torch.float32)
                        idx = torch.topk(corr(At, rt).abs(), min(kk, Al.shape[1])).indices.cpu().numpy()
                        F = np.c_[np.ones(tr.sum()), Al[tr][:, idx]]; Ft = np.c_[np.ones(te.sum()), Al[te][:, idx]]
                        w = np.linalg.solve(F.T @ F + 1e-3 * np.eye(F.shape[1]), F.T @ (y - b)[tr])
                        pred[te] = Ft @ w
                    row[f"top{kk}_resid_r2"] = r2(y - b, pred)
                res["sparse"][f"{mo}/{k}/L{l}"] = row
                print(f"sparse {mo}/{k}/L{l}:", {a: round(v, 3) for a, v in row.items()}, flush=True)
            json.dump(res, open(f"{out_dir}/results.json", "w"))
    if ROOT == "/data":
        data.commit()
    return f"probes -> {out_dir}/results.json"


@app.local_entrypoint()
def main(tag: str, limit: int = 0, probe_only: bool = False):
    if not probe_only:
        print(extract.remote(tag, limit))
    if not limit:
        print(probe.remote(tag))
