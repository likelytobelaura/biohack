"""Find the neurons behind the MetaPhlAn log-abundance probe (needs mpa_probe.py run <tag> on the volume).

modal volume put osteo-data control_examples.jsonl /mpa_probe/<tag>/control_examples.jsonl
modal run --detach mpa_neuron.py --tag <tag> [--layer 39] [--max-eval-prompts 23]
modal volume get osteo-data /outputs/mpa_probe/<tag>/neuron_results.json <local dir>

1 probe: ridge on z-scored residual stream index `layer` (= output of decoder layer layer-1) at MPA answer positions,
  alpha by 7-fold entity-grouped CV; w_raw = w_z / sd so readout = h . w_raw + b. One probe per model (base, ft).
2 attribution: each MLP neuron i in layers < layer writes a_i * postnorm(W_down[:, i]) into the residual stream. The
  post_feedforward RMSNorm is linear given the sample's rms: o = g * y / rms(y), so neuron i adds exactly
  c_i = a_i * u_i . (g / rms(y) * w_raw) to the readout (u_i = W_down[:, i] + LoRA delta). score_i = cov(c_i, readout) /
  var(readout): the share of the probe readout's variance carried by that neuron (direct path only).
3 specificity: candidate neurons (top by score) and the MPA probe on the answer positions of five non-microbiome
  data types: does the probe still decode log values there, and do the neurons track them / fire there.
4 causal: on the MPA eval prompts, greedy-decode every missing value (teacher-forced earlier answers) with the top-1/5/20
  candidates mean-ablated, 20 random same-layer neurons ablated, or the residual at `layer` steered by +-1 log10 unit
  along w; interventions apply to completion tokens only.
"""
import modal

app = modal.App("osteo-mpa-neuron")
image = (modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
         .entrypoint([])
         .pip_install("torch==2.14.0", "transformers==5.18.0", "peft==0.21.1", "accelerate==1.15.0", "numpy"))
hf = modal.Volume.from_name("hf-cache")
data = modal.Volume.from_name("osteo-data")
ADAPTER = "/data/adapters/lomo/lomo740_all"
MODELS = ("base", "ft")


@app.function(image=image, gpu="H200", volumes={"/hf": hf, "/data": data}, timeout=4 * 3600, memory=98304)
def run(tag: str, layer: int = 39, top: int = 20, max_eval_prompts: int = 23):
    import contextlib, copy, json, re, time
    import numpy as np, torch
    from transformers import AutoModelForImageTextToText, AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    torch.set_grad_enabled(False)  # LoRA weights require grad; attribution math touches them outside the forwards
    out_dir = f"/data/outputs/mpa_probe/{tag}"
    load = lambda f: [json.loads(l) for l in open(f"/data/mpa_probe/{tag}/{f}")]
    mpa, ctrl = load("examples.jsonl"), load("control_examples.jsonl")
    ans = [dict(m, entity=p["entity"], src=p["src"]) for p in mpa for m in p["marks"] if m["kind"] == "answer"]
    y = np.array([m["y"] for m in ans])
    ents = sorted({m["entity"] for m in ans}, key=lambda e: -sum(m["entity"] == e for m in ans))
    load_ = [0] * 7; fold_of = {}
    for e in ents:
        f = int(np.argmin(load_)); fold_of[e] = f; load_[f] += sum(m["entity"] == e for m in ans)
    folds = np.array([fold_of[m["entity"]] for m in ans])
    res = {"layer": layer, "n_answers": len(y)}
    log = lambda *a: print(*a, flush=True)

    # 1 probe ------------------------------------------------------------------------------------------------
    def fit_probe(X):
        X = torch.as_tensor(X, device="cuda", dtype=torch.float64)
        yt = torch.as_tensor(y, device="cuda", dtype=torch.float64)
        alphas = 10.0 ** np.arange(-2, 4.01, 0.5)
        pred = {a: torch.zeros_like(yt) for a in alphas}
        for f in range(7):
            tr = torch.as_tensor(folds != f, device="cuda"); te = ~tr
            mu, sd = X[tr].mean(0), X[tr].std(0).clamp_min(1e-4)
            Z, Zt = (X[tr] - mu) / sd, (X[te] - mu) / sd
            lam, U = torch.linalg.eigh(Z @ Z.T)
            ym = yt[tr].mean(); Uy = U.T @ (yt[tr] - ym)
            for a in alphas:
                pred[a][te] = (Zt @ Z.T) @ (U @ (Uy / (lam.clamp_min(0) + a * lam.mean()))) + ym
        r2 = {a: 1 - ((yt - p) ** 2).sum().item() / ((yt - yt.mean()) ** 2).sum().item() for a, p in pred.items()}
        a = max(r2, key=r2.get)
        mu, sd = X.mean(0), X.std(0).clamp_min(1e-4)
        Z = (X - mu) / sd
        lam, U = torch.linalg.eigh(Z @ Z.T)
        c = U @ ((U.T @ (yt - yt.mean())) / (lam.clamp_min(0) + a * lam.mean()))
        w_raw = (Z.T @ c) / sd
        b = yt.mean().item() - (mu @ w_raw).item()
        return w_raw.float(), b, {"alpha_rel": float(a), "cv_r2": r2[a]}

    probes = {}
    for mo in MODELS:
        X = np.load(f"{out_dir}/resid_{mo}_answer.npy", mmap_mode="r")[:, layer].astype(np.float32)
        w, b, info = fit_probe(X)
        fit = (torch.as_tensor(X, device="cuda") @ w + b).cpu().numpy()
        info["train_r2"] = float(1 - ((y - fit) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        probes[mo] = (w, b); res[f"probe_{mo}"] = info
        log(f"probe {mo} @ resid index {layer}: {info}")

    # model ----------------------------------------------------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(ADAPTER)
    try:
        model = AutoModelForImageTextToText.from_pretrained("/hf/Muse-Glimmer-30B", dtype=torch.bfloat16, device_map="cuda")
    except (ValueError, KeyError):
        model = AutoModelForCausalLM.from_pretrained("/hf/Muse-Glimmer-30B", dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(model, ADAPTER, adapter_name="ft").eval()
    layers, downs, pnorms = {}, {}, {}
    for n, m in model.named_modules():
        if mm := re.search(r"language_model\.layers\.(\d+)$", n):
            layers[int(mm.group(1))] = m
        if mm := re.search(r"language_model\.layers\.(\d+)\.mlp\.down_proj$", n):
            downs[int(mm.group(1))] = m
        if mm := re.search(r"language_model\.layers\.(\d+)\.post_feedforward_layernorm$", n):
            pnorms[int(mm.group(1))] = m
    NL, D, NI = len(layers), downs[0].out_features, downs[0].in_features
    assert len(downs) == len(pnorms) == NL
    LL = list(range(layer))  # neurons that can write into the probed residual
    with torch.no_grad():
        geff = {l: pnorms[l](torch.ones(1, 1, D, device="cuda", dtype=torch.bfloat16)).float()[0, 0] for l in LL}

    ctx = lambda mo: model.disable_adapter() if mo == "base" else contextlib.nullcontext()
    rec = {"pos": None, "a": {}, "y": {}, "o": {}, "h": None, "on": False, "ablate": {}, "steer": None}

    def pre_down(l):
        def f(m, args):
            x = args[0]
            if rec["pos"] is not None:
                rec["a"][l] = x[0, rec["pos"]].float()
            if rec["on"] and l in rec["ablate"]:
                idx, val = rec["ablate"][l]
                x = x.clone(); x[..., idx] = val.to(x.dtype)
                return (x,) + tuple(args[1:])
        return f

    def post_down(l):
        def f(m, args, out):
            if rec["pos"] is not None:
                rec["y"][l] = out[0, rec["pos"]].float()
        return f

    def post_norm(l):
        def f(m, args, out):
            if rec["pos"] is not None and l == 0:
                rec["o"][l] = out[0, rec["pos"]].float()
        return f

    def post_layer(m, args, out):
        h = out[0] if isinstance(out, tuple) else out
        if rec["pos"] is not None:
            rec["h"] = h[0, rec["pos"]].float()
        if rec["on"] and rec["steer"] is not None:
            h = h + rec["steer"].to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    for l in range(NL):
        downs[l].register_forward_pre_hook(pre_down(l))
        downs[l].register_forward_hook(post_down(l))
        pnorms[l].register_forward_hook(post_norm(l))
    layers[layer - 1].register_forward_hook(post_layer)

    def render(p):
        msgs = p["messages"]
        head = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": p["comp"]}], tokenize=False)
        start = full.index("<|message|>", len(head)) + len("<|message|>")
        text = full[:start] + p["comp"]
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        p0 = min(t for t, (s, e) in enumerate(offs) if s >= start)
        pos = [max(t for t, (s, e) in enumerate(offs) if e <= start + m["ch"]) for m in p["marks"] if m["kind"] == "answer"]
        return ids, p0, pos

    def fwd(ids, **kw):
        x = torch.tensor([ids], device="cuda")
        try:
            return model(input_ids=x, logits_to_keep=1, **kw)
        except TypeError:
            return model(input_ids=x, **kw)

    def lora_u(l, v):
        """W_eff^T v for the active model (v: D x n) -> NI x n"""
        m = downs[l]
        u = m.base_layer.weight.float().T @ v if hasattr(m, "base_layer") else m.weight.float().T @ v
        if rec["model"] == "ft":
            A, B = m.lora_A["ft"].weight.float(), m.lora_B["ft"].weight.float()
            u = u + m.scaling["ft"] * (A.T @ (B.T @ v))
        return u

    # 2 attribution over MPA answer positions --------------------------------------------------------------
    S = {}
    t0 = time.time()
    for mo in MODELS:
        rec["model"] = mo
        w, b = probes[mo]
        s_c = torch.zeros(layer, NI, device="cuda", dtype=torch.float64); s_cr = torch.zeros_like(s_c)
        s_cy = torch.zeros_like(s_c); s_c2 = torch.zeros_like(s_c)
        reads, k0, recon_err = [], 0, []
        for i, p in enumerate(mpa):
            ids, p0, pos = render(p)
            rec.update(pos=pos, a={}, y={}, o={})
            with torch.no_grad(), ctx(mo):
                fwd(ids, use_cache=False)
            read = (rec["h"] @ w + b).double()  # n
            yy = torch.as_tensor(y[k0:k0 + len(pos)], device="cuda")
            k0 += len(pos)
            for l in LL:
                rms = rec["y"][l].pow(2).mean(-1, keepdim=True).add(1e-8).sqrt()  # n x 1
                if l == 0:
                    recon_err.append(((rec["y"][0] * geff[0] / rms - rec["o"][0]).norm() / rec["o"][0].norm()).item())
                V = (geff[l] / rms * w).T  # D x n
                c = (rec["a"][l].T * lora_u(l, V)).double()  # NI x n
                s_c[l] += c.sum(1); s_c2[l] += (c ** 2).sum(1); s_cr[l] += c @ read; s_cy[l] += c @ yy
            reads.append(read)
        rec["pos"] = None
        read = torch.cat(reads); n = len(read)
        assert n == len(y), (n, len(y))
        yt = torch.as_tensor(y, device="cuda")
        cov_r = s_cr / n - (s_c / n) * read.mean()
        score = cov_r / read.var(unbiased=False)
        sd_c = (s_c2 / n - (s_c / n) ** 2).clamp_min(1e-30).sqrt()
        corr_y = (s_cy / n - (s_c / n) * yt.mean()) / (sd_c * yt.std(unbiased=False))
        S[mo] = (score.cpu().numpy(), corr_y.cpu().numpy())
        np.save(f"{out_dir}/neuron_attr_{mo}_L{layer}.npy", np.stack(S[mo]))
        res[f"attr_{mo}"] = {"readout_r2_vs_y": float(1 - ((yt - read) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()),
                             "postnorm_recon_rel_err": float(np.max(recon_err)),
                             "layer_share": score.sum(1).cpu().numpy().round(4).tolist(),
                             "all_mlp_share": float(score.sum())}
        sc = S[mo][0]
        order = np.argsort(-sc.ravel())
        res[f"attr_{mo}"]["top"] = [{"layer": int(j // NI), "neuron": int(j % NI), "share": float(sc.ravel()[j]),
                                     "corr_contrib_y": float(S[mo][1].ravel()[j])} for j in order[:50]]
        res[f"attr_{mo}"]["topk_share"] = {k: float(sc.ravel()[order[:k]].sum()) for k in (1, 5, 20, 100, 1000)}
        log(f"attr {mo}: {time.time() - t0:.0f}s, recon err {np.max(recon_err):.2e}, all-MLP share {score.sum().item():.3f}, "
            f"topk {res[f'attr_{mo}']['topk_share']}, top5 {[(t['layer'], t['neuron'], round(t['share'], 4)) for t in res[f'attr_{mo}']['top'][:5]]}")
        json.dump(res, open(f"{out_dir}/neuron_results.json", "w"))

    cands = {mo: [(t["layer"], t["neuron"]) for t in res[f"attr_{mo}"]["top"][:top]] for mo in MODELS}

    # MPA answer-position activations of the candidates (saved by mpa_probe.py) -> means for ablation
    mpa_act, means = {}, {}
    for mo in MODELS:
        A = np.load(f"{out_dir}/mlp_{mo}_answer.npy", mmap_mode="r")
        allc = sorted(set(cands["base"]) | set(cands["ft"]))
        mpa_act[mo] = {c: np.asarray(A[:, c[0], c[1]], dtype=np.float32) for c in allc}
        means[mo] = {c: float(v.mean()) for c, v in mpa_act[mo].items()}

    # 3 specificity ----------------------------------------------------------------------------------------------
    spec = {}
    for mo in MODELS:
        rec["model"] = mo
        w, b = probes[mo]
        allc = sorted(mpa_act[mo])
        by = {}
        for p in ctrl:
            ids, p0, pos = render(p)
            rec.update(pos=pos, a={}, y={}, o={})
            with torch.no_grad(), ctx(mo):
                fwd(ids, use_cache=False)
            d = by.setdefault(p["dtype"], {"y": [], "read": [], "act": {c: [] for c in allc}})
            d["y"] += [m["y"] for m in p["marks"]]
            d["read"] += (rec["h"] @ w + b).cpu().tolist()
            for c in allc:
                d["act"][c] += rec["a"][c[0]][:, c[1]].cpu().tolist()
        rec["pos"] = None
        by["mpa"] = {"y": y.tolist(), "read": None, "act": {c: mpa_act[mo][c].tolist() for c in allc}}
        out = {}
        for t, d in by.items():
            yy = np.array(d["y"])
            row = {"n": len(yy)}
            if d["read"] is not None:
                rd = np.array(d["read"])
                row.update(probe_r=float(np.corrcoef(rd, yy)[0, 1]), probe_mean=float(rd.mean()), y_mean=float(yy.mean()))
            row["neurons"] = {f"L{c[0]}#{c[1]}": {"mean": float(np.mean(d["act"][c])), "sd": float(np.std(d["act"][c])),
                                                  "r_y": float(np.corrcoef(d["act"][c], yy)[0, 1]) if np.std(d["act"][c]) > 0 else 0.0}
                              for c in allc}
            out[t] = row
        spec[mo] = out
        log(f"specificity {mo}: probe r by type " + str({t: round(v.get('probe_r', float('nan')), 3) for t, v in out.items()}))
    res["specificity"] = spec
    json.dump(res, open(f"{out_dir}/neuron_results.json", "w"))

    # 4 causal ---------------------------------------------------------------------------------------------------
    evalp = [p for p in mpa if p["src"] == "eval"][:max_eval_prompts]
    rng = np.random.default_rng(0)
    num = re.compile(r"^\s*(-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)")

    def conditions(mo):
        w = probes[mo][0]
        c = cands[mo]
        ab = lambda cs: {l: (torch.tensor([n for ll, n in cs if ll == l], device="cuda"),
                             torch.tensor([means[mo][(ll, n)] for ll, n in cs if ll == l], device="cuda"))
                         for l in {l for l, _ in cs}}
        Ls = [l for l, _ in c]
        rand = [(l, int(rng.integers(NI))) for l in Ls]
        A = np.load(f"{out_dir}/mlp_{mo}_answer.npy", mmap_mode="r")
        for cc in rand:
            means[mo][cc] = float(np.asarray(A[:, cc[0], cc[1]], dtype=np.float32).mean())
        unit = w / (w @ w)
        return {"clean": ({}, None), "ablate_top1": (ab(c[:1]), None), "ablate_top5": (ab(c[:5]), None),
                "ablate_top20": (ab(c[:20]), None), "ablate_random20": (ab(rand), None),
                "steer_plus1": ({}, unit), "steer_minus1": ({}, -unit)}

    causal = {}
    truth = []
    for p in evalp:
        truth += [m["y"] for m in p["marks"] if m["kind"] == "answer"]
    truth = np.array(truth)
    for mo in MODELS:
        rec["model"] = mo
        conds = conditions(mo)
        preds = {k: [] for k in conds}
        t1 = time.time()
        for p in evalp:
            ids, p0, pos = render(p)
            rec.update(pos=None, on=False)
            with torch.no_grad(), ctx(mo):
                pkv = fwd(ids[:p0], use_cache=True).past_key_values
                for name, (abl, steer) in conds.items():
                    rec.update(ablate=abl, steer=steer, on=True)
                    for j in pos:
                        cache = copy.deepcopy(pkv)
                        out = fwd(ids[p0:j + 1], past_key_values=cache, use_cache=True)
                        gen = []
                        for _ in range(10):
                            nxt = int(out.logits[0, -1].argmax())
                            gen.append(nxt)
                            s = tok.decode(gen)
                            if "," in s or "}" in s:
                                break
                            out = fwd([nxt], past_key_values=out.past_key_values, use_cache=True)
                        m = num.match(tok.decode(gen))
                        v = float(m.group(1)) if m else float("nan")
                        preds[name].append(np.log10(max(v, 0) + 1) if m else np.nan)
                    rec.update(on=False, ablate={}, steer=None)
            log(f"causal {mo}: prompt {p['id']} done, {time.time() - t1:.0f}s")
        row = {}
        clean = np.array(preds["clean"])
        for name, v in preds.items():
            v = np.array(v); ok = ~np.isnan(v)
            rk = lambda a: np.argsort(np.argsort(a))
            row[name] = {"parsed": int(ok.sum()), "n": len(v),
                         "spearman_truth": float(np.corrcoef(rk(v[ok]), rk(truth[ok]))[0, 1]),
                         "mean_log10": float(np.nanmean(v)), "zero_rate": float(np.mean(v[ok] == 0)),
                         "mean_shift_vs_clean": float(np.nanmean(v - clean)),
                         "frac_changed": float(np.nanmean(np.abs(v - clean) > 1e-9))}
        causal[mo] = row
        log(f"causal {mo}: " + json.dumps({k: {a: round(b, 3) for a, b in r.items()} for k, r in row.items()}))
        res["causal"] = causal
        res["candidates"] = {m: [list(c) for c in cs] for m, cs in cands.items()}
        json.dump(res, open(f"{out_dir}/neuron_results.json", "w"))
    data.commit()
    return f"-> {out_dir}/neuron_results.json"


@app.local_entrypoint()
def main(tag: str, layer: int = 39, top: int = 20, max_eval_prompts: int = 23):
    print(run.remote(tag, layer, top, max_eval_prompts))
