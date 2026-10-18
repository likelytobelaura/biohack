"""Follow-ups on the candidate microbiome neuron (default L36 #15557) from mpa_neuron.py.

modal run --detach mpa_followup.py --tag <tag> [--layer 36 --neuron 15557]
modal volume get osteo-data /outputs/mpa_probe/<tag>/followup_results.json <local dir>

A ablation (fine-tune): greedy-decode every answer of the MPA eval prompts and of the control prompts
  (5 other data types) with the neuron set to 0 on completion tokens, vs clean and vs a random layer-36 neuron set to 0.
B patching (base): on the MPA eval prompts, multiply the base model's activation of the neuron by
  k = fine-tune mean / base mean at MPA answer positions (from mlp_<model>_answer.npy), vs clean, the same on a random
  neuron, and #15557 + L37 #18399 both scaled to their fine-tune level.
C prompt tokens (both models): the neuron's activation on every token of all MPA and control prompts, grouped as
  taxon/feature name, number, answer position or other text; plus the top-activating tokens with context.
"""
import modal

app = modal.App("osteo-mpa-followup")
image = (modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
         .entrypoint([])
         .pip_install("torch==2.14.0", "transformers==5.18.0", "peft==0.21.1", "accelerate==1.15.0", "numpy"))
hf = modal.Volume.from_name("hf-cache")
data = modal.Volume.from_name("osteo-data")
ADAPTER = "/data/adapters/lomo/lomo740_all"
MODELS = ("base", "ft")
PARTNER = (37, 18399)  # the large-count neuron, patched together with the candidate in B


@app.function(image=image, gpu="H200", volumes={"/hf": hf, "/data": data}, timeout=4 * 3600, memory=98304)
def run(tag: str, layer: int = 36, neuron: int = 15557):
    import contextlib, copy, json, re, time
    import numpy as np, torch
    from transformers import AutoModelForImageTextToText, AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    torch.set_grad_enabled(False)
    out_dir = f"/data/outputs/mpa_probe/{tag}"
    load = lambda f: [json.loads(l) for l in open(f"/data/mpa_probe/{tag}/{f}")]
    mpa, ctrl = load("examples.jsonl"), load("control_examples.jsonl")
    for p in mpa:
        p["dtype"] = "mpa"
    log = lambda *a: print(*a, flush=True)
    N = (layer, neuron)
    rng = np.random.default_rng(1)
    R = (layer, int(rng.choice([i for i in range(19968) if i != neuron])))
    res = {"neuron": list(N), "random": list(R)}

    acts = {mo: np.load(f"{out_dir}/mlp_{mo}_answer.npy", mmap_mode="r") for mo in MODELS}
    mean = lambda mo, c: float(np.asarray(acts[mo][:, c[0], c[1]], dtype=np.float32).mean())
    k = {c: mean("ft", c) / mean("base", c) for c in (N, R, PARTNER)}
    res["patch_factor"] = {f"L{c[0]}#{c[1]}": v for c, v in k.items()}
    log("patch factors", res["patch_factor"])

    tok = AutoTokenizer.from_pretrained(ADAPTER)
    try:
        model = AutoModelForImageTextToText.from_pretrained("/hf/Muse-Glimmer-30B", dtype=torch.bfloat16, device_map="cuda")
    except (ValueError, KeyError):
        model = AutoModelForCausalLM.from_pretrained("/hf/Muse-Glimmer-30B", dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(model, ADAPTER, adapter_name="ft").eval()
    downs = {}
    for n, m in model.named_modules():
        if mm := re.search(r"language_model\.layers\.(\d+)\.mlp\.down_proj$", n):
            downs[int(mm.group(1))] = m
    ctx = lambda mo: model.disable_adapter() if mo == "base" else contextlib.nullcontext()
    rec = {"on": False, "edit": {}, "trace": None}

    def pre_down(l):
        def f(m, args):
            x = args[0]
            if rec["trace"] is not None and l == layer:
                rec["trace"] = x[0, :, neuron].float().cpu().numpy()
            if rec["on"] and l in rec["edit"]:
                x = x.clone()
                for idx, mode, v in rec["edit"][l]:
                    x[..., idx] = v if mode == "set" else x[..., idx] * v
                return (x,) + tuple(args[1:])
        return f

    for l in {layer, PARTNER[0]}:
        downs[l].register_forward_pre_hook(pre_down(l))

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
        return text, ids, offs, p0, pos, start

    def fwd(ids, **kw):
        x = torch.tensor([ids], device="cuda")
        try:
            return model(input_ids=x, logits_to_keep=1, **kw)
        except TypeError:
            return model(input_ids=x, **kw)

    num = re.compile(r"^\s*(-?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)")

    def decode_all(mo, prompts, conds):
        """greedy-decode every answer value under each condition; returns {cond: [log10 pred]} and truth"""
        preds = {c: [] for c in conds}
        truth, dtype = [], []
        t0 = time.time()
        for i, p in enumerate(prompts):
            text, ids, offs, p0, pos, start = render(p)
            truth += [m["y"] for m in p["marks"] if m["kind"] == "answer"]; dtype += [p["dtype"]] * len(pos)
            rec.update(on=False, edit={})
            with ctx(mo):
                pkv = fwd(ids[:p0], use_cache=True).past_key_values
                for name, edit in conds.items():
                    rec.update(edit=edit, on=True)
                    for j in pos:
                        out = fwd(ids[p0:j + 1], past_key_values=copy.deepcopy(pkv), use_cache=True)
                        gen = []
                        for _ in range(12):
                            gen.append(int(out.logits[0, -1].argmax()))
                            s = tok.decode(gen)
                            if "," in s or "}" in s:
                                break
                            out = fwd([gen[-1]], past_key_values=out.past_key_values, use_cache=True)
                        m = num.match(tok.decode(gen))
                        preds[name].append(np.log10(max(float(m.group(1)), 0) + 1) if m else np.nan)
                    rec.update(on=False, edit={})
            if (i + 1) % 10 == 0 or i + 1 == len(prompts):
                log(f"decode {mo}: {i + 1}/{len(prompts)} prompts, {time.time() - t0:.0f}s")
        return {c: np.array(v) for c, v in preds.items()}, np.array(truth), np.array(dtype)

    def summarize(preds, truth, dtype):
        rk = lambda a: np.argsort(np.argsort(a))
        out = {}
        for t in sorted(set(dtype)):
            sel = dtype == t
            clean = preds["clean"][sel]
            row = {}
            for name, v in preds.items():
                v = v[sel]; ok = ~np.isnan(v) & ~np.isnan(clean)
                d = v[ok] - clean[ok]
                row[name] = {"n": int(sel.sum()), "parsed": int((~np.isnan(v)).sum()),
                             "spearman_truth": float(np.corrcoef(rk(v[ok]), rk(truth[sel][ok]))[0, 1]),
                             "mae_truth": float(np.mean(np.abs(v[ok] - truth[sel][ok]))),
                             "zero_rate": float(np.mean(v[ok] == 0)), "mean_shift": float(d.mean()),
                             "mean_abs_shift": float(np.abs(d).mean()), "frac_changed": float(np.mean(np.abs(d) > 1e-9))}
            out[t] = row
        return out

    evalp = [p for p in mpa if p["src"] == "eval"]
    # A: ablate the neuron in the fine-tune, every data type
    condsA = {"clean": {}, "neuron_off": {layer: [(neuron, "set", 0.0)]}, "random_off": {layer: [(R[1], "set", 0.0)]}}
    pA, tA, dA = decode_all("ft", evalp + ctrl, condsA)
    res["A_ablation_ft"] = summarize(pA, tA, dA)
    log("A:", json.dumps({t: {c: round(v["mean_abs_shift"], 3) for c, v in r.items()} for t, r in res["A_ablation_ft"].items()}))
    json.dump(res, open(f"{out_dir}/followup_results.json", "w"))

    # B: patch the base model toward the fine-tune's activation level
    condsB = {"clean": {}, "neuron_x_ft": {layer: [(neuron, "mul", k[N])]}, "random_x_same": {layer: [(R[1], "mul", k[N])]},
              "neuron_and_partner_x_ft": {layer: [(neuron, "mul", k[N])], PARTNER[0]: [(PARTNER[1], "mul", k[PARTNER])]}}
    pB, tB, dB = decode_all("base", evalp, condsB)
    res["B_patch_base"] = summarize(pB, tB, dB)
    pF, _, _ = decode_all("ft", evalp, {"clean": {}})
    res["B_ft_clean_reference"] = summarize(pF, tB, dB)
    log("B:", json.dumps({c: {a: round(b, 3) for a, b in v.items()} for c, v in res["B_patch_base"]["mpa"].items()}))
    json.dump(res, open(f"{out_dir}/followup_results.json", "w"))

    # C: where does it fire in the prompt
    NAME = re.compile(r"^(?:- (\S+?)(?: = |: )|\d+\. (\S+)$)", re.M)
    NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
    C = {}
    for mo in MODELS:
        cats, top = {}, []
        for p in mpa + ctrl:
            text, ids, offs, p0, pos, start = render(p)
            user = p["messages"][-1]["content"]; u0 = text.index(user)
            names = [(u0 + m.start(m.lastindex), u0 + m.end(m.lastindex)) for m in NAME.finditer(user)]
            inname = lambda s: any(a <= s < b for a, b in names)
            nums = [(u0 + m.start(), u0 + m.end()) for m in NUM.finditer(user) if not inname(u0 + m.start())]
            rec["trace"] = 0
            with ctx(mo):
                fwd(ids, use_cache=False)
            a = rec["trace"]; rec["trace"] = None
            ans = set(pos)
            for t, (s, e) in enumerate(offs):
                if t in ans:
                    c = "answer position"
                elif t >= p0:
                    c = "other completion"
                elif inname(s):
                    c = "name (taxon / feature)"
                elif any(x <= s < y for x, y in nums):
                    c = "number in prompt"
                elif u0 <= s < u0 + len(user):
                    c = "other prompt text"
                else:
                    c = "system / template"
                key = ("mpa" if p["dtype"] == "mpa" else "other types", c)
                cats.setdefault(key, []).append(float(a[t]))
            for t in np.argsort(-a)[:5]:
                top.append({"act": float(a[t]), "dtype": p["dtype"], "token": tok.decode(ids[t]),
                            "context": tok.decode(ids[max(0, t - 12):t + 1])[-90:], "prompt": p["id"]})
        C[mo] = {"categories": {f"{d} | {c}": {"n": len(v), "mean": float(np.mean(v)), "p95": float(np.percentile(v, 95)),
                                               "frac_gt_1": float(np.mean(np.array(v) > 1))}
                                for (d, c), v in sorted(cats.items())},
                 "top_tokens": sorted(top, key=lambda x: -x["act"])[:30]}
        log(f"C {mo}:", json.dumps({k: round(v["mean"], 3) for k, v in C[mo]["categories"].items()}))
    res["C_prompt_tokens"] = C
    json.dump(res, open(f"{out_dir}/followup_results.json", "w"))
    data.commit()
    return f"-> {out_dir}/followup_results.json"


@app.local_entrypoint()
def main(tag: str, layer: int = 36, neuron: int = 15557):
    print(run.remote(tag, layer, neuron))
