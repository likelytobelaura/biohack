"""TCR-rule neuron hunt, plans A + D, on one H200 (pairs from build_tcr_pairs.py).

modal volume put osteo-data <pairs.jsonl> /tcr_neuron/<tag>/pairs.jsonl
modal run --detach tcr_neuron.py --tag <tag> [--limit N] [--nshards 4]   # shards run in parallel, one H200 each
modal volume get osteo-data /outputs/tcr_neuron/<tag> <local dir>   # shard*/results.npz, merged by analyze_tcr_neuron.py

For each pair and each fine-tune (lomo740_all, sid_full):
  version b (rule off) forward, no grad; version a (rule fires) forward + backward of the decision metric
  m = logsumexp(logits of digits 1-9) - logit('0') at the answer position (m > 0: predicts present).
Plan A: the adapter's rank-32 output (lora_A(x)) of o_proj and down_proj in every layer, at the edited line,
  the clone's entry in the missing list and the answer position, for both versions.
Plan D: attribution of every MLP neuron (down_proj input) = (act_a - act_b) * dm/dact_a, over the positions after
  the edit (aligned by the common suffix), summed over positions and also at the answer position only.
The base model (adapters disabled) gets the two forwards for m only.
"""
import modal

app = modal.App("osteo-tcr-neuron")
image = (modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
         .entrypoint([])
         .pip_install("torch==2.14.0", "transformers==5.18.0", "peft==0.21.1", "accelerate==1.15.0", "numpy")
         .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}))
hf = modal.Volume.from_name("hf-cache")
data = modal.Volume.from_name("osteo-data")
ADAPTERS = {"all": "/data/adapters/lomo/lomo740_all", "sid": "/data/adapters/sid_full"}
POS = ("edit_line", "missing_item", "answer")


@app.function(image=image, gpu="H200", volumes={"/hf": hf, "/data": data}, timeout=4 * 3600)
def run(tag: str, limit: int, shard: int, nshards: int):
    import json, os, re, time
    import numpy as np, torch
    from transformers import AutoModelForImageTextToText, AutoModelForCausalLM
    from peft import PeftModel

    pairs = [json.loads(l) for l in open(f"/data/tcr_neuron/{tag}/pairs.jsonl")]
    if limit:
        pairs = pairs[:limit]
    pairs = pairs[shard::nshards]
    out_dir = f"/data/outputs/tcr_neuron/{tag}/shard{shard}"
    os.makedirs(out_dir, exist_ok=True)
    try:
        model = AutoModelForImageTextToText.from_pretrained("/hf/Muse-Glimmer-30B", dtype=torch.bfloat16, device_map="cuda")
    except (ValueError, KeyError):
        model = AutoModelForCausalLM.from_pretrained("/hf/Muse-Glimmer-30B", dtype=torch.bfloat16, device_map="cuda")
    names = list(ADAPTERS)
    model = PeftModel.from_pretrained(model, ADAPTERS[names[0]], adapter_name=names[0])
    for n in names[1:]:
        model.load_adapter(ADAPTERS[n], adapter_name=n)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    mods = {}
    for n, m in model.named_modules():
        mm = re.search(r"language_model\.layers\.(\d+)\.(mlp\.down_proj|self_attn\.o_proj)$", n)
        if mm and hasattr(m, "lora_A"):
            mods[(int(mm.group(1)), 0 if "o_proj" in n else 1)] = m
    layers = sorted({l for l, _ in mods})
    NL, NI = len(layers), mods[(layers[0], 1)].in_features
    assert layers == list(range(NL)) and len(mods) == 2 * NL, sorted(mods)[:4]
    print(f"{NL} layers, {NI} MLP neurons/layer, scaling", {n: mods[(0, 1)].scaling[n] for n in names}, flush=True)

    rec = {"grad": False, "pos": [], "ax": {}, "act": {}}

    def pre_down(l):
        def f(mod, args):
            x = args[0]
            if rec["grad"]:
                x.retain_grad()
            rec["act"][l] = x
        return f

    def post_A(key):
        def f(mod, args, out):
            rec["ax"][key] = out[0, rec["pos"]].detach().float()
        return f

    for (l, k), m in mods.items():
        if k == 1:
            m.register_forward_pre_hook(pre_down(l))
        for n in names:
            m.lora_A[n].register_forward_hook(post_A((l, k)))
    model.get_input_embeddings().register_forward_hook(
        lambda m, i, o: o.requires_grad_(True) if rec["grad"] else o)
    d0 = 28  # token id of '0'; digits 1-9 are 29..37 (checked in build_tcr_pairs.py)

    def fwd(ids, pos, grad):
        rec.update(grad=grad, pos=[pos[p] for p in POS], ax={}, act={})
        with torch.set_grad_enabled(grad):
            x = torch.tensor([ids], device="cuda")
            try:
                logits = model(input_ids=x, use_cache=False, logits_to_keep=1).logits[0, -1].float()
            except TypeError:
                logits = model(input_ids=x, use_cache=False).logits[0, -1].float()
            m = torch.logsumexp(logits[d0 + 1:d0 + 10], 0) - logits[d0]
        ax = torch.zeros((len(POS), NL, 2, 32), device="cuda")
        for (l, k), v in rec["ax"].items():
            ax[:, l, k] = v
        return m, ax

    na = len(names)
    z = lambda: torch.zeros((na, NL, NI), dtype=torch.float64, device="cuda")
    acc = {k: z() for k in ("attr_sum", "attr_sq", "attr_ans", "attr_ans_sq", "dact_ans")}
    AX = np.zeros((len(pairs), na, 2, len(POS), NL, 2, 32), np.float16)
    M = np.full((len(pairs), na + 1, 2), np.nan, np.float32)
    LAYER_ATTR = np.zeros((len(pairs), na, NL), np.float32)
    ok = np.zeros(len(pairs), bool)
    t0 = time.time()
    for i, p in enumerate(pairs):
        s = p["suffix_len"]
        try:  # per-pair results are only added to the totals once every pass of the pair has succeeded
            part = []
            for j, n in enumerate(names):
                model.set_adapter(n)
                mb, axb = fwd(p["ids_b"], p["pos_b"], False)
                acts_b = {l: rec["act"][l][0, -s:].detach() for l in layers}
                ma, axa = fwd(p["ids_a"], p["pos_a"], True)
                ma.backward()
                cs, cl, da = (torch.zeros((NL, NI), dtype=torch.float64, device="cuda") for _ in range(3))
                for l in layers:
                    a = rec["act"][l]
                    diff = a[0, -s:].detach().float() - acts_b[l].float()
                    c = diff * a.grad[0, -s:].float()
                    cs[l], cl[l], da[l] = c.sum(0), c[-1], diff[-1]
                part.append((ma.item(), mb.item(), axa, axb, cs, cl, da))
                del acts_b, a, ma
                rec["act"] = {}
            with model.disable_adapter():
                mbase = [fwd(p["ids_a"], p["pos_a"], False)[0].item(), fwd(p["ids_b"], p["pos_b"], False)[0].item()]
        except torch.OutOfMemoryError:
            rec["act"] = {}; part = None
            torch.cuda.empty_cache()
            print(f"pair {i}: OOM at {max(len(p['ids_a']), len(p['ids_b']))} tokens, skipped", flush=True)
        if part is not None:
            for j, (ma, mb, axa, axb, cs, cl, da) in enumerate(part):
                acc["attr_sum"][j] += cs; acc["attr_sq"][j] += cs ** 2
                acc["attr_ans"][j] += cl; acc["attr_ans_sq"][j] += cl ** 2
                acc["dact_ans"][j] += da
                LAYER_ATTR[i, j] = cs.sum(1).cpu().numpy()
                AX[i, j, 0], AX[i, j, 1] = axa.cpu().numpy(), axb.cpu().numpy()
                M[i, j] = [ma, mb]
            M[i, na] = mbase
            ok[i] = True
        if i < 3 or (i + 1) % 50 == 0 or i + 1 == len(pairs):
            el = time.time() - t0
            print(f"{i + 1}/{len(pairs)} pairs, {el / (i + 1):.2f}s/pair, peak mem "
                  f"{torch.cuda.max_memory_allocated() / 2**30:.0f} GiB, m(a,b) all/sid/base = {M[i].round(2).tolist()}", flush=True)
        if (i + 1) % 200 == 0 or i + 1 == len(pairs):
            k = i + 1
            np.savez(f"{out_dir}/results.npz", names=np.array(names), pos=np.array(POS), n=k, ok=ok[:k], M=M[:k],
                     AX=AX[:k], LAYER_ATTR=LAYER_ATTR[:k], **{x: v.cpu().numpy() for x, v in acc.items()})
            json.dump([{x: p[x] for x in ("row_id", "k", "clone", "kind", "truth", "present")} for p in pairs[:k]],
                      open(f"{out_dir}/pairs_meta.json", "w"))
            data.commit()
    return f"shard {shard}: {int(ok.sum())}/{len(pairs)} pairs -> {out_dir}"


@app.local_entrypoint()
def main(tag: str, limit: int = 0, nshards: int = 1):
    for r in run.starmap([(tag, limit, k, nshards) for k in range(nshards)]):
        print(r)
