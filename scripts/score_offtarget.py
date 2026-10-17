"""How much does holding out one data type hurt the other types?
For each hold-out adapter: per-type score minus the all-data reference adapter's score on the same prompts,
with a paired 95% bootstrap CI over prompts. Untrained base shown for scale.

python3 score_offtarget.py fetch                      # pull eval shards from the osteo-data volume
python3 score_offtarget.py score [--ref DIR] [--holdouts flow,rna,tcr,xenium]
"""
import collections, glob, json, os, random, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_compare import LABELS, CAT, BIN, parse, num, spear

R = "/Users/laurabocek/osteo-results"
EV = f"{R}/lomo/evals"
MODAL = os.path.expanduser("~/sutro-problems/.venv/bin/modal")
RUNS = {"all": "lomo740_all", "flow": "lomo740_flow_full", "rna": "lomo740_rna_full",
        "tcr": "lomo740_tcr_full", "xenium": "lomo740_xenium_full"}


def fetch():
    for d in RUNS.values():
        os.makedirs(f"{EV}/{d}", exist_ok=True)
        ls = subprocess.run([MODAL, "volume", "ls", "osteo-data", f"outputs/lomo/{d}"], capture_output=True, text=True).stdout.split()
        for p in ls:
            if p.endswith(".jsonl") and not os.path.exists(f"{EV}/{d}/{os.path.basename(p)}"):
                subprocess.run([MODAL, "volume", "get", "osteo-data", p, f"{EV}/{d}/"], check=True, capture_output=True)
        print(d, len(glob.glob(f"{EV}/{d}/*.jsonl")), "shards")


def load_preds(path):
    files = sorted(glob.glob(f"{path}/*.jsonl")) if os.path.isdir(path) else [path]
    pred = {}
    for f in files:
        for l in open(f):
            r = json.loads(l); pred[r["prompt_id"]] = parse(r.get("text") or "") or {}
    return pred


def metric(lab, pairs):
    """pairs: (pred, truth) for one type -> the primary metric score_compare uses."""
    if lab in CAT:
        return sum(str(p).strip() == str(t).strip() for p, t in pairs) / len(pairs) if pairs else None
    xy = [(num(p), num(t)) for p, t in pairs]
    xy = [(a, b) for a, b in xy if a is not None and b is not None]
    if lab in BIN:
        return sum(round(a) == round(b) for a, b in xy) / len(xy) if xy else None
    return spear([a for a, _ in xy], [b for _, b in xy])


def by_prompt(pred, ts):
    g = collections.defaultdict(list)
    for t in ts:
        p = pred.get(t["prompt_id"], {}).get(str(t["i"]))
        if p is not None:
            g[t["prompt_id"]].append((p, t["truth"]))
    return g


def score(ref_dir, holdouts, nboot=1000):
    truth = collections.defaultdict(list)
    for l in open(f"{R}/baseline_truth.jsonl"):
        t = json.loads(l); truth[t["mod"]].append(t)
    ref, base = load_preds(ref_dir), load_preds(f"{R}/baseline_zeroshot_merged.jsonl")
    out = []
    for h in holdouts:
        ho = load_preds(f"{EV}/{RUNS[h]}")
        if not ho:
            print(f"{h}: no eval shards yet"); continue
        print(f"\n== hold out {LABELS[h]} ({len(ho)} prompts scored) ==")
        print(f"{'type':20s} {'n':>5s} {'base':>6s} {'ref':>6s} {'holdout':>7s} {'Δ vs ref':>8s}  95% CI")
        for lab in LABELS:
            ts = [t for t in truth[lab] if t["prompt_id"] in ho and t["prompt_id"] in ref]
            if not ts:
                continue
            gh, gr, gb = by_prompt(ho, ts), by_prompt(ref, ts), by_prompt(base, ts)
            pids = sorted({t["prompt_id"] for t in ts})
            m = lambda g, ids: metric(lab, [x for i in ids for x in g.get(i, [])])
            mh, mr, mb = m(gh, pids), m(gr, pids), m(gb, pids)
            d = None if mh is None or mr is None else mh - mr
            rng, boots = random.Random(0), []
            for _ in range(nboot if len(pids) > 1 else 0):
                s = [rng.choice(pids) for _ in pids]
                a, b = m(gh, s), m(gr, s)
                if a is not None and b is not None:
                    boots.append(a - b)
            boots.sort()
            ci = (boots[int(.025 * len(boots))], boots[int(.975 * len(boots)) - 1]) if len(boots) > 20 else (None, None)
            f = lambda v: "   n/a" if v is None else f"{v:6.3f}"
            flag = "  <- held out" if lab == h else ("  *" if ci[0] is not None and (ci[0] > 0 or ci[1] < 0) else "")
            print(f"{LABELS[lab]:20s} {len(pids):5d} {f(mb)} {f(mr)} {f(mh):>7s} {f(d):>8s}  "
                  f"[{f(ci[0])}, {f(ci[1])}]{flag}")
            out.append({"holdout": h, "type": lab, "data_type": LABELS[lab], "prompts": len(pids), "base": mb,
                        "ref": mr, "holdout_score": mh, "delta": d, "ci_lo": ci[0], "ci_hi": ci[1], "is_holdout": lab == h})
    json.dump(out, open(f"{R}/lomo/offtarget_scores.json", "w"), indent=1)
    print(f"\n* = 95% CI excludes 0. Saved {R}/lomo/offtarget_scores.json")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["fetch"]:
        fetch()
    else:
        ref = args[args.index("--ref") + 1] if "--ref" in args else f"{EV}/{RUNS['all']}"
        hs = args[args.index("--holdouts") + 1].split(",") if "--holdouts" in args else ["flow", "rna", "tcr", "xenium"]
        score(ref, hs)
