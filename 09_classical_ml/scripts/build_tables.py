"""Observed-value tables per data type: everything Glimmer could see (training rows incl. their answers,
plus the visible values in every eval prompt). Asserts no eval target is in them."""
import collections, json, pickle, re, sys
sys.path.insert(0, "/Users/laurabocek/osteo-results/scripts")
from common import parse_user

R = "/Users/laurabocek/osteo-results"
TRAIN = "/Users/laurabocek/.claude/jobs/6c5c300f/tmp/osteo/sid_full.jsonl"
KEY = {"Single-cell RNA-seq": "scrna", "CITE-seq": "cite", "Bulk tumour RNA-seq": "rna", "DIA mass-spec": "dia",
       "Olink": "olink", "Multiplex immunofluorescence": "mif", "T-cell receptor": "tcr", "Xenium": "xenium",
       "Somatic": "somatic", "Shotgun metagenomics": "mpa", "Spectral flow": "flow", "Germline HLA": "hla", "Gene fusions": "fusion"}
key = lambda u: next(v for k, v in KEY.items() if u.startswith(k))

obs = collections.defaultdict(dict)          # mod -> {(entity, feature): value}
def add(mod, ent, feat, v, src):
    d = obs[mod]
    if (ent, feat) in d and d[(ent, feat)] != v:
        conflicts[src] += 1
    d.setdefault((ent, feat), v)
conflicts = collections.Counter()
for l in open(TRAIN):
    r = json.loads(l); u = r["prompt"][-1]["content"]; m = key(u)
    ent, known, cross, items = parse_user(u)
    for f, v in known.items(): add(m, ent, f, v, "train_known")
    for (e, f), v in cross.items(): add(m, e, f, v, "train_cross")
    ans = json.loads(r["completion"][0]["content"])
    for i, it in enumerate(items, 1):
        if str(i) in ans: add(m, ent, it, str(ans[str(i)]), "train_answer")
for l in open(f"{R}/eval_prompts.jsonl"):
    r = json.loads(l); u = r["messages"][-1]["content"]; m = key(u)
    ent, known, cross, _ = parse_user(u)
    for f, v in known.items(): add(m, ent, f, v, "eval_known")
    for (e, f), v in cross.items(): add(m, e, f, v, "eval_cross")
truth = [json.loads(l) for l in open(f"{R}/baseline_truth.jsonl")]
leak = [t for t in truth if (t["entity"], t["item"]) in obs[t["mod"]]]
print("conflicting duplicate values:", dict(conflicts))
print("eval targets present in observed pool:", len(leak))
for t in leak[:5]: print("  ", t, obs[t["mod"]][(t["entity"], t["item"])])
for m, d in sorted(obs.items()):
    ents = {e for e, _ in d}; feats = {f for _, f in d}
    nonnum = sum(1 for v in d.values() if re.fullmatch(r"-?[\d.]+(e-?\d+)?", str(v)) is None)
    print(f"{m:8s} entities={len(ents):5d} features={len(feats):6d} observed={len(d):7d} non-numeric={nonnum} targets={sum(t['mod']==m for t in truth)}")
pickle.dump((dict(obs), truth), open("/Users/laurabocek/osteo-classical/tables.pkl", "wb"))
