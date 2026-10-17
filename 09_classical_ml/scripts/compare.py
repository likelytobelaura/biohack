"""Score every classical method and the Glimmer runs on the same 24,634 eval targets."""
import glob, json, os, sys
sys.path.insert(0, "/Users/laurabocek/osteo-results/scripts")
from score_compare import score, LABELS

R = "/Users/laurabocek/osteo-results"; OUT = "/Users/laurabocek/osteo-classical"
truth = [json.loads(l) for l in open(f"{R}/baseline_truth.jsonl")]
runs = {"glimmer_base": f"{R}/baseline_zeroshot_merged.jsonl", "glimmer_ft": f"{R}/finetuned_sid_full_eval.jsonl"}
noctx = "/Users/laurabocek/osteo-nim/out_noctx.jsonl.scoring"
if os.path.exists(noctx): runs["glimmer_noctx"] = noctx
for p in sorted(glob.glob(f"{OUT}/preds/*.jsonl")):
    runs[os.path.basename(p)[:-6]] = p
res = {m: score(f, truth)[0] for m, f in runs.items()}
vsel = json.load(open(f"{OUT}/val_picks.json"))
rows = []
for lab in LABELS:
    row = {"key": lab, "data_type": LABELS[lab], "metric": res["glimmer_base"][lab]["metric"],
           "targets": res["glimmer_base"][lab]["targets"], "val_pick": vsel.get(lab)}
    for m in runs:
        if lab in res[m] and not (m == "glimmer_noctx" and lab not in ("flow", "rna", "tcr", "xenium")):
            row[m] = None if res[m][lab]["primary"] is None else round(res[m][lab]["primary"], 4)
            row[m + "_answered"] = round(res[m][lab]["answered"], 3)
    rows.append(row)
json.dump(rows, open(f"{OUT}/compare_classical.json", "w"), indent=1)
cls = [m for m in runs if not m.startswith("glimmer")]
print(f"{'type':18s} {'metric':11s} {'base':>6s} {'FT':>6s} {'noctx':>6s} | {'best_by_val':>11s} ({'pick'}) | best-on-test (oracle, for reference)")
for r in rows:
    f = lambda v: "   -  " if v is None else f"{v:6.3f}"
    cand = [(r.get(m), m) for m in cls if m != "best_by_val" and r.get(m) is not None]
    bo = max(cand) if cand else (None, "")
    print(f"{r['data_type']:18s} {r['metric']:11s} {f(r.get('glimmer_base'))} {f(r.get('glimmer_ft'))} {f(r.get('glimmer_noctx'))} | "
          f"{f(r.get('best_by_val')):>11s} ({r['val_pick']}) | {f(bo[0])} {bo[1]}")
