"""Score the context-free NVIDIA baseline on the 4 LOMO hold-out types next to the with-context zero-shot and sid_full fine-tune."""
import json, sys
sys.path.insert(0, "/Users/laurabocek/osteo-results/scripts")
from score_compare import score, LABELS

R = "/Users/laurabocek/osteo-results"
KEYS = ["flow", "rna", "tcr", "xenium"]
pred = sys.argv[1]
truth = [json.loads(l) for l in open(f"{R}/baseline_truth.jsonl")]
truth = [t for t in truth if t["mod"] in KEYS]
# NIM puts the answer in content; fall back to the reasoning trace if content is empty
tmp = pred + ".scoring"
with open(tmp, "w") as f:
    for l in open(pred):
        r = json.loads(l)
        if "error" not in r:
            f.write(json.dumps({"prompt_id": r["prompt_id"], "text": r["text"] or r["reasoning"]}) + "\n")
runs = {"noctx_base": tmp, "ctx_base": f"{R}/baseline_zeroshot_merged.jsonl", "ft_sid_full": f"{R}/finetuned_sid_full_eval.jsonl"}
res = {m: score(f, truth)[0] for m, f in runs.items()}
rows = []
for k in KEYS:
    row = {"key": k, "data_type": LABELS[k], "targets": res["noctx_base"][k]["targets"]}
    for m in runs:
        for f in ("primary", "answered", "within_10pct", "median_abs_err_asinh"):
            v = res[m][k][f]; row[f"{m}_{f}"] = None if v is None else round(v, 4)
    rows.append(row)
    print(f"{row['data_type']:18s} n={row['targets']:5d}  Spearman: no-context={row['noctx_base_primary']}  "
          f"with-context={row['ctx_base_primary']}  sid_full FT={row['ft_sid_full_primary']}  "
          f"(answered {row['noctx_base_answered']}, within10% {row['noctx_base_within_10pct']})")
json.dump(rows, open(pred.replace(".jsonl", "_scores.json"), "w"), indent=1)
