"""Build the datasets for the Glimmer osteosarc dashboard (claude.ai artifact YTL434mENWCv3anqC5EE2J).
Every score comes from osteo-results/scripts/score_compare.score() on the same 24,634 masked targets.
Rerun after new runs land:  ~/osteo-classical/.venv/bin/python ~/osteo-dash-work/build.py
"""
import collections, glob, json, math, os, random, re, sys
import numpy as np
from scipy.stats import spearmanr

H = os.path.expanduser("~")
R, C, N = f"{H}/osteo-results", f"{H}/osteo-classical", f"{H}/osteo-nim"
OUT = f"{H}/osteo-dash-work/data"
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, f"{R}/scripts")
from score_compare import score, parse, num, LABELS, CAT, BIN

truth = [json.loads(l) for l in open(f"{R}/baseline_truth.jsonl")]
HELD = ["flow", "rna", "tcr", "xenium"]


def nrows(p):
    return sum(1 for _ in open(p)) if os.path.exists(p) else 0


def shards(d):
    """Concatenate a sharded eval dir into one file; return (path, prompts) or (None, 0)."""
    fs = sorted(glob.glob(f"{d}/*.jsonl")) if os.path.isdir(d) else ([d] if os.path.exists(d) else [])
    if not fs:
        return None, 0
    if len(fs) == 1:
        return fs[0], nrows(fs[0])
    dst = f"{OUT}/_tmp_{os.path.basename(d)}.jsonl"
    n = 0
    with open(dst, "w") as f:
        for p in fs:
            for l in open(p):
                f.write(l); n += 1
    return dst, n


# ---- runs on the full eval set -------------------------------------------------------------------
# (id, label, family, file, note)
RUNS = [
    ("glimmer_base", "Glimmer-30B zero-shot", "LLM zero-shot", f"{R}/baseline_zeroshot_merged.jsonl",
     "Muse-Glimmer-30B, untrained, greedy, with the entity's other observed values in the prompt"),
    ("glimmer_sid_full", "Glimmer LoRA sid_full", "LLM fine-tuned", f"{R}/finetuned_sid_full_eval.jsonl",
     "LoRA r32 on all 11,824 training rows; stopped at step 739 of 1,478 (lr mid-schedule)"),
]
for h in ["all"] + HELD:
    d = f"{R}/lomo/evals/lomo740_{h}" + ("" if h == "all" else "_full")
    p, n = shards(d)
    if h == "all" and n < 15900:
        p, n = shards(f"{R}/lomo/evals/lomo740h100_all")      # same all-data reference, re-run with H100 fallback
    if n < 15900:                                             # pulled straight from the Modal volume by this dashboard
        p, n = shards(f"{H}/osteo-dash-work/evals/" + os.path.basename(d))
    if p and n >= 15900:
        lab = "Glimmer LoRA all-data (740 steps)" if h == "all" else f"Glimmer LoRA minus {LABELS[h]}"
        RUNS.append((f"lomo740_{h}", lab, "LLM fine-tuned (hold-out)" if h != "all" else "LLM fine-tuned",
                     p, f"740-step schedule run to completion; trained on {'everything' if h == 'all' else 'everything except ' + LABELS[h]}"))
for m in ("1.3", "1.2"):
    p = f"{R}/spark/muse-spark-{m}-contributor.jsonl"
    n = nrows(p)
    if n:
        RUNS.append((f"spark_{m}", f"Muse Spark {m}", "LLM zero-shot", p,
                     f"Meta Model API, muse-spark-{m}-contributor, temperature 0, reasoning low; {n:,}/15,964 prompts"
                     + ("" if n >= 15964 else " (run still in progress: unanswered prompts count as unanswered)")))
picks = json.load(open(f"{C}/val_picks.json"))
CLASSICAL_NAMES = {"col_mean": "feature mean", "col_median": "feature median", "row_mean": "entity mean",
                   "additive": "additive entity+feature", "gbm_stack": "GBM stack", "best_by_val": "classical (validation pick)"}


def cname(m):
    if m in CLASSICAL_NAMES:
        return CLASSICAL_NAMES[m]
    k = re.match(r"knn_(entity|feature)_(\d+)", m)
    if k:
        return f"{k[1]}-kNN k={k[2]}"
    a = re.match(r"als_(\d+)_([\d.]+)", m)
    if a:
        return f"ALS rank {a[1]} λ{float(a[2]):g}"
    return m.replace("_", " ")


for p in sorted(glob.glob(f"{C}/preds/*.jsonl")):
    m = os.path.basename(p)[:-6]
    if m.startswith("card_"):
        continue                     # ET-SoC-1 card copy of knn_entity (CITE only); identical predictions
    RUNS.append((f"cl_{m}", cname(m), "Classical ML" if m != "best_by_val" else "Classical ML (pick)", p,
                 "Visible values only, asinh space; " + ("per-type method chosen on a 10% validation split" if m == "best_by_val" else "classical imputation")))

res, pts = {}, {}
for rid, _, _, f, _ in RUNS:
    res[rid], pts[rid] = score(f, truth)
    print("scored", rid, file=sys.stderr)

# never-observed share per type (classical RESULTS.md table)
never = {}
for line in open(f"{C}/RESULTS.md"):
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if len(cells) == 6:
        mm = re.match(r"([\d.]+)%", cells[-1])
        if mm:
            never[cells[0]] = float(mm[1]) / 100
NEVER_KEY = {"CITE-seq": "cite", "Xenium": "xenium", "Gene fusions": "fusion", "Bulk RNA TPM": "rna", "Olink NPX": "olink",
             "Flow cytometry": "flow", "Multiplex IF (acc.)": "mif", "DIA proteomics": "dia", "TCR clones": "tcr",
             "scRNA-seq": "scrna", "Somatic WGS": "somatic", "Metagenomics": "mpa", "HLA (exact)": "hla"}
never = {NEVER_KEY[k]: v for k, v in never.items() if k in NEVER_KEY}

r4 = lambda v: None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), 4)

# ---- leaderboard (long) ----------------------------------------------------------------------------
board = []
for rid, label, fam, f, note in RUNS:
    for lab in LABELS:
        r = res[rid].get(lab)
        if not r or (rid.startswith("cl_") and lab in CAT):
            continue
        board.append({"run_id": rid, "run": label, "family": fam, "key": lab, "data_type": LABELS[lab],
                      "metric": r["metric"], "targets": r["targets"], "primary": r4(r["primary"]),
                      "answered": r4(r["answered"]), "within_10pct": r4(r["within_10pct"]),
                      "median_abs_err_asinh": r4(r["median_abs_err_asinh"]),
                      "pick": picks.get(lab) if rid == "cl_best_by_val" else None})
json.dump(board, open(f"{OUT}/leaderboard.json", "w"))

runs_meta = [{"run_id": rid, "run": label, "family": fam, "note": note,
              "prompts": nrows(f) if not f.startswith(OUT) else None} for rid, label, fam, f, note in RUNS]
json.dump(runs_meta, open(f"{OUT}/runs.json", "w"))

# ---- per-type summary (wide): best of each family --------------------------------------------------
def prim(rid, lab):
    r = res.get(rid, {}).get(lab)
    if rid.startswith("cl_") and lab in CAT:
        return None                  # classical methods abstain on HLA alleles (numeric imputers only)
    return None if not r else r4(r["primary"])


TRAIN_PREFIX = {"scrna": "Single-cell RNA-seq", "cite": "CITE-seq", "rna": "Bulk tumour RNA-seq", "dia": "DIA mass-spec",
                "olink": "Olink plasma", "mif": "Multiplex immunofluorescence", "tcr": "T-cell receptor beta",
                "xenium": "Xenium", "somatic": "Somatic tumour/normal", "mpa": "Shotgun metagenomics",
                "flow": "Spectral flow cytometry", "hla": "Germline HLA", "fusion": "Gene fusions"}
train_rows = collections.Counter()
for l in open(f"{H}/biohack/01_sid_full_dataset/sid_full.jsonl"):
    u = json.loads(l)["prompt"][-1]["content"]
    train_rows[next((k for k, p in TRAIN_PREFIX.items() if u.startswith(p)), "other")] += 1


types = []
for lab in LABELS:
    cl = [(prim(r[0], lab), r[0]) for r in RUNS if r[2] == "Classical ML" and prim(r[0], lab) is not None]
    orc = max(cl) if cl else (None, None)
    row = {"key": lab, "data_type": LABELS[lab], "metric": res["glimmer_base"][lab]["metric"],
           "targets": res["glimmer_base"][lab]["targets"], "never_observed": never.get(lab), "train_rows": train_rows[lab],
           "glimmer_base": prim("glimmer_base", lab), "glimmer_sid_full": prim("glimmer_sid_full", lab),
           "lomo740_all": prim("lomo740_all", lab),
           "glimmer_ft": max([v for v in (prim("glimmer_sid_full", lab), prim("lomo740_all", lab)) if v is not None], default=None),
           "spark_1_3": prim("spark_1.3", lab), "spark_1_2": prim("spark_1.2", lab),
           "classical_pick": prim("cl_best_by_val", lab), "classical_pick_method": cname(picks.get(lab, "")) if picks.get(lab) else None,
           "classical_oracle": orc[0], "classical_oracle_method": cname(orc[1][3:]) if orc[1] else None}
    cand = {"Glimmer zero-shot": row["glimmer_base"], "Glimmer fine-tuned": row["glimmer_ft"],
            "Muse Spark 1.3": row["spark_1_3"], "Classical (val pick)": row["classical_pick"]}
    cand = {k: v for k, v in cand.items() if v is not None}
    row["winner"] = max(cand, key=cand.get) if cand else None
    types.append(row)
json.dump(types, open(f"{OUT}/types.json", "w"))

# legacy 'scores' dataset (base vs sid_full) keeps the original section working
scores = []
for lab in LABELS:
    b, f = res["glimmer_base"][lab], res["glimmer_sid_full"][lab]
    scores.append({"data_type": LABELS[lab], "key": lab, "metric": b["metric"], "targets": b["targets"],
                   "base_primary": r4(b["primary"]), "base_answered": r4(b["answered"]), "base_within_10pct": r4(b["within_10pct"]),
                   "base_median_abs_err_asinh": r4(b["median_abs_err_asinh"]),
                   "ft_primary": r4(f["primary"]), "ft_answered": r4(f["answered"]), "ft_within_10pct": r4(f["within_10pct"]),
                   "ft_median_abs_err_asinh": r4(f["median_abs_err_asinh"]),
                   "delta": None if b["primary"] is None or f["primary"] is None else r4(f["primary"] - b["primary"])})
json.dump(scores, open(f"{OUT}/scores.json", "w"))

# ---- points for the scatter (several runs) ---------------------------------------------------------
random.seed(0)
SC = {"glimmer_base": "Glimmer zero-shot", "glimmer_sid_full": "Glimmer fine-tuned", "spark_1.3": "Muse Spark 1.3",
      "cl_best_by_val": "Classical (val pick)"}
points = []
for rid, nm in SC.items():
    if rid not in pts:
        continue
    for lab, p in pts[rid].items():
        for tv, pv in random.sample(p, min(300, len(p))):
            points.append({"model": nm, "data_type": LABELS[lab], "truth": tv, "pred": pv})
json.dump(points, open(f"{OUT}/points.json", "w"))

# ---- context-free baseline (NVIDIA-hosted Glimmer, own-type values stripped) -----------------------
noctx_pred = {}
nerr = collections.Counter(); nall = collections.Counter()
for l in open(f"{N}/out_noctx.jsonl"):
    r = json.loads(l); nall[r["holdout"]] += 1
    if "error" in r:
        nerr[r["holdout"]] += 1; continue
    noctx_pred[r["prompt_id"]] = parse(r.get("text") or r.get("reasoning") or "") or {}
want = collections.Counter(json.loads(l)["holdout"] for l in open(f"{N}/prompts_noctx.jsonl"))


def score_dict(pred, lab):
    """Primary metric of a {prompt_id: {i: value}} prediction dict on one type, plus extras."""
    ts = [t for t in truth if t["mod"] == lab]
    xs, ys, ans = [], [], 0
    for t in ts:
        p = pred.get(t["prompt_id"], {}).get(str(t["i"]))
        if p is None:
            continue
        ans += 1
        a, b = num(p), num(t["truth"])
        if a is not None and b is not None:
            xs.append(a); ys.append(b)
    rho = spearmanr(xs, ys).statistic if len(set(xs)) > 1 and len(xs) > 2 else None
    w10 = sum(abs(a - b) <= .1 * abs(b) + 1e-9 for a, b in zip(xs, ys)) / max(len(xs), 1)
    mae = float(np.median([abs(math.asinh(a) - math.asinh(b)) for a, b in zip(xs, ys)])) if xs else None
    return {"rho": r4(rho), "answered": r4(ans / len(ts)), "within_10pct": r4(w10), "mae_asinh": r4(mae),
            "distinct_preds": len(set(xs)), "pred_zero": r4(sum(a == 0 for a in xs) / max(len(xs), 1)),
            "true_zero": r4(sum(b == 0 for b in ys) / max(len(ys), 1))}


# ---- transfer table: how well is a type predicted when it is missing from training / from context ---
cross = json.load(open(f"{C}/cross_results.json"))
cg = {g["type"]: g for g in cross["gene_link"]}
transfer = []
for h in HELD:
    nc = score_dict(noctx_pred, h)
    base, ft, ho, allr = prim("glimmer_base", h), prim("glimmer_sid_full", h), prim(f"lomo740_{h}", h), prim("lomo740_all", h)
    ref, ref_src = (allr, "lomo740_all") if allr is not None else (ft, "sid_full")
    tr = None if None in (ho, ref, base) or abs(ref - base) < 0.05 else r4((ho - base) / (ref - base))   # tiny gains make the ratio meaningless
    transfer.append({
        "key": h, "data_type": LABELS[h], "targets": res["glimmer_base"][h]["targets"],
        "noctx_prompts_done": nall[h] - nerr[h], "noctx_prompts_total": want[h],
        "noctx_rho": nc["rho"], "noctx_answered": nc["answered"], "noctx_within_10pct": nc["within_10pct"],
        "noctx_distinct_preds": nc["distinct_preds"], "noctx_pred_zero": nc["pred_zero"], "true_zero": nc["true_zero"],
        "base_ctx": base, "sid_full": ft, "holdout_adapter": ho, "all_ref": allr, "ref_used": ref_src,
        "transfer_ratio": tr,
        "classical_pick": prim("cl_best_by_val", h),
        "classical_within": r4(cg[h]["within"]) if h in cg else None,
        "classical_within_cross": r4(cg[h]["within+cross"]) if h in cg else None,
        "spark_1_3": prim("spark_1.3", h)})
json.dump(transfer, open(f"{OUT}/transfer.json", "w"))

# ---- off-target effect of each hold-out, paired bootstrap vs the reference --------------------------
def load_preds(f):
    d = {}
    for l in open(f):
        r = json.loads(l); d[r["prompt_id"]] = parse(r.get("text") or "") or {}
    return d


def metric_arr(lab, p, t):
    if lab in CAT:
        return float(np.mean([str(a).strip() == str(b).strip() for a, b in zip(p, t)])) if len(p) else None
    xy = [(num(a), num(b)) for a, b in zip(p, t)]
    xy = [(a, b) for a, b in xy if a is not None and b is not None]
    if lab in BIN:
        return float(np.mean([round(a) == round(b) for a, b in xy])) if xy else None
    if len(xy) < 3:
        return None
    a, b = np.array(xy).T
    s = spearmanr(a, b).statistic
    return None if s is None or np.isnan(s) else float(s)


runf = {r[0]: r[3] for r in RUNS}
ref_id = "lomo740_all" if "lomo740_all" in runf else "glimmer_sid_full"
refp, basep = load_preds(runf[ref_id]), load_preds(runf["glimmer_base"])
bytype = collections.defaultdict(list)
for t in truth:
    bytype[t["mod"]].append(t)
off = []
for h in HELD:
    if f"lomo740_{h}" not in runf:
        continue
    hp = load_preds(runf[f"lomo740_{h}"])
    for lab in LABELS:
        ts = [t for t in bytype[lab] if all(str(t["i"]) in d.get(t["prompt_id"], {}) for d in (hp, refp))]
        if len(ts) < 3:
            continue
        pids = sorted({t["prompt_id"] for t in ts}); idx = {p: i for i, p in enumerate(pids)}
        g = collections.defaultdict(list)
        for t in ts:
            g[t["prompt_id"]].append(t)
        def m(pred, ids):
            items = [t for i in ids for t in g[i]]
            return metric_arr(lab, [pred[t["prompt_id"]][str(t["i"])] for t in items], [t["truth"] for t in items])
        mh, mr = m(hp, pids), m(refp, pids)
        rng = random.Random(0); boots = []
        for _ in range(300 if len(pids) > 1 else 0):
            s = [rng.choice(pids) for _ in pids]
            a, b = m(hp, s), m(refp, s)
            if a is not None and b is not None:
                boots.append(a - b)
        boots.sort()
        ci = (boots[int(.025 * len(boots))], boots[int(.975 * len(boots)) - 1]) if len(boots) > 20 else (None, None)
        off.append({"holdout": h, "holdout_label": LABELS[h], "key": lab, "data_type": LABELS[lab], "targets": len(ts),
                    "prompts": len(pids), "ref_run": ref_id, "base": prim("glimmer_base", lab), "ref": r4(mr),
                    "holdout_score": r4(mh), "delta": None if mh is None or mr is None else r4(mh - mr),
                    "ci_lo": r4(ci[0]), "ci_hi": r4(ci[1]), "is_holdout": lab == h,
                    "significant": ci[0] is not None and (ci[0] > 0 or ci[1] < 0)})
    print("offtarget", h, file=sys.stderr)
json.dump(off, open(f"{OUT}/offtarget.json", "w"))

# ---- cross-type classical results -------------------------------------------------------------------
cgene = []
for g in cross["gene_link"]:
    for src, v in g["source_alone"].items():
        cgene.append({"target": LABELS[g["type"]], "target_key": g["type"], "source": LABELS[src], "source_key": src,
                      "pair": f"{src}->{g['type']}", "n": v["n"], "source_alone_rho": r4(v["rho"]),
                      "within_feature_mean_rho": r4(v["within_col_mean_same_targets"]),
                      "drop_one_delta": r4(g["drop_one"].get(src))})
json.dump(cgene, open(f"{OUT}/cross_gene.json", "w"))
csum = [{"key": g["type"], "data_type": LABELS[g["type"]], "targets": g["targets"], "within": r4(g["within"]),
         "within_cross": r4(g["within+cross"]), "n_never_observed": g["n_never_observed"],
         "never_within": r4(g["within_never_observed"]), "never_cross": r4(g["cross_never_observed"]),
         "n_gene_covered": g["n_gene_covered"], "covered_within": r4(g["within_gene_covered"]),
         "covered_cross": r4(g["cross_gene_covered"])} for g in cross["gene_link"]]
json.dump(csum, open(f"{OUT}/cross_summary.json", "w"))
dl = [{"pair": f"{d['source']}->{d['target']}", "source": LABELS[d["source"]], "target": LABELS[d["target"]],
       "matched_samples": d["matched_samples"], "targets": d["targets"], "cross_knn_rho": r4(d["cross_knn"]),
       "mean_rho": r4(d["mean_of_same_samples"]), "gain": r4(d["cross_knn"] - d["mean_of_same_samples"]),
       "mae_cross": r4(d["mae_asinh_cross"]), "mae_mean": r4(d["mae_asinh_mean"])} for d in cross["date_link"]]
json.dump(dl, open(f"{OUT}/date_link.json", "w"))

# ---- classical hold-out settings (osteo-classical/scripts/holdout.py) ---------------------------------------
MNAME = {"zero": "predict 0", "gene_score": "other types' gene level", "transfer_gbm": "GBM trained on other types",
         "feat_mean": "feature mean, other samples", "feat_median": "feature median, other samples",
         "carry_forward": "carry forward previous draw", "nearest_date": "nearest draw in time",
         "gbm_link": "GBM: other samples + links", "entity_mean": "sample's own mean", "entity_median": "sample's own median"}
SNAME = {"S1": "S1 type absent", "S2": "S2 new sample", "S3": "S3 type unseen in training"}
ho = json.load(open(f"{C}/holdout_results.json"))["results"]
chold = [{"id": f"{r['setting']}__{r['method']}__{r['key']}", "setting": SNAME[r["setting"]], "setting_key": r["setting"],
          "method": MNAME.get(r["method"], r["method"]), "key": r["key"], "data_type": r["data_type"], "targets": r["targets"],
          "primary": r4(r["primary"]), "within_10pct": r4(r["within_10pct"]), "rho_seen": r4(r.get("rho_seen")),
          "rho_never_seen": r4(r.get("rho_never_seen")), "rho_gene_covered": r4(r.get("rho_gene_covered")),
          "n_gene_covered": r.get("n_gene_covered")} for r in ho]
json.dump(chold, open(f"{OUT}/classical_holdout.json", "w"))
best = {}
for r in chold:
    k = (r["key"], r["setting_key"])
    if r["primary"] is not None and (k not in best or r["primary"] > best[k]["primary"]):
        best[k] = r
csum2 = []
for lab in LABELS:
    if lab == "hla":
        continue
    row = {"key": lab, "data_type": LABELS[lab], "all_visible": prim("cl_best_by_val", lab)}
    for s_ in ("S1", "S2", "S3"):
        b_ = best.get((lab, s_))
        row[s_] = b_["primary"] if b_ else None
        row[s_ + "_method"] = b_["method"] if b_ else None
    csum2.append(row)
json.dump(csum2, open(f"{OUT}/classical_holdout_best.json", "w"))
for t in transfer:
    for s_ in ("S1", "S2", "S3"):
        b_ = best.get((t["key"], s_))
        t[f"classical_{s_}"] = b_["primary"] if b_ else None
json.dump(transfer, open(f"{OUT}/transfer.json", "w"))

for f in glob.glob(f"{OUT}/_tmp_*.jsonl"):
    os.remove(f)
print(json.dumps({"runs": [r[0] for r in RUNS], "ref_for_offtarget": ref_id}, indent=0))
