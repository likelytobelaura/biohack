"""Non-microbiome control prompts for mpa_neuron.py: answer positions of other numeric data types.

python build_control_examples.py <lomo740_all_unseen_rows.jsonl> <out.jsonl> [--per-type 12]
Same format as build_mpa_probe.py (answer marks only, y = log10(value + 1)), plus `dtype`.
Rows come from the lomo740_all never-trained split, so the fine-tune has not seen their answers.
"""
import argparse, json, math, random, re

TYPES = {"rna": "Bulk tumour RNA-seq", "dia": "DIA mass-spec proteomics", "xenium": "Xenium in-situ transcriptomics",
         "cite": "CITE-seq antibody tags", "flow": "Spectral flow cytometry"}
ap = argparse.ArgumentParser()
ap.add_argument("rows"); ap.add_argument("out"); ap.add_argument("--per-type", type=int, default=12)
args = ap.parse_args()
rows = [json.loads(l) for l in open(args.rows)]
rng = random.Random(0)
n = 0
with open(args.out, "w") as f:
    for t, prefix in TYPES.items():
        rs = [r for r in rows if r["prompt"][-1]["content"].startswith(prefix)]
        rng.shuffle(rs)
        for r in rs[:args.per_type]:
            comp = r["completion"][0]["content"]
            marks = [dict(kind="answer", where="comp", ch=m.end(), rule="before", y=math.log10(float(v) + 1))
                     for k, v in json.loads(comp).items() if (m := re.search(r'"%s": ' % k, comp))]
            user = r["prompt"][-1]["content"]
            ent = re.search(r"^Entity: (\S+)$", user, re.M)
            f.write(json.dumps(dict(id=f"row{r['row_id']}", src="control", dtype=t, messages=r["prompt"], comp=comp,
                                    entity=ent.group(1) if ent else t, marks=marks)) + "\n")
            n += len(marks)
print(f"{n} answer marks over {len(TYPES)} types -> {args.out}")
