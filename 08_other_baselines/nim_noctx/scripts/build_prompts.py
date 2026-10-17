"""Context-free baseline prompts for the LOMO hold-out types: drop the masked labels (never in prompts)
and every observed value of the held-out type (the 'Known values' and 'elsewhere in the dataset' blocks)."""
import json, re, sys

PREFIX = {"flow": "Spectral flow cytometry", "rna": "Bulk tumour RNA-seq",
          "tcr": "T-cell receptor beta clones", "xenium": "Xenium in-situ transcriptomics"}
src, dst = sys.argv[1], sys.argv[2]
n = {k: 0 for k in PREFIX}
with open(dst, "w") as f:
    for l in open(src):
        r = json.loads(l)
        u = r["messages"][-1]["content"]
        key = next((k for k, p in PREFIX.items() if u.startswith(p)), None)
        if key is None:
            continue
        head, rest = u.split("\n\nKnown values for this entity:\n", 1)
        _, tail = rest.split("\n\nMissing values to predict:\n", 1)
        new = head + "\n\nNo observed values of this data type are available.\n\nMissing values to predict:\n" + tail
        assert not re.search(r" = |=\d", new.split("Missing values to predict:")[0].split("\nEntity: ")[1]), r["prompt_id"]
        msgs = [dict(m) for m in r["messages"]]
        msgs[-1]["content"] = new
        f.write(json.dumps({"prompt_id": r["prompt_id"], "holdout": key, "messages": msgs}) + "\n")
        n[key] += 1
print(n, sum(n.values()))
