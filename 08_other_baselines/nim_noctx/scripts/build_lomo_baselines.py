"""Full-eval-set baseline per LOMO hold-out: untrained base on every prompt, with the held-out type's prompts
answered context-free (NVIDIA run) and every other prompt taken from the existing with-context zero-shot run.
No eval prompt references another data type, so the other prompts are unchanged by removing the held-out type."""
import json, sys

R = "/Users/laurabocek/osteo-results"
N = "/Users/laurabocek/osteo-nim"
noctx = {}
src = sys.argv[1] if len(sys.argv) > 1 else f"{N}/out_noctx.jsonl"
for l in open(src):
    r = json.loads(l)
    if "error" not in r:
        noctx[r["prompt_id"]] = (r["holdout"], r["text"] or r.get("reasoning", ""))
want = {}
for l in open(f"{N}/prompts_noctx.jsonl"):
    r = json.loads(l); want[r["prompt_id"]] = r["holdout"]
base = [json.loads(l) for l in open(f"{R}/baseline_zeroshot_merged.jsonl")]
assert len(base) == 15964
for h in ("flow", "rna", "tcr", "xenium"):
    ids = [p for p, k in want.items() if k == h]
    have = sum(p in noctx for p in ids)
    with open(f"{N}/baseline_lomo_{h}.jsonl", "w") as f:
        for r in base:
            if want.get(r["prompt_id"]) == h:
                if r["prompt_id"] not in noctx:
                    continue
                r = {"prompt_id": r["prompt_id"], "text": noctx[r["prompt_id"]][1], "source": "nvidia_noctx"}
            else:
                r = {"prompt_id": r["prompt_id"], "text": r["text"], "source": "modal_zeroshot"}
            f.write(json.dumps(r) + "\n")
    print(f"{h}: {have}/{len(ids)} held-out prompts answered context-free -> baseline_lomo_{h}.jsonl"
          + ("" if have == len(ids) else "  (INCOMPLETE)"))
