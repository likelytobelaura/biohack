"""Context-free LOMO baseline on Modal: untrained Muse-Glimmer-30B on the 826 stripped hold-out prompts,
same vLLM image and decoding as the existing zero-shot baseline. Writes ~/osteo-nim/out_noctx_modal.jsonl.

cd ~/osteo-nim && ~/sutro-problems/.venv/bin/modal run scripts/modal_noctx.py
"""
import json, os
import modal

app = modal.App("osteo-noctx-baseline")
image = (modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
         .entrypoint([]).pip_install("vllm==0.31.0"))
hf = modal.Volume.from_name("hf-cache")
N = os.path.expanduser("~/osteo-nim")


@app.function(image=image, gpu="H200", volumes={"/hf": hf}, timeout=2 * 3600)
def run(rows):
    from vllm import LLM, SamplingParams
    llm = LLM(model="/hf/Muse-Glimmer-30B", max_model_len=12288, gpu_memory_utilization=0.92, trust_remote_code=True)
    outs = llm.chat([r["messages"] for r in rows], SamplingParams(temperature=0.0, max_tokens=2048))
    res = [{"prompt_id": r["prompt_id"], "holdout": r["holdout"], "text": o.outputs[0].text,
            "finish": o.outputs[0].finish_reason, "n_output_tokens": len(o.outputs[0].token_ids)}
           for r, o in zip(rows, outs)]
    cut = [i for i, x in enumerate(res) if x["finish"] == "length"]
    if cut:
        outs2 = llm.chat([rows[i]["messages"] for i in cut], SamplingParams(temperature=0.0, max_tokens=8192))
        for i, o in zip(cut, outs2):
            res[i].update(text=o.outputs[0].text, finish=o.outputs[0].finish_reason,
                          n_output_tokens=len(o.outputs[0].token_ids), retried=True)
    print(f"wrote {len(res)} rows, {len(cut)} retried")
    return res


@app.local_entrypoint()
def main():
    rows = [json.loads(l) for l in open(f"{N}/prompts_noctx.jsonl")]
    res = run.remote(rows)
    with open(f"{N}/out_noctx_modal.jsonl", "w") as f:
        for x in res:
            f.write(json.dumps(x) + "\n")
    print(f"saved {len(res)} rows to {N}/out_noctx_modal.jsonl")
