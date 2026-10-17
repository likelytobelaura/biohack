import json
import modal

app = modal.App("osteo-glimmer-compare")
image = (modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
         .entrypoint([]).pip_install("vllm==0.31.0"))
hf = modal.Volume.from_name("hf-cache")
data = modal.Volume.from_name("osteo-data")

BASE = "/hf/Muse-Glimmer-30B"
ADAPTER = "/data/adapters/sid_full"
PROMPTS = "/data/prompts/baseline.jsonl"
OUT = "/data/outputs/compare"
NSHARDS = 16


@app.function(image=image, gpu="H200", volumes={"/hf": hf, "/data": data}, timeout=3 * 3600)
def run_shard(shard: int):
    import os
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    rows = [json.loads(l) for l in open(PROMPTS)]
    rows = rows[shard::NSHARDS]
    llm = LLM(model=BASE, enable_lora=True, max_lora_rank=32, max_model_len=12288,
              gpu_memory_utilization=0.92, trust_remote_code=True)
    lora = LoRARequest("sid_full", 1, ADAPTER)
    os.makedirs(OUT, exist_ok=True)

    for tag, req in (("ft", lora),):
        sp = SamplingParams(temperature=0.0, max_tokens=2048)
        outs = llm.chat([r["messages"] for r in rows], sp, lora_request=req)
        res = [{"prompt_id": r["prompt_id"], "text": o.outputs[0].text, "finish": o.outputs[0].finish_reason,
                "n_prompt_tokens": len(o.prompt_token_ids), "n_output_tokens": len(o.outputs[0].token_ids)}
               for r, o in zip(rows, outs)]
        # retry anything cut off by the length cap with a larger budget, as the baseline did
        cut = [i for i, x in enumerate(res) if x["finish"] == "length"]
        if cut:
            sp2 = SamplingParams(temperature=0.0, max_tokens=8192)
            outs2 = llm.chat([rows[i]["messages"] for i in cut], sp2, lora_request=req)
            for i, o in zip(cut, outs2):
                res[i].update(text=o.outputs[0].text, finish=o.outputs[0].finish_reason,
                              n_output_tokens=len(o.outputs[0].token_ids), retried=True)
        with open(f"{OUT}/{tag}_shard{shard}.jsonl", "w") as f:
            for x in res:
                f.write(json.dumps(x) + "\n")
        data.commit()
        print(f"shard {shard} {tag}: wrote {len(res)} rows, {len(cut)} retried")
    return shard


@app.local_entrypoint()
def main():
    for s in run_shard.map(range(NSHARDS)):
        print("done shard", s)
