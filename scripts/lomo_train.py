"""Leave-one-data-type-out LoRA SFT of Muse-Glimmer-30B on the osteosarc training set.

modal run --detach lomo_train.py --holdout sweep          # all 14 runs: train then eval each
modal run --detach lomo_train.py --holdout rna            # one run (train + eval)
modal run --detach lomo_train.py --holdout all --max-steps 20 --tag smoke --train-only   # smoke test
Holdout keys: all, scrna, cite, rna, dia, olink, mif, tcr, xenium, somatic, mpa, flow, hla, fusion
"""
import modal

app = modal.App("osteo-lomo")
image = (modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
         .entrypoint([])
         .pip_install("torch==2.14.0", "transformers==5.18.0", "trl==1.14.1", "peft==0.21.1",
                      "accelerate==1.15.0", "datasets==5.0.1"))
vllm_image = (modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
              .entrypoint([]).pip_install("vllm==0.31.0"))
hf = modal.Volume.from_name("hf-cache")
data = modal.Volume.from_name("osteo-data")

# first line of each prompt's user message identifies its data type
PREFIX = {
    "scrna": "Single-cell RNA-seq (10x)", "cite": "CITE-seq antibody tags", "rna": "Bulk tumour RNA-seq",
    "dia": "DIA mass-spec proteomics", "olink": "Olink plasma proteomics", "mif": "Multiplex immunofluorescence",
    "tcr": "T-cell receptor beta clones", "xenium": "Xenium in-situ transcriptomics", "somatic": "Somatic tumour/normal WGS",
    "mpa": "Shotgun metagenomics", "flow": "Spectral flow cytometry", "hla": "Germline HLA typing",
    "fusion": "Gene fusions from long-read",
}

TRAIN = r'''
import json, os, sys, torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForImageTextToText, AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

holdout, max_steps, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
prefixes = json.loads(sys.argv[4])
rows = [json.loads(l) for l in open("/data/train/sid_full.jsonl")]
if holdout != "all":
    rows = [r for r in rows if not r["prompt"][-1]["content"].startswith(prefixes[holdout])]
if int(os.environ.get("RANK", 0)) == 0:
    print(f"holdout={holdout} train rows={len(rows)} max_steps={max_steps}", flush=True)

try:
    model = AutoModelForImageTextToText.from_pretrained("/hf/Muse-Glimmer-30B", dtype=torch.bfloat16)
except (ValueError, KeyError):
    model = AutoModelForCausalLM.from_pretrained("/hf/Muse-Glimmer-30B", dtype=torch.bfloat16)

cfg = SFTConfig(
    output_dir=out, max_steps=max_steps, per_device_train_batch_size=1, gradient_accumulation_steps=1,
    learning_rate=1e-4, lr_scheduler_type="cosine", warmup_steps=0.03, weight_decay=0.0,
    optim="adamw_torch_fused", bf16=True, gradient_checkpointing=True, seed=42,
    max_length=6144, packing=False, completion_only_loss=True,
    logging_steps=5, save_strategy="no", report_to="none",
)
lora = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                  target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
tok = AutoTokenizer.from_pretrained("/hf/Muse-Glimmer-30B")
trainer = SFTTrainer(model=model, args=cfg, train_dataset=Dataset.from_list(rows), peft_config=lora, processing_class=tok)
trainer.train()
if trainer.is_world_process_zero():
    trainer.save_model(out)
    json.dump(trainer.state.log_history, open(f"{out}/train_log.json", "w"))
    print(f"saved adapter to {out}", flush=True)
'''


@app.function(image=image, gpu="H200:8", volumes={"/hf": hf, "/data": data}, timeout=4 * 3600)
def train(holdout: str, max_steps: int, tag: str):
    import json, subprocess
    out = f"/data/adapters/lomo/{tag}_{holdout}"
    open("/tmp/train.py", "w").write(TRAIN)
    subprocess.run(["torchrun", "--nproc_per_node=8", "/tmp/train.py", holdout, str(max_steps), out,
                    json.dumps(PREFIX)], check=True)
    data.commit()
    return out


EVAL_SHARD = 1000


@app.function(image=vllm_image, gpu="H200", volumes={"/hf": hf, "/data": data}, timeout=2 * 3600)
def eval_shard(adapter: str, holdout: str, shard: int, nshards: int, out_dir: str):
    import json, os
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    rows = [json.loads(l) for l in open("/data/prompts/baseline.jsonl")]
    if holdout != "all":
        rows = [r for r in rows if r["messages"][-1]["content"].startswith(PREFIX[holdout])]
    rows = rows[shard::nshards]
    llm = LLM(model="/hf/Muse-Glimmer-30B", enable_lora=True, max_lora_rank=32, max_model_len=12288,
              gpu_memory_utilization=0.92, trust_remote_code=True)
    lora = LoRARequest("lomo", 1, adapter)
    outs = llm.chat([r["messages"] for r in rows], SamplingParams(temperature=0.0, max_tokens=2048), lora_request=lora)
    res = [{"prompt_id": r["prompt_id"], "text": o.outputs[0].text, "finish": o.outputs[0].finish_reason,
            "n_output_tokens": len(o.outputs[0].token_ids)} for r, o in zip(rows, outs)]
    cut = [i for i, x in enumerate(res) if x["finish"] == "length"]
    if cut:
        outs2 = llm.chat([rows[i]["messages"] for i in cut], SamplingParams(temperature=0.0, max_tokens=8192), lora_request=lora)
        for i, o in zip(cut, outs2):
            res[i].update(text=o.outputs[0].text, finish=o.outputs[0].finish_reason, n_output_tokens=len(o.outputs[0].token_ids), retried=True)
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/shard{shard}.jsonl", "w") as f:
        for x in res:
            f.write(json.dumps(x) + "\n")
    data.commit()
    return len(res)


@app.function(image=image, volumes={"/data": data}, timeout=12 * 3600)
def pipeline(holdout: str, max_steps: int, tag: str, skip_train: bool = False, adapter: str = ""):
    import json
    adapter = adapter or f"/data/adapters/lomo/{tag}_{holdout}"
    if not skip_train:
        adapter = train.remote(holdout, max_steps, tag)
    rows = [json.loads(l) for l in open("/data/prompts/baseline.jsonl")]
    n = len(rows) if holdout == "all" else sum(r["messages"][-1]["content"].startswith(PREFIX[holdout]) for r in rows)
    nshards = max(1, -(-n // EVAL_SHARD))
    out_dir = f"/data/outputs/lomo/{tag}_{holdout}"
    done = sum(eval_shard.starmap([(adapter, holdout, k, nshards, out_dir) for k in range(nshards)]))
    print(f"{holdout}: trained {adapter}, evaluated {done} prompts in {nshards} shards -> {out_dir}", flush=True)
    return holdout, done


@app.local_entrypoint()
def main(holdout: str = "all", max_steps: int = 740, tag: str = "lomo740", skip_train: bool = False, train_only: bool = False, adapter: str = ""):
    if train_only:
        assert holdout == "all" or holdout in PREFIX, holdout
        print(train.remote(holdout, max_steps, tag))
        return
    keys = ["all"] + list(PREFIX) if holdout == "sweep" else holdout.split(",")
    assert all(k == "all" or k in PREFIX for k in keys), keys
    for h, n in pipeline.starmap([(k, max_steps, tag, skip_train, adapter) for k in keys]):
        print("finished", h, n)
