"""Reconstruct which training rows each lomo740 run saw, by rebuilding its exact data order on CPU.

torchrun --nproc_per_node=8 reconstruct_seen_rows.py <holdout> <out_dir>
Same data filter, SFTConfig seed/batching and library versions as lomo_train.py; a tiny random model stands
in for Muse-Glimmer-30B (the sampler only depends on dataset length, seed and world size, not the model).
Each rank writes the dataset indices of its first 740 batches; the first batches are checked against the real
dataloader output so the index -> row mapping is verified, not assumed.
"""
import json, os, sys
import torch
from datasets import Dataset
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from trl import SFTConfig, SFTTrainer

# keep everything on CPU (this Mac has MPS, and gloo barriers are not implemented for it)
torch.backends.mps.is_available = lambda: False
torch._C._get_accelerator = lambda: torch.device('cpu')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
holdout, out_dir = sys.argv[1], sys.argv[2]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lomo_train.py')).read()
PREFIX = eval(src[src.index('PREFIX = {') + len('PREFIX = '):src.index('\n}\n', src.index('PREFIX = {')) + 2])

rows = [json.loads(l) for l in open(os.path.expanduser('~/osteo-results/lomo/train_sid_full.jsonl'))]
for i, r in enumerate(rows):
    r['row_id'] = i
if holdout != 'all':
    rows = [r for r in rows if not r['prompt'][-1]['content'].startswith(PREFIX[holdout])]
row_ids = [r.pop('row_id') for r in rows]

tok = AutoTokenizer.from_pretrained(os.path.expanduser('~/osteo-results/lomo/adapters/lomo740_all'))
model = LlamaForCausalLM(LlamaConfig(vocab_size=len(tok), hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                     num_attention_heads=2, num_key_value_heads=2))
cfg = SFTConfig(
    output_dir=f'/tmp/recon_{holdout}', max_steps=740, per_device_train_batch_size=1, gradient_accumulation_steps=1,
    learning_rate=1e-4, lr_scheduler_type='cosine', warmup_steps=0.03, weight_decay=0.0, seed=42,
    max_length=6144, packing=False, completion_only_loss=True, logging_steps=5, save_strategy='no', report_to='none',
    use_cpu=True, ddp_backend='gloo',
)
trainer = SFTTrainer(model=model, args=cfg, train_dataset=Dataset.from_list(rows), processing_class=tok)
assert len(trainer.train_dataset) == len(rows), (len(trainer.train_dataset), len(rows))

dl = trainer.get_train_dataloader()
dl.set_epoch(0)
batches = [list(b) for _, b in zip(range(740), dl.batch_sampler)]
# verify: the real dataloader's first batches are exactly those dataset rows
dl.set_epoch(0)
for k, batch in zip(range(5), dl):
    want = trainer.train_dataset[batches[k][0]]['input_ids']
    got = batch['input_ids'][0][batch['attention_mask'][0].bool()].tolist() if 'attention_mask' in batch else batch['input_ids'][0].tolist()
    assert got[:len(want)] == want, f'rank mismatch at batch {k}'
rank = int(os.environ.get('RANK', 0))
os.makedirs(out_dir, exist_ok=True)
json.dump({'rank': rank, 'n_rows': len(rows), 'row_ids': [row_ids[i] for b in batches for i in b]},
          open(f'{out_dir}/{holdout}_rank{rank}.json', 'w'))
print(f'rank {rank}: {len(batches)} batches, {sum(map(len, batches))} rows, verified first 5 batches', flush=True)
