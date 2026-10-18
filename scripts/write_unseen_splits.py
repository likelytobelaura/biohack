"""Turn reconstruct_seen_rows.py output into saved splits: rows each lomo740 run trained on vs never saw.

python3 write_unseen_splits.py <rank-json dir> run1 run2 ...   -> ~/osteo-results/lomo/splits/
"""
import json, os, sys, glob, collections, hashlib

src_dir, runs = sys.argv[1], sys.argv[2:]
R = os.path.expanduser('~/osteo-results/lomo')
OUT = f'{R}/splits'
os.makedirs(OUT, exist_ok=True)
lines = open(f'{R}/train_sid_full.jsonl').read().splitlines()
rows = [json.loads(l) for l in lines]
dtype = [r['prompt'][-1]['content'].split('\n')[0].split('.')[0] for r in rows]
summary = {'train_file': 'train/sid_full.jsonl (osteo-data volume) = 01_sid_full_dataset', 'train_rows': len(rows),
           'train_sha256': hashlib.sha256(open(f'{R}/train_sid_full.jsonl', 'rb').read()).hexdigest(), 'runs': {}}
unseen_sets = {}
for run in runs:
    parts = sorted(glob.glob(f'{src_dir}/{run}_rank*.json'))
    assert len(parts) == 8, (run, len(parts))
    meta = [json.load(open(p)) for p in parts]
    seen = [i for m in meta for i in m['row_ids']]
    assert len(seen) == 8 * 740 and len(set(seen)) == len(seen), (run, len(seen), len(set(seen)))
    eligible = {i for i in range(len(rows))} if run == 'all' else None
    if eligible is None:  # rows the run could have drawn = its filtered training set (n_rows of it)
        n = meta[0]['n_rows']
        held = {'flow': 'Spectral flow cytometry', 'rna': 'Bulk tumour RNA-seq', 'tcr': 'T-cell receptor beta clones',
                'xenium': 'Xenium in-situ transcriptomics'}[run]
        eligible = {i for i, r in enumerate(rows) if not r['prompt'][-1]['content'].startswith(held)}
        assert len(eligible) == n, (run, len(eligible), n)
    seen_set = set(seen)
    unseen = sorted(i for i in range(len(rows)) if i not in seen_set)
    unseen_sets[run] = set(unseen)
    with open(f'{OUT}/lomo740_{run}_unseen_rows.jsonl', 'w') as f:
        for i in unseen:
            f.write(json.dumps({'row_id': i, **rows[i]}) + '\n')
    with open(f'{OUT}/lomo740_{run}_seen_rows.jsonl', 'w') as f:
        for i in sorted(seen_set):
            f.write(json.dumps({'row_id': i, **rows[i]}) + '\n')
    json.dump({'seen_row_ids_in_training_order_by_rank': {m['rank']: m['row_ids'] for m in meta},
               'unseen_row_ids': unseen}, open(f'{OUT}/lomo740_{run}_row_ids.json', 'w'))
    by_type = collections.Counter(dtype[i] for i in unseen)
    summary['runs'][f'lomo740_{run}'] = {'eligible_rows': len(eligible), 'seen_rows': len(seen_set), 'unseen_rows': len(unseen),
                                        'unseen_includes_heldout_type_rows': len(unseen) - (len(eligible) - len(seen_set)),
                                        'unseen_by_type': dict(by_type.most_common())}
    print(run, 'seen', len(seen_set), 'unseen', len(unseen))
common = set.intersection(*unseen_sets.values()) if unseen_sets else set()
with open(f'{OUT}/never_trained_by_any_lomo740.jsonl', 'w') as f:
    for i in sorted(common):
        f.write(json.dumps({'row_id': i, **rows[i]}) + '\n')
summary['never_trained_by_any_of'] = [f'lomo740_{r}' for r in runs]
summary['never_trained_by_any_rows'] = len(common)
summary['never_trained_by_any_by_type'] = dict(collections.Counter(dtype[i] for i in common).most_common())
json.dump(summary, open(f'{OUT}/splits_summary.json', 'w'), indent=1)
print('never trained by any:', len(common))
