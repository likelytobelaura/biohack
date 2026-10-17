# Score Muse Spark runs (and the Glimmer zero-shot baseline) against ~/osteo-golden labels.
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from score_compare import score, LABELS
H = os.path.expanduser('~')
truth = [json.loads(l) for l in open(f'{H}/osteo-golden/labels.csv'.replace('labels.csv', 'golden.jsonl'))]
truth = [{'prompt_id': r['prompt_id'], 'i': int(i), 'mod': r['data_type'], 'truth': v}
         for r in truth for i, v in r['gold'].items()]
files = {'glimmer_zeroshot': f'{H}/osteo-results/baseline_zeroshot_merged.jsonl'}
files.update({a: f'{H}/osteo-results/spark/{a}.jsonl' for a in sys.argv[1:]})
res = {m: score(f, truth)[0] for m, f in files.items()}
rows = []
for lab in LABELS:
    row = {'key': lab, 'data_type': LABELS[lab], 'metric': res['glimmer_zeroshot'][lab]['metric'],
           'targets': res['glimmer_zeroshot'][lab]['targets']}
    for m in files:
        for k in ('primary', 'answered', 'within_10pct'):
            v = res[m][lab][k]; row[f'{m}_{k}'] = None if v is None else round(v, 4)
    rows.append(row)
json.dump(rows, open(f'{H}/osteo-results/spark/spark_scores.json', 'w'), indent=1)
print(f"{'data type':20s} {'metric':11s} " + ' '.join(f'{m[:26]:>26s}' for m in files))
for r in rows:
    print(f"{r['data_type']:20s} {r['metric']:11s} " + ' '.join(
        f"{('-' if r[f'{m}_primary'] is None else format(r[f'{m}_primary'], '.3f')) + ' (' + format(r[f'{m}_answered'], '.0%') + ')':>26s}" for m in files))
