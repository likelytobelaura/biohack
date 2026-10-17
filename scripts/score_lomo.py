"""Score the leave-one-data-type-out runs: python score_lomo.py <lomo outputs dir> holdout1 holdout2 ..."""
import json, sys, random, collections, glob
from score_compare import parse, num, spear, LABELS, CAT, BIN
from common import D

R = '/Users/laurabocek/osteo-results'
BOOT = 1000


def preds(files):
    p = {}
    for f in files:
        for l in open(f):
            r = json.loads(l); p[r['prompt_id']] = parse(r['text']) or {}
    return p


def metric(lab, pairs):
    if lab in CAT:
        return sum(str(a).strip() == str(b).strip() for a, b in pairs) / max(len(pairs), 1)
    xy = [(num(a), num(b)) for a, b in pairs]
    xy = [(a, b) for a, b in xy if a is not None and b is not None]
    if lab in BIN:
        return sum(round(a) == round(b) for a, b in xy) / max(len(xy), 1)
    return spear([a for a, b in xy], [b for a, b in xy])


if __name__ == '__main__':
    out_root, holdouts = sys.argv[1], sys.argv[2:]
    truth = collections.defaultdict(list)
    for l in open(D + '/truth.jsonl'):
        t = json.loads(l); truth[t['mod']].append(t)
    models = {'base': preds([f'{R}/baseline_zeroshot_merged.jsonl']),
              'sid_full': preds([f'{R}/finetuned_sid_full_eval.jsonl']),
              'all': preds(glob.glob(f'{out_root}/lomo740_all/shard*.jsonl'))}
    rows = []
    random.seed(0)
    for h in holdouts:
        models['holdout'] = preds(glob.glob(f'{out_root}/lomo740_{h}/shard*.jsonl'))
        ts = truth[h]
        by_prompt = collections.defaultdict(list)
        for t in ts:
            by_prompt[t['prompt_id']].append(t)
        pids = list(by_prompt)

        def score_on(pid_list):
            res = {}
            for m, p in models.items():
                pairs = [(p.get(t['prompt_id'], {}).get(str(t['i'])), t['truth']) for pid in pid_list for t in by_prompt[pid]]
                pairs = [(a, b) for a, b in pairs if a is not None]
                res[m] = metric(h, pairs)
            return res

        point = score_on(pids)
        boots = collections.defaultdict(list)
        for _ in range(BOOT):
            s = score_on([random.choice(pids) for _ in pids])
            for m, v in s.items():
                if v is not None:
                    boots[m].append(v)
            if None not in (s['holdout'], s['all'], s['base']) and s['all'] != s['base']:
                boots['transfer'].append((s['holdout'] - s['base']) / (s['all'] - s['base']))
        ci = lambda v: (round(sorted(v)[int(.025 * len(v))], 4), round(sorted(v)[int(.975 * len(v)) - 1], 4)) if v else (None, None)
        tr = (point['holdout'] - point['base']) / (point['all'] - point['base']) if point['all'] != point['base'] else None
        row = {'data_type': LABELS[h], 'key': h, 'targets': len(ts), 'prompts': len(pids),
               'metric': 'exact match' if h in CAT else 'accuracy' if h in BIN else 'Spearman ρ',
               'transfer': None if tr is None else round(tr, 4)}
        row['transfer_lo'], row['transfer_hi'] = ci(boots['transfer'])
        for m in ('base', 'sid_full', 'all', 'holdout'):
            row[m] = None if point[m] is None else round(point[m], 4)
            row[f'{m}_lo'], row[f'{m}_hi'] = ci(boots[m])
        row['answered_holdout'] = round(sum(models['holdout'].get(t['prompt_id'], {}).get(str(t['i'])) is not None for t in ts) / len(ts), 4)
        rows.append(row)
        print(f"{LABELS[h]:18s} base {row['base']}  holdout {row['holdout']} [{row['holdout_lo']},{row['holdout_hi']}]  "
              f"all {row['all']}  sid_full {row['sid_full']}  transfer {row['transfer']} [{row['transfer_lo']},{row['transfer_hi']}]")
    json.dump(rows, open(f'{R}/lomo/lomo_scores.json', 'w'), indent=1)
