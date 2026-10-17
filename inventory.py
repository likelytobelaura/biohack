"""Rebuild ~/biohack/INVENTORY.md and inventory.json; pull finished hold-out runs from Modal.

python3 ~/biohack/inventory.py            # refresh everything (safe to re-run any time)
python3 ~/biohack/inventory.py --offline  # skip Modal; re-index local files only
"""
import json, os, sys, glob, hashlib, subprocess, collections, datetime

B = os.path.expanduser('~/biohack')
R = os.path.expanduser('~/osteo-results')
MODAL = os.path.expanduser('~/sutro-problems/.venv/bin/modal')
VOL = 'osteo-data'
EXP = f'{B}/06_heldout_experiments'
CACHE = f'{B}/.hash_cache.json'

LABELS = {'scrna': 'scRNA-seq (10x)', 'cite': 'CITE-seq tags', 'rna': 'Bulk RNA-seq TPM', 'tcr': 'TCR beta clones',
          'dia': 'DIA proteomics', 'mif': 'Multiplex IF', 'olink': 'Olink plasma NPX', 'xenium': 'Xenium',
          'somatic': 'Somatic WGS calls', 'mpa': 'Metagenomics', 'flow': 'Flow cytometry', 'hla': 'HLA typing',
          'fusion': 'Gene fusions', 'all': 'All data (reference)'}
# the sweep launched 2026-10-17 13:23 (job 6c5c300f, app ap-AIzYiRcj5xsmQdDteujfMa); new runs are picked up from Modal
PLANNED = ['flow', 'rna', 'tcr', 'xenium', 'all']

# section, relative path, what it is
ITEMS = [
    ('1. Sid\'s full dataset', '01_sid_full_dataset/sid_full.jsonl',
     'SFT training set built from Sid\'s data minus the 20% mask; {prompt, completion}. Same file as osteo-data:train/sid_full.jsonl'),
    ('1. Sid\'s full dataset', '01_sid_full_dataset/source/manifest.txt', 'every object URL in s3://sid-sijbrandij-osteosarc-dataset'),
    ('1. Sid\'s full dataset', '01_sid_full_dataset/source/keys.txt', 'bucket keys listing used for reconstruction'),
    ('1. Sid\'s full dataset', '01_sid_full_dataset/source/h5keys.txt', 'keys of the 10x .h5 matrices downloaded'),
    ('1. Sid\'s full dataset', '01_sid_full_dataset/source/src/', 'non-h5 source tables downloaded from the bucket'),
    ('1. Sid\'s full dataset', '01_sid_full_dataset/source/h5/', '10x feature-barcode matrices (scRNA / CITE-seq)'),
    ('2. Golden labels (masked data)', '02_golden_labels/golden.jsonl', 'one row per eval prompt: messages, items, gold, completion'),
    ('2. Golden labels (masked data)', '02_golden_labels/labels.csv', 'flat: one row per (prompt_id, item number)'),
    ('2. Golden labels (masked data)', '02_golden_labels/truth.jsonl', 'one row per masked target (prompt_id, i, item, entity, mod, truth); scorers read this'),
    ('2. Golden labels (masked data)', '02_golden_labels/validated_files.json', 'which bucket file each entity\'s truth came from'),
    ('2. Golden labels (masked data)', '02_golden_labels/build_golden.py', 'rebuilds golden.jsonl + labels.csv'),
    ('2. Golden labels (masked data)', '02_golden_labels/README_golden.md', 'how the gold labels were built + caveats'),
    ('3. Baseline eval prompts', '03_baseline_prompts/baseline.jsonl',
     '20% random mask as prompts; {prompt_id, messages}. = osteo-data:prompts/baseline.jsonl (= osteo-results/eval_prompts.jsonl, reformatted)'),
    ('4. Baseline eval responses', '04_baseline_responses/baseline_zeroshot.jsonl', 'Muse-Glimmer-30B zero-shot, first pass'),
    ('4. Baseline eval responses', '04_baseline_responses/baseline_zeroshot_retry.jsonl', 're-runs of the 54 prompts truncated (finish=length)'),
    ('4. Baseline eval responses', '04_baseline_responses/baseline_zeroshot_merged.jsonl', 'first pass with the 54 retries swapped in; USE THIS ONE'),
    ('5. Baseline report (score.py)', '05_baseline_report/REPORT.md', 'written summary of the baseline'),
    ('5. Baseline report (score.py)', '05_baseline_report/score_py_output.txt', 'raw stdout of scripts/score.py'),
    ('5. Baseline report (score.py)', '05_baseline_report/baseline_scores.json', 'per-data-type metrics from score.py'),
    ('7. Full-data fine-tune (sid_full)', '07_finetune_sid_full/finetuned_sid_full_eval.jsonl',
     'sid_full adapter on all eval prompts (= osteo-data:outputs/baseline_sid_full.jsonl)'),
    ('7. Full-data fine-tune (sid_full)', '07_finetune_sid_full/compare_scores.json', 'base vs fine-tune per data type (scripts/score_compare.py)'),
    ('7. Full-data fine-tune (sid_full)', '07_finetune_sid_full/compare_microbiome.json', 'base vs fine-tune, metagenomics detail'),
    ('7. Full-data fine-tune (sid_full)', '07_finetune_sid_full/compare_points.json', 'per-target (truth, base, fine-tune) points for plots'),
    ('7. Full-data fine-tune (sid_full)', '07_finetune_sid_full/adapter/adapter_model.safetensors',
     'LoRA r=32 weights, final (training stopped itself at step 739/1478); base = Muse-Glimmer-30B'),
    ('7. Full-data fine-tune (sid_full)', '07_finetune_sid_full/adapter/', 'adapter + configs + train_log.json + checkpoint-739 (optimizer state)'),
    ('7. Full-data fine-tune (sid_full)', '07_finetune_sid_full/dashboard_src/',
     'source + data of https://claude.ai/artifact/YTL434mENWCv3anqC5EE2J'),
]


# live sources other jobs write to -> durable copy here (re-copied whenever size/mtime changes)
H = os.path.expanduser('~')
SYNC = [
    (f'{H}/osteo-results/lomo', '06_heldout_experiments/osteo_results_lomo', 'hold-out adapters/evals the LOMO job (6c5c300f) pulls locally'),
    (f'{H}/osteo-results/spark', '08_other_baselines/muse_spark', 'Muse Spark 1.2 / 1.3 (api.meta.ai, contributor) on golden.jsonl prompts; job 7d63451c'),
    (f'{H}/osteo-nim', '08_other_baselines/nim_noctx', 'NVIDIA NIM, prompts with the known values stripped (826 prompts); job c76b4182'),
    (f'{H}/osteo-results/scripts', 'scripts', 'project scripts'),
]
SKIP = ('.nvkey', '__pycache__', '.DS_Store', '.pyc')
# everything under these roots must be either mirrored or deliberately skipped; anything else is reported as unfiled
WATCH = [f'{H}/osteo-results', f'{H}/osteo-nim', f'{H}/osteo-golden']
FILED_ELSEWHERE = {'eval_prompts.jsonl', 'baseline_truth.jsonl', 'validated_files.json', 'baseline_zeroshot.jsonl',
                   'baseline_zeroshot_retry.jsonl', 'baseline_zeroshot_merged.jsonl', 'baseline_scores.json',
                   'finetuned_sid_full_eval.jsonl', 'compare_scores.json', 'compare_microbiome.json', 'compare_points.json',
                   'golden.jsonl', 'labels.csv', 'build_golden.py', 'README.md', 'lomo_scores.json'}


def sync():
    copied = []
    for src, rel, _ in SYNC:
        for f in glob.glob(f'{src}/**', recursive=True):
            if not os.path.isfile(f) or any(k in f for k in SKIP):
                continue
            dst = f'{B}/{rel}/{os.path.relpath(f, src)}'
            a = os.stat(f)
            if os.path.exists(dst) and os.path.getsize(dst) == a.st_size and int(os.stat(dst).st_mtime) >= int(a.st_mtime):
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            subprocess.run(['cp', '-cp', f, dst], check=True)
            copied.append(f'{rel}/{os.path.relpath(f, src)}')
    for f in ('finetuned_sid_full_eval.jsonl', 'compare_scores.json', 'compare_microbiome.json', 'compare_points.json'):
        if os.path.exists(f'{R}/{f}') and os.path.getmtime(f'{R}/{f}') > os.path.getmtime(f'{B}/07_finetune_sid_full/{f}'):
            subprocess.run(['cp', '-cp', f'{R}/{f}', f'{B}/07_finetune_sid_full/{f}'], check=True); copied.append(f)
    return copied


def unfiled():
    synced = [s for s, _, _ in SYNC] + [f'{R}/adapters']
    out = []
    for root in WATCH:
        for f in glob.glob(f'{root}/**', recursive=True):
            if (os.path.isfile(f) and not any(f.startswith(s + '/') for s in synced) and not any(k in f for k in SKIP)
                    and os.path.basename(f) not in FILED_ELSEWHERE):
                out.append(f.replace(H, '~'))
    return out


def tree(rel):
    out = []
    for f in sorted(glob.glob(f'{B}/{rel}/**', recursive=True)):
        if os.path.isfile(f):
            out.append({'path': os.path.relpath(f, B), 'bytes': os.path.getsize(f), 'rows': rows(f)})
    return out


def rows(p):
    if not p.endswith(('.jsonl', '.csv', '.txt')):
        return None
    with open(p, 'rb') as f:
        return sum(1 for _ in f)


def sha(p, cache):
    st = os.stat(p); key = f'{p}|{st.st_size}|{int(st.st_mtime)}'
    if key not in cache:
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            for blk in iter(lambda: f.read(1 << 20), b''):
                h.update(blk)
        cache[key] = h.hexdigest()
    return cache[key]


def describe(rel, cache):
    p = f'{B}/{rel}'
    if not os.path.exists(p):
        return {'path': rel, 'missing': True}
    if os.path.isdir(p):
        fs = [f for f in glob.glob(f'{p}/**', recursive=True) if os.path.isfile(f)]
        return {'path': rel, 'files': len(fs), 'bytes': sum(os.path.getsize(f) for f in fs)}
    return {'path': rel, 'bytes': os.path.getsize(p), 'rows': rows(p), 'sha256': sha(p, cache)}


def human(n):
    for u in ('B', 'K', 'M', 'G'):
        if n < 1024 or u == 'G':
            return f'{n:.0f}{u}' if u == 'B' else f'{n:.1f}{u}'
        n /= 1024


def modal(*a):
    out = subprocess.run([MODAL, *a], capture_output=True, text=True)
    return out.stdout if out.returncode == 0 else None


def vol_ls(path):
    out = modal('volume', 'ls', VOL, path, '--json')
    return json.loads(out) if out else []


def subset(src, dst, pids):
    if not os.path.exists(src):
        return 0
    n = 0
    with open(src) as f, open(dst, 'w') as g:
        for l in f:
            if json.loads(l)['prompt_id'] in pids:
                g.write(l); n += 1
    return n


def heldout(offline):
    truth = collections.defaultdict(list)
    for l in open(f'{B}/02_golden_labels/truth.jsonl'):
        t = json.loads(l); truth[t['mod']].append(t)
    remote = {}
    app = None
    if not offline:
        for e in vol_ls('adapters/lomo'):
            n = os.path.basename(e['filename'])
            remote.setdefault(n, {})['adapter_files'] = len(vol_ls(f'adapters/lomo/{n}'))
        for e in vol_ls('outputs/lomo'):
            n = os.path.basename(e['filename'])
            remote.setdefault(n, {})['shards'] = len(vol_ls(f'outputs/lomo/{n}'))
        apps = json.loads(modal('app', 'list', '--json') or '[]')
        app = next((a for a in apps if a['description'] == 'osteo-lomo' and a['stopped_at'] is None), None)
    keys = list(dict.fromkeys(PLANNED + [n.split('_', 1)[1] for n in remote if n.startswith('lomo740_')]))
    exps = []
    for h in keys:
        name = f'lomo740_{h}'
        d = f'{EXP}/{name}'
        r = remote.get(name, {})
        os.makedirs(d, exist_ok=True)
        if r.get('shards'):
            tmp = f'{d}/.pull'
            subprocess.run(['rm', '-rf', tmp]); os.makedirs(tmp)
            if modal('volume', 'get', VOL, f'outputs/lomo/{name}', tmp + '/') is not None:
                subprocess.run(['rm', '-rf', f'{d}/outputs']); os.rename(f'{tmp}/{name}', f'{d}/outputs')
            subprocess.run(['rm', '-rf', tmp])
        local_shards = sorted(glob.glob(f'{d}/outputs/shard*.jsonl'))
        n_out = sum(rows(f) for f in local_shards)
        ts = truth.get(h, []) if h != 'all' else [t for v in truth.values() for t in v]
        pids = {t['prompt_id'] for t in ts}
        if h != 'all':
            b = f'{d}/baselines'; os.makedirs(b, exist_ok=True)
            with open(f'{b}/truth_{h}.jsonl', 'w') as g:
                g.writelines(json.dumps(t) + '\n' for t in ts)
            subset(f'{B}/03_baseline_prompts/baseline.jsonl', f'{b}/prompts_{h}.jsonl', pids)
            subset(f'{B}/04_baseline_responses/baseline_zeroshot_merged.jsonl', f'{b}/base_zeroshot_{h}.jsonl', pids)
            subset(f'{R}/finetuned_sid_full_eval.jsonl', f'{b}/sid_full_finetune_{h}.jsonl', pids)
            subset_all = sorted(glob.glob(f'{EXP}/lomo740_all/outputs/shard*.jsonl'))
            if subset_all:
                with open(f'{b}/all_reference_{h}.jsonl', 'w') as g:
                    for f in subset_all:
                        g.writelines(l for l in open(f) if json.loads(l)['prompt_id'] in pids)
        if n_out and n_out >= len(pids):
            status = 'done'
        elif n_out:
            status = f'evaluating ({n_out}/{len(pids)} prompts pulled)'
        elif r.get('adapter_files'):
            status = 'trained, eval pending'
        elif name in remote:
            status = 'training'
        else:
            status = 'queued' if app else ('not on Modal' if not offline else 'unknown (offline)')
        exps.append({'name': name, 'holdout': h, 'label': LABELS.get(h, h),
                     'trains_on': 'all 11,824 rows' if h == 'all' else f'everything except {LABELS.get(h, h)}',
                     'eval_targets': len(ts), 'eval_prompts': len(pids), 'status': status,
                     'adapter': f'{VOL}:adapters/lomo/{name}', 'outputs_remote': f'{VOL}:outputs/lomo/{name}',
                     'outputs_local_rows': n_out})
    scores = None
    if os.path.exists(f'{R}/lomo_scores.json'):
        subprocess.run(['cp', '-c', f'{R}/lomo_scores.json', f'{EXP}/lomo_scores.json'])
        scores = {s['key']: s for s in json.load(open(f'{EXP}/lomo_scores.json'))}
    return exps, scores, app


def main():
    offline = '--offline' in sys.argv
    copied = sync()
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    items = [dict(section=s, desc=d, **describe(p, cache)) for s, p, d in ITEMS]
    json.dump(cache, open(CACHE, 'w'))
    exps, scores, app = heldout(offline)
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    mirrors = {rel: tree(rel) for _, rel, _ in SYNC if rel != 'scripts'}
    loose = unfiled()
    for f in loose:  # keep a copy of anything new until it gets a proper home
        src = f.replace('~', H, 1); dst = f'{B}/_incoming/{os.path.relpath(src, H)}'
        if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True); subprocess.run(['cp', '-cp', src, dst])
    json.dump({'generated': now, 'items': items, 'heldout': exps, 'lomo_scores': scores, 'sweep_app': app,
               'mirrors': mirrors, 'unfiled': loose, 'copied_this_run': copied}, open(f'{B}/inventory.json', 'w'), indent=1)

    L = [f'# Biohack data inventory (osteosarcoma imputation)', '',
         f'Generated {now} by `python3 ~/biohack/inventory.py`. Re-run it after each hold-out run finishes.', '',
         'Model: Muse-Glimmer-30B (Modal volume `hf-cache:/Muse-Glimmer-30B`). Data: Sid Sijbrandij\'s public',
         'osteosarcoma bucket `s3://sid-sijbrandij-osteosarc-dataset` (us-west-2, `--no-sign-request`).',
         'Eval = a 20% random mask: 24,634 targets in 15,964 prompts, 13 data types.', '']
    def table(its):
      sec = None
      for it in its:
        if it['section'] != sec:
            sec = it['section']; L.extend(['', f'## {sec}', '', '| file | size | rows | sha256 | what |', '|---|---:|---:|---|---|'])
        if it.get('missing'):
            L.append(f"| `{it['path']}` | **MISSING** | | | {it['desc']} |"); continue
        size = human(it['bytes']) + (f" ({it['files']} files)" if 'files' in it else '')
        rw = f"{it['rows']:,}" if it.get('rows') is not None else ''
        L.append(f"| `{it['path']}` | {size} | {rw} | {it.get('sha256', '')[:12]} | {it['desc']} |")

    def mirror(rel):
        fs = mirrors[rel]
        if not fs:
            L.append(f'_`{rel}/`: nothing yet._'); return
        L.extend(['', f'`{rel}/`', '', '| file | size | rows |', '|---|---:|---:|'])
        L.extend(f"| `{f['path'][len(rel) + 1:]}` | {human(f['bytes'])} | {'' if f['rows'] is None else format(f['rows'], ',')} |" for f in fs)

    table([i for i in items if i['section'][0] in '12345'])
    L += ['', '## 6. Hold-out experiments (leave-one-data-type-out)', '',
          'LoRA r=32, 740 steps, 8×H200, same recipe as `sid_full` but with one data type removed from training;',
          'each is then evaluated on that type\'s masked targets only. `lomo740_all` is the matched reference trained on everything.',
          'Script: `scripts/lomo_train.py`; scorer: `scripts/score_lomo.py`. transfer = (holdout − base) / (all − base).',
          f"Sweep app: `{app['app_id']}` ({app['state']})" if app else 'Sweep app: not running.', '',
          '| run | trains on | eval targets | status | local outputs |', '|---|---|---:|---|---|']
    for e in exps:
        L.append(f"| `{e['name']}` | {e['trains_on']} | {e['eval_targets']:,} | {e['status']} | "
                 f"{'`06_heldout_experiments/' + e['name'] + '/outputs/`' if e['outputs_local_rows'] else '–'} |")
    L += ['', 'Baselines for each hold-out are in `06_heldout_experiments/lomo740_<type>/baselines/`, sliced to that type\'s prompts:',
          '`prompts_<type>`, `truth_<type>`, `base_zeroshot_<type>` (Glimmer zero-shot), `sid_full_finetune_<type>` (full-data',
          'adapter, 739 steps), and `all_reference_<type>` once `lomo740_all` has finished. The train-median reference is in',
          '`05_baseline_report/baseline_scores.json`.', '']
    if scores:
        L += ['| hold-out | metric | base | sid_full | all | holdout | transfer [95% CI] |', '|---|---|---:|---:|---:|---:|---|']
        for s in scores.values():
            L.append(f"| {s['data_type']} | {s['metric']} | {s['base']} | {s['sid_full']} | {s['all']} | {s['holdout']} | "
                     f"{s['transfer']} [{s['transfer_lo']}, {s['transfer_hi']}] |")
    else:
        L.append('_No hold-out scores yet (`~/osteo-results/lomo_scores.json` not written)._')
    L.append('')
    L.append('Local mirror of `~/osteo-results/lomo/` (written by the LOMO job):')
    mirror('06_heldout_experiments/osteo_results_lomo')
    table([i for i in items if i['section'][0] == '7'])
    L.extend(['', '## 8. Other baselines', '',
              'Mirrored from the jobs running them; partial files are re-copied on every refresh until the runs finish.'])
    for src, rel, d in SYNC:
        if rel.startswith('08_'):
            L.extend(['', f'**{rel.split("/")[1]}**: {d}. Source `{src.replace(H, "~")}` (API key file excluded).'])
            mirror(rel)
    L.extend(['', '## Not yet filed', ''])
    L.extend([f'- `{f}` (copied to `_incoming/`)' for f in loose] or ['_Nothing: every file under ~/osteo-results, ~/osteo-nim and ~/osteo-golden is mirrored here._'])
    L += ['', '## Links', '',
          '- Comparison dashboard: https://claude.ai/artifact/YTL434mENWCv3anqC5EE2J (source in `07_finetune_sid_full/dashboard_src/`)',
          '- Working copies the original scripts still read: `~/.claude/jobs/6c5c300f/tmp/osteo/` (temporary; this folder is the durable copy).', '']
    open(f'{B}/INVENTORY.md', 'w').write('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
