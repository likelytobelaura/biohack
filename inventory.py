"""Rebuild ~/biohack/INVENTORY.md and inventory.json; pull finished hold-out runs from Modal.

python3 ~/biohack/inventory.py            # refresh everything (safe to re-run any time)
python3 ~/biohack/inventory.py --offline  # skip Modal; re-index local files only
"""
import json, os, sys, glob, hashlib, subprocess, collections, datetime, time

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
    ('7. sid_full: LoRA on all data, stopped at 50%', '07_finetune_sid_full/finetuned_sid_full_eval.jsonl',
     'sid_full adapter on all eval prompts (= osteo-data:outputs/baseline_sid_full.jsonl)'),
    ('7. sid_full: LoRA on all data, stopped at 50%', '07_finetune_sid_full/compare_scores.json', 'base vs sid_full adapter per data type (scripts/score_compare.py)'),
    ('7. sid_full: LoRA on all data, stopped at 50%', '07_finetune_sid_full/compare_microbiome.json', 'base vs sid_full adapter, metagenomics detail'),
    ('7. sid_full: LoRA on all data, stopped at 50%', '07_finetune_sid_full/compare_points.json', 'per-target (truth, base, sid_full) points for plots'),
    ('7. sid_full: LoRA on all data, stopped at 50%', '07_finetune_sid_full/adapter/adapter_model.safetensors',
     'LoRA adapter weights (r=32, a delta on Muse-Glimmer-30B, not a standalone model) after 739 of 1,478 planned steps '
     '(0.5 epoch, stopped mid-schedule, LR not decayed); byte-identical to checkpoint-739/adapter_model.safetensors'),
    ('7. sid_full: LoRA on all data, stopped at 50%', '07_finetune_sid_full/adapter/', 'the adapter above + configs + train_log.json; checkpoint-739/ = same weights + optimizer/scheduler/RNG state to resume'),
    ('7. sid_full: LoRA on all data, stopped at 50%', '07_finetune_sid_full/dashboard_src/',
     'source + data of https://claude.ai/artifact/YTL434mENWCv3anqC5EE2J'),
]


# live sources other jobs write to -> durable copy here (re-copied whenever size/mtime changes)
H = os.path.expanduser('~')
SYNC = [
    (f'{H}/osteo-results/lomo', '06_heldout_experiments/osteo_results_lomo', 'hold-out adapters/evals the LOMO job (6c5c300f) pulls locally'),
    (f'{H}/osteo-results/spark', '08_other_baselines/muse_spark', 'Muse Spark 1.2 / 1.3 (api.meta.ai, contributor) on golden.jsonl prompts; job 7d63451c'),
    (f'{H}/osteo-nim', '08_other_baselines/nim_noctx', 'NVIDIA NIM, prompts with the known values stripped (826 prompts) + NIM hold-out baselines; job c76b4182'),
    (f'{H}/osteo-data-volume', '11_modal_osteo_data', 'full copy of the Modal volume osteo-data, same layout (mirrored locally by the LOMO job)'),
    (f'{H}/osteo-classical', '09_classical_ml', 'classical imputation sweep (kNN etc.) + ET-SoC-1 card work (aifoundry3 card 0; card results copied back to card/from_aifoundry3/); job c76b4182'),
    (f'{H}/osteo-results/scripts', 'scripts', 'project scripts'),
]
# Laura's AI Foundry lab account (Tailscale SSH); pulled whole, minus dotfiles and the public repo clone
LAB = 'likelytobelaura@aifoundry3'
ET = f'{B}/10_esperanto_et_soc1/aifoundry3'
ETREF = 'f0a40879800d8d21955d061b92ad974570455595'
ESPERANTO = [('osteo', 'osteo card job: S = A·Aᵀ on card 0 (kernel `aat/`, build, sim logs, `runs/`, `cite_S_card.f32`, NOTES.md, DONE)'),
             ('work/mnist-et', 'MNIST trained end to end on the card (difficulty 1 passed, 111 ms/call); NOTES.md'),
             ('work/hello', 'first card program'), ('work/ref', 'reference files'),
             ('runs/mnist', 'raw simulator logs of the MNIST runs (kept deliberately: the only raw run records)'),
             ('runs/hello', 'hello run'), ('runs/first-hour', 'first-hour lab test'),
             ('data', 'datasets used on the card'), ('lab-start.md', 'the lab\'s START brief the lab Claude followed')]


def pull_lab():
    os.makedirs(ET, exist_ok=True)
    src = [f'{LAB}:{p}' for p in ('osteo', 'work', 'runs', 'data', 'lab-start.md')]
    try:
        r = subprocess.run(['rsync', '-a', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=10', *src, ET + '/'],
                           capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return 'rsync timed out; previous copy kept'
    return f"last pulled {datetime.datetime.now():%Y-%m-%d %H:%M}" if r.returncode == 0 else \
        f'aifoundry3 unreachable ({r.stderr.strip().splitlines()[-1][:80] if r.stderr.strip() else r.returncode}); previous copy kept'


SKIP = ('.nvkey', '__pycache__', '.DS_Store', '.pyc', '/.venv/', '.scoring')
# everything under these roots must be either mirrored or deliberately skipped; anything else is reported as unfiled
WATCH = [f'{H}/osteo-results', f'{H}/osteo-nim', f'{H}/osteo-golden', f'{H}/osteo-classical']
FILED_ELSEWHERE = {'eval_prompts.jsonl', 'baseline_truth.jsonl', 'validated_files.json', 'baseline_zeroshot.jsonl',
                   'baseline_zeroshot_retry.jsonl', 'baseline_zeroshot_merged.jsonl', 'baseline_scores.json',
                   'finetuned_sid_full_eval.jsonl', 'compare_scores.json', 'compare_microbiome.json', 'compare_points.json',
                   'golden.jsonl', 'labels.csv', 'build_golden.py', 'README.md', 'lomo_scores.json'}


# mirrors whose deletions are carried over (the source's owner removes bad files there); others only ever add
PRUNE = {'09_classical_ml'}


def prune(src, rel):
    # drop mirrored files whose source is gone; skip entirely if the source looks missing or mostly wiped
    dst_root = f'{B}/{rel}'
    have = [f for f in glob.glob(f'{dst_root}/**', recursive=True) if os.path.isfile(f)]
    gone = [f for f in have if not os.path.exists(f'{src}/{os.path.relpath(f, dst_root)}')]
    if not os.path.isdir(src) or not os.listdir(src) or len(gone) > len(have) / 4:
        if gone:
            print(f'prune {rel}: {len(gone)} of {len(have)} files missing at {src}; not deleting', file=sys.stderr)
        return []
    for f in gone:
        os.remove(f)
    return [f'deleted {os.path.relpath(f, B)}' for f in gone]


def sync():
    copied = []
    for src, rel, _ in SYNC:
        for f in glob.glob(f'{src}/**', recursive=True):
            if not os.path.isfile(f) or any(k in f for k in SKIP):
                continue
            dst = f'{B}/{rel}/{os.path.relpath(f, src)}'
            a = os.stat(f)
            if time.time() - a.st_mtime < 120:  # still being written (e.g. a Modal download in progress); next refresh gets it
                continue
            if os.path.exists(dst) and os.path.getsize(dst) == a.st_size and int(os.stat(dst).st_mtime) >= int(a.st_mtime):
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            subprocess.run(['cp', '-cp', f, dst], check=True)
            copied.append(f'{rel}/{os.path.relpath(f, src)}')
        if rel in PRUNE:
            copied += prune(src, rel)
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
    # the local copy of the whole volume (section 11) is the source of truth for status; Modal only adds what's newer
    VM = f'{B}/11_modal_osteo_data'
    local = {os.path.basename(p) for p in glob.glob(f'{VM}/outputs/lomo/lomo740*') + glob.glob(f'{VM}/adapters/lomo/lomo740*')}
    names = list(dict.fromkeys([f'lomo740_{h}' for h in PLANNED] + sorted(n for n in set(remote) | local if n.startswith('lomo740'))))
    exps = []
    for name in names:
        # lomo740h100_all = the all-data reference's full eval, re-launched with H100 fallback
        h = 'all_h100' if name == 'lomo740h100_all' else name.split('_', 1)[1]
        d = f'{EXP}/{name}'
        r = remote.get(name, {})
        os.makedirs(d, exist_ok=True)
        if r.get('shards'):
            tmp = f'{d}/.pull'
            subprocess.run(['rm', '-rf', tmp]); os.makedirs(tmp)
            if modal('volume', 'get', VOL, f'outputs/lomo/{name}', tmp + '/') is not None:
                subprocess.run(['rm', '-rf', f'{d}/outputs']); os.rename(f'{tmp}/{name}', f'{d}/outputs')
            subprocess.run(['rm', '-rf', tmp])
        n_out = max(sum(rows(f) for f in glob.glob(f'{d}/outputs/shard*.jsonl')),
                    sum(rows(f) for f in glob.glob(f'{VM}/outputs/lomo/{name}/shard*.jsonl')))
        # lomo740_<type>_full = the <type> hold-out adapter re-evaluated on every prompt, to measure off-target effects
        off = h.endswith('_full')
        base = h[:-len('_full')] if off else 'all' if h == 'all_h100' else h
        everything = h in ('all', 'all_h100') or off
        ts = [t for v in truth.values() for t in v] if everything else truth.get(h, [])
        pids = {t['prompt_id'] for t in ts}
        if not everything:
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
        trained = bool(r.get('adapter_files')) or os.path.exists(f'{VM}/adapters/lomo/{name}/adapter_model.safetensors')
        full = 'lomo740h100_all' if h == 'all' else f'lomo740_{h}_full'
        out_dir = next((p for p in (f'{d}/outputs', f'{VM}/outputs/lomo/{name}') if glob.glob(f'{p}/shard*.jsonl')), None)
        if n_out and n_out >= len(pids):
            status = 'done'
        elif n_out:
            status = f'evaluating ({n_out:,}/{len(pids):,} prompts so far)'
        elif trained and (full in names):
            status = f'trained; evaluated in `{full}` (all prompts)'
        elif trained:
            status = 'trained, eval pending'
        elif name in remote:
            status = 'training'
        else:
            status = 'queued' if app else 'not on the volume'
        exps.append({'name': name, 'holdout': base, 'off_target_eval': off, 'label': LABELS.get(base, base),
                     'trains_on': ('all 11,824 rows' + (' (eval of `lomo740_all` re-run with H100 fallback)' if h == 'all_h100' else ''))
                                  if base == 'all' else f'everything except {LABELS.get(base, base)}'
                                  + (' (adapter of `lomo740_' + base + '`; eval on ALL prompts, off-target)' if off else ''),
                     'eval_targets': len(ts), 'eval_prompts': len(pids), 'status': status,
                     'adapter': f'{VOL}:adapters/lomo/{name}', 'outputs_remote': f'{VOL}:outputs/lomo/{name}',
                     'outputs_local_rows': n_out, 'outputs_path': out_dir and os.path.relpath(out_dir, B)})
    scores = None
    # score_lomo.py writes into lomo/ since 2026-10-17 15:10; older runs wrote to the osteo-results root
    src = next((p for p in (f'{R}/lomo/lomo_scores.json', f'{R}/lomo_scores.json') if os.path.exists(p)), None)
    if src:
        subprocess.run(['cp', '-cp', src, f'{EXP}/lomo_scores.json'])
        scores = {s.get('key', s.get('data_type')): s for s in json.load(open(f'{EXP}/lomo_scores.json'))}
    return exps, scores, app


def main():
    offline = '--offline' in sys.argv
    copied = sync()
    lab = 'offline run: not pulled' if offline else pull_lab()
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
               'mirrors': mirrors, 'unfiled': loose, 'copied_this_run': copied, 'esperanto_pull': lab}, open(f'{B}/inventory.json', 'w'), indent=1)

    L = [f'# Biohack data inventory (osteosarcoma imputation)', '',
         f'Generated {now} by `inventory.py` (a watcher re-runs it and pushes to GitHub every 10 min while runs are going).', '',
         'Model: Muse-Glimmer-30B (Modal volume `hf-cache:/Muse-Glimmer-30B`). Data: Sid Sijbrandij\'s public',
         'osteosarcoma bucket `s3://sid-sijbrandij-osteosarc-dataset` (us-west-2, `--no-sign-request`).',
         'Eval = a 20% random mask: 24,634 targets in 15,964 prompts, 13 data types.', '',
         '**Where it lives.** GitHub (this repo): code, reports, scores, every model answer file, the per-hold-out prompt and',
         'gold-label slices, and a full text copy of the Modal volume in `11_modal_osteo_data/` (incl. the training set',
         '`train/sid_full.jsonl` and the eval prompts `prompts/baseline.jsonl`). HF',
         '([likelytobelaura/osteo-glimmer-lora](https://huggingface.co/likelytobelaura/osteo-glimmer-lora)): every adapter\'s',
         'weights, optimizer state and tokenizer. Local only (sizes and sha256 below): the full gold labels (`02_golden_labels/`),',
         'raw bucket files (`01_sid_full_dataset/source/`), card result matrices (`*.f32`), raw ET-SoC-1 simulator logs and datasets.',
         'Folders 01 and 03 hold the same training set and prompts as section 11 (git-ignored there, committed in 11).', '']
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
          'LoRA r=32 (same settings as `sid_full`), 8×H200, a 740-step schedule run to completion (5,920 rows seen), one data type removed;',
          'each is then evaluated on that type\'s masked targets only. `lomo740_all` is the matched reference trained on everything:\n'
          'the finished counterpart of `sid_full`, which saw the same half epoch but stopped mid-schedule.',
          'Script: `scripts/lomo_train.py`; scorer: `scripts/score_lomo.py`. transfer = (holdout − base) / (all − base).',
          f"Sweep app: `{app['app_id']}` ({app['state']})" if app else 'Sweep app: not running.', '',
          '| run | trains on | eval targets | status | local outputs |', '|---|---|---:|---|---|']
    for e in exps:
        L.append(f"| `{e['name']}` | {e['trains_on']} | {e['eval_targets']:,} | {e['status']} | "
                 f"{'`' + e['outputs_path'] + '/`' if e['outputs_path'] else '–'} |")
    L += ['', 'Baselines for each hold-out are in `06_heldout_experiments/lomo740_<type>/baselines/`, sliced to that type\'s prompts:',
          '`prompts_<type>`, `truth_<type>`, `base_zeroshot_<type>` (Glimmer zero-shot), `sid_full_finetune_<type>` (sid_full',
          'adapter, stopped at step 739/1478), and `all_reference_<type>` once `lomo740_all` has finished. The train-median reference is in',
          '`05_baseline_report/baseline_scores.json`.',
          '`lomo740_<type>_full` runs re-evaluate a hold-out adapter on every prompt; their per-type change vs `lomo740_all`',
          '(off-target effect of dropping that type) is in `osteo_results_lomo/offtarget_scores.json` (`scripts/score_offtarget.py`).',
          'Full-eval outputs as the LOMO job pulls them: `osteo_results_lomo/evals/` (incl. `lomo740h100_all`, the all-data reference',
          're-run with H100 fallback; a later `lomo740_all` full eval would duplicate it).', '']
    if scores:
        L += ['| hold-out | metric | base | sid_full | all | holdout | transfer [95% CI] |', '|---|---|---:|---:|---:|---:|---|']
        for s in scores.values():
            g = lambda k: s.get(k, '–')
            L.append(f"| {g('data_type')} | {g('metric')} | {g('base')} | {g('sid_full')} | {g('all')} | {g('holdout')} | "
                     f"{g('transfer')} [{g('transfer_lo')}, {g('transfer_hi')}] |")
    else:
        L.append('_No hold-out scores yet (`~/osteo-results/lomo/lomo_scores.json` not written; `preview_noflow_scores.json` is an early preview it supersedes)._')
    L.append('')
    L.append('Local mirror of `~/osteo-results/lomo/` (written by the LOMO job):')
    mirror('06_heldout_experiments/osteo_results_lomo')
    table([i for i in items if i['section'][0] == '7'])
    L.extend(['', '## 8. Other baselines  ·  9. Classical ML + ET-SoC-1', '',
              'Mirrored from the jobs running them; partial files are re-copied on every refresh until the runs finish.'])
    for src, rel, d in SYNC:
        if rel.startswith(('08_', '09_')):
            L.extend(['', f'**{rel.split("/")[-1]}**: {d}. Source `{src.replace(H, "~")}` (API key and .venv excluded).'])
            mirror(rel)
    L.extend(['', '## 10. Esperanto ET-SoC-1 (AI Foundry lab, aifoundry3 card 0)', '',
              f'Everything Laura\'s lab account `likelytobelaura@aifoundry3` holds, pulled with rsync on every refresh ({lab}).',
              'The account only reaches aifoundry3 (aifoundry1/2 refuse it), so this is all of it. Not copied: dotfiles',
              '(`~/.claude`, `.claude.json` hold the lab Claude\'s login) and `~/et-soc1-prototyping`, an unmodified clone of',
              f'https://github.com/yaroslavvb/et-soc1-prototyping at `{ETREF}`. `osteo/` is the osteo card job (also in section 9).', '',
              '| folder | files | size | what |', '|---|---:|---:|---|'])
    for name, what in ESPERANTO:
        fs = [f for f in glob.glob(f'{ET}/{name}/**', recursive=True) if os.path.isfile(f)] if os.path.isdir(f'{ET}/{name}') else \
             ([f'{ET}/{name}'] if os.path.exists(f'{ET}/{name}') else [])
        L.append(f"| `{name}` | {len(fs):,} | {human(sum(os.path.getsize(f) for f in fs)) if fs else '–'} | {what} |")
    V = f'{B}/11_modal_osteo_data'
    L.extend(['', '## 11. Modal volume `osteo-data` (full backup)', '',
              'Everything on the volume, same layout, mirrored from `~/osteo-data-volume`. Text files are in git; weights,',
              'optimizer state and `tokenizer.json` are on HF (`hf_volume_backup.py` uploads each adapter folder once its copy',
              'matches the volume\'s file sizes, then checks every file on HF against it).', '',
              '| volume folder | files | size | backed up on |', '|---|---:|---:|---|'])
    groups = collections.defaultdict(list)
    for f in mirrors.get('11_modal_osteo_data', []):
        p = f['path'].split('/')[1:]
        groups['/'.join(p[:3] if p[0] == 'adapters' and p[1] == 'lomo' else p[:2] if p[0] in ('adapters', 'outputs') and len(p) > 2 else p[:1])].append(f)
    for g, fs in sorted(groups.items()):
        where = 'HF (`' + g.split('/')[-1] + '/`) + text in git' if g.startswith('adapters/') else 'GitHub'
        L.append(f"| `{g}` | {len(fs)} | {human(sum(f['bytes'] for f in fs))} | {where} |")
    L.extend(['', '## Not yet filed', ''])
    L.extend([f'- `{f}` (copied to `_incoming/`)' for f in loose] or ['_Nothing: every file under ' + ', '.join(w.replace(H, '~') for w in WATCH) + ' is mirrored here._'])
    L += ['', '## Links', '',
          '- Comparison dashboard: https://claude.ai/artifact/YTL434mENWCv3anqC5EE2J (source in `07_finetune_sid_full/dashboard_src/`)',
          '- Working copies the original scripts still read: `~/.claude/jobs/6c5c300f/tmp/osteo/` (temporary; this folder is the durable copy).', '']
    open(f'{B}/INVENTORY.md', 'w').write('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
