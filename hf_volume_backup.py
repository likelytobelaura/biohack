"""Back up every adapter on the Modal volume osteo-data to HF, from the local mirror ~/osteo-data-volume.

~/biohack/.venv/bin/python hf_volume_backup.py     # re-runnable; uploads only adapters whose mirror matches the volume
An adapter is uploaded only when every one of its files exists locally with the size the volume reports and its
.safetensors are not truncated; afterwards each file on HF is checked against the local copy (sha256 for LFS files).
PEFT's per-adapter README.md is not uploaded (the Hub rejects its local base_model path); it is in git instead.
"""
import glob, hashlib, json, os, subprocess, sys
from huggingface_hub import HfApi
from hf_upload import REPO, complete, token

V = os.path.expanduser('~/osteo-data-volume')
MODAL = os.path.expanduser('~/sutro-problems/.venv/bin/modal')
SKIP = ('README.md', '.DS_Store')


def volume_sizes():
    sizes, todo = {}, ['/adapters']
    while todo:
        out = subprocess.run([MODAL, 'volume', 'ls', 'osteo-data', todo.pop(), '--json'], capture_output=True, text=True, check=True).stdout
        for e in json.loads(out):
            if 'dir' in str(e['type']).lower():
                todo.append(e['filename'])
            else:
                sizes[e['filename']] = e['size']
    return sizes


def adapter_dir(p):
    # the adapter a file belongs to: its folder, or the folder above a checkpoint-N/
    d = os.path.dirname(p)
    return os.path.dirname(d) if os.path.basename(d).startswith('checkpoint-') else d


def hf_name(d):
    # adapters/sid_full -> sid_full, adapters/lomo/lomo740_rna -> lomo740_rna, adapters/lomo/smoke_all -> smoke_all
    return os.path.basename(d)


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


if __name__ == '__main__':
    api = HfApi(token=token())
    assert api.whoami()['name'] == 'likelytobelaura'
    sizes = volume_sizes()
    dirs = sorted({adapter_dir(p) for p in sizes})
    ok = True
    for d in dirs:
        name = hf_name(d)
        want = {p: s for p, s in sizes.items() if p.startswith(d + '/') and os.path.basename(p) not in SKIP}
        bad = [p for p, s in want.items() if not os.path.exists(f'{V}/{p}') or os.path.getsize(f'{V}/{p}') != s
               or (p.endswith('.safetensors') and not complete(f'{V}/{p}'))]
        if bad:
            print(f'{name}: mirror not complete ({len(bad)} of {len(want)} files), skipped'); ok = False; continue
        api.upload_folder(folder_path=f'{V}/{d}', path_in_repo=name, repo_id=REPO, ignore_patterns=list(SKIP) + [f'**/{s}' for s in SKIP],
                          commit_message=f'Back up {name} from Modal osteo-data:{d}')
        remote = {s.rfilename: s for s in api.model_info(REPO, files_metadata=True).siblings}
        miss = []
        for p in want:
            r = remote.get(f'{name}/{os.path.relpath(p, d)}')
            if r is None or r.size != want[p] or (r.lfs and r.lfs.sha256 != sha256(f'{V}/{p}')):
                miss.append(p)
        print(f'{name}: {len(want) - len(miss)}/{len(want)} files verified on HF' + (f'; MISMATCH {miss[:3]}' if miss else ''))
        ok &= not miss
    info = api.model_info(REPO)
    print(f'HF revision {info.sha} | {len(info.siblings)} files | all adapters backed up and verified: {ok}')
    sys.exit(0 if ok else 1)
