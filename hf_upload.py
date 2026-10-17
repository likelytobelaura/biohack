"""Upload every LoRA adapter in ~/biohack to the HF repo likelytobelaura/osteo-glimmer-lora.

~/biohack/.venv/bin/python hf_upload.py          # upload anything new or changed (safe to re-run)
Token: $HF_TOKEN, else Keychain item `hf-likelytobelaura` (the saved stationhouse token is never used).
"""
import os, sys, glob, json, subprocess
from huggingface_hub import HfApi

H = os.path.expanduser('~')
B = f'{H}/biohack'
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = 'likelytobelaura/osteo-glimmer-lora'
# local adapter dir -> folder in the HF repo
ADAPTERS = {f'{B}/07_finetune_sid_full/adapter': 'sid_full'}
ADAPTERS.update({d: os.path.basename(d) for d in sorted(glob.glob(f'{B}/06_heldout_experiments/osteo_results_lomo/adapters/lomo740_*'))})
# tokenizer.json is the base model's own (27 MB), not part of the adapter; the per-adapter README.md is PEFT
# boilerplate whose base_model is a local path, which the Hub rejects (the repo-level model card covers it)
IGNORE = ['tokenizer.json', '**/tokenizer.json', 'README.md', '**/README.md', '.DS_Store']


def token():
    t = os.environ.get('HF_TOKEN') or subprocess.run(['security', 'find-generic-password', '-s', 'hf-likelytobelaura', '-w'],
                                                       capture_output=True, text=True).stdout.strip()
    if not t:
        sys.exit('no HF_TOKEN and no Keychain item hf-likelytobelaura')
    return t


def complete(p):
    # a safetensors header records where its last tensor ends; a file caught mid-copy is shorter than that
    with open(p, 'rb') as f:
        n = int.from_bytes(f.read(8), 'little')
        meta = json.loads(f.read(n))
    end = max(v['data_offsets'][1] for k, v in meta.items() if k != '__metadata__')
    return os.path.getsize(p) == 8 + n + end


def done(d):
    # weights + config on disk, weights not truncated, and identical in size to the copy the LOMO job pulled
    w = f'{d}/adapter_model.safetensors'
    if not (os.path.exists(w) and os.path.exists(f'{d}/adapter_config.json')) or not complete(w):
        return False
    src = f'{H}/osteo-results/lomo/adapters/{os.path.basename(d)}/adapter_model.safetensors'
    return not os.path.exists(src) or (complete(src) and os.path.getsize(src) == os.path.getsize(w))


if __name__ == '__main__':
    api = HfApi(token=token())
    who = api.whoami()['name']
    assert who == 'likelytobelaura', f'token belongs to {who}, expected likelytobelaura'
    api.create_repo(REPO, exist_ok=True)  # public since 2026-10-17, by the user's choice
    api.upload_file(path_or_fileobj=f'{HERE}/hf_model_card.md', path_in_repo='README.md', repo_id=REPO,
                    commit_message='Update model card')
    api.upload_file(path_or_fileobj=f'{HERE}/hf_LICENSE', path_in_repo='LICENSE', repo_id=REPO,
                    commit_message='Add base model license (Apache 2.0)')
    for d, name in ADAPTERS.items():
        if not done(d):
            print(f'{name}: not finished or still copying, skipped'); continue
        api.upload_folder(folder_path=d, path_in_repo=name, repo_id=REPO, ignore_patterns=IGNORE,
                          commit_message=f'Add/update {name} adapter')
        print(f'{name}: uploaded')
    print('files:', len(api.list_repo_files(REPO)), f'https://huggingface.co/{REPO}')
