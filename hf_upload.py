"""Upload every LoRA adapter in ~/biohack to the private HF repo likelytobelaura/osteo-glimmer-lora.

~/biohack/.venv/bin/python hf_upload.py          # upload anything new or changed (safe to re-run)
Token: $HF_TOKEN, else Keychain item `hf-likelytobelaura` (the saved stationhouse token is never used).
"""
import os, sys, glob, subprocess
from huggingface_hub import HfApi

B = os.path.expanduser('~/biohack')
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = 'likelytobelaura/osteo-glimmer-lora'
# local adapter dir -> folder in the HF repo
ADAPTERS = {f'{B}/07_finetune_sid_full/adapter': 'sid_full'}
ADAPTERS.update({d: os.path.basename(d) for d in sorted(glob.glob(f'{B}/06_heldout_experiments/osteo_results_lomo/adapters/lomo740_*'))})
# tokenizer.json is the base model's own (27 MB), not part of the adapter
IGNORE = ['tokenizer.json', '**/tokenizer.json', '.DS_Store']


def token():
    t = os.environ.get('HF_TOKEN') or subprocess.run(['security', 'find-generic-password', '-s', 'hf-likelytobelaura', '-w'],
                                                       capture_output=True, text=True).stdout.strip()
    if not t:
        sys.exit('no HF_TOKEN and no Keychain item hf-likelytobelaura')
    return t


def done(d):
    # a run's adapter is complete once its weights and config are both on disk
    return os.path.exists(f'{d}/adapter_model.safetensors') and os.path.exists(f'{d}/adapter_config.json')


if __name__ == '__main__':
    api = HfApi(token=token())
    who = api.whoami()['name']
    assert who == 'likelytobelaura', f'token belongs to {who}, expected likelytobelaura'
    api.create_repo(REPO, private=True, exist_ok=True)
    assert api.repo_info(REPO).private, f'{REPO} is not private; refusing to upload'
    api.upload_file(path_or_fileobj=f'{HERE}/hf_model_card.md', path_in_repo='README.md', repo_id=REPO,
                    commit_message='Update model card')
    for d, name in ADAPTERS.items():
        if not done(d):
            print(f'{name}: not finished, skipped'); continue
        api.upload_folder(folder_path=d, path_in_repo=name, repo_id=REPO, ignore_patterns=IGNORE,
                          commit_message=f'Add/update {name} adapter')
        print(f'{name}: uploaded')
    print('files:', len(api.list_repo_files(REPO)), f'https://huggingface.co/{REPO}')
