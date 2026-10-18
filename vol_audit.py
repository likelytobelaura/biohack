"""Is every file on the Modal volume osteo-data backed up on GitHub (11_modal_osteo_data/, origin/inventory) or HF?

~/biohack/.venv/bin/python -I vol_audit.py      # exit 0 only if nothing is missing
Match = same size; HF LFS files must also match the local mirror's sha256 (checked by hf_volume_backup.py).
"""
import json, os, subprocess, sys
from huggingface_hub import HfApi
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hf_volume_backup import adapter_dir, hf_name, MODAL

B = os.path.expanduser('~/biohack')


def volume_sizes():
    sizes, todo = {}, ['/']
    while todo:
        out = subprocess.run([MODAL, 'volume', 'ls', 'osteo-data', todo.pop(), '--json'], capture_output=True, text=True, check=True).stdout
        for e in json.loads(out):
            (todo.append(e['filename']) if 'dir' in str(e['type']).lower() else sizes.__setitem__(e['filename'], e['size']))
    return sizes


subprocess.run(['git', '-C', B, 'fetch', '-q', 'origin'], check=True)
tree = subprocess.run(['git', '-C', B, 'ls-tree', '-r', '-l', 'origin/inventory', '11_modal_osteo_data/'],
                      capture_output=True, text=True, check=True).stdout
git = {l.split(None, 4)[4].strip()[len('11_modal_osteo_data/'):]: int(l.split(None, 4)[3]) for l in tree.splitlines()}
hf = {s.rfilename: s.size for s in HfApi().model_info('likelytobelaura/osteo-glimmer-lora', files_metadata=True).siblings}
vol = volume_sizes()
gaps, where = [], {'GitHub': 0, 'HF': 0, 'both': 0}
for p, s in sorted(vol.items()):
    in_git = git.get(p) == s
    in_hf = p.startswith('adapters/') and hf.get(f'{hf_name(adapter_dir(p))}/{os.path.relpath(p, adapter_dir(p))}') == s
    if in_git and in_hf:
        where['both'] += 1
    elif in_git or in_hf:
        where['GitHub' if in_git else 'HF'] += 1
    else:
        gaps.append((p, s))
print(f'volume: {len(vol)} files, {sum(vol.values())/1e9:.2f} GB | GitHub only {where["GitHub"]}, HF only {where["HF"]}, both {where["both"]}, MISSING {len(gaps)}')
for p, s in gaps:
    print(f'  missing: {p} ({s/1e6:.1f} MB)')
sys.exit(1 if gaps else 0)
