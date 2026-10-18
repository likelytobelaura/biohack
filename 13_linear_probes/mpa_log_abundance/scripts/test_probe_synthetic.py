# synthetic check of mpa_probe.probe: y planted in residual dim 3 at layer 2 and MLP neuron 7 at layer 1 (ft only)
# python test_probe_synthetic.py <scratch dir, wiped> <examples.jsonl>   (torch + numpy + modal; CPU, a few minutes)
import json, os, sys, shutil
import numpy as np
root = sys.argv[1]; tag = 't'
shutil.rmtree(root, ignore_errors=True)
os.makedirs(f'{root}/mpa_probe/{tag}'); od = f'{root}/outputs/mpa_probe/{tag}'; os.makedirs(od)
src = [json.loads(l) for l in open(sys.argv[2])]
with open(f'{root}/mpa_probe/{tag}/examples.jsonl', 'w') as f:
    for p in src: f.write(json.dumps(p) + '\n')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['MPA_ROOT'] = root
import mpa_probe as M
rng = np.random.default_rng(1)
for k in M.KINDS:
    mk = M.flat_marks(src, k); y = np.array([m['y'] for m in mk]); N = len(y)
    for mo in M.MODELS:
        X = rng.normal(size=(N, 4, 40)).astype(np.float16)
        X[:, 2, 3] = (y - y.mean()) / y.std() + 0.3 * rng.normal(size=N)
        np.save(f'{od}/resid_{mo}_{k}.npy', X)
for k in M.MLP_KINDS:
    mk = M.flat_marks(src, k if k == 'answer' else 'known_name_mlp'); y = np.array([m['y'] for m in mk]); N = len(y)
    for mo in M.MODELS:
        A = rng.normal(size=(N, 3, 60)).astype(np.float16)
        if mo == 'ft': A[:, 1, 7] = y + 0.2 * rng.normal(size=N)
        np.save(f'{od}/mlp_{mo}_{k}.npy', A)
print(M.probe.local(tag))
r = json.load(open(f'{od}/results.json'))
for key, c in r['probes'].items():
    print(key, [round(x['r2'], 2) for x in c], 'resid', [round(x['resid_r2'], 2) for x in c])
print(r['controls'])
print('top ft/answer/y', r['neurons']['ft/answer/y']['top'][:2]); print('top base/answer/y', r['neurons']['base/answer/y']['top'][:1])
print(r['sparse'])
