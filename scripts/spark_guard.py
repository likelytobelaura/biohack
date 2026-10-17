# Budget guard for the Spark runs: every 30s, total logged cost; kill both runners at the cap.
import json, glob, os, time, subprocess, sys
CAP = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
STOP = CAP - 0.5   # margin for up to 64 in-flight requests not yet logged
D = os.path.expanduser('~/osteo-results/spark')
def cost():
    tot = 0.0; n = 0
    for f in glob.glob(D + '/*.jsonl'):
        for l in open(f):
            u = json.loads(l)['usage']; k = (u.get('prompt_tokens_details') or {}).get('cached_tokens', 0)
            tot += ((u['prompt_tokens'] - k) * 0.10 + k * 0.002 + u['completion_tokens'] * 0.20) / 1e6; n += 1
    return tot, n
while True:
    c, n = cost()
    with open(D + '/cost.txt', 'w') as f: f.write(f'{time.strftime("%H:%M:%S")} ${c:.2f} over {n} rows (cap ${CAP:.2f})\n')
    alive = subprocess.run(['pgrep', '-f', 'spark_run.py'], capture_output=True).stdout.strip()
    if c >= STOP:
        subprocess.run(['pkill', '-f', 'spark_run.py'])
        print(f'BUDGET STOP: ${c:.2f} >= ${STOP:.2f}; killed runners', flush=True); break
    if not alive:
        print(f'runners finished; final logged cost ${c:.2f}', flush=True); break
    time.sleep(30)
