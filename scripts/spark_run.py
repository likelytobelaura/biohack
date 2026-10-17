# Zero-shot baseline on Meta Model API (Muse Spark) over the saved eval prompts; resumable.
import json, os, sys, time, random, threading, subprocess, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

MODEL = sys.argv[1]                      # e.g. muse-spark-1.3-contributor
PROMPTS = os.path.expanduser('~/osteo-golden/golden.jsonl')
OUT = os.path.expanduser(f'~/osteo-results/spark/{MODEL}.jsonl')
RPM = int(os.environ.get('RPM', 140))    # contributor limit is 150/min per model
LIMIT = int(os.environ.get('LIMIT', 0))  # >0: only the first N prompts (smoke test)
URL = 'https://api.meta.ai/v1/chat/completions'
KEY = os.environ.get('MODEL_API_KEY') or subprocess.run(
    ['security', 'find-generic-password', '-s', 'meta-api-key', '-w'], capture_output=True, text=True).stdout.strip()
assert KEY, 'no MODEL_API_KEY and no Keychain item meta-api-key'

rows = [json.loads(l) for l in open(PROMPTS)]
rows = rows[:LIMIT] if LIMIT else rows
done = set()
if os.path.exists(OUT):
    done = {json.loads(l)['prompt_id'] for l in open(OUT)}
todo = [r for r in rows if r['prompt_id'] not in done]
print(f'{MODEL}: {len(done)} done, {len(todo)} to go', flush=True)

lock = threading.Lock(); gap = 60.0 / RPM; nxt = [time.monotonic()]
def throttle():
    with lock:
        now = time.monotonic(); t = max(now, nxt[0]); nxt[0] = t + gap
    time.sleep(max(0, t - now))

def call(messages, max_tokens):
    body = json.dumps({'model': MODEL, 'messages': messages, 'temperature': 0, 'seed': 0,
                       'reasoning_effort': 'low', 'max_completion_tokens': max_tokens}).encode()
    for attempt in range(8):
        throttle()
        req = urllib.request.Request(URL, body, {'Authorization': f'Bearer {KEY}', 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors='replace')[:300]
            if e.code in (429, 500, 503):
                ra = e.headers.get('Retry-After')
                time.sleep(float(ra) if ra else min(60, 2 ** attempt) + random.random()); continue
            raise RuntimeError(f'HTTP {e.code}: {msg}')
        except (urllib.error.URLError, TimeoutError) as e:
            time.sleep(min(60, 2 ** attempt) + random.random())
    raise RuntimeError('gave up after retries')

wlock = threading.Lock(); n = [0]; err = [0]
def work(r):
    try:
        res = call(r['messages'], 8192); ch = res['choices'][0]; retried = False
        if ch['finish_reason'] == 'length':   # same rule as the Glimmer baseline: retry cut-offs with a bigger budget
            res = call(r['messages'], 16384); ch = res['choices'][0]; retried = True
        out = {'prompt_id': r['prompt_id'], 'text': ch['message'].get('content') or '', 'finish': ch['finish_reason'],
               'usage': res.get('usage'), 'retried': retried}
    except Exception as e:
        with wlock: err[0] += 1
        print(f"error {r['prompt_id']}: {e}", flush=True); return
    with wlock:
        with open(OUT, 'a') as f: f.write(json.dumps(out) + '\n')
        n[0] += 1
        if n[0] % 250 == 0: print(f'{MODEL}: {len(done) + n[0]}/{len(rows)} ({err[0]} errors)', flush=True)

with ThreadPoolExecutor(32) as ex:
    list(ex.map(work, todo))
print(f'{MODEL}: finished, {len(done) + n[0]}/{len(rows)} written, {err[0]} errors (rerun to retry them)', flush=True)
