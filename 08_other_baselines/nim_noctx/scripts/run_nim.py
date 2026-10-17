"""Run prompts through meta/muse-glimmer-30b on NVIDIA's hosted API (resumable; same decoding as the Modal baseline)."""
import json, os, sys, time, threading, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

URL = "https://integrate.api.nvidia.com/v1/chat/completions"
KEY = open(os.path.expanduser("~/osteo-nim/.nvkey")).read().strip()
src, dst, workers = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 8
lock = threading.Lock()


def call(messages, max_tokens):
    body = json.dumps({"model": "meta/muse-glimmer-30b", "messages": messages, "temperature": 0.0,
                       "max_tokens": max_tokens}).encode()
    for attempt in range(12):
        req = urllib.request.Request(URL, body, {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            msg = e.read()[:300]
            if e.code not in (429, 500, 502, 503, 504):
                raise RuntimeError(f"HTTP {e.code}: {msg}")
        except Exception as e:
            msg = repr(e)
        time.sleep(min(120, 5 * 2 ** attempt))
    raise RuntimeError(f"gave up: {msg}")


def run(r):
    try:
        d = call(r["messages"], 2048)
        ch = d["choices"][0]
        res = {"prompt_id": r["prompt_id"], "holdout": r["holdout"], "finish": ch["finish_reason"]}
        if ch["finish_reason"] == "length":
            ch = call(r["messages"], 8192)["choices"][0]
            res.update(finish=ch["finish_reason"], retried=True)
        m = ch["message"]
        res.update(text=m.get("content") or "", reasoning=m.get("reasoning_content") or "")
    except Exception as e:
        res = {"prompt_id": r["prompt_id"], "holdout": r["holdout"], "error": str(e)}
    with lock, open(dst, "a") as f:
        f.write(json.dumps(res) + "\n")
    return res


done = set()
if os.path.exists(dst):
    for l in open(dst):
        x = json.loads(l)
        if "error" not in x:
            done.add(x["prompt_id"])
todo = [r for r in map(json.loads, open(src)) if r["prompt_id"] not in done]
print(f"{len(done)} done, {len(todo)} to run, {workers} workers", flush=True)
t0, n, err = time.time(), 0, 0
with ThreadPoolExecutor(workers) as ex:
    for res in ex.map(run, todo):
        n += 1; err += "error" in res
        if n % 10 == 0 or "error" in res:
            print(f"{n}/{len(todo)} err={err} {time.time() - t0:.0f}s {res.get('error', '')[:200]}", flush=True)
print("finished", n, "errors", err, flush=True)
