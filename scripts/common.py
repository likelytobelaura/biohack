import json,re,collections,pickle,os
D='/Users/laurabocek/.claude/jobs/6c5c300f/tmp/osteo'
def parse_user(u):
    ent=re.search(r'^Entity: (.+)$',u,re.M).group(1).strip()
    known={m.group(1):m.group(2) for m in re.finditer(r'^- (.+?) = (\S+)$',u,re.M)}
    cross={}
    for m in re.finditer(r'^- (.+?): (.+=.+)$',u,re.M):
        for kv in m.group(2).split(', '):
            if '=' in kv: e,v=kv.rsplit('=',1); cross[(e.strip(),m.group(1))]=v
    miss=re.search(r'Missing values to predict:\n(.*?)\n\n',u,re.S)
    items=[re.sub(r'^\d+\.\s*','',l) for l in miss.group(1).splitlines()] if miss else []
    return ent,known,cross,items
def load():
    p=D+'/tables.pkl'
    if os.path.exists(p): return pickle.load(open(p,'rb'))
    train=collections.defaultdict(dict); evalt=collections.defaultdict(list)
    for l in open(D+'/sid_full.jsonl'):
        r=json.loads(l); u=r['prompt'][-1]['content']; mod=u.split('\n')[0]
        ent,known,cross,items=parse_user(u)
        for k,v in known.items(): train[(mod,ent)][k]=v
        for (e,k),v in cross.items(): train[(mod,e)].setdefault(k,v)
        ans=json.loads(r['completion'][0]['content'])
        for i,it in enumerate(items,1):
            if str(i) in ans: train[(mod,ent)][it]=ans[str(i)]
    for l in open(D+'/baseline.jsonl'):
        r=json.loads(l); u=r['messages'][-1]['content']; mod=u.split('\n')[0]
        ent,_,_,items=parse_user(u)
        for i,it in enumerate(items,1): evalt[(mod,ent)].append((r['prompt_id'],i,it))
    out=(dict(train),dict(evalt)); pickle.dump(out,open(p,'wb')); return out
def close(a,b):
    try: a=float(a); b=float(b)
    except: return str(a).strip()==str(b).strip()
    return abs(a-b)<=1e-3*max(abs(a),abs(b))+1e-4
def score(train,mod,ent,src):
    kv=train.get((mod,ent),{}); n=ok=0
    for k,v in kv.items():
        if k in src: n+=1; ok+=close(v,src[k])
    return ok,n,len(kv)
