import json,os
from common import D
P=D+'/truth.jsonl'
def save(mod,evalt,ent2src,label):
    rows=[];miss=0
    for (m,e),lst in evalt.items():
        if m!=mod or e not in ent2src: continue
        src=ent2src[e]
        for pid,i,it in lst:
            if it in src: rows.append({'prompt_id':pid,'i':i,'item':it,'entity':e,'mod':label,'truth':src[it]})
            else: miss+=1
    old=[r for r in map(json.loads,open(P)) if r['mod']!=label] if os.path.exists(P) else []
    with open(P,'w') as f:
        for r in old+rows: f.write(json.dumps(r)+'\n')
    print(label,'saved',len(rows),'missing',miss)
