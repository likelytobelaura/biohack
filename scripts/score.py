import json,re,collections,math,statistics as st
from common import *
train,evalt=load()
def parse(t):
    for m in reversed(list(re.finditer(r'\{[^{}]*\}',t))):
        try: return json.loads(m.group())
        except: pass
    return None
pred={}
for l in open('/Users/laurabocek/osteo-results/baseline_zeroshot_merged.jsonl'):
    r=json.loads(l); pred[r['prompt_id']]=parse(r['text']) or {}
truth=[json.loads(l) for l in open(D+'/truth.jsonl')]
# train-median reference per (modality, item)
mods={m for m,_ in train}
med=collections.defaultdict(list)
for (m,e),kv in train.items():
    for k,v in kv.items():
        try: med[(m[:20],k)].append(float(v))
        except: pass
def f(x):
    try: return float(x)
    except: return None
def rank(a):
    o=sorted(range(len(a)),key=lambda i:a[i]); r=[0]*len(a); i=0
    while i<len(o):
        j=i
        while j+1<len(o) and a[o[j+1]]==a[o[i]]: j+=1
        for k in range(i,j+1): r[o[k]]=(i+j)/2
        i=j+1
    return r
def spear(x,y):
    if len(x)<3: return float('nan')
    rx,ry=rank(x),rank(y); mx,my=st.mean(rx),st.mean(ry)
    num=sum((a-mx)*(b-my) for a,b in zip(rx,ry)); den=math.sqrt(sum((a-mx)**2 for a in rx)*sum((b-my)**2 for b in ry))
    return num/den if den else float('nan')
modname={}
full={}
for (m,e),lst in evalt.items():
    modname[e]=m
    for pid,i,it in lst: full[(pid,i)]=it
by=collections.defaultdict(list)
for t in truth: by[t['mod']].append(t)
rows=[]
CAT={'hla'}; BIN={'mif'}
for lab,ts in sorted(by.items(),key=lambda x:-len(x[1])):
    n=len(ts); ans=0; xs=[];ys=[];ms=[];err=[];merr=[];w10=0;exact=0;mexact=0
    for t in ts:
        p=pred.get(t['prompt_id'],{}).get(str(t['i']))
        mod=modname[t['entity']]
        key=full[(t['prompt_id'],t['i'])]
        mv=med.get((mod[:20],key))
        if p is None: continue
        ans+=1
        if lab in CAT:
            exact+= str(p).strip()==str(t['truth']).strip(); continue
        pv,tv=f(p),f(t['truth'])
        if pv is None or tv is None: continue
        xs.append(pv); ys.append(tv); e=abs(math.asinh(pv)-math.asinh(tv)); err.append(e)
        w10+= abs(pv-tv)<=0.1*abs(tv)+1e-9
        if lab in BIN or (lab=='somatic' and t['item'].startswith('PRESENT')): exact+= round(pv)==round(tv)
        if mv: m_=st.median(mv); merr.append(abs(math.asinh(m_)-math.asinh(tv))); ms.append((m_,tv))
    r={'data_type':lab,'targets':n,'answered':ans}
    if lab in CAT: r['exact_match']=exact/max(ans,1)
    else:
        r['spearman']=spear(xs,ys); r['median_abs_err_asinh']=st.median(err) if err else None
        r['within_10pct']=w10/max(len(xs),1)
        if lab in BIN: r['accuracy']=exact/max(len(xs),1)
        if merr: r['train_median_ref_median_abs_err_asinh']=st.median(merr); r['train_median_ref_spearman']=spear([a for a,b in ms],[b for a,b in ms]); r['ref_coverage']=len(ms)/max(len(xs),1)
    rows.append(r)
tot=sum(r['targets'] for r in rows); ans=sum(r['answered'] for r in rows)
json.dump(rows,open('/Users/laurabocek/osteo-results/baseline_scores.json','w'),indent=1)
print(f'TOTAL targets {tot}, answered {ans} ({ans/tot:.1%})')
print(f"{'type':9s} {'n':>5s} {'ans%':>5s} {'rho':>6s} {'MAE*':>6s} {'<10%':>5s} {'acc':>5s} | {'ref rho':>7s} {'ref MAE*':>8s}")
for r in rows:
    g=lambda k,fmt: (fmt%r[k]) if r.get(k) is not None else '   -'
    print(f"{r['data_type']:9s} {r['targets']:5d} {r['answered']/r['targets']:5.0%} {g('spearman','%6.3f')} {g('median_abs_err_asinh','%6.3f')} {g('within_10pct','%5.2f')} {g('accuracy','%5.2f') if 'accuracy' in r else g('exact_match','%5.2f')} | {g('train_median_ref_spearman','%7.3f')} {g('train_median_ref_median_abs_err_asinh','%8.3f')}")
