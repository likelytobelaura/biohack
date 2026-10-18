import json,csv,re,glob,math,collections,random
src=open('build_snapshots.py').read()
ANT=eval(src[src.index('ANTIGEN = {')+10:src.index('}',src.index('ANTIGEN = {'))+1]); ANT['CD29']='ITGB1'
def gene_of(item):
    n=item.split('|')[-1]; n=re.sub(r'^(Hu|HuMs|HuMsRt)\.','',n)
    for k in sorted(ANT,key=len,reverse=True):
        if n==k or n.startswith(k+'_'): return ANT[k]
    return None
P={}
for l in open('out/eval_snap.jsonl'):
    x=json.loads(l)
    if x['data_type']=='cite': P[x['prompt_id'],x['cond']]=x['messages'][1]['content']
lab=collections.defaultdict(dict)
for r in csv.DictReader(open('out/eval_snap_labels.csv')):
    if r['data_type']=='cite': lab[r['prompt_id'],r['cond']][r['i']]=(r['item'],float(r['truth']))
def preds(m):
    out={}
    for f in glob.glob(f'evals/snapeval_{m}/*.jsonl')+glob.glob(f'evals/xtype/snapeval_{m}/*.jsonl'):
        for l in open(f):
            x=json.loads(l); mm=re.search(r'\{.*\}',x['text'],re.S)
            try: out[x['prompt_id'],x['cond']]=json.loads(mm.group(0))
            except Exception: out[x['prompt_id'],x['cond']]={}
    return out
def rk(v):
    o=sorted(range(len(v)),key=lambda i:v[i]);r=[0]*len(v);i=0
    while i<len(o):
        j=i
        while j+1<len(o) and v[o[j+1]]==v[o[i]]: j+=1
        for k in range(i,j+1): r[o[k]]=(i+j)/2
        i=j+1
    return r
def spear(a,b):
    ra,rb=rk(a),rk(b);ma,mb=sum(ra)/len(ra),sum(rb)/len(rb)
    sa=math.sqrt(sum((x-ma)**2 for x in ra));sb=math.sqrt(sum((x-mb)**2 for x in rb))
    return float('nan') if sa*sb==0 else sum((x-ma)*(y-mb) for x,y in zip(ra,rb))/(sa*sb)
def rna_of(u):
    if 'RNA of THIS cell' not in u: return {}
    sec=u[u.find('RNA of THIS cell'):].split('\n\n')[0]
    return {m.group(1):float(m.group(2)) for m in re.finditer(r'- ENSG\d+\|(\S+) = (\S+)',sec)}
rows=[]
for (pid,c),items in lab.items():
    if c!='full': continue
    rna=rna_of(P[pid,'full']); rsw=rna_of(P[pid,'swap'])
    for i,(item,tr) in items.items():
        g=gene_of(item)
        if g and g in rna: rows.append((pid,i,item,g,rna[g],tr,rsw.get(g)))
tot=sum(len(v) for (p,c),v in lab.items() if c=='full')
print(f'CITE targets with their own antigen mRNA in the same-cell context: {len(rows)} of {tot}; mRNA nonzero in {sum(r[4]>0 for r in rows)}')
print('ceiling: rho(mRNA count, true ADT) = %.3f'%spear([r[4] for r in rows],[r[5] for r in rows]))
random.seed(0)
for m in ['lomo740_all','lomo740_all_snap']:
    p=preds(m); S=[]
    for pid,i,item,g,mr,tr,_ in rows:
        try: a=float(p[pid,'full'][i]); b=float(p[pid,'own'][i])
        except Exception: continue
        S.append((mr,a,b,tr))
    shift=[a-b for mr,a,b,tr in S]
    r_shift=spear([s[0] for s in S],shift)
    bs=[]
    for _ in range(500):
        smp=[random.choice(S) for _ in S]; bs.append(spear([s[0] for s in smp],[s[1]-s[2] for s in smp]))
    bs=sorted(x for x in bs if x==x)
    print(f"{m}: n={len(S)} full rho {spear([s[1] for s in S],[s[3] for s in S]):.3f} | own rho {spear([s[2] for s in S],[s[3] for s in S]):.3f} | "
          f"preds changed {sum(abs(x)>0 for x in shift)/len(S):.0%} | rho(mRNA, shift) {r_shift:.3f} [{bs[12]:.3f},{bs[487]:.3f}]")
