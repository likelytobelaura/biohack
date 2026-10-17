import json,re,sys,csv,collections
prompts_p,truth_p,out=sys.argv[1:4]
tr=collections.defaultdict(dict); meta={}
for l in open(truth_p):
    t=json.loads(l); tr[t['prompt_id']][t['i']]=t; meta[t['prompt_id']]=(t['mod'],t['entity'])
def val(s):
    try:
        x=float(s); return int(x) if x.is_integer() and re.fullmatch(r'-?\d+',str(s).strip()) else x
    except: return s
n=full=part=none=items=lab=0; bymod=collections.Counter()
with open(out+'/golden.jsonl','w') as g, open(out+'/labels.csv','w',newline='') as c:
    w=csv.writer(c); w.writerow(['prompt_id','i','data_type','entity','item','truth'])
    for l in open(prompts_p):
        r=json.loads(l); u=r['messages'][-1]['content']; n+=1
        m=re.search(r'Missing values to predict:\n(.*?)\n\n',u,re.S)
        its=[re.sub(r'^\d+\.\s*','',x) for x in m.group(1).splitlines()] if m else []
        t=tr.get(r['prompt_id'],{}); items+=len(its); lab+=len(t)
        if not t: none+=1; continue
        if len(t)==len(its): full+=1
        else: part+=1
        mod,ent=meta[r['prompt_id']]; bymod[mod]+=len(t)
        ans={str(i):val(t[i]['truth']) for i in sorted(t)}
        g.write(json.dumps({'prompt_id':r['prompt_id'],'data_type':mod,'entity':ent,'messages':r['messages'],
            'items':{str(i):x for i,x in enumerate(its,1)},'gold':ans,'n_items':len(its),'n_labelled':len(t),
            'completion':[{'role':'assistant','content':json.dumps(ans)}]})+'\n')
        for i in sorted(t): w.writerow([r['prompt_id'],i,mod,ent,t[i]['item'],t[i]['truth']])
print(json.dumps({'prompts':n,'fully_labelled':full,'partly_labelled':part,'unlabelled':none,'items':items,'labelled_items':lab,'by_type':dict(bymod)},indent=1))
