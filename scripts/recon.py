import sys,glob,os,re,csv,collections
import pandas as pd
from common import *
from save_truth import save
train,evalt=load()
S=D+'/src'
def mod_of(prefix): return [m for m,_ in train if m.startswith(prefix)][0]
def check_save(prefix,label,cands,emap=None):
    """cands: {cand_name:{item:val}}; emap: optional fixed entity->cand name, else best by score."""
    mod=mod_of(prefix); ents=sorted({e for m,e in list(train)+list(evalt) if m==mod})
    best={};tot_ok=tot_n=0;unver=[]
    for e in ents:
        if emap: c=emap(e); sc=score(train,mod,e,cands.get(c,{})) if c in cands else (0,0,0)
        else: c,sc=max(((c,score(train,mod,e,d)) for c,d in cands.items()),key=lambda x:x[1][0])
        ok,n,k=sc; tot_ok+=ok; tot_n+=k
        flag='' if k and ok>=k-1 else '  <-- imperfect'
        print(f'  {e:45s} -> {c}: {ok}/{n} matched of {k} known{flag}')
        if c in cands and ((k and ok/k>0.99) or (k==0 and emap)): best[e]=cands[c]
        if k==0 and emap and c in cands: unver.append(e)
    print(f'{label}: overall {tot_ok}/{tot_n} known values reproduced; eval-only entities name-matched (unverified): {len(unver)}')
    save(mod,evalt,best,label)
def tcr():
    c={}
    df=pd.read_csv(S+'/tcr/complete_TCRs.txt',sep='\t',index_col=0)
    mon={'May':'05','June':'06','July':'07','August':'08','September':'09'}
    for col in df.columns:
        c['tcr_freq_2025-'+mon[col[:-4]]]={k:v for k,v in df[col].items()}
    for f in glob.glob(S+'/tcr/*_all_expanders.tsv'):
        name='tcr_stim_'+os.path.basename(f).replace('_all_expanders.tsv','')
        t=pd.read_csv(f,sep='\t'); d={}
        for col in t.columns:
            lc=col.lower()
            suf=('baseline' if lc.endswith('_baseline') else 'post_stim_total' if lc.endswith('post_stim_total') else
                 'post_stim_N' if lc.endswith('post_stim_n') else 'post_stim_P' if lc.endswith('post_stim_p') else None)
            if suf:
                for tr,v in zip(t['TRB'],t[col]): d[f'{tr}|{suf}']=v
        c[name]=d
    check_save('T-cell','tcr',c,emap=lambda e:e)
def hla():
    t=pd.read_csv(S+'/dragen/hla.tsv',sep='\t'); c=collections.defaultdict(dict); cnt=collections.Counter()
    for r in t.itertuples():
        e=f'dragen_{r.pair}_{r.sample}'; cnt[(e,r.gene)]+=1
        c[e][f'HLA-{r.gene}_{cnt[(e,r.gene)]}']=r.allele
    o=pd.read_csv(S+'/dragen/optitype_T3.tsv',sep='\t').iloc[0]
    c['optitype_T3']={f'HLA-{g}_{n}':o[f'{g}{n}'] for g in 'ABC' for n in (1,2)}
    check_save('Germline','hla',dict(c),emap=lambda e:e)
def dia():
    t=pd.read_csv(S+'/dia/T0_B1_pg_matrix.tsv',sep='\t'); key=t['Protein.Group']+'|'+t['Genes'].fillna('')
    c={col:{k:v for k,v in zip(key,t[col]) if pd.notna(v)} for col in t.columns[4:]}
    check_save('DIA','dia',c,emap=lambda e:e)
def mpa():
    c={}
    for f in glob.glob(S+'/mpa/*.csv'):
        m=re.search(r'_(Saliva|Stool)_(\d+)\.(\d+)\.(\d+)_',f); e=f'{m[1].lower()}_{m[4]}-{int(m[2]):02d}-{int(m[3]):02d}'
        t=pd.read_csv(f); val=t.columns[-1]
        c[e]={f'SGB{s}':v for s,v in zip(t['SGB_ID'],t[val])}
    mod=mod_of('Shotgun')
    # items look like 'SGB6014|s__X'; key on SGB id
    for k in [k for k in train if k[0]==mod]: train[k]={i.split('|')[0]:v for i,v in train[k].items()}
    ev={k:[(p,i,it.split('|')[0]) for p,i,it in v] for k,v in evalt.items() if k[0]==mod}
    evalt.update(ev)
    check_save('Shotgun','mpa',c,emap=lambda e:e)
def fus():
    c={}
    for f in glob.glob(S+'/fus/*.tsv'):
        t=pd.read_csv(f,sep='\t'); tp,kind=os.path.basename(f).split('.')[:2]
        c[f'{tp}_tumor_ont/{kind}']={f'{a}|{b}|{d}':float(n) for a,b,d,n in zip(t['#FusionName'],t['LeftBreakpoint'],t['RightBreakpoint'],t['num_LR'])}
    check_save('Gene fusions','fusion',c)
def somatic():
    dd=S+'/dragen/'; c=collections.defaultdict(dict); pairs=['personalis','ucla_T1','ucla_T1_organoid','ucla_T2']
    m=pd.read_csv(dd+'00_metrics.tsv',sep='\t')
    for r in m.to_dict('records'):
        for k,v in r.items():
            if k!='pair': c[r['pair']][f'METRIC|{k}']=v
    L=pd.read_csv(dd+'longitudinal.tsv',sep='\t')
    for r in L.to_dict('records'):
        for p in pairs: c[p][f"PRESENT|{r['source']}|{r['gene']}|{r['variant_id']}"]=r[p]
    for r in pd.read_csv(dd+'snv_top.tsv',sep='\t').to_dict('records'):
        k=f"SNV|{r['gene']}|{r['chrom']}:{r['pos']}:{r['ref']}>{r['alt']}"
        for f in ['tumor_vaf','tumor_dp','normal_vaf','normal_dp']: c[r['pair']][f'{k}|{f}']=r[f]
    for r in pd.read_csv(dd+'cnv_top.tsv',sep='\t').to_dict('records'):
        for f in ['copy_number','minor_cn','seg_mean']: c[r['pair']][f"CNV|{r['gene']}|{r['seg_id']}|{f}"]=r[f]
    for r in pd.read_csv(dd+'sv_top.tsv',sep='\t').to_dict('records'):
        for f in ['tumor_pr','tumor_sr','tumor_vaf']: c[r['pair']][f"SV|{r['gene_a']}|{r['sv_id']}|{f}"]=r[f]
    check_save('Somatic','somatic',dict(c),emap=lambda e:e)

def flow():
    x=pd.read_excel(S+'/flow/combined_data_all.xlsx',header=None); hdr=x.iloc[1]; c=collections.defaultdict(dict)
    for i in range(2,len(x)):
        d=x.iloc[i,0]
        if not hasattr(d,'strftime'): continue
        e='pbmc_'+d.strftime('%Y-%m-%d')
        for j in range(5,x.shape[1]):
            h,v=hdr[j],x.iloc[i,j]
            if isinstance(h,str) and pd.notna(v) and str(v).strip() not in ('','-'): c[e].setdefault(h.strip(),v)
    check_save('Spectral','flow',dict(c),emap=lambda e:e)
def mif():
    t=pd.read_csv(S+'/mif/SIN-01_protein_marker_predictions.csv.gz'); marks=list(t.columns[4:]); c={}
    for r in t.itertuples(index=False):
        c[f'{r.sample}|cell{r.label}']=dict(zip(marks,r[4:]))
    check_save('Multiplex','mif',c,emap=lambda e:e)
def olink():
    t=pd.read_csv(S+'/olink/20260901_all_proteins.csv'); c=collections.defaultdict(dict)
    for p,v,d in zip(t.protein_name,t.providence_npx,t.date_collected): c['plasma_'+d].setdefault(p,v)
    check_save('Olink','olink',dict(c),emap=lambda e:e)
import h5py,numpy as np
H=D+'/h5'
def _h5(f):
    h=h5py.File(f,'r'); g=h['matrix']
    bc=[b.decode() for b in g['barcodes'][:]]
    ids=[x.decode() for x in g['features/id'][:]]; nm=[x.decode() for x in g['features/name'][:]]
    ft=[x.decode() for x in g['features/feature_type'][:]]
    return h,g,{b:i for i,b in enumerate(bc)},ids,nm,ft
def _col(g,j):
    a,b=g['indptr'][j],g['indptr'][j+1]
    return g['indices'][a:b],g['data'][a:b]
VAL=D+'/validated_files.json'
def percell(prefix,label,ftype,files):
    mod=mod_of(prefix)
    samples=collections.defaultdict(lambda:{'train':[], 'eval':[]})
    for (m,e) in train:
        if m==mod: samples[e.rsplit(':',1)[0]]['train'].append(e)
    for (m,e) in evalt:
        if m==mod: samples[e.rsplit(':',1)[0]]['eval'].append(e)
    idx={f:_h5(f) for f in files}
    val=json.load(open(VAL)) if os.path.exists(VAL) else {}
    def build(f,ents):
        h,g,b2i,ids,nm,ft=idx[f]; out={}
        if ftype not in ft: return out
        for e in ents:
            b=e.rsplit(':',1)[1]
            if b not in b2i: continue
            r,v=_col(g,b2i[b]); vals=dict(zip(r.tolist(),v.tolist()))
            out[e]={f'{ids[i]}|{nm[i]}':vals.get(i,0) for i in range(len(ids)) if ft[i]==ftype}
            out[e].update({ids[i]:vals.get(i,0) for i in range(len(ids)) if ft[i]==ftype and ids[i]==nm[i]})
        return out
    cands={}; rep=[]
    for s_,d in sorted(samples.items()):
        if d['train']:
            tb=[e.rsplit(':',1)[1] for e in d['train']]; best=None
            for f in [f for f,(h,g,b2i,*_) in idx.items() if all(b in b2i for b in tb)]:
                bt=build(f,d['train'])
                o=sum(score(train,mod,e,bt.get(e,{}))[0] for e in d['train']); k=sum(len(train[(mod,e)]) for e in d['train'])
                if best is None or o>best[0]: best=(o,k,f)
            if best and best[1] and best[0]/best[1]>=0.99:
                cands.update(build(best[2],d['eval'])); val[s_]=best[2]; st='OK'
            else: st='REJECTED'
            rep.append((s_,best[0] if best else 0,best[1] if best else 0,st,os.path.basename(best[2])[:70] if best else '-'))
        else:
            f=val.get(s_)
            if f and f in idx:
                b=build(f,d['eval']); cands.update(b)
                rep.append((s_,0,0,'NO TRAIN CELLS; used file validated in other modality' if b else 'NO TRAIN CELLS; validated file lacks '+ftype,os.path.basename(f)[:70]))
            else: rep.append((s_,0,0,'NO TRAIN CELLS; unrecoverable','-'))
    json.dump(val,open(VAL,'w'),indent=1)
    for r in rep:
        if r[3]!='OK': print('  %-40s %6d/%-6d %s  %s'%r)
    print('  samples OK: %d / %d'%(sum(r[3]=='OK' for r in rep),len(rep)))
    check_save(prefix,label,cands,emap=lambda e:e)
def _files(pat): return sorted(glob.glob(H+'/'+pat))
def scrna(): percell('Single-cell','scrna','Gene Expression',_files('hudson_lab*')+_files('kamil*')+_files('ucsf*')+_files('T1__*'))
def cite(): percell('CITE','cite','Antibody Capture',_files('hudson_lab*')+_files('kamil*')+_files('ucsf*')+_files('T1__*'))
def xenium(): percell('Xenium','xenium','Gene Expression',_files('xenium*'))
