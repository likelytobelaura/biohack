import gzip,glob,os
from common import *
train,evalt=load()
mod=[m for m,_ in train if m.startswith('Bulk tumour RNA')][0]
srcs={}
for f in glob.glob(D+'/src/rna/*.gct.gz'):
    d={}
    for i,l in enumerate(gzip.open(f,'rt')):
        if i<3: continue
        a=l.rstrip('\n').split('\t'); d[a[0]+'|'+a[1]]=a[2]
    srcs[os.path.basename(f).split('.')[0]]=d
ents=sorted({e for m,e in list(train)+list(evalt) if m==mod})
for e in ents:
    print(e, [(s,)+score(train,mod,e,d) for s,d in srcs.items()])
best={e:max(srcs.values(),key=lambda d:score(train,mod,e,d)[0]) for e in ents}
from save_truth import save
save(mod,evalt,best,'rna')
