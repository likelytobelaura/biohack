"""Examples for the MetaPhlAn log-abundance probe (mpa_probe.py).

python build_mpa_probe.py <out.jsonl>
Prompts: MetaPhlAn training rows lomo740_all never trained on (lomo/splits) + MetaPhlAn eval prompts with gold
answers (~/osteo-golden). Eval completions are written in the training format (json floats) and teacher-forced.

Each prompt gets char marks (where = 'user' content or 'comp' completion; rule 'before' = last token ending at or
before ch, 'cover' = last token starting before ch). Label y = log10(count + 1):
  known_name  end of the taxon name of a known line, before ' = value' (model has not read the value yet)
  known_value last token of the known value just read (sanity: does it encode the number it read)
  answer      teacher-forced answer position, right before missing item k's value (the decision)
Baseline features per mark: elsewhere = mean log10(v+1) of the taxon's values in other samples listed in the
prompt ("missing features elsewhere" lines; NaN if absent). The taxon prior (out-of-fold) is computed in mpa_probe.py.
"""
import json, math, os, re, sys

H = os.path.expanduser('~')
MPA = 'Shotgun metagenomics'
KNOWN = re.compile(r'^- (\S+) = (\S+)$', re.M)
ELSE = re.compile(r'^- (\S+): (\S+=.*)$', re.M)
ly = lambda v: math.log10(float(v) + 1)


def marks(user, comp, items):
    out = []
    kb = re.search(r'Known values for this entity:\n(.*?)\n\n', user, re.S)
    for m in KNOWN.finditer(user, kb.start(1), kb.end(1)):
        y, taxon = ly(m.group(2)), m.group(1)
        out.append(dict(kind='known_name', where='user', ch=m.end(1), rule='before', y=y, taxon=taxon))
        out.append(dict(kind='known_value', where='user', ch=m.end(2), rule='cover', y=y, taxon=taxon))
    els = {m.group(1): [ly(kv.rsplit('=', 1)[1]) for kv in m.group(2).split(', ')] for m in ELSE.finditer(user)}
    ans = json.loads(comp)
    for k, taxon in items.items():
        m = re.search(r'"%s": ' % k, comp)
        e = els.get(taxon)
        out.append(dict(kind='answer', where='comp', ch=m.end(), rule='before', y=ly(ans[k]), taxon=taxon,
                        elsewhere=sum(e) / len(e) if e else None))
    return out


def items_of(user):
    lines = re.search(r'Missing values to predict:\n(.*?)\n\n', user, re.S).group(1).splitlines()
    return {l.split('. ', 1)[0]: l.split('. ', 1)[1] for l in lines}


prompts = []
for l in open(f'{H}/osteo-results/lomo/splits/lomo740_all_unseen_rows.jsonl'):
    r = json.loads(l)
    user = r['prompt'][-1]['content']
    if user.startswith(MPA):
        comp = r['completion'][0]['content']
        prompts.append(dict(id=f"row{r['row_id']}", src='train_unseen', messages=r['prompt'], comp=comp,
                            items=items_of(user)))
for l in open(f'{H}/osteo-golden/golden.jsonl'):
    r = json.loads(l)
    if r['data_type'] == 'mpa':
        comp = json.dumps({k: float(v) for k, v in r['gold'].items()})
        prompts.append(dict(id=r['prompt_id'], src='eval', messages=r['messages'], comp=comp, items=r['items']))

n = {}
with open(sys.argv[1], 'w') as f:
    for p in prompts:
        user = p['messages'][-1]['content']
        p['entity'] = re.search(r'^Entity: (\S+)$', user, re.M).group(1)
        p['marks'] = marks(user, p['comp'], p.pop('items'))
        for m in p['marks']:
            n[m['kind']] = n.get(m['kind'], 0) + 1
        f.write(json.dumps(p) + '\n')
ents = {p['entity'] for p in prompts}
print(f"{len(prompts)} prompts ({sum(p['src'] == 'eval' for p in prompts)} eval), {len(ents)} entities, marks {n}")
