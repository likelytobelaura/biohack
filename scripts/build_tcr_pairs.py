"""Matched TCR prompt pairs for the TCR-rule neuron hunt (plans A + D, tcr_neuron.py).

python build_tcr_pairs.py <rows.jsonl> <out.jsonl> [--max-pairs N] [--per-row K]
Rows are training-format ({prompt, completion}, optional row_id), e.g. lomo/splits/lomo740_all_unseen_rows.jsonl.

Each pair is one target clone k in one prompt, in two versions that differ only in that clone's
"missing features elsewhere" line:
  a = rule fires: every other month shown is 0
  b = rule off:   one month is nonzero
kind zero2nz: the real line is all zero (a is real, b edits the first month to a nonzero value)
kind nz2zero: the real line has exactly one nonzero month (b is real, a sets it to 0)
Both versions are teacher-forced with the true answers for items < k and end right before item k's value,
so the next token is the decision: '0' (absent) vs a nonzero digit (present).
Token ids are computed here with the run's tokenizer; a causal model gives identical activations before the
edit, and positions after it are aligned by the common suffix (stored as suffix_len).
"""
import argparse, json, os, random, re
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument('rows'); ap.add_argument('out')
ap.add_argument('--max-pairs', type=int, default=1200); ap.add_argument('--per-row', type=int, default=4)
args = ap.parse_args()
tok = AutoTokenizer.from_pretrained(os.path.expanduser('~/osteo-results/lomo/adapters/lomo740_all'))
DIGIT0 = tok.convert_tokens_to_ids('0')
LINE = re.compile(r'^- (\S+): (.+)$', re.M)


def months(s):
    return [kv.rsplit('=', 1) for kv in s.split(', ')]


def render(prompt, user, completion_prefix):
    msgs = prompt[:-1] + [{'role': 'user', 'content': user}]
    head = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(msgs + [{'role': 'assistant', 'content': completion_prefix}], tokenize=False)
    start = full.index('<|message|>', len(head)) + len('<|message|>')
    return full[:start] + completion_prefix


def encode(text, marks):
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc['input_ids'], enc['offset_mapping']
    pos = {}
    for name, ch in marks.items():  # token covering the char just before `ch` (end of that span)
        pos[name] = max(i for i, (s, e) in enumerate(offs) if s < ch)
    return ids, pos


rows = [json.loads(l) for l in open(args.rows)]
rows = [r for r in rows if r['prompt'][-1]['content'].startswith('T-cell receptor beta clones')]
rng = random.Random(0)
rng.shuffle(rows)
pairs = []
for r in rows:
    user, comp = r['prompt'][-1]['content'], r['completion'][0]['content']
    ans = json.loads(comp)
    items = [re.sub(r'^\d+\.\s*', '', l) for l in
             re.search(r'Missing values to predict:\n(.*?)\n\n', user, re.S).group(1).splitlines()]
    lines = {m.group(1): m for m in LINE.finditer(user) if '=' in m.group(2)}
    nz_by_month = {}
    for m in lines.values():
        for mo, v in months(m.group(2)):
            if float(v) != 0:
                nz_by_month.setdefault(mo, v)
    cands = []
    for k, clone in enumerate(items, 1):
        m = lines.get(clone)
        if m is None or str(k) not in ans:
            continue
        kv = months(m.group(2))
        nz = [i for i, (_, v) in enumerate(kv) if float(v) != 0]
        if not nz:
            mo = kv[0][0]
            edited = list(kv); edited[0] = [mo, nz_by_month.get(mo, '1.294e-05')]
            cands.append((k, clone, m, 'zero2nz', kv, edited))
        elif len(nz) == 1:
            edited = list(kv); edited[nz[0]] = [kv[nz[0]][0], '0']
            cands.append((k, clone, m, 'nz2zero', edited, kv))
    rng.shuffle(cands)
    for k, clone, m, kind, kv_a, kv_b in cands[:args.per_row]:
        cprefix = comp[:comp.index(f'"{k}": ') + len(f'"{k}": ')]
        out = {'row_id': r.get('row_id'), 'k': k, 'clone': clone, 'kind': kind, 'truth': ans[str(k)],
               'present': float(ans[str(k)]) != 0}
        for ver, kv in (('a', kv_a), ('b', kv_b)):
            line = f'- {clone}: ' + ', '.join(f'{mo}={v}' for mo, v in kv)
            u = user[:m.start()] + line + user[m.end():]
            text = render(r['prompt'], u, cprefix)
            miss = re.search(rf'^{k}\. {re.escape(clone)}$', u, re.M)
            off = len(text) - len(cprefix) - len(u)  # user text sits inside the rendered prompt
            off = text.index(u[:200]) if off < 0 or text[off:off + len(u)] != u else off
            ids, pos = encode(text, {'edit_line': off + m.start() + len(line), 'missing_item': off + miss.end(),
                                     'answer': len(text)})
            assert pos['answer'] == len(ids) - 1
            check = tok(text + '0', add_special_tokens=False)['input_ids']
            assert check[:-1] == ids and check[-1] == DIGIT0, 'decision token is not a bare digit after the prefix'
            out[f'ids_{ver}'], out[f'pos_{ver}'] = ids, pos
        a, b = out['ids_a'], out['ids_b']
        pre = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
        suf = next(i for i, (x, y) in enumerate(zip(a[::-1], b[::-1])) if x != y)
        assert pre + suf <= min(len(a), len(b)) and out["pos_a"]["edit_line"] >= pre - 1, (pre, suf, len(a), len(b), out["pos_a"], kind, tok.decode(a[pre-5:len(a)-suf+5]), tok.decode(b[pre-5:len(b)-suf+5]))
        out['prefix_len'], out['suffix_len'] = pre, suf
        pairs.append(out)
    if len(pairs) >= args.max_pairs:
        break

pairs = pairs[:args.max_pairs]
with open(args.out, 'w') as f:
    for p in pairs:
        f.write(json.dumps(p) + '\n')
kinds = {k: sum(p['kind'] == k for p in pairs) for k in ('zero2nz', 'nz2zero')}
print(f'{len(pairs)} pairs from {len({p["row_id"] for p in pairs})} rows; {kinds}; '
      f'present among zero2nz: {sum(p["present"] for p in pairs if p["kind"] == "zero2nz")}/{kinds["zero2nz"]}; '
      f'max len {max(max(len(p["ids_a"]), len(p["ids_b"])) for p in pairs)}')
