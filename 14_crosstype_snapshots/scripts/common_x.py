"""Shared parsing / timepoint mapping for the cross-type (timepoint snapshot) experiment."""
import csv, json, re, collections

B = '/Users/laurabocek/biohack'
UNSEEN = f'{B}/06_heldout_experiments/osteo_results_lomo/splits/lomo740_all_unseen_rows.jsonl'
SPLITIDS = f'{B}/06_heldout_experiments/osteo_results_lomo/splits/lomo740_all_row_ids.json'
FULL = f'{B}/01_sid_full_dataset/sid_full.jsonl'
GOLD = f'{B}/02_golden_labels/golden.jsonl'
LABELS = f'{B}/02_golden_labels/labels.csv'
H5DIR = f'{B}/01_sid_full_dataset/source/h5/'
VALID = f'{B}/02_golden_labels/validated_files.json'

TYPE = [("scrna", "Single-cell RNA-seq"), ("cite", "CITE-seq antibody tags"), ("rna", "Bulk tumour RNA-seq"),
        ("dia", "DIA mass-spec proteomics"), ("olink", "Olink plasma proteomics"), ("mif", "Multiplex immunofluorescence"),
        ("tcr", "T-cell receptor beta clones"), ("xenium", "Xenium in-situ transcriptomics"), ("somatic", "Somatic tumour/normal WGS"),
        ("mpa", "Shotgun metagenomics"), ("flow", "Spectral flow cytometry"), ("hla", "Germline HLA typing"),
        ("fusion", "Gene fusions from long-read")]


def dtype(user):
    for k, p in TYPE:
        if user.startswith(p):
            return k
    raise ValueError(user[:80])


def parse(user):
    """-> dict(type, desc, entity, known[(name,val_str)], targets[name])"""
    L = user.split('\n')
    out = dict(type=dtype(user), desc=L[0], entity=None, known=[], targets=[])
    sec = None
    for l in L[1:]:
        if l.startswith('Entity: '):
            out['entity'] = l[8:].strip(); continue
        if l.startswith('Known values'):
            sec = 'k'; continue
        if l.startswith('The missing features elsewhere'):
            sec = 'e'; continue
        if l.startswith('Missing values to predict'):
            sec = 't'; continue
        if l.startswith('Answer as JSON'):
            sec = None; continue
        if sec == 'k' and l.startswith('- ') and not l.startswith('- (') and ' = ' in l:
            n, v = l[2:].rsplit(' = ', 1); out['known'].append((n, v))
        elif sec == 't':
            m = re.match(r'(\d+)\. (.*)$', l)
            if m: out['targets'].append(m.group(2))
    return out


def gold_set():
    """(type, entity, item) for every eval target, under both the label name and the prompt's item name."""
    g = set()
    for r in csv.DictReader(open(LABELS)):
        g.add((r['data_type'], r['entity'], r['item']))
    for l in open(GOLD):
        x = json.loads(l)
        for n in x['items'].values():
            g.add((x['data_type'], x['entity'], n))
    return g


MON = {'Jan': 1, 'January': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'April': 4, 'May': 5, 'Jun': 6, 'June': 6, 'Jul': 7,
       'July': 7, 'Aug': 8, 'August': 8, 'Sept': 9, 'Sep': 9, 'Nov': 11, 'November': 11, 'Dec': 12, 'December': 12}


def sample_of(t, e):
    return e.rsplit(':', 1)[0] if t in ('cite', 'scrna', 'xenium') else e


def date_of(t, e):
    """blood/stool/saliva collection date (YYYY-MM-DD) or None (tumour / germline)."""
    s = sample_of(t, e)
    if s.startswith('tumor_') or s.startswith('xenium'):
        return None
    m = re.search(r'(\d{4}-\d{2}-\d{2})', s)
    if m: return m.group(1)
    m = re.search(r'tcr_freq_(\d{4}-\d{2})$', s)
    if m: return m.group(1)          # month only; resolved in visit_of
    m = re.search(r'pbmc_([A-Za-z]+)(\d{4})_(\d{2})(\d{2})(\d{2})_', s)
    if m: return f'{m.group(2)}-{MON[m.group(1)]:02d}-{m.group(5)}'   # June2025_260625 -> 2025-06-25
    m = re.search(r'pbmc_([A-Za-z]+)(\d{4})', s)
    if m and m.group(1) in MON: return f'{m.group(2)}-{MON[m.group(1)]:02d}-15'
    return None


def tumour_of(t, e):
    s = sample_of(t, e)
    if t in ('olink', 'flow', 'tcr', 'mpa') or s.startswith('pbmc'):
        return None
    if t in ('dia', 'mif'): return 'T0'
    if s == 'personalis' or 'personalis' in s.lower(): return 'T0'
    m = re.search(r'(?:^|[_:])T(\d)', s)
    return f'T{m.group(1)}' if m else None
