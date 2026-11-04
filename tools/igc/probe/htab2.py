"""htab2.py: the guideline arms of the H-trust test, at the same points as htab.py (A: software's value inserted; B: the model writes its own).
  arm C  the value inserted, then the tokens that start a re-check (wait, hmm, actually, verify, ...) are taken out of the next 60 steps
         (NoWait / CGRS style: a logit-level act, only after a value software has verified);
  arm E  the value inserted, then a short guideline sentence in the model's context: "This value is exact and final." (no word the re-check
         counter looks for);
Same prefix, same sampling, same outcomes as htab.py; the A and B rows are read back from htab.json when it is there.
Each sample's first 300 characters are saved so the cases can be READ, not only counted."""
import json, os, re, sys, time
import numpy as np
import torch
sys.path.insert(0, '/content')
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

MODEL, IDS, POINTS, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
K, MAXNEW, HOLD = 3, 240, 60
GUIDE = '\nThis value is exact and final.\n'
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map='cuda:0').eval()
rows = {json.loads(l)['id']: json.loads(l) for l in open(IDS, encoding='utf-8')}
pts = json.load(open(POINTS))
MARK = re.compile(r"\b(wait|hmm|double[- ]?check|re-?check|re-?verify|verify|let me (?:re|double|check|verify)|re-?calculate|re-?compute|again)\b", re.I)
norm = lambda s: re.sub(r'[\s$]+', '', s)
EOS = [t for t in (tok.eos_token_id, tok.convert_tokens_to_ids('<|im_end|>')) if t is not None]
STARTERS = re.compile(r'^\s*(wait|hmm|actually|alternatively|verify|check|recheck)\s*$', re.I)
banned = [i for i, s in enumerate(tok.convert_ids_to_tokens(list(range(model.lm_head.weight.shape[0]))))
          if s is not None and STARTERS.match(tok.convert_tokens_to_string([s]))]
print('re-check starter tokens taken out in arm C:', len(banned), flush=True)
GUIDE_IDS = tok.encode(GUIDE, add_special_tokens=False)


class Suppress(LogitsProcessor):
    def __init__(self, ids, hold):
        self.ids, self.hold, self.step = torch.tensor(ids, device='cuda:0'), hold, 0

    def __call__(self, input_ids, scores):
        if self.step < self.hold:
            scores[:, self.ids] = -float('inf')
        self.step += 1
        return scores


def gen(prefix, k, seed, procs=None):
    x = torch.tensor([prefix], device='cuda:0').repeat(k, 1)
    torch.manual_seed(seed)
    with torch.no_grad():
        o = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=MAXNEW, pad_token_id=EOS[0], eos_token_id=EOS,
                           do_sample=True, temperature=0.6, top_p=0.95, top_k=20, logits_processor=procs)
    return o[:, x.shape[1]:].cpu().tolist()


def prefix_with_value(p):
    r = rows[p['id']]
    return r['prompt_ids'] + r['out_ids'][:p['start'] + p['n']]


res, t0 = [], time.time()
for i, p in enumerate(pts):
    ev = norm(p['expr'])
    for arm in ('C', 'E'):
        if arm == 'C':
            gens = gen(prefix_with_value(p), K, 1000 + i, LogitsProcessorList([Suppress(banned, HOLD)]))
        else:
            gens = gen(prefix_with_value(p) + GUIDE_IDS, K, 1000 + i)
        for g in gens:
            text = tok.decode(g, skip_special_tokens=True)
            res.append(dict(i=i, id=p['id'], group=p['group'], arm=arm, m300=bool(MARK.search(text[:300])), m700=bool(MARK.search(text[:700])),
                            same=bool(len(ev) >= 5 and ev in norm(text[:900])), head=text[:300]))
    if i % 5 == 0:
        print(f'  point {i + 1}/{len(pts)}  {time.time() - t0:.0f}s', flush=True)
old = json.load(open('/content/htab.json'))['samples'] if os.path.exists('/content/htab.json') else []
json.dump(dict(points=pts, samples=res, earlier=old), open(OUT, 'w'))


def rate(sel, k):
    n = len(sel)
    c = sum(bool(r[k]) for r in sel)
    return f'{c}/{n} ({100 * c / max(n, 1):.0f}%)'


allr = [r for r in old if not r.get('no_value')] + res
print('\n== re-check within 300 / 700 chars, same expression again within 900 chars')
print(f'{"group":6s} {"arm":4s} {"re-check<=300":>16s} {"re-check<=700":>16s} {"same expr again":>18s}')
for g in ('low', 'high'):
    for arm in ('A', 'B', 'C', 'E'):
        sel = [r for r in allr if r['group'] == g and r['arm'] == arm]
        if sel:
            print(f'{g:6s} {arm:4s} {rate(sel, "m300"):>16s} {rate(sel, "m700"):>16s} {rate(sel, "same"):>18s}')
