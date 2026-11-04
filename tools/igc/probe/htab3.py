"""htab3.py: the controls for htab2.py. Arm E of htab2 put a sentence on a new line after the value and the model began that line with "Wait"
in a third of the samples; arms A, B and C went on in the line of the value. So the sentence and the new line were not told apart.
  arm N  the value, then only a newline;
  arm G  the value, then " (exact)" and a newline (the form of the runner's inline value when it is not plain, tir4b `--inline-plain` off).
Same points, same prefix, same sampling and outcomes as htab.py; the A and B rows are read back from htab.json, and E from htab2.json.
Each sample's first 300 characters are saved so the cases can be read."""
import json, os, re, sys, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL, IDS, POINTS, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
K, MAXNEW = 3, 240
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map='cuda:0').eval()
rows = {json.loads(l)['id']: json.loads(l) for l in open(IDS, encoding='utf-8')}
pts = json.load(open(POINTS))
MARK = re.compile(r"\b(wait|hmm|double[- ]?check|re-?check|re-?verify|verify|let me (?:re|double|check|verify)|re-?calculate|re-?compute|again)\b", re.I)
norm = lambda s: re.sub(r'[\s$]+', '', s)
EOS = [t for t in (tok.eos_token_id, tok.convert_tokens_to_ids('<|im_end|>')) if t is not None]
NL = chr(10)
TAILS = {'N': tok.encode(NL, add_special_tokens=False), 'G': tok.encode(' (exact)' + NL, add_special_tokens=False)}
print('tails', {k: v for k, v in TAILS.items()}, flush=True)


def gen(prefix, k, seed):
    x = torch.tensor([prefix], device='cuda:0').repeat(k, 1)
    torch.manual_seed(seed)
    with torch.no_grad():
        o = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=MAXNEW, pad_token_id=EOS[0], eos_token_id=EOS,
                           do_sample=True, temperature=0.6, top_p=0.95, top_k=20)
    return o[:, x.shape[1]:].cpu().tolist()


def prefix_with_value(p):
    r = rows[p['id']]
    return r['prompt_ids'] + r['out_ids'][:p['start'] + p['n']]


res, t0 = [], time.time()
for i, p in enumerate(pts):
    ev = norm(p['expr'])
    for arm in ('N', 'G'):
        for g in gen(prefix_with_value(p) + TAILS[arm], K, 1000 + i):
            text = tok.decode(g, skip_special_tokens=True)
            res.append(dict(i=i, id=p['id'], group=p['group'], arm=arm, m300=bool(MARK.search(text[:300])), m700=bool(MARK.search(text[:700])),
                            same=bool(len(ev) >= 5 and ev in norm(text[:900])), first=re.split(r'\W+', text.strip())[0].lower(), head=text[:300]))
    if i % 5 == 0:
        print(f'  point {i + 1}/{len(pts)}  {time.time() - t0:.0f}s', flush=True)
older = []
for name in ('/content/htab.json', '/content/htab2.json'):
    if os.path.exists(name):
        older += [r for r in json.load(open(name))['samples'] if not r.get('no_value')]
json.dump(dict(points=pts, samples=res, earlier=older), open(OUT, 'w'))


def rate(sel, k):
    n = len(sel)
    c = sum(bool(r[k]) for r in sel)
    return f'{c}/{n} ({100 * c / max(n, 1):.0f}%)'


allr = older + res
print('\n== re-check within 300 / 700 chars, same expression again within 900 chars')
print(f'{"group":6s} {"arm":4s} {"re-check<=300":>16s} {"re-check<=700":>16s} {"same expr again":>18s}')
for g in ('low', 'high'):
    for arm in ('A', 'B', 'N', 'G', 'E', 'C'):
        sel = [r for r in allr if r['group'] == g and r['arm'] == arm]
        if sel:
            print(f'{g:6s} {arm:4s} {rate(sel, "m300"):>16s} {rate(sel, "m700"):>16s} {rate(sel, "same"):>18s}')
print('\n== the first word of the continuation (a new line begins it in N, G and E)')
import collections
for arm in ('N', 'G'):
    c = collections.Counter(r['first'] for r in res if r['arm'] == arm)
    print(f'  {arm}:', c.most_common(6))
