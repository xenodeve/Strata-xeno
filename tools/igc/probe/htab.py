"""htab.py: H-trust as an intervention at the SAME point, on the failing items (pc5-full: 0 of 7 right, every one hit the token limit or stuck).
For each injection point of the stored trace (software's value written right after the model's own `=`):
  arm A  the context as the stored trace has it: software's value inserted, then the model continues (K samples);
  arm B  the context stops at the `=`: the model writes its own value and continues (K samples) - the normal item, where the model does
         not receive software's result.
Same prefix, same sampling (T 0.6, top-p 0.95, top-k 20), 240 new tokens. Per sample: a re-check word within 300 / 700 characters after the
value, and the same expression written again within 900 characters. In arm B also: did the model's own value equal software's?
Part 1: forcing the value's tokens through a logits processor (P = 1 on the token) against inserting them: do greedy continuations agree?
Points: group 'low' = the model gave software's value P < 0.5 (it would have written something else), 'high' = P >= 0.9, same trace rows."""
import json, re, sys, time
import numpy as np
import torch
try:
    from tools.igc.mathexpr import evaluate
except ImportError:   # on a VM the file sits next to the script
    sys.path.insert(0, '/content')
    from mathexpr import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

MODEL, IDS, POINTS, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
K, MAXNEW = 3, 240
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map='cuda:0').eval()
rows = {json.loads(l)['id']: json.loads(l) for l in open(IDS, encoding='utf-8')}
pts = json.load(open(POINTS))
MARK = re.compile(r"\b(wait|hmm|double[- ]?check|re-?check|re-?verify|verify|let me (?:re|double|check|verify)|re-?calculate|re-?compute|again)\b", re.I)
VAL = re.compile(r"^\s*\$?(-?\d[\d,]*\.?\d*(?:\s*/\s*\d+)?)")
norm = lambda s: re.sub(r'[\s$]+', '', s)
EOS = [t for t in (tok.eos_token_id, tok.convert_tokens_to_ids('<|im_end|>')) if t is not None]


def gen(prefix, k, greedy=False, max_new=MAXNEW, seed=0, procs=None):
    x = torch.tensor([prefix], device='cuda:0').repeat(k, 1)
    torch.manual_seed(seed)
    kw = dict(do_sample=False) if greedy else dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20)
    with torch.no_grad():
        o = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=max_new, pad_token_id=EOS[0], eos_token_id=EOS,
                           logits_processor=procs, **kw)
    return o[:, x.shape[1]:].cpu().tolist()


class Force(LogitsProcessor):
    """The value's tokens by the sampler: all the probability on the forced token, as a logit override would."""
    def __init__(self, forced):
        self.f, self.step = forced, 0

    def __call__(self, input_ids, scores):
        if self.step < len(self.f):
            s = torch.full_like(scores, -float('inf'))
            s[:, self.f[self.step]] = 0.0
            scores = s
        self.step += 1
        return scores


def prefix_of(p, with_value):
    r = rows[p['id']]
    end = p['start'] + p['n']
    return r['prompt_ids'] + r['out_ids'][:end if with_value else p['start']]


# ---- part 1: forcing through the sampler against inserting the text
t0 = time.time()
same, tested, first_div = 0, 0, []
for p in pts[:30]:
    r = rows[p['id']]
    forced = r['out_ids'][p['start']:p['start'] + p['n']]
    a = gen(prefix_of(p, True), 1, greedy=True, max_new=48)[0]
    b = gen(prefix_of(p, False), 1, greedy=True, max_new=len(forced) + 48, procs=LogitsProcessorList([Force(forced)]))[0]
    ok_forced = b[:len(forced)] == forced
    cont = b[len(forced):len(forced) + 48]
    n = min(len(a), len(cont))
    tested += 1
    if ok_forced and a[:n] == cont[:n]:
        same += 1
    else:
        d = next((i for i in range(n) if a[i] != cont[i]), None)
        first_div.append(d if ok_forced else 'forced-failed')
print(f'PART1 forcing by the sampler vs inserting: greedy 48 tokens identical in {same} of {tested} points; first divergence at {first_div}; {time.time()-t0:.0f}s', flush=True)

# ---- part 2: the same point, software's value against the model's own
res = []
t0 = time.time()
for i, p in enumerate(pts):
    ev = norm(p['expr'])
    sw = None
    try:
        sw = evaluate(p['value'].replace(',', '')) if p['value'] else None
    except Exception:
        pass
    for arm in ('A', 'B'):
        gens = gen(prefix_of(p, arm == 'A'), K, seed=1000 + i)
        for g in gens:
            text = tok.decode(g, skip_special_tokens=True)
            if arm == 'A':
                after, own_equal = text, None
            else:
                m = VAL.match(text)
                if not m:
                    res.append(dict(i=i, id=p['id'], group=p['group'], arm=arm, no_value=True))
                    continue
                after = text[m.end():]
                try:
                    own = evaluate(m.group(1).replace(',', '').replace(' ', ''))
                except Exception:
                    own = None
                own_equal = (own is not None and sw is not None and own == sw)
            res.append(dict(i=i, id=p['id'], group=p['group'], arm=arm, own_equal=own_equal,
                            m300=bool(MARK.search(after[:300])), m700=bool(MARK.search(after[:700])),
                            same=bool(len(ev) >= 5 and ev in norm(after[:900]))))
    if i % 5 == 0:
        print(f'  point {i + 1}/{len(pts)}  {time.time() - t0:.0f}s', flush=True)
json.dump(dict(points=pts, samples=res), open(OUT, 'w'))


def rate(sel, k):
    n = len(sel)
    c = sum(bool(r[k]) for r in sel)
    return f'{c}/{n} ({100 * c / max(n, 1):.0f}%)'


print('\n== arm B (the model writes its own value): how often is it software\'s value?')
for g in ('low', 'high'):
    sel = [r for r in res if r['arm'] == 'B' and r['group'] == g and not r.get('no_value')]
    print(f'  {g}: own value equals software\'s in {rate(sel, "own_equal")} of samples')
print('\n== re-check within 300 / 700 chars, same expression again within 900 chars')
print(f'{"group":6s} {"arm":4s} {"re-check<=300":>16s} {"re-check<=700":>16s} {"same expr again":>18s}')
for g in ('low', 'high'):
    for arm in ('A', 'B'):
        sel = [r for r in res if r['group'] == g and r['arm'] == arm and not r.get('no_value')]
        print(f'{g:6s} {arm:4s} {rate(sel, "m300"):>16s} {rate(sel, "m700"):>16s} {rate(sel, "same"):>18s}')
print('\n== only the points where the model\'s own value differs from software\'s (B samples), against A at the same points')
diff_pts = {(r['i']) for r in res if r['arm'] == 'B' and r.get('own_equal') is False}
for g in ('low', 'high'):
    for arm in ('A', 'B'):
        sel = [r for r in res if r['group'] == g and r['arm'] == arm and r['i'] in diff_pts and not r.get('no_value') and
               (arm == 'A' or r.get('own_equal') is False)]
        print(f'{g:6s} {arm:4s} {rate(sel, "m300"):>16s} {rate(sel, "m700"):>16s} {rate(sel, "same"):>18s}')
