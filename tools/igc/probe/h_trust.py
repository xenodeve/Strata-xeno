"""h_trust.py MODEL ids.jsonl[,ids.jsonl...] OUT.json: H-trust test 1 (#298, the developer's hypothesis).

H-trust: a model re-checks a value software gave it because the value arrives in a form the model can tell is not its own.
Two readings that this separates:
  form   - an injected value is re-checked more than a value the model wrote, at the SAME own-probability;
  belief - a value is re-checked when the model gave it a low probability, whoever wrote it.

Data: the stored token ids of a reasoning arm (tir4b --ids): prompt_ids, out_ids, spans [start, end, kind].
- injected values: the 'inline value' spans (software's value, written into the thinking);
- written values: every number that follows '=' in the model's own text (outside every span), in the same files.
For each value: the model's probability of those tokens, teacher-forced in the real context (raw model, T 1),
and whether a re-check marker appears in the next RECHECK_TOKENS tokens.
Run on one GPU (CUDA_VISIBLE_DEVICES set by the caller)."""
import bisect, json, re, sys
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL, FILES, OUT = sys.argv[1], sys.argv[2].split(','), sys.argv[3]
RECHECK_TOKENS = 80
MARK = re.compile(r"\b(wait|hmm|double[- ]?check|re-?check|re-?verify|verify|let me (?:re|double|check|verify)|re-?calculate|re-?compute|again)\b", re.I)
NUM = re.compile(r"=\s*\$?(-?\d[\d,]*\.?\d*)")

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map='cuda:0').eval()


def pieces(ids):
    """Decoded text of each token, the joined text, and the char offset where each token starts."""
    ps = [tok.decode([t]) for t in ids]
    offs, n = [], 0
    for p in ps:
        offs.append(n)
        n += len(p)
    return ps, ''.join(ps), offs


def logprobs(prompt_ids, out_ids, want):
    """log P(out_ids[i] | everything before) for every i in `want` (positions in out_ids), one forward over the whole row."""
    x = torch.tensor([prompt_ids + out_ids], device='cuda:0')
    with torch.no_grad():
        h = model.model(input_ids=x, use_cache=False).last_hidden_state[0]
        res = {}
        base = len(prompt_ids)
        want = sorted(set(want))
        for k in range(0, len(want), 512):
            chunk = want[k:k + 512]
            logits = model.lm_head(h[[base + i - 1 for i in chunk]]).float()
            lp = torch.log_softmax(logits, -1)
            tgt = torch.tensor([out_ids[i] for i in chunk], device='cuda:0')
            got = lp[torch.arange(len(chunk)), tgt].cpu().tolist()
            res.update(zip(chunk, got))
    return res


rows = []
for path in FILES:
    arm = path.split('/')[-2] if '/' in path else path
    for line in open(path, encoding='utf-8'):
        r = json.loads(line)
        out = r['out_ids']
        spans = r['spans']
        covered = np.zeros(len(out) + 1, dtype=bool)
        for s, e, kind in spans:
            covered[s:e] = True
        ps, text, offs = pieces(out)
        vals = []   # (start, end, kind)
        for s, e, kind in spans:
            if kind == 'inline value' and e > s:
                vals.append((s, e, 'injected'))
        for m in NUM.finditer(text):
            a, b = m.span(1)
            s = bisect.bisect_right(offs, a) - 1
            e = bisect.bisect_left(offs, b)
            if e > s and not covered[s:e].any() and not (e - s > 12):
                vals.append((s, e, 'written'))
        want = [i for s, e, _ in vals for i in range(s, e)]
        lp = logprobs(r['prompt_ids'], out, want)
        for s, e, kind in vals:
            l = [lp[i] for i in range(s, e)]
            after = ''.join(ps[e:e + RECHECK_TOKENS])
            rows.append(dict(arm=arm, id=r['id'], kind=kind, start=s, n=e - s, mean_lp=float(np.mean(l)), sum_lp=float(np.sum(l)),
                             min_lp=float(np.min(l)), first_lp=float(l[0]), recheck=bool(MARK.search(after)),
                             value=''.join(ps[s:e]).strip()[:20]))
        print(arm, r['id'], 'injected', sum(v[2] == 'injected' for v in vals), 'written', sum(v[2] == 'written' for v in vals), flush=True)
json.dump(rows, open(OUT, 'w'))


def rate(sel):
    n = len(sel)
    k = sum(r['recheck'] for r in sel)
    return f'{k}/{n} ({100 * k / n:.0f} %)' if n else '0/0'


def auc(sel, key='mean_lp'):
    y = [r['recheck'] for r in sel]
    if len(set(y)) < 2:
        return 'n/a (one class)'
    return f'{roc_auc_score(y, [-r[key] for r in sel]):.3f} (low probability -> re-check; {sum(y)} of {len(y)} re-checked)'


inj = [r for r in rows if r['kind'] == 'injected']
wri = [r for r in rows if r['kind'] == 'written']
print('\n== re-check within', RECHECK_TOKENS, 'tokens after the value')
print('injected values        :', rate(inj))
print('model-written values   :', rate(wri))
print('\n== AUROC of the value\'s own probability for a re-check afterwards')
print('injected :', auc(inj))
print('written  :', auc(wri))
print('\n== same own-probability, who wrote it (mean log-prob per token)')
edges = [-1e9, -4.0, -1.5, -0.5, 1e9]
for lo, hi in zip(edges[:-1], edges[1:]):
    a = [r for r in inj if lo <= r['mean_lp'] < hi]
    b = [r for r in wri if lo <= r['mean_lp'] < hi]
    print(f'  mean_lp in [{lo:g}, {hi:g}):  injected {rate(a):>16}   written {rate(b):>16}')
print('\n== by arm')
for arm in sorted({r['arm'] for r in rows}):
    sel = [r for r in rows if r['arm'] == arm]
    print(f'  {arm}: injected {rate([r for r in sel if r["kind"] == "injected"])}, written {rate([r for r in sel if r["kind"] == "written"])}')
