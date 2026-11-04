"""detect_cov2.py: detection coverage for three kinds of events, one table (#298; the developer: detect as widely as possible first).

    python detect_cov2.py --model M --traces a.jsonl,b.jsonl --out out.json --layers 8,16,24,31 [--router] [--dry]

A trace line is {id, prompt_ids, out_ids} (tir4b --ids) or {id, prompt_text, text, right} (a stored record plus the question).
Events found in the text alone (the traces are control arms: no MIP in them):
- comp     an arithmetic `=` (mathexpr reads the left side, it has an operator); onset = first token of the expression; right / wrong value
- reflect  a re-check starts (wait, hmm, let me double-check, verify, again ...); onset = the first token of the word, at least 60 chars apart
- loop     ten or more lines in a row that already appeared earlier in the trace; onset = the first of them
Signals, teacher-forced on the trace (the state at token t predicts token t+1), each fitted on 2/3 of the rows and scored on the other 1/3,
three folds so every row is tested once:
- next      mass of the tokens that start the event kind in the next-token distribution (digits / operators for comp, reflection words for
            reflect; none for loop); stands in for the MTP draft proposals
- text_lr   the strongest text-only detector: logistic regression on the last 12 token ids
- probe_L   a linear probe on the residual stream after layer L
- router    which experts the router picked (MoE only)
Metric per kind and window k: AUROC; onsets caught out of all onsets (an alarm in the k tokens before the onset) with the threshold set so
1 % of the test positions alarm; median lead; precision; `random` = 1 - 0.99^k.
Also: wrong values (digit confidence, probe before the first digit); the end of the prompt (probe at the last prompt token against whether the
trace's answer was right, leave-one-out) when the traces carry `right`.
--dry: random features, CPU, no model: only to run the code."""
import argparse, bisect, json, re, sys, time
from fractions import Fraction
import numpy as np
import torch
import torch.nn.functional as F
try:
    from tools.igc.mathexpr import evaluate, expression_before
except ImportError:   # on a VM the file sits next to the script
    sys.path.insert(0, '/content')
    from mathexpr import evaluate, expression_before

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--traces', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--layers', default='8,16,24,31')
ap.add_argument('--router', action='store_true')
ap.add_argument('--dry', action='store_true')
ap.add_argument('--maxtok', type=int, default=30000)
ap.add_argument('--limit', type=int, default=0)
args = ap.parse_args()
LAYERS = [int(x) for x in args.layers.split(',')]
KS, FA, FOLDS = (2, 8, 32), 0.01, 3
LOOP_RUN = 10   # lines in a row that already appeared: on the 36 control rows 10 flags 11 rows, 10 of the 14 the runner called stuck, 1 right one
DEV = 'cpu' if args.dry else 'cuda:0'
rng = np.random.default_rng(0)
torch.manual_seed(0)

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(args.model)
if not args.dry:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=DEV).eval()
    lm = model.model.language_model if hasattr(model.model, 'language_model') else model.model
    blocks = lm.layers

# ------------------------------------------------------------------------------------------------ events
WRITTEN_END = re.compile(r'[=\n$;]|\\approx|\\neq|\\quad|,\s')
DECIMALS = re.compile(r'\.(\d+)\s*$')
OPERATOR = re.compile(r'[-+*/×÷^]|\\times|\\cdot|\\div|\\frac')
MARK = re.compile(r"\b(wait|hmm|double[- ]?check|re-?check|re-?verify|verify|let me (?:re|double|check|verify)|re-?calculate|re-?compute|again)\b", re.I)


def comp_events(text):
    out = []
    for m in re.finditer('=', text):
        eq = m.start()
        lo = max(0, eq - 300)
        head = text[lo:eq]
        try:
            got = expression_before(head)
        except Exception:
            got = None
        if not got:
            continue
        expr, val = got
        if val is None or not OPERATOR.search(expr):
            continue
        s = head.rfind(expr)
        if s < 0:
            continue
        right = vstart = None
        seg = WRITTEN_END.split(text[m.end():m.end() + 160])[0].strip()
        if seg:
            try:
                w = evaluate(seg)
            except Exception:
                w = None
            if w is not None:
                tail = DECIMALS.search(seg)
                dec = len(tail.group(1)) if tail else 0
                rounded = dec > 0 or '.' in expr or 'approx' in head[-60:]
                right = bool(w == val or (rounded and abs(w - val) <= Fraction(5, 10 ** (dec + 1))))
                vstart = m.end() + (len(text[m.end():]) - len(text[m.end():].lstrip()))
        out.append(dict(onset=lo + s, right=right, vstart=vstart))
    return out


def reflect_events(text):
    out, last = [], -10 ** 9
    for m in MARK.finditer(text):
        if m.start() - last >= 60:
            out.append(dict(onset=m.start()))
            last = m.start()
    return out


def loop_events(text):
    out, seen, run, pos, first = [], set(), 0, 0, None
    in_run = False
    for ln in text.split('\n'):
        s = ln.strip()
        if len(s) >= 25:
            if s in seen:
                if run == 0:
                    first = pos
                run += 1
                if run == LOOP_RUN and not in_run:
                    out.append(dict(onset=first))
                    in_run = True
            else:
                run, in_run = 0, False
            seen.add(s)
        pos += len(ln) + 1
    return out


EXTRACT = {'comp': comp_events, 'reflect': reflect_events, 'loop': loop_events}

# ------------------------------------------------------------------------------------------------ data
V = 248320 if args.dry else model.lm_head.weight.shape[0]
START = {'comp': re.compile(r'^[\s\d\.\,\+\-\*/=\^\(\)×÷\$]+$'), 'reflect': re.compile(r'^\s*(wait|hmm|actually|alternatively|verify|check|recheck|but)\b', re.I)}
start_mask = {}
if not args.dry:
    toks = tok.convert_ids_to_tokens(list(range(V)))
    strs = [tok.convert_tokens_to_string([p]) if p is not None else '' for p in toks]
    for kind, pat in START.items():
        mk = torch.zeros(V, dtype=torch.bool)
        for i, s in enumerate(strs):
            if s.strip() and pat.match(s) and (kind != 'comp' or re.search(r'[\d\+\-\*/=\^×÷]', s)):
                mk[i] = True
        start_mask[kind] = mk.to(DEV)
        print('start tokens', kind, int(mk.sum()), flush=True)


def load_traces():
    rows = []
    for path in args.traces.split(','):
        for line in open(path, encoding='utf-8'):
            r = json.loads(line)
            if 'out_ids' in r:
                p_ids, o_ids = r['prompt_ids'], r['out_ids'][:args.maxtok]
            else:
                msgs = [{'role': 'user', 'content': r['prompt_text']}]
                p_ids = tok.encode(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True),
                                   add_special_tokens=False)
                o_ids = tok.encode(r['text'], add_special_tokens=False)[:args.maxtok]
            rows.append(dict(id=r['id'], prompt_ids=p_ids, out_ids=o_ids, right=r.get('right'), src=path.split('/')[-1]))
            if args.limit and len(rows) >= args.limit:
                return rows
    return rows


def features(r):
    base, out = len(r['prompt_ids']), r['out_ids']
    n = len(out)
    if args.dry:
        d = 32
        return dict(caps={L: torch.randn(n, d).half() for L in LAYERS}, hp={L: torch.randn(d) for L in LAYERS},
                    router=(torch.randint(0, 256, (n, 6, 8)).to(torch.uint8) if args.router else None),
                    top1=np.random.rand(n), ent=np.random.rand(n), nxt={k: np.random.rand(n) for k in ('comp', 'reflect')})
    x = torch.tensor([r['prompt_ids'] + out], device=DEV)
    caps, hp, hooks = {}, {}, []
    for L in LAYERS:
        def hook(m_, i_, o, L=L):
            h = o[0] if isinstance(o, tuple) else o
            caps[L] = h[0, base:].to(torch.float16).cpu()
            hp[L] = h[0, base - 1].float().cpu()
        hooks.append(blocks[L - 1].register_forward_hook(hook))
    rt = None
    use_router = args.router
    try:
        with torch.no_grad():
            try:
                o = model.model(input_ids=x, use_cache=False, **(dict(output_router_logits=True) if use_router else {}))
            except TypeError:
                use_router = False
                o = model.model(input_ids=x, use_cache=False)
            if use_router and getattr(o, 'router_logits', None) is not None:
                rt = torch.stack([q[base:].topk(8, -1).indices.to(torch.uint8) for q in o.router_logits], 1).cpu()
            h = o.last_hidden_state[0, base:]
            nxt = {k: [] for k in start_mask}
            ent, top1 = [], []
            for k in range(0, h.shape[0], 1024):
                lp = torch.log_softmax(model.lm_head(h[k:k + 1024]).float(), -1)
                p = lp.exp()
                for kind, mk in start_mask.items():
                    nxt[kind].append((p * mk).sum(-1).cpu())
                ent.append(-(p * lp).sum(-1).cpu())
                top1.append(p.max(-1).values.cpu())
    finally:
        for hk in hooks:
            hk.remove()
    return dict(caps=caps, hp=hp, router=rt, top1=torch.cat(top1).numpy(), ent=torch.cat(ent).numpy(),
                nxt={k: torch.cat(v).numpy() for k, v in nxt.items()})


rows = load_traces()
print('traces', len(rows), 'tokens', sum(len(r['out_ids']) for r in rows), flush=True)
for r in rows:
    ps = [tok.decode([t]) for t in r['out_ids']]
    offs, n = [], 0
    for p in ps:
        offs.append(n)
        n += len(p)
    text = ''.join(ps)
    r['n'] = len(r['out_ids'])
    r['events'] = {}
    for kind, fn in EXTRACT.items():
        evs = fn(text)
        for e in evs:
            e['tok'] = bisect.bisect_right(offs, e['onset']) - 1
            if e.get('vstart') is not None:
                e['vtok'] = bisect.bisect_right(offs, e['vstart']) - 1
        r['events'][kind] = evs
    t0 = time.time()
    r.update(features(r))
    print(r['id'], r['src'], 'tokens', r['n'], {k: len(v) for k, v in r['events'].items()}, f'{time.time() - t0:.0f}s', flush=True)
    r['ids_np'] = np.array(r['out_ids'])

# ------------------------------------------------------------------------------------------------ learning on the GPU
def labels(r, kind, k):
    y = np.zeros(r['n'], dtype=bool)
    for e in r['events'][kind]:
        y[max(0, e['tok'] - k):e['tok']] = True
    return y


def fit_dense(X, y, l2=1e-3, steps=200):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mu) / sd
    w = torch.zeros(X.shape[1], device=DEV, requires_grad=True)
    b = torch.zeros(1, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.02)
    yt = y.float()
    for _ in range(steps):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(Xn @ w + b, yt) + l2 * (w ** 2).sum()
        loss.backward()
        opt.step()
    return lambda Z: ((Z - mu) / sd) @ w.detach() + b.detach()


def fit_bag(idx, y, D, l2=1e-5, steps=150):
    emb = torch.nn.EmbeddingBag(D, 1, mode='sum', padding_idx=0).to(DEV)
    torch.nn.init.zeros_(emb.weight)
    with torch.no_grad():
        emb.weight[0] = 0
    b = torch.zeros(1, device=DEV, requires_grad=True)
    opt = torch.optim.Adam(list(emb.parameters()) + [b], lr=0.05)
    yt = y.float()
    for _ in range(steps):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(emb(idx).squeeze(-1) + b, yt) + l2 * (emb.weight ** 2).sum()
        loss.backward()
        opt.step()
    return lambda Z: (emb(Z).squeeze(-1) + b).detach()


DT = 1 << 18
def text_idx(r):
    ids = r['ids_np'].astype(np.int64)
    n = len(ids)
    pad = V
    sh = lambda j: np.concatenate([np.full(j, pad), ids[:n - j]]) if j else ids
    cols = [(sh(j) + j * (V + 1)) % (DT - 1) + 1 for j in range(4)]
    cols.append((ids * 1000003 + sh(1)) % (DT - 1) + 1)
    cols += [(sh(j) + 77 * (V + 1)) % (DT - 1) + 1 for j in range(4, 12)]
    return torch.tensor(np.stack(cols, 1), device=DEV)


def router_idx(r):
    rt = r['router'].long()
    return (torch.arange(rt.shape[1]).view(1, -1, 1) * 256 + rt + 1).reshape(rt.shape[0], -1).to(DEV)


def sample(rows_, kind, k):
    out = []
    for r in rows_:
        y = labels(r, kind, k)
        pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
        neg = rng.choice(neg, size=min(len(neg), 10 * max(len(pos), 1)), replace=False)
        idx = np.concatenate([pos, neg])
        out.append((r, idx, y[idx]))
    return out


def make_scorers(kind, k, train_rows):
    sm = sample(train_rows, kind, k)
    ytr = torch.tensor(np.concatenate([s[2] for s in sm]), device=DEV)
    if ytr.sum() < 3:
        return {}
    sc = {}
    if kind in start_mask or args.dry:
        sc['next'] = lambda r: torch.tensor(r['nxt'][kind] if kind in r['nxt'] else r['nxt']['comp'])
    X = {L: torch.cat([r['caps'][L][idx].float() for r, idx, _ in sm]).to(DEV) for L in LAYERS}
    for L in LAYERS:
        f = fit_dense(X[L], ytr)
        sc[f'probe_L{L}'] = (lambda r, f=f, L=L: torch.cat([f(r['caps'][L][i:i + 8192].float().to(DEV)).cpu() for i in range(0, r['n'], 8192)]))
    TX = torch.cat([text_idx(r)[torch.tensor(idx, device=DEV)] for r, idx, _ in sm])
    ft = fit_bag(TX, ytr, DT)
    sc['text_lr'] = lambda r: ft(text_idx(r)).cpu()
    if args.router and sm[0][0]['router'] is not None:
        RX = torch.cat([router_idx(r)[torch.tensor(idx, device=DEV)] for r, idx, _ in sm])
        fr = fit_bag(RX, ytr, 40 * 256 + 1 if not args.dry else 6 * 256 + 1)
        sc['router'] = lambda r: fr(router_idx(r)).cpu()
    return sc


def auroc(y, s):
    y = np.asarray(y, bool)
    s = np.asarray(s, float)
    order = np.argsort(s, kind='mergesort')
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    n1, n0 = y.sum(), (~y).sum()
    return float('nan') if n1 == 0 or n0 == 0 else float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


order = rng.permutation(len(rows))
fold_of = {int(i): j % FOLDS for j, i in enumerate(order)}
results = []
for kind in EXTRACT:
    total = sum(len(r['events'][kind]) for r in rows)
    print(f'\n######## {kind}: {total} onsets in {len(rows)} traces', flush=True)
    if total < 10:
        continue
    for k in KS:
        agg = {}
        for f in range(FOLDS):
            tr = [r for i, r in enumerate(rows) if fold_of[i] != f]
            te = [r for i, r in enumerate(rows) if fold_of[i] == f]
            scorers = make_scorers(kind, k, tr)
            for name, sc in scorers.items():
                scores = [np.asarray(sc(r)) for r in te]
                thr = np.quantile(np.concatenate(scores), 1 - FA)
                a = agg.setdefault(name, dict(y=[], s=[], caught=0, onsets=0, leads=[], ahit=0, atot=0))
                for r, s in zip(te, scores):
                    y = labels(r, kind, k)
                    al = s >= thr
                    a['y'].append(y)
                    a['s'].append(s)
                    a['atot'] += int(al.sum())
                    a['ahit'] += int((al & y).sum())
                    for e in r['events'][kind]:
                        o = e['tok']
                        a['onsets'] += 1
                        w = np.flatnonzero(al[max(0, o - k):o])
                        if len(w):
                            a['caught'] += 1
                            a['leads'].append(o - (max(0, o - k) + w[0]))
        for name, a in agg.items():
            res = dict(kind=kind, k=k, signal=name, auroc=round(auroc(np.concatenate(a['y']), np.concatenate(a['s'])), 3),
                       caught=a['caught'], onsets=a['onsets'], lead=(float(np.median(a['leads'])) if a['leads'] else None),
                       precision=round(a['ahit'] / max(a['atot'], 1), 3), random=round(1 - (1 - FA) ** k, 3))
            results.append(res)
            print(f'{kind:7s} k={k:2d} {name:10s} AUROC {res["auroc"]:.3f}  caught {res["caught"]}/{res["onsets"]}  '
                  f'(random {res["random"]:.3f})  median lead {res["lead"]}  precision {res["precision"]:.3f}', flush=True)

# ------------------------------------------------------------------------------------------------ a wrong value
print('\n######## wrong values')
W = []
for ri, r in enumerate(rows):
    for e in r['events']['comp']:
        if e['right'] is not None and e.get('vtok') is not None and e['vtok'] >= 1:
            W.append((ri, e['vtok'] - 1, not e['right']))
yw = np.array([w[2] for w in W])
print('values read', len(W), 'wrong', int(yw.sum()))
wrong_res = {}
if 3 <= yw.sum() < len(yw):
    for name, key in (('digit confidence (1 - top-1 prob)', 'top1'), ('entropy', 'ent')):
        sc = np.array([(1 - rows[ri][key][t]) if key == 'top1' else rows[ri][key][t] for ri, t, _ in W])
        wrong_res[name] = round(auroc(yw, sc), 3)
        print(f'  {name}: AUROC {wrong_res[name]}')
    folds = np.array([fold_of[w[0]] for w in W])
    for L in LAYERS:
        X = torch.stack([rows[ri]['caps'][L][t].float() for ri, t, _ in W]).to(DEV)
        sc = np.zeros(len(W))
        for f in range(FOLDS):
            trm, tem = folds != f, folds == f
            if yw[trm].sum() < 2 or tem.sum() == 0:
                continue
            fn = fit_dense(X[torch.tensor(trm, device=DEV)], torch.tensor(yw[trm], device=DEV), l2=1e-2)
            sc[tem] = fn(X[torch.tensor(tem, device=DEV)]).cpu().numpy()
        wrong_res[f'probe_L{L}'] = round(auroc(yw, sc), 3)
        print(f'  probe L{L}: AUROC {wrong_res[f"probe_L{L}"]}')

# ------------------------------------------------------------------------------------------------ the end of the prompt
print('\n######## end of the prompt: does the trace end with a right answer?')
lab = [(i, r) for i, r in enumerate(rows) if r.get('right') is not None and str(r['right']) in ('True', 'False', 'true', 'false')]
yp = np.array([str(r['right']).lower() == 'true' for _, r in lab])
prompt_res = {}
print('traces with a right / wrong label:', len(lab), 'right', int(yp.sum()))
if len(lab) >= 12 and 3 <= yp.sum() <= len(yp) - 3:
    length = np.array([len(r['prompt_ids']) for _, r in lab], float)
    prompt_res['prompt length'] = round(auroc(~yp, length), 3)
    print(f'  prompt length (longer -> wrong): AUROC {prompt_res["prompt length"]}')
    for L in LAYERS:
        X = torch.stack([r['hp'][L] for _, r in lab]).to(DEV)
        sc = np.zeros(len(lab))
        for j in range(len(lab)):   # leave one out
            keep = torch.tensor([i != j for i in range(len(lab))], device=DEV)
            fn = fit_dense(X[keep], torch.tensor(~yp[keep.cpu().numpy()], device=DEV), l2=1e-1, steps=100)
            sc[j] = float(fn(X[j:j + 1]).cpu())
        prompt_res[f'probe_L{L}'] = round(auroc(~yp, sc), 3)
        print(f'  probe at the last prompt token, L{L} (leave one out): AUROC for "wrong" {prompt_res[f"probe_L{L}"]}')
json.dump(dict(model=args.model, traces=len(rows), results=results, wrong=wrong_res, prompt=prompt_res), open(args.out, 'w'))
