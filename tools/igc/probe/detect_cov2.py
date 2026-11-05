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
from pathlib import Path
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
ap.add_argument('--train-traces', default='', help='fit every detector on these (the control arm), then read --traces arm by arm; without it the traces are cross-validated by trace')
ap.add_argument('--out', required=True)
ap.add_argument('--layers', default='8,16,24,31')
ap.add_argument('--router', action='store_true')
ap.add_argument('--dry', action='store_true')
ap.add_argument('--check', action='store_true', help='test every detector against the real decode (alignment, extractors, shifted labels) and stop')
ap.add_argument('--maxtok', type=int, default=30000)
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--loop-run', type=int, default=10, help='how many lines in a row that already appeared make a loop')
args = ap.parse_args()
LAYERS = [int(x) for x in args.layers.split(',')]
KS, FA, FOLDS = (2, 8, 32), 0.01, 3
LOOP_RUN = args.loop_run   # lines in a row that already appeared. On the 36 control rows: 10 flags 11 rows (10 of the 14 the runner called stuck, 1 right one), 6 flags 17 (12 stuck, 5 right)
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


def load_traces(paths, role):
    rows = []
    for path in paths.split(','):
        for line in open(path, encoding='utf-8'):
            r = json.loads(line)
            if 'out_ids' in r:
                p_ids, o_ids = r['prompt_ids'], r['out_ids'][:args.maxtok]
            else:
                msgs = [{'role': 'user', 'content': r['prompt_text']}]
                p_ids = tok.encode(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True),
                                   add_special_tokens=False)
                o_ids = tok.encode(r['text'], add_special_tokens=False)[:args.maxtok]
            rows.append(dict(id=r['id'], prompt_ids=p_ids, out_ids=o_ids, right=r.get('right'), src=path.split('/')[-1], role=role, spans=r.get('spans', []),
                             arm=Path(path).parent.name or Path(path).stem))
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
                    top1=np.random.rand(n), ent=np.random.rand(n), am=np.random.randint(0, 1000, n), nxt={k: np.random.rand(n) for k in ('comp', 'reflect')})
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
            ent, top1, am = [], [], []
            for k in range(0, h.shape[0], 1024):
                lp = torch.log_softmax(model.lm_head(h[k:k + 1024]).float(), -1)
                p = lp.exp()
                for kind, mk in start_mask.items():
                    nxt[kind].append((p * mk).sum(-1).cpu())
                ent.append(-(p * lp).sum(-1).cpu())
                top1.append(p.max(-1).values.cpu())
                am.append(lp.argmax(-1).cpu())
    finally:
        for hk in hooks:
            hk.remove()
    return dict(caps=caps, hp=hp, router=rt, top1=torch.cat(top1).numpy(), ent=torch.cat(ent).numpy(), am=torch.cat(am).numpy(),
                nxt={k: torch.cat(v).numpy() for k, v in nxt.items()})


APPLY = bool(args.train_traces)
rows = (load_traces(args.train_traces, 'train') if APPLY else []) + load_traces(args.traces, 'test')
print('traces', len(rows), 'tokens', sum(len(r['out_ids']) for r in rows), flush=True)
for r in rows:
    ps = [tok.decode([t]) for t in r['out_ids']]
    offs, n = [], 0
    for p in ps:
        offs.append(n)
        n += len(p)
    text = ''.join(ps)
    r['text'], r['offs'], r['pieces'] = text, offs, ps
    r['n'] = len(r['out_ids'])
    r['events'] = {}
    for kind, fn in EXTRACT.items():
        evs = fn(text)
        for e in evs:
            e['tok'] = bisect.bisect_right(offs, e['onset']) - 1
            if e.get('vstart') is not None:
                e['vtok'] = bisect.bisect_right(offs, e['vstart']) - 1
        own = np.zeros(r['n'] + 1, dtype=bool)
        for s_, e_, _ in r['spans']:
            own[s_:e_] = True
        r['events'][kind] = [e for e in evs if not own[min(e['tok'], r['n'])]]
    t0 = time.time()
    r.update(features(r))
    print(r['id'], r['src'], 'tokens', r['n'], {k: len(v) for k, v in r['events'].items()}, f'{time.time() - t0:.0f}s', flush=True)
    r['ids_np'] = np.array(r['out_ids'])

if not args.dry:   # the features are in memory: the weights are not needed for the learning (a 35B model left no room for it)
    del model, lm, blocks
    import gc
    gc.collect()
    torch.cuda.empty_cache()

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
    for L in LAYERS:
        XL = torch.cat([r['caps'][L][idx].float() for r, idx, _ in sm]).to(DEV)   # one layer at a time: all of them together did not fit beside the model
        f = fit_dense(XL, ytr)
        del XL
        torch.cuda.empty_cache() if not args.dry else None
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



def run_checks():
    print('\n######## CHECK 1: the state at token t must predict token t+1 (top-1 against what the trace says came next)')
    hit = [(r['am'][:-1] == np.array(r['out_ids'][1:])).mean() for r in rows]
    s1 = [(r['am'][:-2] == np.array(r['out_ids'][2:])).mean() for r in rows]
    s0 = [(r['am'] == np.array(r['out_ids'])).mean() for r in rows]
    print(f'  aligned (t -> t+1): mean {np.mean(hit):.3f}, min {np.min(hit):.3f}, max {np.max(hit):.3f} over {len(rows)} traces; '
          f'shifted t -> t+2: {np.mean(s1):.3f}; shifted t -> t: {np.mean(s0):.3f}   (a right alignment is far above both)')
    # the prompt end predicts the first token of the trace
    print('\n######## CHECK 2: the start-token masses of "next" sit where the next token really is a start token')
    for kind in start_mask:
        mk = start_mask[kind].cpu().numpy() if not args.dry else np.zeros(V, bool)
        on, off, on_m1, on_p1 = [], [], [], []
        for r in rows:
            nxt_is = mk[np.array(r['out_ids'][1:])] if len(mk) > 1 else np.zeros(r['n'] - 1, bool)
            m = r['nxt'][kind][:-1]
            on += list(m[nxt_is]); off += list(m[~nxt_is])
            on_m1 += list(r['nxt'][kind][1:][nxt_is])   # the same starters, read one position late
        print(f'  {kind}: mass where the next token is one of its starters {np.mean(on) if on else float("nan"):.3f} ({len(on)} positions), elsewhere {np.mean(off):.3f}; '
              f'one position late {np.mean(on_m1) if on_m1 else float("nan"):.3f}')
    print('\n######## CHECK 3: the token index of every event points at the text it names; 8 random events of each kind to READ')
    for kind in EXTRACT:
        bad, total = 0, 0
        for r in rows:
            for e in r['events'][kind]:
                total += 1
                piece = r['pieces'][e['tok']].strip()
                if not piece or not r['text'][e['onset']:e['onset'] + 40].lstrip().startswith(piece[:max(1, min(len(piece), 6))]):
                    bad += 1
        print(f'  {kind}: {total} events, the token at the index does not open the named text in {bad}')
        pool = [(r, e) for r in rows for e in r['events'][kind]]
        for r, e in [pool[i] for i in rng.choice(len(pool), size=min(8, len(pool)), replace=False)] if pool else []:
            a = e['onset']
            print('     ', r['id'], repr(r['text'][max(0, a - 50):a]) + ' >>> ' + repr(r['text'][a:a + 70]))
    print('\n######## CHECK 4: what the extractors miss (lines with an arithmetic `=` and a number, outside every extracted window); 8 to READ')
    import re as _re
    pat = _re.compile(r'=[ ]*-?[0-9]')
    numeric_left = _re.compile(r'[0-9 .,()+*/^-]+')   # digits and operators only: no letter, no backslash command (a symbolic equation is not a computation)
    miss, seen = [], 0
    for r in rows:
        spots = [e['onset'] for e in r['events']['comp']]
        for m in pat.finditer(r['text']):
            left = _re.split(r'[\n$]|=|\\times|\\cdot', r['text'][max(0, m.start() - 70):m.start()])[-1].strip()
            if not (numeric_left.fullmatch(left) and _re.search(r'[0-9)][ ]*[+*/^-][ ]*[0-9(]', left)):
                continue
            seen += 1
            if not any(abs(m.start() - o) < 90 for o in spots):
                miss.append((r['id'], r['text'][max(0, m.start() - 40):m.end() + 20]))
    print(f'  {seen} arithmetic-looking `=` lines, {len(miss)} with no comp event near them ({100 * len(miss) / max(seen, 1):.0f} %)')
    for i in (rng.choice(len(miss), size=min(8, len(miss)), replace=False) if miss else []):
        print('     ', miss[i][0], repr(miss[i][1]))
    print('\n######## CHECK 5: a detector trained on labels shifted by s tokens, scored on the true labels (s = 0 must be the best); labels shuffled must be ~0.5')
    L = LAYERS[-1]
    half = [r for i, r in enumerate(rows) if i % 2 == 0], [r for i, r in enumerate(rows) if i % 2 == 1]
    for kind in ('comp', 'reflect'):
        if sum(len(r['events'][kind]) for r in rows) < 20:
            continue
        for what in ('probe', 'router') if (args.router and rows[0]['router'] is not None) else ('probe',):
            line = []
            for sh in (-4, -2, 0, 2, 4, 'shuffled'):
                ys, ss = [], []
                for tr, te in ((half[0], half[1]), (half[1], half[0])):
                    xs, yl = [], []
                    for r in tr:
                        y = labels(r, kind, 2)
                        if sh == 'shuffled':
                            ys_ = rng.permutation(y)
                        elif sh >= 0:
                            ys_ = np.concatenate([y[sh:], np.zeros(sh, bool)])
                        else:
                            ys_ = np.concatenate([np.zeros(-sh, bool), y[:sh]])
                        pos, neg = np.flatnonzero(ys_), np.flatnonzero(~ys_)
                        neg = rng.choice(neg, size=min(len(neg), 10 * max(len(pos), 1)), replace=False)
                        idx = np.concatenate([pos, neg])
                        xs.append((r, idx)); yl.append(ys_[idx])
                    ytr = torch.tensor(np.concatenate(yl), device=DEV)
                    if ytr.sum() < 3:
                        continue
                    if what == 'probe':
                        f = fit_dense(torch.cat([r['caps'][L][idx].float() for r, idx in xs]).to(DEV), ytr)
                        score = lambda r, f=f: torch.cat([f(r['caps'][L][i:i + 8192].float().to(DEV)).cpu() for i in range(0, r['n'], 8192)]).numpy()
                    else:
                        f = fit_bag(torch.cat([router_idx(r)[torch.tensor(idx, device=DEV)] for r, idx in xs]), ytr, 40 * 256 + 1 if not args.dry else 6 * 256 + 1)
                        score = lambda r, f=f: f(router_idx(r)).cpu().numpy()
                    for r in te:
                        ys.append(labels(r, kind, 2)); ss.append(score(r))
                line.append(f'{sh}: {auroc(np.concatenate(ys), np.concatenate(ss)):.3f}')
            print(f'  {kind} {what}' + (f' L{L}' if what == "probe" else '') + ' (k=2, AUROC on the true labels)  ' + '   '.join(line))
    print('\n######## CHECK 6: the digit-confidence state (the one before a value\'s first digit) predicts that digit')
    W = [(r, e['vtok'] - 1) for r in rows for e in r['events']['comp'] if e.get('vtok') is not None and e['vtok'] >= 1]
    agree = np.mean([r['am'][t] == r['out_ids'][t + 1] for r, t in W]) if W else float('nan')
    early = np.mean([r['am'][t - 1] == r['out_ids'][t + 1] for r, t in W if t >= 1]) if W else float('nan')
    print(f'  {len(W)} values: top-1 equals the value\'s first token in {agree:.3f}; one position earlier {early:.3f}')
    print('\n######## CHECK 7: tokens of the software\'s own text are not events')
    n_own = sum(int(sum(e_ - s_ for s_, e_, _ in r['spans'])) for r in rows)
    print(f'  {n_own} tokens inside the software\'s own spans in these traces; events kept only outside them')


if args.check:
    run_checks()
    sys.exit(0)

results = []


# Detectors read together (the developer, 2026-11-05: the signals err in different places, so the signals together should be sharper). Parameter-free: each member's
# score becomes its rank percentile among all positions of the set being read, and the fused score is the mean of the members' percentiles. Nothing is fitted on the
# set, so there is nothing to over-fit; the threshold is the same 1 %-of-positions rule as for the single signals.
FUSIONS = {'fuse_all': None, 'fuse_next_p24_p31': ('next', 'probe_L24', 'probe_L31'), 'fuse_next_p24_text': ('next', 'probe_L24', 'text_lr'),
           'fuse_p24_p31_text': ('probe_L24', 'probe_L31', 'text_lr')}


def fused(allscores):
    """{fusion name: [fused score per trace]} from {signal: [score per trace]}; a fusion with fewer than two of its members present is skipped."""
    out = {}
    for fname, members in FUSIONS.items():
        names = [n for n in allscores if n != 'router'] if members is None else [m for m in members if m in allscores]
        if len(names) < 2:
            continue
        pct = []
        for n in names:
            pooled = np.concatenate(allscores[n])
            order = np.argsort(pooled, kind='mergesort')
            rank = np.empty(len(pooled))
            rank[order] = (np.arange(len(pooled)) + 0.5) / len(pooled)
            pct.append(rank)
        mean = np.mean(pct, 0)
        cuts = np.cumsum([0] + [len(s) for s in allscores[names[0]]])
        out[fname] = [mean[cuts[i]:cuts[i + 1]] for i in range(len(cuts) - 1)]
    return out


def accumulate(agg, te, scorers, kind, k):
    """Score the traces `te` with each detector, and with the fusions of them; the threshold is set so 1 % of this set's positions alarm."""
    allscores = {name: [np.asarray(sc(r)) for r in te] for name, sc in scorers.items()}
    allscores.update(fused(allscores))
    for name, scores in allscores.items():
        thr = np.quantile(np.concatenate(scores), 1 - FA)
        a = agg.setdefault(name, dict(y=[], s=[], caught=0, onsets=0, leads=[], ahit=0, atot=0))
        for r, s_ in zip(te, scores):
            y = labels(r, kind, k)
            al = s_ >= thr
            a['y'].append(y)
            a['s'].append(s_)
            a['atot'] += int(al.sum())
            a['ahit'] += int((al & y).sum())
            for e in r['events'][kind]:
                o = e['tok']
                a['onsets'] += 1
                w = np.flatnonzero(al[max(0, o - k):o])
                if len(w):
                    a['caught'] += 1
                    a['leads'].append(o - (max(0, o - k) + w[0]))


def emit(agg, kind, k, arm=''):
    for name, a in agg.items():
        res = dict(arm=arm, kind=kind, k=k, signal=name, auroc=round(auroc(np.concatenate(a['y']), np.concatenate(a['s'])), 3),
                   caught=a['caught'], onsets=a['onsets'], lead=(float(np.median(a['leads'])) if a['leads'] else None),
                   precision=round(a['ahit'] / max(a['atot'], 1), 3), random=round(1 - (1 - FA) ** k, 3))
        results.append(res)
        print(f'{arm:14s} {kind:7s} k={k:2d} {name:10s} AUROC {res["auroc"]:.3f}  caught {res["caught"]}/{res["onsets"]}  '
              f'(random {res["random"]:.3f})  median lead {res["lead"]}  precision {res["precision"]:.3f}', flush=True)


if APPLY:
    train_rows = [r for r in rows if r['role'] == 'train']
    arms = {}
    for r in rows:
        if r['role'] == 'test':
            arms.setdefault(r['arm'], []).append(r)
    print('detectors fitted on', len(train_rows), 'control traces; read on', {a: len(v) for a, v in arms.items()}, flush=True)
    for kind in EXTRACT:
        total = sum(len(r['events'][kind]) for r in train_rows)
        print(f'\n######## {kind}: {total} onsets in the control traces; per arm: ' +
              str({a: sum(len(r['events'][kind]) for r in v) for a, v in arms.items()}), flush=True)
        if total < 10:
            continue
        for k in KS:
            scorers = make_scorers(kind, k, train_rows)
            for arm, te in arms.items():
                agg = {}
                accumulate(agg, te, scorers, kind, k)
                emit(agg, kind, k, arm)
    print('\n######## wrong values, per arm (digit confidence, 1 - top-1 probability, at the state before the first digit)')
    wrong_res = {}
    for arm, te in arms.items():
        W = [(r, e['vtok'] - 1, not e['right']) for r in te for e in r['events']['comp'] if e['right'] is not None and e.get('vtok') is not None and e['vtok'] >= 1]
        yw = np.array([w[2] for w in W])
        if 3 <= yw.sum() < len(yw):
            wrong_res[arm] = round(auroc(yw, np.array([1 - r['top1'][t] for r, t, _ in W])), 3)
        print(f'  {arm}: values read {len(W)}, wrong {int(yw.sum()) if len(W) else 0}, AUROC {wrong_res.get(arm)}')
    json.dump(dict(model=args.model, mode='apply', arms={a: len(v) for a, v in arms.items()}, results=results, wrong=wrong_res), open(args.out, 'w'))
    sys.exit(0)

order = rng.permutation(len(rows))
fold_of = {int(i): j % FOLDS for j, i in enumerate(order)}
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
            accumulate(agg, te, make_scorers(kind, k, tr), kind, k)
        emit(agg, kind, k)

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
