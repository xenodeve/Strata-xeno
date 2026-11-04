"""steer_lb.py MODEL TRACES OUT: software's value written into the residual stream, on the computations of real thinking traces (#298, the third level of the
2026-11-04 plan: the mechanism that decides what the model believes, not the logit and not the text).

num_rw2.py did it on a clean `a * b =` prompt with no thinking. Here the points are the `=` of the model's own thinking (tir4b --ids traces, the control
arm), and the class means are fitted on those traces, so nothing is calibrated on a prompt the model does not meet in use.

1. Teacher-forced pass over each trace. At every `=` whose expression `inline_found` reads and whose value the model wrote as an integer of 3+ digits,
   keep the state at LAYERS for the 3 positions that predict the first three digits. Class means mu[L][j][d] over TRAIN rows, from the points where the
   model wrote the value software gives (the digit it wrote is the digit it held).
2. Point test (TEST rows): from the prefix up to the first digit, greedy, the digits steered by alpha * (mu[tgt_j] - mu[model_j]) at layer L.
   `fix` = the model wrote another value, target = software's; `control` = the model wrote software's value, target = software's + 111 (does the push
   write anything it is told). caught = the first 3 digits are the target's.
3. Behaviour test (the point of the exercise): from the same prefix, three arms, S samples each, T 0.6 / top-p 0.95 / top-k 20, NEW tokens:
   A the model alone | B software's value inserted as text | C the digits steered (the model writes the value itself) and then left alone.
   Per arm: the value written is software's (digits), a re-check word within 300 characters after the value (the H-trust measure), and a re-write of the value.
Needs a GPU for the model alone (no vLLM running beside it: vLLM's memory profile fails if another process frees memory)."""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path.cwd()))
from tools.igc.tir4b import inline_found

ap = argparse.ArgumentParser()
ap.add_argument('model')
ap.add_argument('traces')
ap.add_argument('out')
ap.add_argument('--layers', default='24,28,31')
ap.add_argument('--alphas', default='5,10')
ap.add_argument('--max-points', type=int, default=60, help='per kind (fix / control) and per layer x alpha')
ap.add_argument('--behave', type=int, default=30, help='fix points for the behaviour test (0: skip)')
ap.add_argument('--samples', type=int, default=4)
ap.add_argument('--new', type=int, default=200)
ap.add_argument('--max-prefix', type=int, default=14000)
ap.add_argument('--rows', type=int, default=0)
args = ap.parse_args()
LAYERS = [int(x) for x in args.layers.split(',')]
ALPHAS = [float(x) for x in args.alphas.split(',')]
K = 3
MARK = re.compile(r"\b(wait|hmm|double[- ]?check|re-?check|re-?verify|verify|let me (?:re|double|check|verify)|re-?calculate|re-?compute|again)\b", re.I)
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map='cuda:0').eval()
lm = model.model.language_model if hasattr(model.model, 'language_model') else model.model
blocks = lm.layers
DIG = {d: tok.encode(str(d), add_special_tokens=False)[0] for d in range(10)}
DIGIT_OF = {v: k for k, v in DIG.items()}
EQ = {i for i in range(len(tok)) if tok.decode([i]).strip() == '='}


def points_of(row):
    """The computations of one trace whose value the model wrote as an integer of 3+ digits: positions, the model's digits, software's."""
    prompt, out = row['prompt_ids'], row['out_ids'][:args.max_prefix - len(row['prompt_ids'])]
    pts = []
    for p, t in enumerate(out):
        if t not in EQ:
            continue
        found = inline_found(tok.decode(out[max(0, p - 300):p]))
        if found is None:
            continue
        expr, sval = found
        if not re.fullmatch(r'-?\d{3,}', sval):
            continue
        j = p + 1
        while j < min(len(out), p + 4) and out[j] not in DIGIT_OF:
            j += 1   # a space, a `$` or a minus before the first digit
        if j + K > len(out) or any(out[j + i] not in DIGIT_OF for i in range(K)):
            continue
        n = 0
        while j + n < len(out) and out[j + n] in DIGIT_OF:
            n += 1
        written = ''.join(str(DIGIT_OF[out[j + i]]) for i in range(n))
        neg_model = tok.decode(out[p + 1:j]).strip().startswith(('-', '$-', '−'))
        if neg_model != sval.startswith('-'):
            continue   # the sign is a different question (17 sign flips in logit8); this test is about digits
        sd = sval.lstrip('-')
        pts.append(dict(p=p, q=j, expr=expr[:80], software=sd, model=written, right=written == sd, same_len=len(written) == len(sd)))
    return prompt, out, pts


def collect(prompt, out, pts):
    """Teacher-forced pass; the state at LAYERS for the positions that predict the first K digits of every point."""
    ids = torch.tensor([prompt + out], device='cuda:0')
    need = sorted({len(prompt) + pt['q'] + i - 1 for pt in pts for i in range(K)})
    got, hooks = {}, []
    for L in LAYERS:
        def hook(m_, i_, o_, L=L):
            h = o_[0] if isinstance(o_, tuple) else o_
            got[L] = h[0, need].float().cpu().numpy()
        hooks.append(blocks[L - 1].register_forward_hook(hook))
    try:
        with torch.no_grad():
            lm(input_ids=ids, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    at = {pos: k for k, pos in enumerate(need)}
    for pt in pts:
        pt['st'] = {L: [got[L][at[len(prompt) + pt['q'] + i - 1]] for i in range(K)] for L in LAYERS}


def generate(prefix, steps, steer=None, sample=None, batch=1):
    """Greedy (sample=None) or sampled continuation of `prefix`, `batch` copies at once. steer: {L: [delta for step 0..K-1]} added to the last position."""
    cur = {}
    hooks = []
    for L in LAYERS:
        def hook(m_, i_, o_, L=L):
            h = o_[0] if isinstance(o_, tuple) else o_
            if L in cur:
                h[:, -1] += cur[L]
            return o_
        hooks.append(blocks[L - 1].register_forward_hook(hook))
    x = torch.tensor([prefix] * batch, device='cuda:0')
    out, past = [[] for _ in range(batch)], None
    gen = torch.Generator(device='cuda:0')
    gen.manual_seed(sample if sample is not None else 0)
    try:
        with torch.no_grad():
            for step in range(steps):
                cur.clear()
                if steer and step < K:
                    cur.update({L: d[step] for L, d in steer.items()})
                o = model(input_ids=x, past_key_values=past, use_cache=True, logits_to_keep=1)   # the logits of one position, not 14,000 x 248,000
                past = o.past_key_values
                lg = o.logits[:, -1].float()
                if sample is None:
                    t = lg.argmax(-1)
                else:
                    lg = lg / 0.6
                    v, ix = lg.topk(20, -1)
                    pr = torch.softmax(v, -1)
                    cs = pr.cumsum(-1)
                    pr = torch.where(cs - pr > 0.95, torch.zeros_like(pr), pr)
                    t = ix.gather(1, torch.multinomial(pr / pr.sum(-1, keepdim=True), 1, generator=gen)).squeeze(1)
                for b in range(batch):
                    out[b].append(int(t[b]))
                x = t[:, None]
    finally:
        for h in hooks:
            h.remove()
    return out


def digits_of(gen):
    d = []
    for t in gen:
        if t not in DIGIT_OF:
            break
        d.append(DIGIT_OF[t])
    return ''.join(map(str, d))


def after_value(text):
    m = re.match(r'\s*\$?\s*-?\d[\d,.]*', text)
    return text[m.end():] if m else text


def main():
    t0 = time.monotonic()
    rows = [json.loads(l) for l in Path(args.traces).read_text(encoding='utf-8').splitlines() if l.strip()]
    rows = rows[:args.rows] if args.rows else rows
    cut = int(0.7 * len(rows))
    data = []
    for k, row in enumerate(rows):
        prompt, out, pts = points_of(row)
        if pts:
            collect(prompt, out, pts)
        data.append(dict(id=row['id'], prompt=prompt, out=out, pts=pts))
        print(f'  trace {k + 1}/{len(rows)} {row["id"]}: {len(pts)} points ({sum(p["right"] for p in pts)} the model wrote software\'s value)', flush=True)
    train, test = data[:cut], data[cut:]
    mu = {L: [[None] * 10 for _ in range(K)] for L in LAYERS}
    for L in LAYERS:
        for j in range(K):
            for d in range(10):
                sel = [pt['st'][L][j] for r in train for pt in r['pts'] if pt['right'] and pt['model'][j] == str(d)]
                mu[L][j][d] = np.mean(sel, 0) if len(sel) >= 3 else None
    print(f'train {len(train)} traces, test {len(test)}; class means with 3+ samples: '
          + ', '.join(f'L{L}: {sum(m is not None for j in range(K) for m in mu[L][j])}/{K * 10}' for L in LAYERS), flush=True)
    fix = [(r, pt) for r in test for pt in r['pts'] if not pt['right'] and pt['same_len']]
    ctl = [(r, pt) for r in test for pt in r['pts'] if pt['right']]
    print(f'test points: fix {len(fix)} (the model wrote another value of the same length), control {len(ctl)}', flush=True)
    rng = random.Random(3)
    rng.shuffle(fix)
    rng.shuffle(ctl)
    result = dict(train=len(train), test=len(test), fix_points=len(fix), control_points=len(ctl), table=[], behave=[])

    def deltas(L, alpha, pt, target):
        out = {}
        for j in range(K):
            a, b = mu[L][j][int(target[j])], mu[L][j][int(pt['model'][j])]
            if a is None or b is None:
                return None
            out[j] = torch.tensor(alpha * (a - b), device='cuda:0', dtype=torch.bfloat16)
        return [out[j] for j in range(K)]

    best = None
    for L in LAYERS:
        for alpha in ALPHAS:
            row = dict(layer=L, alpha=alpha)
            for name, pool in (('fix', fix), ('control', ctl)):
                tried = hit = alone = 0
                for r, pt in pool[:args.max_points]:
                    target = pt['software'] if name == 'fix' else str(int(pt['software'][:K]) + 111)[:K].rjust(K, '0')
                    dl = deltas(L, alpha, pt, target)
                    if dl is None:
                        continue
                    prefix = r['prompt'] + r['out'][:pt['q']]
                    g = generate(prefix, K + 1, steer={L: dl})[0]
                    tried += 1
                    hit += digits_of(g)[:K] == target[:K]
                    alone += digits_of(generate(prefix, K + 1)[0])[:K] == target[:K]
                row[name] = dict(tried=tried, caught=hit, unsteered_already=alone)
            result['table'].append(row)
            print(f'layer {L:2d} alpha {alpha:4.1f}: fix caught {row["fix"]["caught"]}/{row["fix"]["tried"]} (the model alone: {row["fix"]["unsteered_already"]}); '
                  f'control (told to write +111) {row["control"]["caught"]}/{row["control"]["tried"]}', flush=True)
            score = row['fix']['caught']
            if best is None or score > best[0]:
                best = (score, L, alpha)
    if args.behave and best and best[0] > 0:
        _, L, alpha = best
        print(f'\nbehaviour test at layer {L}, alpha {alpha}: {args.behave} fix points, {args.samples} samples per arm, {args.new} new tokens', flush=True)
        stats = {a: dict(n=0, software=0, mark=0) for a in 'ABC'}
        for r, pt in fix[:args.behave * 2]:
            if stats['A']['n'] >= args.behave * args.samples:
                break
            dl = deltas(L, alpha, pt, pt['software'])
            if dl is None:
                continue
            prefix = r['prompt'] + r['out'][:pt['q']]
            alone = digits_of(generate(prefix, K + 1)[0])[:K] == pt['software'][:K]
            if alone:
                continue   # the model alone already writes software's value here: no wrong value to correct
            ins = tok.encode(pt['software'], add_special_tokens=False)
            arms = dict(A=generate(prefix, args.new, sample=11, batch=args.samples),
                        B=[ins + g for g in generate(prefix + ins, args.new - len(ins), sample=12, batch=args.samples)],
                        C=generate(prefix, args.new, steer={L: dl}, sample=13, batch=args.samples))
            for a, gens in arms.items():
                for g in gens:
                    text = tok.decode(g)
                    s = stats[a]
                    s['n'] += 1
                    s['software'] += digits_of(g)[:len(pt['software'])] == pt['software']
                    s['mark'] += bool(MARK.search(after_value(text)[:300]))
            result['behave'].append(dict(id=r['id'], expr=pt['expr'], software=pt['software'], model=pt['model'],
                                         A=tok.decode(arms['A'][0])[:260], B=tok.decode(arms['B'][0])[:260], C=tok.decode(arms['C'][0])[:260]))
        for a, name in (('A', 'the model alone'), ('B', 'value inserted as text'), ('C', 'digits steered (the model writes it)')):
            s = stats[a]
            print(f'  {a} {name:38s} n {s["n"]:3d}  wrote software\'s value {s["software"]:3d} ({100 * s["software"] / max(s["n"], 1):4.1f} %)  '
                  f're-check word <=300 chars after: {s["mark"]:3d} ({100 * s["mark"] / max(s["n"], 1):4.1f} %)')
        result['behave_stats'] = stats
    Path(args.out).write_text(json.dumps(result, indent=1), encoding='utf-8')
    print(f'\nwrote {args.out} in {time.monotonic() - t0:.0f} s')


if __name__ == '__main__':
    main()
