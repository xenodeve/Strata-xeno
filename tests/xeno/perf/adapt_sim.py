"""Phase 4 estimate: static EXL3 placement vs an oracle vs decayed-LFU swaps with a per-round budget.

Rounds = verify windows (layer 0 starts one). After each round, up to SWAP experts that missed the primary tier
and have the highest decayed count replace the primary residents with the lowest decayed count, if the newcomer
beats the victim by a hysteresis margin. Secondary stays static. Hits are counted before the swap takes effect.
"""
import os, sys
from collections import Counter, defaultdict
from route_tools import records
from blend_profile import exl3, ranked, BENCH, P, S


def rounds(trace):
    cur = []
    for layer, _, _, ids in records(trace):
        if layer == 0 and cur:
            yield cur; cur = []
        cur.append((layer, ids))
    if cur: yield cur


def simulate(trace, swap, decay=0.97, margin=1.0):
    rank = ranked(0.0, exl3(), {})
    prim, sec = set(rank[:P]), set(rank[P:P + S])
    score = defaultdict(float)
    h = Counter()
    for rd in rounds(trace):
        for key in list(score): score[key] *= decay
        for layer, ids in rd:
            for e in ids:
                if e < 0: continue
                k = (layer, e)
                h["p" if k in prim else "s" if k in sec else "c"] += 1
                score[k] += 1.0
        if swap:
            cand = sorted((k for k in score if k not in prim), key=lambda k: -score[k])[:swap]
            victims = sorted(prim, key=lambda k: score.get(k, 0.0))[:swap]
            for c, v in zip(cand, victims):
                if score[c] > score.get(v, 0.0) + margin:
                    prim.discard(v); prim.add(c)
                    sec.discard(c)   # a promoted secondary expert leaves the secondary tier
    t = sum(h.values())
    return h["p"] / t, h["s"] / t, h["c"] / t


def oracle(trace):
    c = Counter()
    for layer, _, _, ids in records(trace):
        for e in ids:
            if e >= 0: c[(layer, e)] += 1
    r = [k for k, _ in c.most_common()]
    prim, sec = set(r[:P]), set(r[P:P + S])
    t = sum(c.values())
    return sum(v for k, v in c.items() if k in prim) / t, sum(v for k, v in c.items() if k not in prim and k not in sec) / t


if __name__ == "__main__":
    for b, p in BENCH.items():
        o = oracle(p)
        line = f"{b:5} oracle P{o[0]:.1%} C{o[1]:.1%} |"
        for sw in (0, 4, 8, 16, 32):
            pp, ss, cc = simulate(p, sw)
            line += f" swap{sw}: P{pp:.1%} C{cc:.1%} |"
        print(line, flush=True)
