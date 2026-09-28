"""Blend EXL3 router counts with Strata training traces; evaluate held-out on the benchmark traces.

  python blend_profile.py sweep            # w in a grid: CPU / primary share per benchmark prompt
  python blend_profile.py write W OUT.bin  # write the blended profile for weight W
"""
import json, os, sys
from collections import Counter
from route_tools import NL, NE, counts, records, write_profile

T = os.environ["TEMP"]
TR = os.path.join(T, "strata-claude-traces")
TRAIN = [os.path.join(TR, n + ".trace") for n in ("bash-files", "cpp-bug", "json-review", "py-async", "thai-net", "thai-plan")]
BENCH = {b: os.path.join(TR, f"bench-{b}.trace") for b in ("code", "thai", "sky", "long")}
P, S = 6653, 6602


def exl3():
    d = json.load(open(r"D:\Github\exllamav3-xeno\xeno\router-stats.json"))
    c = {}
    for l in range(NL):
        for e, n in enumerate(d[f"model.language_model.layers.{l}.mlp"]): c[(l, e)] = n
    return c


def ranked(w, ex, tr):
    te, tt = sum(ex.values()), max(1, sum(tr.values()))
    allp = [(l, e) for l in range(NL) for e in range(NE)]
    return sorted(allp, key=lambda p: -((1 - w) * ex.get(p, 0) / te + w * tr.get(p, 0) / tt))


def evaluate(rank, trace):
    prim, sec = set(rank[:P]), set(rank[P:P + S])
    h = Counter()
    for layer, _, _, ids in records(trace):
        for e in ids:
            if e >= 0: h["p" if (layer, e) in prim else "s" if (layer, e) in sec else "c"] += 1
    t = sum(h.values())
    return h["p"] / t, h["c"] / t


def main(a):
    ex, (tr, _) = exl3(), counts(TRAIN)
    if a[0] == "sweep":
        for w in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
            r = ranked(w, ex, tr)
            res = {b: evaluate(r, p) for b, p in BENCH.items()}
            mean_c = sum(c for _, c in res.values()) / len(res)
            print(f"w {w:.1f}  " + "  ".join(f"{b} P{p:.1%} C{c:.1%}" for b, (p, c) in res.items()) + f"  mean CPU {mean_c:.2%}")
    elif a[0] == "write":
        write_profile(a[2], ranked(float(a[1]), ex, tr)); print("wrote", a[2])


if __name__ == "__main__":
    main(sys.argv[1:])
