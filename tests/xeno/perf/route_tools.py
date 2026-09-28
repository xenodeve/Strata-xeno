"""Routing traces from `strata generate --route-trace`: count, rank into an STRP profile, and simulate static tiers.

Trace record: int16 layer, int16 n_tok, int16 k, then n_tok*k int16 expert ids (-1 = none).

  python route_tools.py stats   TRACE...
  python route_tools.py profile OUT.bin TRACE... [--prior OLD.bin]     # rank by routed count, prior breaks ties
  python route_tools.py simulate PROFILE.bin PRIMARY SECONDARY TRACE... # static placement: primary = first PRIMARY
                                                                        # ranked, secondary = next SECONDARY
"""
import struct, sys
from collections import Counter

NL, NE = 48, 512


def records(path):
    b = open(path, "rb").read()
    i, n = 0, len(b) // 2
    v = struct.unpack(f"<{n}h", b[: n * 2])
    while i + 3 <= n:
        layer, nt, k = v[i], v[i + 1], v[i + 2]
        m = nt * k
        yield layer, nt, k, v[i + 3 : i + 3 + m]
        i += 3 + m


def counts(paths):
    c, windows = Counter(), 0
    for p in paths:
        for layer, _, _, ids in records(p):
            if layer == 0: windows += 1
            for e in ids:
                if e >= 0: c[(layer, e)] += 1
    return c, windows


def read_profile(path):
    b = open(path, "rb").read()
    assert b[:4] == b"STRP"
    _, nl, ne, _, n = struct.unpack("<5I", b[4:24])
    raw = struct.unpack(f"<{n * 2}H", b[24 : 24 + n * 4])
    return [(raw[2 * i], raw[2 * i + 1]) for i in range(n)]


def write_profile(path, ranked):
    with open(path, "wb") as f:
        f.write(b"STRP" + struct.pack("<5I", 1, NL, NE, len(ranked), len(ranked)))
        for l, e in ranked: f.write(struct.pack("<2H", l, e))


def main(a):
    if a[0] == "stats":
        c, w = counts(a[1:])
        total = sum(c.values())
        print(f"windows {w}  routed entries {total}  distinct (layer,expert) {len(c)} of {NL * NE}")
        top = sorted(c.values(), reverse=True)
        for n in (1000, 2000, 4000, 6653, 13255):
            print(f"  top {n:>6}: {sum(top[:n]) / total:.1%} of entries")
    elif a[0] == "profile":
        out, rest = a[1], a[2:]
        prior = []
        if "--prior" in rest:
            j = rest.index("--prior"); prior = read_profile(rest[j + 1]); rest = rest[:j] + rest[j + 2 :]
        c, _ = counts(rest)
        prank = {p: i for i, p in enumerate(prior)}
        allp = [(l, e) for l in range(NL) for e in range(NE)]
        ranked = sorted(allp, key=lambda p: (-c.get(p, 0), prank.get(p, 10 ** 9), p))
        write_profile(out, ranked)
        print(f"wrote {out}: {len(ranked)} ranked, {sum(1 for p in allp if c.get(p))} seen in the trace")
    elif a[0] == "simulate":
        prof, P, S, traces = read_profile(a[1]), int(a[2]), int(a[3]), a[4:]
        prim, sec = set(prof[:P]), set(prof[P : P + S])
        hit = Counter()
        for p in traces:
            for layer, _, _, ids in records(p):
                for e in ids:
                    if e < 0: continue
                    key = (layer, e)
                    hit["primary" if key in prim else "secondary" if key in sec else "cpu"] += 1
        t = sum(hit.values())
        print("  ".join(f"{k} {hit[k] / t:.1%}" for k in ("primary", "secondary", "cpu")) + f"  of {t} entries")


if __name__ == "__main__":
    main(sys.argv[1:])
