"""#11 N0: offline four-tier replay of real routing traces - VRAM0 / VRAM1 / a bounded RAM cache / NVMe.

The GPU tiers are static, as at runtime: VRAM0 holds the first P and VRAM1 the next S experts of the EXL3-ranked
profile. The RAM cache of C experts starts with the next C ranked experts; everything else lives on NVMe. A miss
loads the expert from NVMe once per verify window, however many tokens routed it (the N5 grouping). It is then
admitted to the RAM cache, which evicts by policy:
  lru    least recently used
  lfu    lowest routed count so far
  dlfu   lowest decayed count (x0.97 per window)
  prior  dlfu seeded with the EXL3 counts
Reported per RAM size and policy: the share of routed entries served by each tier, NVMe loads per verify token, and
NVMe MB per verify token. The session replay concatenates every trace, so it exercises topic switches (#23).

  python nvme_sim.py                 # all traces as one session, plus each benchmark trace alone
"""
import os, sys
from collections import defaultdict
from route_tools import records
from blend_profile import exl3, ranked, TR, P, S

BLOB_MB = 1382400 / 2 ** 20
SIZES_GIB = (4, 8, 12, 16)
POLICIES = ("lru", "lfu", "dlfu", "prior")


def windows(paths):
    """Verify windows as lists of (n_tok, ids) per layer, across the traces in order."""
    for path in paths:
        cur = []
        for layer, nt, _, ids in records(path):
            if layer == 0 and cur:
                yield cur
                cur = []
            cur.append((nt, ids, layer))
        if cur:
            yield cur


def simulate(paths, cap, policy, rank, prior):
    vram0, vram1 = set(rank[:P]), set(rank[P:P + S])
    ram = set(rank[P + S:P + S + cap])
    last, count, score = {}, defaultdict(int), defaultdict(float)
    if policy == "prior":
        top = max(prior.values()) if prior else 1
        for k, v in prior.items(): score[k] = 8.0 * v / top
    hits = {"vram0": 0, "vram1": 0, "ram": 0, "nvme": 0}
    loads = tokens = 0
    t = 0
    for w in windows(paths):
        t += 1
        if policy in ("dlfu", "prior"):
            for k in list(score): score[k] *= 0.97
        loaded = set()
        tokens += w[0][0]
        for nt, ids, layer in w:
            for e in ids:
                if e < 0: continue
                key = (layer, e)
                count[key] += 1; score[key] += 1.0; last[key] = t
                if key in vram0: hits["vram0"] += 1
                elif key in vram1: hits["vram1"] += 1
                elif key in ram: hits["ram"] += 1
                else:
                    hits["nvme"] += 1
                    if key not in loaded:
                        loaded.add(key)
                        loads += 1
        # admit this window's loads, evicting by policy (ties broken by rank order, i.e. arbitrary but fixed)
        for key in loaded:
            if key in ram or cap == 0: continue
            if len(ram) >= cap:
                if policy == "lru": victim = min(ram, key=lambda k: last.get(k, 0))
                elif policy == "lfu": victim = min(ram, key=lambda k: count.get(k, 0))
                else: victim = min(ram, key=lambda k: score.get(k, 0.0))
                ram.discard(victim)
            ram.add(key)
    total = sum(hits.values()) or 1
    return {k: v / total for k, v in hits.items()}, loads / max(tokens, 1), loads * BLOB_MB / max(tokens, 1), tokens


def main():
    rank = ranked(0.0, exl3(), {})
    prior = dict(exl3())
    traces = sorted(os.path.join(TR, f) for f in os.listdir(TR) if f.endswith(".trace"))
    sets = [("session (all 10 traces)", traces)] + [(os.path.basename(p)[:-6], [p]) for p in traces if "bench-" in p]
    print(f"VRAM0 {P} + VRAM1 {S} experts static; RAM cache sizes in GiB; blob {BLOB_MB:.3f} MB")
    for name, paths in sets:
        print(f"\n== {name}")
        print(f"{'RAM':>4} {'policy':>6}  {'vram0':>6} {'vram1':>6} {'ram':>6} {'nvme':>6}  {'loads/tok':>9} {'MB/tok':>7}")
        for gib in SIZES_GIB:
            cap = int(gib * 1024 / BLOB_MB)
            for pol in POLICIES:
                h, lpt, mbt, _ = simulate(paths, cap, pol, rank, prior)
                print(f"{gib:>4} {pol:>6}  {h['vram0']:6.3f} {h['vram1']:6.3f} {h['ram']:6.3f} {h['nvme']:6.3f}  "
                      f"{lpt:9.2f} {mbt:7.1f}")


if __name__ == "__main__":
    main()
