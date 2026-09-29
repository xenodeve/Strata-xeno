"""#34: would a lazy exact refill of the borrowed cache tail pay?  From decode route traces.

Usage:
    python tests/xeno/perf/lazy_refill_sim.py PROFILE.bin KEEP_FROM SLOTS TRACE... [--ms-per-expert 1.3] [--blob-mb 1.3824]

The prompt path borrows the expert cache's slots [KEEP_FROM, SLOTS), which hold the profile's ranked experts
[KEEP_FROM, SLOTS). A lazy refill leaves them empty after the prompt and loads one only when decode routes to it.
It is exact only if the GPU still computes it, so each first touch is a stall: the window's layer waits for an NVMe
read and a DMA (the misses of one layer are read together). This counts, per trace, the tail experts decode
touches after N windows, the bytes that are then read (against the whole tail a bulk refill reads) and the stalls
(window-layers with at least one first touch) x the per-read latency.
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, __import__("os").path.dirname(__file__))
from route_tools import read_profile, records  # noqa: E402


def tail(ranked, keep_from: int, slots: int) -> set:
    return set(ranked[keep_from:slots])


def _windows(path: str):
    """Yield per window: {layer: set of routed ids}."""
    cur: dict[int, set] = {}
    last_layer = -1
    for layer, _, _, ids in records(path):
        if layer <= last_layer and cur:
            yield cur
            cur = {}
        cur.setdefault(layer, set()).update(e for e in ids if e >= 0)
        last_layer = layer
    if cur:
        yield cur


def first_touches(path: str, tail_set: set) -> list[int]:
    seen, out = set(), []
    for w in _windows(path):
        n = 0
        for layer, ids in w.items():
            for e in ids:
                if (layer, e) in tail_set and (layer, e) not in seen:
                    seen.add((layer, e))
                    n += 1
        out.append(n)
    return out


def stall_layers(path: str, tail_set: set) -> list[int]:
    seen, out = set(), []
    for w in _windows(path):
        n = 0
        for layer, ids in w.items():
            new = {(layer, e) for e in ids if (layer, e) in tail_set} - seen
            if new:
                n += 1
                seen |= new
        out.append(n)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("profile")
    ap.add_argument("keep_from", type=int)
    ap.add_argument("slots", type=int)
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--ms-per-expert", type=float, default=1.3)
    ap.add_argument("--blob-mb", type=float, default=1.3824)
    a = ap.parse_args(argv)
    t = tail(read_profile(a.profile), a.keep_from, a.slots)
    print(f"tail: {len(t)} experts ({len(t) * a.blob_mb / 1000:.2f} GB for a bulk refill)")
    marks = (8, 16, 32, 64, 128, 256)
    print(f"{'trace':16}{'windows':>8}" + "".join(f"{'@' + str(m):>9}" for m in marks) + f"{'total':>8}{'MB':>8}"
          f"{'stalls':>8}{'stall ms':>10}")
    for tr in a.traces:
        ft, st = first_touches(tr, t), stall_layers(tr, t)
        cum, c = [], 0
        for v in ft:
            c += v
            cum.append(c)
        cols = "".join(f"{(cum[m - 1] if len(cum) >= m else cum[-1]):9d}" for m in marks)
        name = tr.replace("\\", "/").rsplit("/", 1)[-1]
        print(f"{name:16}{len(ft):8d}{cols}{c:8d}{c * a.blob_mb:8.0f}{sum(st):8d}{sum(st) * a.ms_per_expert:10.0f}")
    print("columns @N: tail experts first touched within the first N windows (a window is 1-5 tokens)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
