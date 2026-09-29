"""#44: which path each decode layer waited for, from one STRATA_TIMELINE run.

timeline.py sums the decode stages per round; this pairs, per layer, the main thread's "cpu experts" stage with its
"dispatch plan", its "cpu pool" and the 4070's GPU span ("4070 experts", which includes its H2D and D2H), and reports
where the stage's time goes and how often the 4070 finished after the CPU pool.

Usage:
    python tests/xeno/perf/decode_paths.py run.json [run.json ...]
"""
from __future__ import annotations

import bisect
import json
import sys
from collections import defaultdict


def load(path: str) -> list[dict]:
    text = open(path, encoding="utf-8").read().rstrip().rstrip(",")
    return json.loads(text if text.endswith("]") else text + "]")   # the engine leaves the array open


def breakdown(events: list[dict]) -> dict:
    by: dict[str, list] = defaultdict(list)
    for e in events:
        if e.get("ph") == "X":
            by[e["name"]].append((e["ts"], e["ts"] + e["dur"]))
    for v in by.values():
        v.sort()
    gpu = by["4070 experts"]
    starts = [g[0] for g in gpu]
    acc: dict[str, float] = defaultdict(float)
    n = later = 0
    for (c0, c1), (p0, p1), (q0, q1) in zip(by["cpu experts"], by["dispatch plan"], by["cpu pool"]):
        i = bisect.bisect_left(starts, p0)
        if i >= len(gpu) or gpu[i][0] > c1:
            continue
        g0, g1 = gpu[i]
        n += 1
        later += g1 > q1
        for k, v in (("stage", c1 - c0), ("plan", p1 - p0), ("plan start -> 4070 GPU start", g0 - p0),
                     ("4070 GPU busy", g1 - g0), ("plan end -> pool start", q0 - p1), ("pool", q1 - q0),
                     ("pool end -> stage end", c1 - q1), ("4070 GPU end -> stage end", c1 - g1)):
            acc[k] += v
    rounds = len(by["decode round"])
    return {"layers": n, "rounds": rounds, "4070 after pool": later / n if n else 0.0,
            "ms": {k: round(v / 1000 / max(1, rounds), 6) for k, v in acc.items()}}


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    for path in argv:
        r = breakdown(load(path))
        print(f"{path}: {r['layers']} layers with a 4070 share over {r['rounds']} rounds (ms per round)")
        for k, v in r["ms"].items():
            print(f"  {k:32}{v:8.2f}")
        print(f"  the 4070 ended after the CPU pool in {r['4070 after pool']:.0%} of those layers")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
