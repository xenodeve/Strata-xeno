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


def sync_split(events: list[dict]) -> dict:
    """The host's "4070 wait" (the blocking event sync in finish()), split by whether the 4070's GPU span had already
    ended when the wait began: if it had, the wait is the sync call's own cost; if not, it is the GPU's remaining time
    plus the host's wake-up after it (the overshoot)."""
    gpu = sorted((e["ts"], e["ts"] + e["dur"]) for e in events if e.get("ph") == "X" and e["name"] == "4070 experts")
    starts = [g[0] for g in gpu]
    rounds = max(1, sum(1 for e in events if e.get("ph") == "X" and e["name"] == "decode round"))
    done, running = [], []
    for e in events:
        if e.get("ph") != "X" or e["name"] != "4070 wait":
            continue
        i = bisect.bisect_right(starts, e["ts"]) - 1
        if i < 0:
            continue
        end = gpu[i][1]
        (done if end <= e["ts"] else running).append((e["dur"], end - e["ts"]))
    med = lambda v: sorted(v)[len(v) // 2] if v else 0.0   # noqa: E731
    out = {"done": {"n": len(done), "ms_per_round": round(sum(d for d, _ in done) / 1000 / rounds, 6)},
           "running": {"n": len(running), "ms_per_round": round(sum(d for d, _ in running) / 1000 / rounds, 6)}}
    if running:
        out["running"]["gpu_left_us"] = med([g for _, g in running])
        out["running"]["overshoot_us"] = med([d - g for d, g in running])
    return out


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    for path in argv:
        ev = load(path)
        r = breakdown(ev)
        print(f"{path}: {r['layers']} layers with a 4070 share over {r['rounds']} rounds (ms per round)")
        for k, v in r["ms"].items():
            print(f"  {k:32}{v:8.2f}")
        print(f"  the 4070 ended after the CPU pool in {r['4070 after pool']:.0%} of those layers")
        s = sync_split(ev)
        n = s["done"]["n"] + s["running"]["n"]
        if n:
            print(f"  4070 sync, GPU already done: {s['done']['n'] / n:.0%} of waits, {s['done']['ms_per_round']:.2f} ms/round")
            print(f"  4070 sync, GPU still running: {s['running']['n'] / n:.0%} of waits, "
                  f"{s['running']['ms_per_round']:.2f} ms/round (median GPU left {s['running'].get('gpu_left_us', 0):.1f} us, "
                  f"median wake-up after it {s['running'].get('overshoot_us', 0):.1f} us)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
