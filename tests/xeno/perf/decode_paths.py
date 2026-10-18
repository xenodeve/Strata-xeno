"""#44: which path each decode layer waited for, from one STRATA_TIMELINE run.

timeline.py sums the decode stages per round; this pairs, per layer, the main thread's "cpu experts" stage with its
"dispatch plan", its "cpu pool" and the 4070's GPU span ("4070 experts", which includes its H2D and D2H), and reports
where the stage's time goes and how often the 4070 finished after the CPU pool; the 4070 sync split by cause; and,
from the "gpu0 decode" lane, the primary GPU's idle time between its graphs charged to the host work it waited for.

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
    if not text.endswith("]") and not text.endswith("}"):   # a server stopped mid-write: drop the partial event
        text = text[:text.rfind("},") + 1]
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


def window_split(events: list[dict]) -> dict:
    """#227 window graph: the 4070 runs one graph per window and has no "4070 experts" span; each layer's work is a
    "4070 segment" on the gpu1 lane (args a = layer), tied to the host clock by the runner.  Per host "cpu experts"
    stage with a 4070 share, pair its "plan 4070 launch", "4070 finish" (starts when the CPU part is done) and
    "4070 wait" with the segment of the same layer, and split the wait: the 4070 starting late after the plan, the
    4070 still computing when the CPU part ended (the straggle a deferral of its output could hide at most), and
    the wait left after the segment had ended (the done flag's readback)."""
    xs = [e for e in events if e.get("ph") == "X"]
    rounds = max(1, sum(1 for e in xs if e["name"] == "decode round"))
    seg: dict[int, list] = defaultdict(list)
    for e in xs:
        if e["name"] == "4070 segment":
            seg[e["args"]["a"]].append((e["ts"], e["ts"] + e["dur"], e["args"].get("b", 0)))
    for v in seg.values():
        v.sort()
    stages = sorted((e for e in xs if e["name"] == "cpu experts"), key=lambda e: e["ts"])
    starts = [e["ts"] for e in stages]
    inner: list[dict] = [{} for _ in stages]
    for e in xs:
        if e["name"] in ("plan 4070 launch", "4070 finish", "4070 wait"):
            i = bisect.bisect_right(starts, e["ts"]) - 1
            if i >= 0 and e["ts"] <= stages[i]["ts"] + stages[i]["dur"]:
                inner[i][e["name"]] = e
    acc: dict[str, float] = defaultdict(float)
    straggles: list[float] = []
    steps: list[tuple] = []
    n = later = 0
    for st, sp in zip(stages, inner):
        if not all(k in sp for k in ("plan 4070 launch", "4070 finish", "4070 wait")):
            continue
        plan, fin, wait = sp["plan 4070 launch"], sp["4070 finish"], sp["4070 wait"]
        lay = seg.get(wait["args"]["a"], [])
        wend = wait["ts"] + wait["dur"]
        k = bisect.bisect_right([s[0] for s in lay], wend) - 1   # the layer's last segment to start before the wait ended
        if k < 0 or lay[k][1] < st["ts"]:
            continue
        g0, g1, gb = lay[k]
        cpu_end = fin["ts"]
        n += 1
        later += g1 > cpu_end
        straggles.append(max(0.0, g1 - cpu_end))
        steps.append((max(0.0, g1 - cpu_end), gb // 1000, gb % 1000))   # straggle, resident groups, staged blobs
        for key, v in (("plan -> segment start", g0 - (plan["ts"] + plan["dur"])), ("segment", g1 - g0),
                       ("4070 straggle past cpu", max(0.0, g1 - cpu_end)), ("4070 wait", wait["dur"]),
                       ("wait with the 4070 done", wend - max(g1, wait["ts"])), ("4070 finish", fin["dur"])):
            acc[key] += v
    # the most a segment shorter by d us could take off the host's wait: each step's straggle, capped at d
    shave = {d: round(sum(min(d, s) for s in straggles) / 1000 / rounds, 6) for d in SHAVE_US}
    return {"layers": n, "rounds": rounds, "4070 after cpu": later / n if n else 0.0,
            "ms": {k: round(v / 1000 / rounds, 6) for k, v in acc.items()}, "shave ms": shave, "steps": steps}


def fetch_overlap(w: dict, fetch_us: float, group_us: float, base_us: float = 0.0) -> float:
    """The most hiding the staged blobs' fetch under the resident groups' compute could take off the host's wait, in
    ms/round: per step that staged, min(the fetch, the resident compute (base_us + group_us per group, nothing without
    a resident group), the step's straggle)."""
    return round(sum(min(fetch_us * st, base_us + group_us * g if g > 0 else 0.0, s)
                     for s, g, st in w["steps"] if st > 0) / 1000 / w["rounds"], 6)


SHAVE_US = (5, 10, 20, 40)
GPU_DECODE = ("verify graph", "commit graph", "mtp round", "mtp step")   # the "gpu0 decode" lane (#44)
EDGE_HOST = ("verify stage", "verify launch", "verify tail", "head sampling", "verify commit", "mtp draft", "emit tokens",
             "adapt join", "adapt apply", "verify capture", "ple gather", "suffix propose")


def edge(events: list[dict]) -> dict:
    """The primary GPU at the rounds' edge: its busy time per graph, and each idle gap between two of its graphs
    charged to the main thread's edge spans that overlap it (the host work the GPU waited for), innermost span first."""
    xs = [e for e in events if e.get("ph") == "X"]
    rounds = max(1, sum(1 for e in xs if e["name"] == "decode round"))
    gpu = sorted((e["ts"], e["ts"] + e["dur"], e["name"]) for e in xs if e["name"] in GPU_DECODE)
    host = sorted((e["ts"], e["ts"] + e["dur"], e["name"]) for e in xs if e["name"] in EDGE_HOST)
    busy: dict[str, float] = defaultdict(float)
    for s, e, n in gpu:
        busy[n] += e - s
    idle = 0.0
    during: dict[str, float] = defaultdict(float)
    for (_, a, _), (b, _, _) in zip(gpu, gpu[1:]):
        if b <= a:
            continue
        idle += b - a
        inside = [(max(s, a), min(e, b), s, n) for s, e, n in host if e > a and s < b]
        cuts = sorted({a, b} | {x for lo, hi, _, _ in inside for x in (lo, hi)})
        for lo, hi in zip(cuts, cuts[1:]):   # each piece goes to the innermost span covering it (the latest start)
            cover = [(s0, n) for l0, h0, s0, n in inside if l0 <= lo and h0 >= hi]
            during[max(cover)[1] if cover else "(no host span)"] += hi - lo
    ms = lambda v: round(v / 1000 / rounds, 6)   # noqa: E731
    return {"gpu busy": {k: ms(v) for k, v in busy.items()}, "gpu idle": ms(idle),
            "idle during": {k: ms(v) for k, v in during.items()}}


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
        w = window_split(ev)
        if w["layers"]:
            print(f"  window graph: {w['layers']} layers with a 4070 segment; the 4070 ended after the CPU part in "
                  f"{w['4070 after cpu']:.0%} of them (ms per round)")
            for k, v in w["ms"].items():
                print(f"    {k:30}{v:8.2f}")
            print(f"    hiding the staged fetch (57 us/blob) under the resident groups (7 us/group) takes at most "
                  f"{fetch_overlap(w, 57.0, 7.0):.2f} ms/round, {fetch_overlap(w, 57.0, 7.0, 40.0):.2f} with the "
                  f"expert kernels' own 40 us")
            print("    a segment shorter by d us takes at most (ms/round): " +
                  ", ".join(f"{d} us {v:.2f}" for d, v in w["shave ms"].items()))
        s = sync_split(ev)
        n = s["done"]["n"] + s["running"]["n"]
        if n:
            print(f"  4070 sync, GPU already done: {s['done']['n'] / n:.0%} of waits, {s['done']['ms_per_round']:.2f} ms/round")
            print(f"  4070 sync, GPU still running: {s['running']['n'] / n:.0%} of waits, "
                  f"{s['running']['ms_per_round']:.2f} ms/round (median GPU left {s['running'].get('gpu_left_us', 0):.1f} us, "
                  f"median wake-up after it {s['running'].get('overshoot_us', 0):.1f} us)")
        g = edge(ev)
        if g["gpu busy"]:
            print(f"  primary GPU at the round edge: busy " +
                  ", ".join(f"{k} {v:.2f}" for k, v in g["gpu busy"].items()) + f"; idle {g['gpu idle']:.2f} ms/round, during:")
            for k, v in sorted(g["idle during"].items(), key=lambda x: -x[1]):
                print(f"    {k:28}{v:8.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
