"""The whole-pipeline latency budget from one STRATA_TIMELINE run (#33).

Usage:
    STRATA_TIMELINE=run.json strata generate ...        (or serve: the file grows after every request)
    python tests/xeno/perf/timeline.py run.json [--top 15] [--json summary.json] [--perfetto closed.json]

The engine records every host thread (main, copy issuer, stagers, pool workers, the 4070 launcher, the adaptive tier)
and the GPU lanes (the prompt path's compute phases, every expert copy on the x4 copy engine, the 4070's expert work)
on one clock.  This script turns that into the budget an optimisation starts from:

  * prompt path: GPU time by phase and by layer; the copy engine's busy time and every idle gap split into
    "not issued yet (host)" - with what the issuing thread was doing instead - "in the issue call" and
    "issued, not started (device)"; the largest gaps by layer; the host thread's own budget;
  * decode: ms per round of every stage (verify window, wait for the GPU, CPU experts and their dispatch stages, the
    pool phases, commit, draft, adaptive tier), the unaccounted rest, the slowest layers, the pool workers' wake-up
    latency and stragglers;
  * every lane's utilisation.

Every number is read from the run's own spans.  --perfetto writes the trace as a closed JSON array for
ui.perfetto.dev (the engine leaves the array open so that it can append after each request).
"""
from __future__ import annotations

import argparse
import bisect
import heapq
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass

EPS = 1e-6


@dataclass
class Span:
    lane: str
    name: str
    t0: float   # ms
    t1: float
    a: int = -1
    b: int = -1

    @property
    def dur(self) -> float:
        return self.t1 - self.t0


def read_events(path: str) -> list:
    """The raw records; an open array (the engine's) is closed, a torn last line (a crash mid-write) dropped."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    lines = text.rstrip().splitlines()
    while lines:
        s = "\n".join(lines).rstrip()
        if s.endswith(","):
            s = s[:-1]
        if not s.endswith("]"):
            s += "\n]"
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            lines.pop()
    return []


class Timeline:
    def __init__(self, events: list):
        names = {}
        for e in events:
            if e.get("ph") == "M" and e.get("name") == "thread_name":
                names[(e.get("pid"), e.get("tid"))] = e["args"]["name"]
        self.lanes: dict[str, list[Span]] = defaultdict(list)
        self.instants: list[Span] = []
        for e in events:
            ph = e.get("ph")
            if ph not in ("X", "i"):
                continue
            key = (e.get("pid"), e.get("tid"))
            lane = names.get(key, f"{key[0]}:{key[1]}")
            args = e.get("args") or {}
            t0 = e["ts"] / 1000.0
            s = Span(lane, e["name"], t0, t0 + e.get("dur", 0.0) / 1000.0, int(args.get("a", -1)), int(args.get("b", -1)))
            (self.lanes[lane] if ph == "X" else self.instants).append(s)
        for v in self.lanes.values():
            v.sort(key=lambda s: (s.t0, -s.dur))
        self._t0 = {ln: [s.t0 for s in v] for ln, v in self.lanes.items()}
        self._maxdur = {ln: max((s.dur for s in v), default=0.0) for ln, v in self.lanes.items()}
        self._inner: dict[str, tuple[list, list]] = {}

    def lane(self, name: str) -> list[Span]:
        return self.lanes.get(name, [])

    def find(self, name: str) -> list[Span]:
        out = [s for v in self.lanes.values() for s in v if s.name == name]
        out.sort(key=lambda s: s.t0)
        return out

    def lanes_like(self, part: str) -> list[str]:
        return [n for n in self.lanes if part in n]

    # windowed lookups (the lanes are sorted by start): a serve trace holds a million spans and thousands of rounds
    def inside(self, lane: str, t0: float, t1: float) -> list[Span]:
        """The spans of `lane` that lie within [t0, t1]."""
        spans, starts = self.lane(lane), self._t0.get(lane, [])
        out, i = [], bisect.bisect_left(starts, t0 - EPS)
        while i < len(spans) and starts[i] <= t1 + EPS:
            if spans[i].t1 <= t1 + EPS:
                out.append(spans[i])
            i += 1
        return out

    def overlapping(self, lane: str, t0: float, t1: float) -> list[Span]:
        """The spans of `lane` that overlap (t0, t1)."""
        spans, starts = self.lane(lane), self._t0.get(lane, [])
        out, i = [], bisect.bisect_left(starts, t0 - self._maxdur.get(lane, 0.0) - EPS)
        while i < len(spans) and starts[i] < t1 - EPS:
            if spans[i].t1 > t0 + EPS:
                out.append(spans[i])
            i += 1
        return out

    # ---- innermost attribution: at every instant of a lane, the shortest span covering it
    def _segments(self, lane: str) -> tuple[list, list]:
        if lane in self._inner:
            return self._inner[lane]
        spans = self.lane(lane)
        bounds = sorted({s.t0 for s in spans} | {s.t1 for s in spans})
        starts = sorted(range(len(spans)), key=lambda i: spans[i].t0)
        heap, segs, j = [], [], 0
        for x0, x1 in zip(bounds, bounds[1:]):
            while j < len(starts) and spans[starts[j]].t0 <= x0 + EPS:
                s = spans[starts[j]]
                heapq.heappush(heap, (s.dur, -s.t0, starts[j]))
                j += 1
            while heap and spans[heap[0][2]].t1 <= x0 + EPS:
                heapq.heappop(heap)
            if heap:
                name = spans[heap[0][2]].name
                if segs and segs[-1][2] == name and abs(segs[-1][1] - x0) < EPS:
                    segs[-1][1] = x1
                else:
                    segs.append([x0, x1, name])
        self._inner[lane] = (segs, [s[0] for s in segs])
        return self._inner[lane]

    def innermost(self, lane: str, g0: float, g1: float) -> dict[str, float]:
        """ms of [g0, g1] charged to the innermost span of `lane` covering each instant ("(no span)" if none)."""
        out: dict[str, float] = defaultdict(float)
        if g1 <= g0:
            return out
        segs, starts = self._segments(lane)
        i = max(0, bisect.bisect_right(starts, g0) - 1)
        covered = 0.0
        while i < len(segs) and segs[i][0] < g1:
            x0, x1, name = segs[i]
            o = min(x1, g1) - max(x0, g0)
            if o > 0:
                out[name] += o
                covered += o
            i += 1
        if g1 - g0 - covered > EPS:
            out["(no span)"] += g1 - g0 - covered
        return out


def load(path: str) -> Timeline:
    """The engine's file, and serve/server.py's <file>.server.json beside it when there is one (same clock)."""
    events = read_events(path)
    server = path + ".server.json"
    if os.path.exists(server):
        events += read_events(server)
    return Timeline(events)


def union_ms(spans: list[Span], lo: float = float("-inf"), hi: float = float("inf")) -> float:
    total, end = 0.0, float("-inf")
    for s in sorted(spans, key=lambda s: s.t0):
        a, b = max(s.t0, lo), min(s.t1, hi)
        if b <= a:
            continue
        if a > end:
            total += b - a
            end = b
        elif b > end:
            total += b - end
            end = b
    return total


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    return ys[min(len(ys) - 1, int(q * (len(ys) - 1) + 0.5))]


def add(d: dict, other: dict) -> None:
    for k, v in other.items():
        d[k] = d.get(k, 0.0) + v


# ---------------------------------------------------------------------------------------------------- prompt path
GAP_KINDS = ("not issued yet (host)", "in the issue call", "issued, not started (device)")


def prefill_report(tl: Timeline, top: int = 15) -> list[dict]:
    copy_lanes = tl.lanes_like("copy engine")
    # #178: CUDA0's compute lanes of the prompt path, one per wave lane ("gpu0 compute (prefill)", "... (prefill, lane 2)";
    # the 4070's "(prefill split...)" lanes are not CUDA0's).  A run takes the lane of its own wave lane: the run on a
    # "prompt wave lane" thread is lane 2.  (Matching "compute (prefill)" alone dropped lane 2 and gave its run lane 1's.)
    gpu0_compute = [ln for ln in tl.lanes_like("compute (prefill") if "split" not in ln]
    compute_of = {second: [ln for ln in gpu0_compute if ("lane 2" in ln) == second] for second in (False, True)}
    # #178: each copy matches its own issuer's span - the 4070 split issuer's "split issue host", the others' "copy
    # issue" - and the two kinds carry their keys in a different order: normalised once by copy_key
    split_lanes = {ln for ln in copy_lanes if "split" in ln}
    issues_of = {kind: tl.find(kind) for kind in ("copy issue", "split issue host")}
    issue_t0_of = {kind: [s.t0 for s in v] for kind, v in issues_of.items()}

    def copy_key(c: Span) -> tuple[int, int]:   # (entry, layer): split copies and their issues carry (layer, expert)
        return (c.b, c.a) if c.lane in split_lanes or c.name == "split issue host" else (c.a, c.b)

    out = []
    for run in tl.find("prefill run"):
        compute_lanes = compute_of["wave lane" in run.lane]
        copies = sorted((s for ln in copy_lanes for s in tl.overlapping(ln, run.t0, run.t1)
                         if s.name.startswith("copy ")), key=lambda s: s.t0)
        r: dict = {"t0_ms": run.t0, "wall_ms": run.dur, "tokens": run.a, "copies": len(copies),
                   "copy_busy_ms": union_ms(copies, run.t0, run.t1),
                   "copies_by_kind": dict(Counter(s.name for s in copies))}
        gaps = {k: 0.0 for k in GAP_KINDS}
        gaps["unmatched"] = 0.0
        cause: dict[str, float] = defaultdict(float)
        top_gaps, latency = [], []

        def issue_of(c: Span):
            kind = "split issue host" if c.lane in split_lanes else "copy issue"
            issues, issue_t0 = issues_of[kind], issue_t0_of[kind]
            i = bisect.bisect_right(issue_t0, c.t0 + EPS) - 1
            while i >= 0 and issues[i].t0 >= run.t0 - EPS:
                if copy_key(issues[i]) == copy_key(c):   # entry and layer (entries restart per layer)
                    return issues[i]
                i -= 1
            return None

        prev_end = None
        for c in copies:
            iss = issue_of(c)
            if iss is not None and c.t0 > iss.t1:
                latency.append(c.t0 - iss.t1)
            if prev_end is not None and c.t0 > prev_end + EPS:
                g0, g1 = prev_end, c.t0
                parts = {k: 0.0 for k in GAP_KINDS}
                why: dict[str, float] = {}
                if iss is None:
                    gaps["unmatched"] += g1 - g0
                else:
                    late = max(0.0, min(iss.t0, g1) - g0)
                    call = max(0.0, min(iss.t1, g1) - max(g0, iss.t0))
                    dev = max(0.0, g1 - max(g0, iss.t1))
                    parts = dict(zip(GAP_KINDS, (late, call, dev)))
                    if late > 0:
                        why = tl.innermost(iss.lane, g0, g0 + late)
                        add(cause, why)
                add(gaps, parts)
                main_cause = max(why.items(), key=lambda kv: kv[1])[0] if why else None
                kind = max(parts.items(), key=lambda kv: kv[1])[0]
                entry, layer = copy_key(c)
                top_gaps.append({"ms": g1 - g0, "t_ms": g0 - run.t0, "entry": entry, "layer": layer,
                                 "cause": main_cause if kind == GAP_KINDS[0] and main_cause else kind,
                                 "parts": parts})
            prev_end = c.t1 if prev_end is None else max(prev_end, c.t1)
        if not gaps["unmatched"]:
            del gaps["unmatched"]
        r["copy_gaps"] = gaps
        r["host_late_cause"] = dict(cause)
        r["issue_to_start_ms"] = {"n": len(latency), "p50": pct(latency, 0.5), "p95": pct(latency, 0.95),
                                  "max": max(latency) if latency else 0.0}
        top_gaps.sort(key=lambda g: -g["ms"])
        r["top_copy_gaps"] = top_gaps[:top]
        phase: dict[str, float] = defaultdict(float)
        layer_phase: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for ln in compute_lanes:
            for s in tl.overlapping(ln, run.t0, run.t1):
                phase[s.name] += s.dur
                layer_phase[s.a][s.name] += s.dur
        r["phase_ms"] = dict(phase)
        r["layer_phase_ms"] = {k: dict(v) for k, v in sorted(layer_phase.items())}
        # each copy against the compute phase running at its midpoint: a copy that stretches while a long kernel
        # runs shows up against that kernel's phase
        comp = sorted((s for ln in compute_lanes for s in tl.overlapping(ln, run.t0, run.t1)), key=lambda s: s.t0)
        cstart = [s.t0 for s in comp]
        by: dict[str, list] = defaultdict(list)
        for c in copies:
            mid = 0.5 * (c.t0 + c.t1)
            i = bisect.bisect_right(cstart, mid) - 1
            by[comp[i].name if i >= 0 and comp[i].t1 >= mid else "(none)"].append(c.dur)
        r["copy_by_phase"] = {k: {"n": len(v), "ms": sum(v), "p50": pct(v, 0.5)} for k, v in by.items()}
        per_layer: dict[int, list] = defaultdict(list)
        for c in copies:
            per_layer[copy_key(c)[1]].append(c)
        r["layer_copy_ms"] = {l: union_ms(v, run.t0, run.t1) for l, v in sorted(per_layer.items())}
        r["host_ms"] = dict(tl.innermost(run.lane, run.t0, run.t1))
        out.append(r)
    return out


# ------------------------------------------------------------------------------------------------------ decode
def decode_report(tl: Timeline) -> dict:
    rounds = tl.find("decode round")
    r: dict = {"rounds": len(rounds)}
    if not rounds:
        return r
    per: dict[str, float] = defaultdict(float)
    unacc = 0.0
    layer: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for rd in rounds:
        kids = [s for s in tl.inside(rd.lane, rd.t0, rd.t1) if s is not rd and s.name != "decode round"]
        for s in kids:
            per[s.name] += s.dur
            if s.name in ("wait gpu", "cpu experts"):
                layer[s.a][s.name].append(s.dur)
        unacc += rd.dur - union_ms(kids, rd.t0, rd.t1)
    n = len(rounds)
    r["round_ms"] = sum(s.dur for s in rounds) / n
    r["per_round_ms"] = {k: v / n for k, v in per.items()}
    r["unaccounted_ms"] = unacc / n
    r["layer_ms"] = {l: {k: sum(v) / len(v) for k, v in d.items()} for l, d in sorted(layer.items())}
    # GPU lanes busy per round (the 4070's share, ...)
    gpu = {}
    for ln in tl.lanes:
        if ln.startswith("gpu") and "prefill" not in ln:
            gpu[ln] = sum(union_ms(tl.overlapping(ln, rd.t0, rd.t1), rd.t0, rd.t1) for rd in rounds) / n
    r["gpu_busy_per_round_ms"] = gpu
    r["workers"] = workers_report(tl)
    return r


def workers_report(tl: Timeline) -> dict:
    """Pool batches, grouped by the host's publish (the start of every worker's wake span)."""
    batches: dict[float, list] = defaultdict(list)
    wakes, per_worker = [], defaultdict(list)
    for ln in tl.lanes_like("pool worker"):
        spans = tl.lane(ln)
        wake_by_end = {round(s.t1, 6): s for s in spans if s.name == "wake"}
        for s in spans:
            if s.name == "wake":
                wakes.append(s.dur)
            elif s.name == "drain":
                w = wake_by_end.get(round(s.t0, 6))
                if w is not None:
                    batches[round(w.t0, 6)].append((ln, s))
                    per_worker[ln].append(s.t1 - w.t0)
    strag = [max(d.t1 for _, d in b) - min(d.t1 for _, d in b) for b in batches.values() if len(b) > 1]
    out = {"batches": len(batches), "straggler_ms": sum(strag) / len(strag) if strag else 0.0,
           "straggler_p95_ms": pct(strag, 0.95), "wake_mean_ms": sum(wakes) / len(wakes) if wakes else 0.0,
           "wake_max_ms": max(wakes) if wakes else 0.0, "wake_p95_ms": pct(wakes, 0.95)}
    if per_worker:
        means = {w: sum(v) / len(v) for w, v in per_worker.items()}
        out["slowest_worker"] = max(means, key=means.get)
        out["worker_finish_ms"] = dict(sorted(means.items()))
    return out


def utilisation(tl: Timeline) -> dict:
    out = {}
    for ln, spans in tl.lanes.items():
        if not spans:
            continue
        lo, hi = min(s.t0 for s in spans), max(s.t1 for s in spans)
        busy = union_ms(spans)
        out[ln] = {"busy_ms": busy, "span_ms": hi - lo, "busy_pct": 100.0 * busy / (hi - lo) if hi > lo else 0.0,
                   "spans": len(spans)}
    return out


def gpu_phase_report(tl: Timeline) -> dict:
    """Every GPU lane (the ones added later too: the 4070's split streams, #32) split by span name, so a lane's waits
    are not read as its work."""
    out = {}
    for ln, spans in tl.lanes.items():
        if not ln.startswith("gpu") or not spans:
            continue
        d: dict[str, float] = defaultdict(float)
        for s in spans:
            d[s.name] += s.dur
        out[ln] = dict(d)
    return out


def layer_window(tl: Timeline, layer: int) -> list[dict]:
    """One layer across every lane whose spans carry it as `a`: per (lane, name) the window from the first start to
    the last end, the busy time and the count, ordered by start - which lane waited on which (#32)."""
    rows: dict[tuple, dict] = {}
    for ln, spans in tl.lanes.items():
        for s in spans:
            if s.a != layer:
                continue
            r = rows.setdefault((ln, s.name), {"lane": ln, "name": s.name, "t0": s.t0, "t1": s.t1, "busy": 0.0, "n": 0})
            r["t0"] = min(r["t0"], s.t0)
            r["t1"] = max(r["t1"], s.t1)
            r["busy"] += s.dur
            r["n"] += 1
    return sorted(rows.values(), key=lambda r: r["t0"])


def thread_report(tl: Timeline) -> dict:
    """Host threads other than main, grouped by name without its number ("pool worker *"), each lane's active range
    split by innermost span: where a helper thread's time goes (a stager that waits for buffers, not for memcpy)."""
    groups: dict[str, dict] = {}
    for ln, spans in tl.lanes.items():
        if not spans or ln == "main" or ln.startswith("gpu") or ln == "timeline anchors":
            continue
        base = ln.rstrip("0123456789").rstrip()
        key = base + " *" if base != ln else ln
        g = groups.setdefault(key, {"lanes": 0, "ms": defaultdict(float)})
        g["lanes"] += 1
        add(g["ms"], tl.innermost(ln, min(s.t0 for s in spans), max(s.t1 for s in spans)))
    return {k: {"lanes": v["lanes"], "ms": dict(v["ms"])} for k, v in groups.items()}


def server_report(tl: Timeline) -> list[dict]:
    """serve/server.py's requests: HTTP to first token and to the end, split into template+tokenize, the FIFO wait
    and the engine round trip; the engine's own "request" inside the round trip; the rest is pipes and parsing."""
    out = []
    firsts = sorted(s.t0 for s in tl.instants if s.name == "first token")
    engine_reqs = [s for s in tl.find("request")]
    for h in tl.find("http request"):
        r = {"t0_ms": h.t0, "http_ms": h.dur}
        for s in tl.inside(h.lane, h.t0, h.t1):
            if s.name in ("template+tokenize", "queue wait", "engine request"):
                r[s.name + "_ms"] = r.get(s.name + "_ms", 0.0) + s.dur
                if s.name == "engine request":
                    side = [e for e in engine_reqs if e.t0 >= s.t0 - EPS and e.t1 <= s.t1 + EPS]
                    if side:
                        r["engine_side_ms"] = sum(e.dur for e in side)
                        r["pipe_ms"] = s.dur - r["engine_side_ms"]
        i = bisect.bisect_left(firsts, h.t0)
        if i < len(firsts) and firsts[i] <= h.t1:
            r["ttft_ms"] = firsts[i] - h.t0
        out.append(r)
    return out


def startup_report(tl: Timeline) -> list[tuple[str, float]]:
    firsts = [s.t0 for n in ("prefill run", "decode round", "request") for s in tl.find(n)[:1]]
    end = min(firsts) if firsts else float("inf")
    return [(s.name, s.dur) for s in tl.lane("main") if s.t1 <= end + EPS and s.name not in ("prompt read (batched)",)]


def export_perfetto(src: str, out: str) -> None:
    with open(out, "w", encoding="utf-8") as f:
        json.dump(read_events(src), f)


# -------------------------------------------------------------------------------------------------------- text
def table(rows: list[tuple], head: tuple) -> str:
    cols = [head] + [tuple(str(c) for c in r) for r in rows]
    w = [max(len(r[i]) for r in cols) for i in range(len(head))]
    return "\n".join("  " + "  ".join(c.ljust(w[i]) if i == 0 else c.rjust(w[i]) for i, c in enumerate(r)) for r in cols)


def ranked(d: dict, total: float | None = None, limit: int = 0) -> list[tuple]:
    items = sorted(d.items(), key=lambda kv: -kv[1])
    if limit:
        items = items[:limit]
    return [(k, f"{v:.1f}", f"{100.0 * v / total:.1f}%" if total else "") for k, v in items]


def render(tl: Timeline, top: int) -> str:
    out = []
    st = startup_report(tl)
    if st:
        out.append("== startup (main thread) ==")
        out.append(table([(n, f"{ms:.0f}") for n, ms in st], ("step", "ms")))
    for r in server_report(tl):
        out.append("== server request at {:.1f} s: ".format(r["t0_ms"] / 1000) + ", ".join(
            f"{k[:-3] if k.endswith('_ms') else k} {v:.0f} ms" for k, v in r.items() if k != "t0_ms") + " ==")
    for req in tl.find("request"):
        out.append(f"== request at {req.t0 / 1000:.1f} s: {req.a} prompt tokens, {req.b} generated, {req.dur:.0f} ms ==")
    for i, r in enumerate(prefill_report(tl, top)):
        tps = r["tokens"] / (r["wall_ms"] / 1000.0) if r["wall_ms"] > 0 and r["tokens"] > 0 else 0.0
        out.append(f"\n== prompt path run {i}: {r['tokens']} tokens in {r['wall_ms']:.0f} ms ({tps:.1f} tok/s) ==")
        tot = sum(r["phase_ms"].values())
        if tot:
            out.append(f"GPU compute lane by phase ({tot:.0f} ms):")
            out.append(table(ranked(r["phase_ms"], tot), ("phase", "ms", "share")))
        out.append(f"host thread ({r['wall_ms']:.0f} ms, innermost span):")
        out.append(table(ranked(r["host_ms"], r["wall_ms"], 12), ("span", "ms", "share")))
        if r["copies"]:
            busy = r["copy_busy_ms"]
            out.append(f"copy engine: {r['copies']} copies {r['copies_by_kind']}, busy {busy:.0f} ms "
                       f"({100.0 * busy / r['wall_ms']:.1f}% of the run)")
            out.append(table(ranked(r["copy_gaps"], r["wall_ms"]), ("idle between copies", "ms", "of run")))
            if r["host_late_cause"]:
                out.append("  not issued yet - what the issuing thread was doing:")
                out.append(table(ranked(r["host_late_cause"], r["wall_ms"], 10), ("span", "ms", "of run")))
            out.append("  copies by the compute phase running at their midpoint:")
            out.append(table([(k, v["n"], f"{v['ms']:.0f}", f"{v['ms'] / v['n']:.3f}", f"{v['p50']:.3f}")
                              for k, v in sorted(r["copy_by_phase"].items(), key=lambda kv: -kv[1]["ms"])],
                             ("phase", "copies", "ms", "mean", "p50")))
            lat = r["issue_to_start_ms"]
            out.append(f"  issue -> copy start: p50 {lat['p50']:.2f} ms, p95 {lat['p95']:.2f} ms, max {lat['max']:.2f} ms")
            out.append("  largest gaps:")
            out.append(table([(f"{g['t_ms']:.0f}", g["layer"], g["entry"], f"{g['ms']:.1f}", g["cause"])
                              for g in r["top_copy_gaps"]], ("at ms", "layer", "entry", "ms", "cause")))
        if r["layer_phase_ms"]:
            names = [k for k, _ in sorted(r["phase_ms"].items(), key=lambda kv: -kv[1])[:6]]
            rows = []
            for l, d in r["layer_phase_ms"].items():
                rows.append((l, f"{sum(d.values()):.1f}", f"{r['layer_copy_ms'].get(l, 0.0):.1f}",
                             *[f"{d.get(n, 0.0):.1f}" for n in names]))
            out.append("GPU compute lane per layer, and the copy engine's busy time for the layer's experts (ms):")
            out.append(table(rows, ("layer", "compute", "copy busy", *names)))
    d = decode_report(tl)
    if d["rounds"]:
        out.append(f"\n== decode: {d['rounds']} rounds, {d['round_ms']:.2f} ms per round ==")
        out.append(table(ranked(d["per_round_ms"], d["round_ms"]) + [("(unaccounted)", f"{d['unaccounted_ms']:.2f}", "")],
                         ("span (inclusive, nested)", "ms/round", "of round")))
        lay = d["layer_ms"]
        if lay:
            worst = sorted(lay.items(), key=lambda kv: -kv[1].get("cpu experts", 0.0))[:5]
            out.append("slowest layers by CPU experts (ms per window):")
            out.append(table([(l, f"{v.get('wait gpu', 0):.2f}", f"{v.get('cpu experts', 0):.2f}") for l, v in worst],
                             ("layer", "wait gpu", "cpu experts")))
        if d["gpu_busy_per_round_ms"]:
            out.append("GPU lanes busy per round: " + ", ".join(f"{k} {v:.2f} ms" for k, v in d["gpu_busy_per_round_ms"].items()))
        w = d["workers"]
        if w["batches"]:
            out.append(f"pool: {w['batches']} batches; wake-up mean {w['wake_mean_ms']:.3f} ms (p95 {w['wake_p95_ms']:.3f}, "
                       f"max {w['wake_max_ms']:.3f}); straggler mean {w['straggler_ms']:.3f} ms (p95 "
                       f"{w['straggler_p95_ms']:.3f}); slowest {w.get('slowest_worker')}")
    th = thread_report(tl)
    if th:
        out.append("\n== helper threads (each lane's active time, innermost span) ==")
        for k, g in sorted(th.items()):
            tot = sum(g["ms"].values())
            parts = ", ".join(f"{n} {100.0 * v / tot:.1f}%" for n, v in sorted(g["ms"].items(), key=lambda kv: -kv[1])[:4])
            out.append(f"  {k} ({g['lanes']} lanes, {tot:.0f} ms): {parts}")
    gp = gpu_phase_report(tl)
    if gp:
        out.append("\n== GPU lanes by phase (ms) ==")
        for ln, d in sorted(gp.items()):
            tot = sum(d.values())
            out.append(f"  {ln} ({tot:.0f} ms): " + ", ".join(
                f"{n} {v:.0f}" for n, v in sorted(d.items(), key=lambda kv: -kv[1])[:8]))
    u = utilisation(tl)
    out.append("\n== lanes ==")
    out.append(table([(k, f"{v['busy_ms']:.0f}", f"{v['span_ms']:.0f}", f"{v['busy_pct']:.1f}%", v["spans"])
                      for k, v in sorted(u.items())], ("lane", "busy ms", "active ms", "busy", "spans")))
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("trace")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--layer", type=int, help="lay one layer out across every lane (who waited on whom)")
    ap.add_argument("--json", help="write the reports as JSON (for comparing runs)")
    ap.add_argument("--perfetto", help="write the trace as a closed JSON array for ui.perfetto.dev")
    a = ap.parse_args(argv)
    tl = load(a.trace)
    print(render(tl, a.top))
    if a.layer is not None:
        w = layer_window(tl, a.layer)
        if w:
            base = min(r["t0"] for r in w)
            print(f"\n== layer {a.layer} across the lanes (ms from its first span) ==")
            print(table([(f"{r['lane']}: {r['name']}", f"{r['t0'] - base:.1f}", f"{r['t1'] - base:.1f}",
                          f"{r['busy']:.1f}", r["n"]) for r in w], ("lane: span", "from", "to", "busy", "n")))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"prefill": prefill_report(tl, a.top), "decode": decode_report(tl), "lanes": utilisation(tl),
                       "threads": thread_report(tl), "server": server_report(tl)},
                      f, indent=1, default=str)
    if a.perfetto:
        export_perfetto(a.trace, a.perfetto)
    return 0


if __name__ == "__main__":
    sys.exit(main())
