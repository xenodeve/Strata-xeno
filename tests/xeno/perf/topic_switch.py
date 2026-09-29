"""#23: the topic-switch benchmark: how fast the expert tiers recover when a conversation changes topic.

Usage:
    python tests/xeno/perf/topic_switch.py run OUT_DIR --exe strata.exe [--arms static,p2,p4,p8]
    python tests/xeno/perf/topic_switch.py report OUT_DIR

`run` starts `strata --serve` once per arm (the dual-GPU serving flags, --adapt-swaps set per arm) with
STRATA_TIMELINE, sends a code prompt and then a Thai prompt as two requests of one engine session, and keeps each
arm's timeline. It refuses to start an arm with less than 12 GB of free RAM. `report` reads the per-window tier
entries the engine records (the "tiers primary/cpu" and "tiers 4070/pcie" instants, cumulative) and prints per
request: windows, tokens, ms, the primary-VRAM share of the routed entries (first windows vs steady state) and the
recovery: windows until the mean share of the next `span` windows is back within `within` of the steady state
(the median of the segment's last half).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(__file__))
import timeline as tlmod  # noqa: E402


@dataclass
class Window:
    t: float
    primary: int
    cpu: int
    g4070: int
    pcie: int

    @property
    def primary_share(self) -> float:
        n = self.primary + self.cpu + self.g4070 + self.pcie
        return self.primary / n if n else 0.0


@dataclass
class Segment:
    t0: float
    t1: float
    tokens: int
    windows: list[Window]


def load(path: str) -> tlmod.Timeline:
    return tlmod.load(path)


def windows(tl: tlmod.Timeline) -> list[Window]:
    a = sorted((s for s in tl.instants if s.name == "tiers primary/cpu"), key=lambda s: s.t0)
    b = sorted((s for s in tl.instants if s.name == "tiers 4070/pcie"), key=lambda s: s.t0)
    out, prev = [], (0, 0, 0, 0)
    for x, y in zip(a, b):
        cur = (x.a, x.b, y.a, y.b)
        d = [c - p for c, p in zip(cur, prev)]
        if any(v < 0 for v in d):   # a new engine process (counters restart)
            d = list(cur)
        out.append(Window(x.t0, d[0], d[1], d[2], d[3]))
        prev = cur
    return out


def segments(tl: tlmod.Timeline) -> list[Segment]:
    ws = windows(tl)
    return [Segment(r.t0, r.t1, r.b, [w for w in ws if r.t0 <= w.t <= r.t1]) for r in tl.find("request")]


def recovery_windows(shares: list[float], within: float = 0.05, span: int = 8) -> int:
    if len(shares) < span:
        return -1
    steady = statistics.median(shares[len(shares) // 2:])
    for i in range(len(shares) - span + 1):
        if sum(shares[i:i + span]) / span >= steady - within:
            return i
    return -1


def report(out_dir: str, within: float, span: int) -> None:
    for name in sorted(os.listdir(out_dir)):
        if not name.endswith(".json"):
            continue
        segs = segments(load(os.path.join(out_dir, name)))
        print(f"== {name[:-5]}")
        for i, s in enumerate(segs):
            sh = [w.primary_share for w in s.windows]
            if not sh:
                continue
            first = sum(sh[:span]) / min(span, len(sh))
            steady = statistics.median(sh[len(sh) // 2:])
            cpu = sum(w.cpu for w in s.windows) / max(1, sum(w.primary + w.cpu + w.g4070 + w.pcie for w in s.windows))
            print(f"  request {i}: {len(sh)} windows, {s.tokens} tokens in {s.t1 - s.t0:.0f} ms "
                  f"({1000 * s.tokens / (s.t1 - s.t0):.1f} tok/s incl. prompt); primary share first {span} windows "
                  f"{first:.3f}, steady {steady:.3f}; CPU share {cpu:.3f}; recovery {recovery_windows(sh, within, span)} windows")


ARMS = {"static": ["--adapt-swaps", "0"], "p2": ["--adapt-swaps", "2"], "p4": ["--adapt-swaps", "4"],
        "p8": ["--adapt-swaps", "8"]}


def free_gb() -> int:
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB)"],
                         capture_output=True, text=True).stdout.strip()
    return int(out or 0)


def run(out_dir: str, exe: str, arms: list[str], base: list[str], prompts: list[str], max_new: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    ids = [open(p, encoding="utf-8").read().replace("\n", ",").replace(" ", ",").strip(",") for p in prompts]
    for arm in arms:
        if free_gb() < 12:
            print(f"{arm}: SKIPPED, less than 12 GB of RAM free")
            continue
        tl_path = os.path.join(out_dir, arm + ".json")
        if os.path.exists(tl_path):
            os.remove(tl_path)
        env = dict(os.environ, STRATA_TIMELINE=tl_path, CUDA_VISIBLE_DEVICES="1,0")
        with open(os.path.join(out_dir, arm + ".stderr"), "w") as err:
            p = subprocess.Popen([exe, "--serve", *base, *ARMS[arm]], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=err, text=True, bufsize=1, env=env)
            for line in p.stdout:
                if line.startswith("READY"):
                    break
            done = []
            for t in ids:
                p.stdin.write(f"GEN {max_new} {','.join(x for x in t.split(',') if x)}\n")
                p.stdin.flush()
                for line in p.stdout:
                    if line.startswith("DONE") or line.startswith("ERR"):
                        done.append(line.strip())
                        break
            p.stdin.close()
            p.wait(timeout=120)
        print(f"{arm}: " + " | ".join(done))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("out_dir")
    r.add_argument("--exe", required=True)
    r.add_argument("--arms", default="static,p2,p4,p8")
    r.add_argument("--prompts", nargs=2, required=True, help="the first-topic and second-topic prompt id files")
    r.add_argument("--max-new", type=int, default=256)
    r.add_argument("--base", required=True, help="the engine args (a JSON list), without --adapt-swaps")
    q = sub.add_parser("report")
    q.add_argument("out_dir")
    q.add_argument("--within", type=float, default=0.05)
    q.add_argument("--span", type=int, default=8)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run(a.out_dir, a.exe, a.arms.split(","), json.loads(a.base), a.prompts, a.max_new)
    else:
        report(a.out_dir, a.within, a.span)
    return 0


if __name__ == "__main__":
    sys.exit(main())
