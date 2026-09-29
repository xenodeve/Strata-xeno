"""#37: the sizes of the prompt parts the server actually reads, from its log.

Usage:
    python tests/xeno/perf/read_sizes.py D:\\Github\\Strata\\strata-xeno.log [more logs...]

The engine prints one line per prompt part (root / history / new turn / reread), always:
    strata serve: prompt part new turn: 1199 tokens [12800, 13999) of 14000 (batched) in 1500.0 ms
Split + wave only runs chunks of 2,048 tokens or more, so the share of prefill time spent in such parts bounds what
the dual-GPU prompt path can save in daily use.
"""
from __future__ import annotations

import re
import sys

LINE = re.compile(r"strata serve: prompt part (root|history|new turn|reread): (\d+) tokens \[(\d+), (\d+)\) of (\d+) "
                  r"\((batched|windows)\) in ([\d.]+) ms")
BINS = [("< 2K", 0, 2048), ("2-4K", 2048, 4096), ("4-8K", 4096, 8192), ("> 8K", 8192, 1 << 62)]
SPLIT_MIN = 2048   # STREAM_ALL_MIN: the smallest chunk that runs split


def parse(text: str) -> list[dict]:
    out = []
    for m in LINE.finditer(text):
        kind, tok, a, b, total, path, ms = m.groups()
        out.append({"kind": kind, "tokens": int(tok), "start": int(a), "end": int(b), "total": int(total),
                    "path": path, "ms": float(ms)})
    return out


def histogram(parts: list[dict]) -> dict:
    h = {name: {"parts": 0, "tokens": 0, "ms": 0.0} for name, _, _ in BINS}
    for p in parts:
        for name, lo, hi in BINS:
            if lo <= p["tokens"] < hi:
                h[name]["parts"] += 1
                h[name]["tokens"] += p["tokens"]
                h[name]["ms"] += p["ms"]
                break
    return h


def split_eligible_share(parts: list[dict]) -> float:
    total = sum(p["ms"] for p in parts)
    return sum(p["ms"] for p in parts if p["tokens"] >= SPLIT_MIN) / total if total > 0 else 0.0


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    parts = []
    for path in argv:
        parts += parse(open(path, encoding="utf-8", errors="replace").read())
    if not parts:
        print("no prompt-part lines (the engine predates #37, or no request was served)")
        return 1
    total_ms = sum(p["ms"] for p in parts)
    print(f"{len(parts)} prompt parts, {sum(p['tokens'] for p in parts)} tokens, {total_ms / 1000:.1f} s of prompt reading")
    print(f"{'bin':6}{'parts':>7}{'tokens':>10}{'seconds':>9}{'time %':>8}")
    for name, v in histogram(parts).items():
        print(f"{name:6}{v['parts']:7d}{v['tokens']:10d}{v['ms'] / 1000:9.1f}{100 * v['ms'] / total_ms:8.1f}")
    print(f"prefill time in parts of {SPLIT_MIN}+ tokens (what split + wave can act on): "
          f"{100 * split_eligible_share(parts):.1f} %")
    kinds = {}
    for p in parts:
        kinds.setdefault(p["kind"], []).append(p["tokens"])
    print("by kind: " + ", ".join(f"{k} {len(v)} (median {sorted(v)[len(v) // 2]})" for k, v in sorted(kinds.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
