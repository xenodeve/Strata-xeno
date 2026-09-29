"""#44 D2: where the primary GPU's decode time goes, per region of a layer, from an Nsight Systems capture.

A decode layer on the primary runs in a fixed order, and four kernels mark its regions:

  trunk                  merge_mapped of the previous layer .. doorbell_publish: hyper-connection, attention / GDN,
                         the router (and, at a round's edge, the head, the sampler and the MTP draft)
  publish routes         doorbell_publish_kernel: the routes out to the host
  shared expert          doorbell .. the first wait_flag: the shared expert's FFN and its gate
  wait plan              the first wait_flag_ge_kernel: the host's dispatch plan
  routed experts (5060)  .. the second wait_flag: the plan read and the primary's own routed experts
  wait partials          the second wait_flag_ge_kernel: the CPU pool's and the 4070's partials
  merge partials         merge_mapped_kernel: the partials read from mapped host memory

Each kernel is charged its duration plus the idle gap before the next kernel, so the regions add up to the capture;
the idle column is that gap alone (at a round's edge it is the GPU waiting for the host's verify tail and draft).

Usage:
    python tests/xeno/perf/decode_gpu_layers.py CAPTURE.sqlite [--layers 48] [--top 6]
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter, defaultdict

REGIONS = ["trunk", "publish routes", "shared expert", "wait plan", "routed experts (5060)", "wait partials",
           "merge partials"]


def split(kernels: list[tuple[int, int, str]]) -> dict:
    """kernels: (start, end, name) of the primary GPU in ns, any order."""
    ks = sorted(kernels)
    ns: dict[str, int] = defaultdict(int)
    idle: dict[str, int] = defaultdict(int)
    by: dict[str, Counter] = defaultdict(Counter)
    region = None   # nothing is charged before the first doorbell
    waits = 0
    layers = 0
    for i, (s, e, name) in enumerate(ks):
        cost = (ks[i + 1][0] if i + 1 < len(ks) else e) - s
        if name == "doorbell_publish_kernel":
            layers += 1
            waits = 0
            here, region = "publish routes", "shared expert"
        elif name == "wait_flag_ge_kernel" and region is not None:
            waits += 1
            here = "wait plan" if waits == 1 else "wait partials"
            region = "routed experts (5060)" if waits == 1 else region
        elif name == "merge_mapped_kernel" and region is not None:
            here, region = "merge partials", "trunk"
        else:
            here = region
        if here is None:
            continue
        ns[here] += cost
        idle[here] += cost - (e - s)
        by[here][name] += e - s
    return {"layers": layers, "ns": dict(ns), "idle": dict(idle), "kernels": {k: dict(v) for k, v in by.items()}}


def load(path: str) -> tuple[list, list]:
    db = sqlite3.connect(path)
    rows = db.execute("""select k.deviceId, k.start, k.end, s.value from CUPTI_ACTIVITY_KIND_KERNEL k
                         join StringIds s on s.id = k.shortName""").fetchall()
    per: dict[int, list] = defaultdict(list)
    for d, s, e, n in rows:
        per[d].append((s, e, n))
    primary = max(per, key=lambda d: sum(1 for _, _, n in per[d] if n == "doorbell_publish_kernel"))
    other = [k for d, v in per.items() if d != primary for k in v]
    return per[primary], other


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--top", type=int, default=6)
    a = ap.parse_args(argv)
    prim, other = load(a.capture)
    r = split(prim)
    rounds = r["layers"] / a.layers
    span = (max(e for _, e, _ in prim) - min(s for s, _, _ in prim)) / 1e6
    print(f"{a.capture}: {r['layers']} layers = {rounds:.1f} rounds; primary span {span / rounds:.2f} ms/round; "
          f"other GPU busy {sum(e - s for s, e, _ in other) / 1e6 / rounds:.2f} ms/round")
    print(f"  {'region':24}{'ms/round':>9}{'share':>7}{'idle':>7}   top kernels (busy ms/round)")
    total = sum(r["ns"].values())
    for reg in REGIONS:
        v = r["ns"].get(reg, 0)
        top = sorted(r["kernels"].get(reg, {}).items(), key=lambda x: -x[1])[: a.top]
        tops = ", ".join(f"{n.replace('_kernel', '')} {t / 1e6 / rounds:.2f}" for n, t in top)
        print(f"  {reg:24}{v / 1e6 / rounds:9.2f}{v / total:7.1%}{r['idle'].get(reg, 0) / 1e6 / rounds:7.2f}   {tops}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
