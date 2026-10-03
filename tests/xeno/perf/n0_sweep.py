"""#91: the N0 sweep - every policy at RAM caches of 2-16 GiB over commit-aware traces (#85/#86), grouped by workload.

Each trace replays from its own boot state.  The boot order is the runtime's host-tier order as far as the trace knows
it: the profile ranking (`--expert-profile`), then every other expert by index, GPU-owned experts skipped; the trace's
own boot set (-6, exact at the run's cap) checks that approximation at that cap.

  python n0_sweep.py --pack PACK --profile PROFILE.bin --out REPORT.md code=a.bin,b.bin thai=c.bin agent=d.bin
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import n0sim  # noqa: E402
from route_tools import read_profile  # noqa: E402

CAPS = (2, 4, 6, 8, 12, 16)
POLICIES = [("baseline", {}), ("spec", {})] + [("window_lfu", {"k": k}) for k in (1, 2, 4, 8, 16)] + \
           [("window", {"k": 4}), ("wtinylfu", {"window_frac": 0.01})]


def label(pol, params):
    return pol + "".join(f" {k}={v}" for k, v in params.items())


def boot_order(profile, n_expert, n_layer):
    seen, order = set(), []
    for key in list(profile) + [(l, e) for l in range(n_layer) for e in range(n_expert)]:
        if key not in seen:
            seen.add(key)
            order.append(key)
    return order


def run(traces, sizes, ne, order, cap_gib, pol, params, **kw):
    agg = dict(loads=0, bytes=0, emitted=0, rounds=0, rej=0, start_h=0, start_l=0, share=0.0, seed=0)
    dists = []
    for t in traces:
        s = n0sim.simulate(t, sizes, int(cap_gib * 2 ** 30), order, t.owned, ne, policy=pol, phases=(1,), **params,
                           **kw)
        agg["loads"] += s.loads
        agg["bytes"] += s.load_bytes
        agg["emitted"] += s.emitted
        agg["rounds"] += s.rounds
        agg["rej"] += s.rejected_only_loads
        agg["start_h"] += s.start_hits
        agg["start_l"] += s.start_loads
        agg["share"] += s.window_share * s.rounds
        agg["seed"] += s.seed_loads
        dists += s.reload_distances
    agg["dist"] = statistics.median(dists) if dists else None
    return agg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pack", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("groups", nargs="+", help="workload=trace,trace,...")
    a = ap.parse_args(argv)
    t0 = time.time()
    sizes, ne = n0sim.pack_sizes(a.pack)
    order = boot_order(read_profile(a.profile), ne, len(sizes))
    groups = []
    for g in a.groups:
        name, paths = g.split("=", 1)
        groups.append((name, paths.split(","), [n0sim.load(p, n_expert=ne) for p in paths.split(",")]))
    out = [f"# N0 sweep (#91)", "", f"pack `{a.pack}`, profile `{a.profile}`; decode windows only (phase 1); "
           f"each trace from its own boot state; sizes from `native_experts.txt`.", ""]
    for name, paths, traces in groups:
        out += [f"## {name}", "", "traces: " + ", ".join(f"`{os.path.basename(p)}`" for p in paths), ""]
        n_win = sum(len(t.windows) for t in traces)
        n_dec = sum(1 for t in traces for w in t.windows if w.phase == 1)
        out += [f"{n_win} windows ({n_dec} decode), orphans {sum(t.orphans for t in traces)}, "
                f"dangling {sum(t.dangling for t in traces)}", ""]
        # locality and pollution: the two questions that decide the lever
        loc = {}
        for t in traces:
            for k, v in n0sim.locality(t, ks=(1, 2, 4, 8, 16), phases=(1,)).items():
                loc.setdefault(k, []).append(v)
        out += ["| k | overlap of R_t with the last k committed tokens (mean over traces) |", "|---:|---:|"]
        out += [f"| {k} | {statistics.mean(v):.3f} |" for k, v in loc.items()]
        out += [""]
        # the boot approximation, at the runs' own cap (6 GiB)
        ex_loads = sum(n0sim.simulate(t, sizes, 6 * 2 ** 30, t.boot, t.owned, ne, phases=(1,)).loads for t in traces)
        ap_loads = run(traces, sizes, ne, order, 6, "baseline", {})["loads"]
        out += [f"boot check at 6 GiB: baseline loads with the trace's exact boot set {ex_loads}, with the profile "
                f"order {ap_loads} ({(ap_loads - ex_loads) / max(ex_loads, 1) * 100:+.1f} %)", ""]
        out += ["| cap GiB | policy | loads/token | MB/token | loads/layer/round | vs baseline | pollution | "
                "start hit rate | window share | median reload distance |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for cap in CAPS:
            base = None
            for pol, params in POLICIES:
                r = run(traces, sizes, ne, order, cap, pol, params)
                lpt = r["loads"] / max(r["emitted"], 1)
                if base is None:
                    base = lpt
                rel = f"{(lpt - base) / base * 100:+.1f} %" if base else "-"
                share = r["share"] / r["rounds"] if r["rounds"] else 0.0
                sh = r["start_h"] / (r["start_h"] + r["start_l"]) if r["start_h"] + r["start_l"] else 0.0
                out.append(f"| {cap} | {label(pol, params)} | {lpt:.2f} | {r['bytes'] / max(r['emitted'], 1) / 1e6:.1f}"
                           f" | {r['loads'] / max(r['rounds'], 1) / len(sizes):.2f} | {rel} | "
                           f"{r['rej'] / max(r['loads'], 1):.3f} | {sh:.3f} | {share:.3f} | {r['dist']} |")
            if any(w.phase == 0 for t in traces for w in t.windows):   # the prompt-tail seed, where prompt windows exist
                for kk in (4, 16, 64):
                    r = run(traces, sizes, ne, order, cap, "spec", {}, seed_k=kk)
                    lpt = r["loads"] / max(r["emitted"], 1)
                    sh = r["start_h"] / (r["start_h"] + r["start_l"]) if r["start_h"] + r["start_l"] else 0.0
                    out.append(f"| {cap} | spec + seed_k={kk} (seed loads {r['seed']}) | {lpt:.2f} | "
                               f"{r['bytes'] / max(r['emitted'], 1) / 1e6:.1f} | "
                               f"{r['loads'] / max(r['rounds'], 1) / len(sizes):.2f} | "
                               f"{(lpt - base) / base * 100:+.1f} % | {r['rej'] / max(r['loads'], 1):.3f} | "
                               f"{sh:.3f} | - | {r['dist']} |")
        out += [""]
    out += [f"sweep time {time.time() - t0:.0f} s"]
    open(a.out, "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
