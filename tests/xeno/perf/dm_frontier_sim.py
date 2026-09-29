"""#41: would a k-order frontier shrink Dm?  Replays prefill route traces.

Usage:
    python tests/xeno/perf/dm_frontier_sim.py TRACE [--min-tokens 4096]

Today Dm holds every (token, k) row of a layer (T*K*N f32, 839 MB at 8K) because moe_combine adds a token's rows in
router-rank order k = 0..9 - the fmaf order byte identity depends on.  A frontier buffer would keep only rows that
cannot be added yet: row (t, k) waits until rows (t, 0..k-1) are produced.  Experts produce their rows one expert at
a time, in some order; the peak number of waiting rows is what the buffer would need.

For each layer record of chunks of at least --min-tokens tokens it reports that peak as a share of T*K for:
  id order (today's), and order_by_mean_rank (experts whose rows sit at low ranks first) - a heuristic, not an optimum.
The expert order is free for the weights (each expert streams once in any order), so a good order costs nothing.
The trace format is STRATA_PREFILL_ROUTE_TRACE's (tests/xeno/perf/prefill_route_sim.py read_trace).
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, __import__("os").path.dirname(__file__))


def peak_held(routes: list[list[int]], order: list[int]) -> int:
    """routes[t][k] = expert id of token t's rank-k route.  Rows (t, k) arrive expert by expert in `order`; a token's
    sum takes its rows strictly in k order.  Returns the peak number of rows waiting (after each expert's rows are
    taken as far as they can be)."""
    by_expert: dict[int, list[tuple[int, int]]] = {}
    for t, ks in enumerate(routes):
        for k, e in enumerate(ks):
            if e >= 0:
                by_expert.setdefault(e, []).append((t, k))
    K = max((len(ks) for ks in routes), default=0)
    have = [[False] * K for _ in routes]
    nxt = [0] * len(routes)   # the next rank each token's sum needs
    held = peak = 0
    for e in order:
        for t, k in by_expert.get(e, ()):
            have[t][k] = True
            held += 1
        for t, _ in by_expert.get(e, ()):
            while nxt[t] < K and (have[t][nxt[t]] or routes[t][nxt[t]] < 0):
                if routes[t][nxt[t]] >= 0:
                    held -= 1
                nxt[t] += 1
        peak = max(peak, held)
    return peak


def order_by_mean_rank(routes: list[list[int]], n_expert: int) -> list[int]:
    s = [0.0] * n_expert
    c = [0] * n_expert
    for ks in routes:
        for k, e in enumerate(ks):
            if e >= 0:
                s[e] += k
                c[e] += 1
    return sorted((e for e in range(n_expert) if c[e]), key=lambda e: s[e] / c[e])


def main(argv: list[str] | None = None) -> int:
    from prefill_route_sim import read_trace   # noqa: E402

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("trace")
    ap.add_argument("--min-tokens", type=int, default=4096)
    a = ap.parse_args(argv)
    rows = []
    for L in read_trace(a.trace):
        T, K, E = L.T, L.K, len(L.rows)
        if T < a.min_tokens:
            continue
        routes = [list(L.ids[t * K:(t + 1) * K]) for t in range(T)]
        ids_order = sorted({e for ks in routes for e in ks if e >= 0})
        p_id = peak_held(routes, ids_order)
        p_rank = peak_held(routes, order_by_mean_rank(routes, E))
        rows.append((L.layer, T, T * K, p_id, p_rank))
        print(f"layer {L.layer:2d} T {T:5d}: id order {p_id / (T * K):6.1%}   mean-rank order {p_rank / (T * K):6.1%}",
              flush=True)
    if not rows:
        print(f"no layer record with T >= {a.min_tokens}")
        return 1
    tk = sum(r[2] for r in rows)
    print(f"{len(rows)} layer records: peak held rows / T*K - id order {max(r[3] / r[2] for r in rows):.1%} max, "
          f"{sum(r[3] for r in rows) / tk:.1%} mean; mean-rank order {max(r[4] / r[2] for r in rows):.1%} max, "
          f"{sum(r[4] for r in rows) / tk:.1%} mean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
