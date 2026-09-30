"""#41: a static per-layer expert order for STRATA_EXPERT_ORDER, from prefill route traces.

Usage:
    python tests/xeno/perf/make_expert_order.py OUT.bin TRACE [TRACE ...]      (STRATA_PREFILL_ROUTE_TRACE files)

Each layer's experts run by their mean router rank (the k of their (token, k) rows), lowest first; experts the traces
never routed come last, in id order; ties break by id.  With the Dm frontier this keeps the waiting rows near 30 % of
T*K instead of 49 % in id order (dm_frontier_sim, 2026-09-30: an order from a 2K prompt gave 29.7 % on an 8K one).
The file: int32 magic 0x4F585053, layers, experts, then layers x experts expert ids (position -> expert).
"""
from __future__ import annotations

import struct
import sys

MAGIC = 0x4F585053


def order_from_routes(routes: dict, n_layers: int, n_expert: int) -> list[list[int]]:
    """routes: layer -> list of per-token rank lists (expert id at each k, -1 for none)."""
    out = []
    for l in range(n_layers):
        s = [0.0] * n_expert
        c = [0] * n_expert
        for ks in routes.get(l, []):
            for k, e in enumerate(ks):
                if 0 <= e < n_expert:
                    s[e] += k
                    c[e] += 1
        seen = sorted((e for e in range(n_expert) if c[e]), key=lambda e: (s[e] / c[e], e))
        out.append(seen + [e for e in range(n_expert) if not c[e]])
    return out


def write_order(path: str, order: list[list[int]]) -> None:
    n_expert = len(order[0])
    with open(path, "wb") as f:
        f.write(struct.pack("<3i", MAGIC, len(order), n_expert))
        for layer in order:
            f.write(struct.pack(f"<{n_expert}i", *layer))


def read_order(path: str) -> list[list[int]]:
    b = open(path, "rb").read()
    magic, layers, n_expert = struct.unpack_from("<3i", b, 0)
    assert magic == MAGIC
    v = struct.unpack_from(f"<{layers * n_expert}i", b, 12)
    return [list(v[l * n_expert:(l + 1) * n_expert]) for l in range(layers)]


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    sys.path.insert(0, __import__("os").path.dirname(__file__))
    from prefill_route_sim import read_trace   # noqa: E402
    routes: dict = {}
    n_layers = n_expert = 0
    for tr in argv[1:]:
        for L in read_trace(tr):
            n_layers = max(n_layers, L.layer + 1)
            n_expert = max(n_expert, len(L.rows))
            routes.setdefault(L.layer, []).extend(list(L.ids[t * L.K:(t + 1) * L.K]) for t in range(L.T))
    order = order_from_routes(routes, n_layers, n_expert)
    write_order(argv[0], order)
    print(f"{argv[0]}: {n_layers} layers x {n_expert} experts from {sum(len(v) for v in routes.values())} token-layers")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
