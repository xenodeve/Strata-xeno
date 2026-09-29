"""The 2-GPU prompt-path simulator (#32): which card should run each MoE layer, from a real routing trace.

Usage:
    STRATA_PREFILL_ROUTE_TRACE=route.bin strata.exe ...       (one record per MoE layer of each prompt chunk)
    python tests/xeno/perf/prefill_route_sim.py route.bin [--x16-gbs 13] [--speed-4070 1.3] ...

For every layer it counts, per policy, the expert bytes each card must stream and over which link, the activation
relay bytes (the cards have no P2P path: 5060 -> host -> 4070 and back, over both links), and the MMQ rows per card.
It then predicts a finish time as the slowest of the concurrent resources: the x4 link, the x16 link, the 5060's
MMQ and the 4070's MMQ. That is optimistic (it ignores the order between a copy and the product that needs it), so
compare policies with it; do not quote it as a prompt time.

Policies: whole_5060 (today), whole_4070 (the layer's experts and a relay of its activations to the 4070),
expert_split (the 5060 keeps the experts it holds, the 4070 computes the rest: #32's accelerator shape),
split_25 / split_50 (that share of the tokens to the 4070, each card loading the unique experts its tokens route
to), and best_per_layer (the cheapest of those per layer).

The 4070's MMQ speed relative to the 5060 is UNMEASURED (an assumption); the x16 bandwidth is measured with a
kernel running (not yet under decode load, #22). The report prints them next to the result.
"""
from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass

MAGIC = 0x52505453


@dataclass
class Layer:
    pos0: int
    layer: int
    T: int
    K: int
    rows: list[int]
    here: list[bool]    # resident on the 5060 (the prompt path's own cache)
    peer: list[bool]    # owned by the 4070 (its only copy)
    ids: list[int]      # T * K routed ids, token-major


@dataclass
class Params:
    blob_mb: float = 1.3824            # one expert (measured: the pack's blob)
    x4_gbs: float = 6.7                # 5060 x4 H2D (measured: 0.206 ms per blob, tl2)
    x16_gbs: float = 19.1              # 4070 x16 H2D (measured: 22.7 GB/s alone, 19.1 during a VRAM-streaming kernel,
                                       # xeno_copy_overlap_probe 2026-09-29; under decode load still #22)
    h_bytes: int = 2560                # hidden size
    relay_bytes_per_value: int = 4     # f32 activations each way (a Q8 relay would be ~1.06)
    ms_per_row_5060: float = 23.2 / 81920   # 5060 MMQ gate/up + down per routed row (measured: tl2, 8K chunk)
    speed_4070: float = 1.3            # 4070 MMQ speed vs the 5060: UNMEASURED


@dataclass
class Cost:
    x4_mb: float = 0.0
    x16_mb: float = 0.0
    relay_mb: float = 0.0
    rows_5060: int = 0
    rows_4070: int = 0
    ms: float = 0.0


def read_trace(path: str) -> list[Layer]:
    data = open(path, "rb").read()
    out, o = [], 0
    while o + 24 <= len(data):
        magic, pos0, layer, T, K, E = struct.unpack_from("<6i", data, o)
        if magic != MAGIC:
            raise ValueError(f"bad record magic at byte {o}")
        o += 24
        ex = struct.unpack_from(f"<{3 * E}i", data, o)
        o += 12 * E
        ids = list(struct.unpack_from(f"<{T * K}i", data, o))
        o += 4 * T * K
        out.append(Layer(pos0, layer, T, K, list(ex[0::3]), [bool(x) for x in ex[1::3]],
                         [bool(x) for x in ex[2::3]], ids))
    return out


def finish(c: Cost, p: Params) -> float:
    return max(c.x4_mb / p.x4_gbs, c.x16_mb / p.x16_gbs, c.rows_5060 * p.ms_per_row_5060,
               c.rows_4070 * p.ms_per_row_5060 / p.speed_4070)


def split_experts(L: Layer, share_4070: float) -> tuple[set[int], set[int]]:
    """Unique experts routed by the 5060's tokens (the first 1 - share) and by the 4070's (the rest)."""
    cut = L.T - int(round(L.T * share_4070))
    a, b = set(), set()
    for t in range(L.T):
        (a if t < cut else b).update(L.ids[t * L.K:(t + 1) * L.K])
    return a, b


def _streams(experts, L: Layer, card: str, p: Params, c: Cost) -> None:
    for e in experts:
        if L.rows[e] == 0:
            continue
        if card == "5060":
            if L.here[e]:
                continue
            c.x4_mb += p.blob_mb
            if L.peer[e]:              # its only copy is on the 4070: out over x16, in over x4
                c.x16_mb += p.blob_mb
        else:
            if L.peer[e]:
                continue
            c.x16_mb += p.blob_mb
            if L.here[e]:              # its only copy is on the 5060: out over x4, in over x16
                c.x4_mb += p.blob_mb


def _relay(tokens: int, p: Params, c: Cost) -> None:
    mb = tokens * p.h_bytes * p.relay_bytes_per_value / 1e6 * 2   # activations there, results back
    c.relay_mb += mb
    c.x4_mb += mb
    c.x16_mb += mb


def whole_5060(L: Layer, p: Params) -> Cost:
    c = Cost()
    _streams(range(len(L.rows)), L, "5060", p, c)
    c.rows_5060 = sum(L.rows)
    c.ms = finish(c, p)
    return c


def whole_4070(L: Layer, p: Params) -> Cost:
    c = Cost()
    _streams(range(len(L.rows)), L, "4070", p, c)
    _relay(L.T, p, c)
    c.rows_4070 = sum(L.rows)
    c.ms = finish(c, p)
    return c


def split(L: Layer, p: Params, share_4070: float) -> Cost:
    c = Cost()
    a, b = split_experts(L, share_4070)
    _streams(a, L, "5060", p, c)
    _streams(b, L, "4070", p, c)
    n4070 = int(round(L.T * share_4070))
    _relay(n4070, p, c)
    c.rows_4070 = n4070 * L.K
    c.rows_5060 = L.T * L.K - c.rows_4070
    c.ms = finish(c, p)
    return c


def expert_split(L: Layer, p: Params) -> Cost:
    """#32's shape: the 5060 keeps the experts it holds, the 4070 computes every other routed expert (host ones over
    x16, its own for free); the tokens that route to at least one 4070 expert are relayed there and back."""
    c = Cost()
    on_4070 = [e for e in range(len(L.rows)) if L.rows[e] and not L.here[e]]
    _streams(on_4070, L, "4070", p, c)
    far = set(on_4070)
    tokens = sum(1 for t in range(L.T) if any(e in far for e in L.ids[t * L.K:(t + 1) * L.K]))
    _relay(tokens, p, c)
    c.rows_4070 = sum(L.rows[e] for e in on_4070)
    c.rows_5060 = sum(L.rows) - c.rows_4070
    c.ms = finish(c, p)
    return c


POLICIES = {"whole_5060": whole_5060, "whole_4070": whole_4070, "expert_split": expert_split,
            "split_25": lambda L, p: split(L, p, 0.25), "split_50": lambda L, p: split(L, p, 0.5)}


def totals(layers: list[Layer], p: Params) -> dict:
    out = {k: {"ms": 0.0, "x4_mb": 0.0, "x16_mb": 0.0, "relay_mb": 0.0} for k in POLICIES}
    out["best_per_layer"] = {"ms": 0.0, "x4_mb": 0.0, "x16_mb": 0.0, "relay_mb": 0.0, "picks": {}}
    for L in layers:
        costs = {k: f(L, p) for k, f in POLICIES.items()}
        for k, c in costs.items():
            for f in ("ms", "x4_mb", "x16_mb", "relay_mb"):
                out[k][f] += getattr(c, f)
        k_best = min(costs, key=lambda k: costs[k].ms)
        b = out["best_per_layer"]
        for f in ("ms", "x4_mb", "x16_mb", "relay_mb"):
            b[f] += getattr(costs[k_best], f)
        b["picks"][k_best] = b["picks"].get(k_best, 0) + 1
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("trace")
    d = Params()
    for f, v in vars(d).items():
        ap.add_argument("--" + f.replace("_", "-"), type=type(v), default=v)
    a = ap.parse_args(argv)
    p = Params(**{f: getattr(a, f) for f in vars(d)})
    layers = read_trace(a.trace)
    chunks = sorted({L.pos0 for L in layers})
    print(f"{len(layers)} MoE layer records, {len(chunks)} chunk(s) of up to {max(L.T for L in layers)} tokens")
    print(f"params: x4 {p.x4_gbs} GB/s, x16 {p.x16_gbs} GB/s, 4070 MMQ x{p.speed_4070} (UNMEASURED), "
          f"relay {p.relay_bytes_per_value} B/value, blob {p.blob_mb} MB")
    routed = sum(sum(1 for r in L.rows if r) for L in layers) / len(layers)
    print(f"experts routed per layer: {routed:.0f} of {len(layers[0].rows)}; resident on the 5060 "
          f"{sum(sum(L.here) for L in layers) / len(layers):.0f}, owned by the 4070 {sum(sum(L.peer) for L in layers) / len(layers):.0f}")
    t = totals(layers, p)
    print(f"\n{'policy':16}{'predicted ms':>14}{'x4 MB':>10}{'x16 MB':>10}{'relay MB':>10}")
    for k, v in t.items():
        print(f"{k:16}{v['ms']:14.0f}{v['x4_mb']:10.0f}{v['x16_mb']:10.0f}{v['relay_mb']:10.0f}")
    print("best per layer picks:", t["best_per_layer"]["picks"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
