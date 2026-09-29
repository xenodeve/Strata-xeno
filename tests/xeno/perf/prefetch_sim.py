"""#44: could a predictor send a layer's CPU-served experts to a GPU before the router asks, like a drafter?

Replays the decode rounds of `--route-trace` files against a static placement (the first PRIMARY + SECONDARY
experts of a ranked profile sit on the GPUs; the rest are CPU-served).  For each (round, layer) a predictor names K
experts ahead of the router; the layer counts as covered only when EVERY distinct CPU-served expert of it is among the
K, because a layer waits for its slowest path and one leftover expert keeps the CPU on it.

Predictors (expert ids only, causal):
  last   the CPU experts this layer used in the latest rounds, newest first
  cross  the experts that the previous layer's experts led to in the training traces (the previous layer's router has
         run by the time this layer's prefetch must start)
  both   half from each

Usage:
    python tests/xeno/perf/prefetch_sim.py PROFILE.bin PRIMARY SECONDARY --train T... --eval E... [--k 4 8 16]
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from route_tools import read_profile, records  # noqa: E402

BLOB_MB = 1.3824
X16_GBS = 24.0   # 4070 H2D under decode (#22)
DECODE_MAX_TOK = 8   # records with more tokens are the prompt


def score(cpu: set, predicted: list) -> dict:
    hit = len(cpu & set(predicted))
    return {"layer": hit == len(cpu), "experts": hit, "of": len(cpu)}


def cpu_experts(layer: int, ids, placed: set) -> set:
    return {e for e in ids if e >= 0 and (layer, e) not in placed}


class LastRound:
    def __init__(self, k: int, depth: int = 4):
        self.k, self.depth = k, depth
        self.hist: dict[int, list[list[int]]] = defaultdict(list)

    def observe(self, layer: int, experts) -> None:
        h = self.hist[layer]
        h.append(list(dict.fromkeys(experts)))
        del h[:-self.depth]

    def predict(self, layer: int, prev_layer_experts) -> list:
        out: list[int] = []
        for seen in reversed(self.hist.get(layer, [])):
            for e in seen:
                if e not in out:
                    out.append(e)
                    if len(out) == self.k:
                        return out
        return out


class CrossLayer:
    def __init__(self, k: int):
        self.k = k
        self.n: dict[tuple[int, int], Counter] = defaultdict(Counter)

    def train(self, rnd) -> None:
        """rnd: [(layer, experts)] in layer order; experts are this layer's targets (CPU-served ones in the sim)."""
        by = dict(rnd)
        for layer, targets in rnd:
            for p in set(by.get(layer - 1, [])):
                for e in set(targets):
                    self.n[(layer, p)][e] += 1

    def predict(self, layer: int, prev_layer_experts) -> list:
        s: Counter = Counter()
        for p in set(prev_layer_experts):
            s.update(self.n.get((layer, p), {}))
        return [e for e, _ in sorted(s.items(), key=lambda x: (-x[1], x[0]))[: self.k]]


class Both:
    def __init__(self, k: int, cross: CrossLayer):
        self.k, self.last, self.cross = k, LastRound(k), cross

    def observe(self, layer, experts):
        self.last.observe(layer, experts)

    def predict(self, layer, prev_layer_experts):
        out = self.last.predict(layer, prev_layer_experts)[: self.k // 2]
        for e in self.cross.predict(layer, prev_layer_experts):
            if len(out) == self.k:
                break
            if e not in out:
                out.append(e)
        return out


def decode_rounds(path: str):
    """Each decode round as {layer: routed ids} (the prompt's records are skipped)."""
    cur: dict[int, list] = {}
    for layer, nt, _, ids in records(path):
        if nt > DECODE_MAX_TOK:
            continue
        if layer == 0 and cur:
            yield cur
            cur = {}
        cur[layer] = list(ids)
    if cur:
        yield cur


def placement(profile: str, primary: int, secondary: int) -> set:
    return set(map(tuple, read_profile(profile)[: primary + secondary]))


def evaluate(path: str, placed: set, pred, observes: bool) -> dict:
    layers = covered = free = cpu_n = cpu_hit = copies = rounds = 0
    ring: dict[int, set] = {}
    for rnd in decode_rounds(path):
        rounds += 1
        prev_all: list = []
        for layer in sorted(rnd):
            ids = rnd[layer]
            cpu = cpu_experts(layer, ids, placed)
            p = pred.predict(layer, prev_all)
            s = score(cpu, p)
            layers += 1
            free += not cpu
            covered += s["layer"]
            cpu_n += s["of"]
            cpu_hit += s["experts"]
            copies += len(set(p) - ring.get(layer, set()))
            ring[layer] = set(p)
            if observes:
                pred.observe(layer, [e for e in ids if e >= 0 and (layer, e) not in placed])
            prev_all = [e for e in ids if e >= 0]
    return {"rounds": rounds, "layers": layers, "free": free / layers, "covered": covered / layers,
            "experts": cpu_hit / max(1, cpu_n), "copies_per_round": copies / max(1, rounds)}


def oracle(path: str, placed: set, k: int) -> float:
    n = ok = 0
    for rnd in decode_rounds(path):
        for layer, ids in rnd.items():
            n += 1
            ok += len(cpu_experts(layer, ids, placed)) <= k
    return ok / max(1, n)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("primary", type=int)
    ap.add_argument("secondary", type=int)
    ap.add_argument("--train", nargs="*", default=[])
    ap.add_argument("--eval", nargs="+", required=True)
    ap.add_argument("--k", nargs="+", type=int, default=[4, 8, 16])
    a = ap.parse_args(argv)
    placed = placement(a.profile, a.primary, a.secondary)
    for k in a.k:
        cross = CrossLayer(k)
        for t in a.train:
            for rnd in decode_rounds(t):
                cross.train([(l, [e for e in rnd[l] if e >= 0 and (l, e) not in placed]) for l in sorted(rnd)])
        print(f"K = {k} per layer ({k * BLOB_MB:.1f} MB)")
        print(f"  {'trace':14}{'no CPU':>8}{'oracle':>8}  layers covered: {'last':>6}{'cross':>7}{'both':>7}"
              f"   CPU experts covered: {'last':>6}{'cross':>7}{'both':>7}   copies/round (both)  x16 ms")
        for e in a.eval:
            last = evaluate(e, placed, LastRound(k), True)
            cr = evaluate(e, placed, cross, False)
            bo = evaluate(e, placed, Both(k, cross), True)
            name = os.path.basename(e).replace(".trace", "")
            ms = bo["copies_per_round"] * BLOB_MB / X16_GBS
            print(f"  {name:14}{last['free']:8.1%}{oracle(e, placed, k):8.1%}  {'':16}{last['covered']:6.1%}"
                  f"{cr['covered']:7.1%}{bo['covered']:7.1%}   {'':21}{last['experts']:6.1%}{cr['experts']:7.1%}"
                  f"{bo['experts']:7.1%}   {bo['copies_per_round']:8.0f}            {ms:5.1f}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
