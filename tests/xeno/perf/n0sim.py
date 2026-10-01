"""#84 N0 (v2): replay commit-aware --dump-routing traces (#85/#86) through a byte-bounded model of the host tier.

Trace format (include/strata/core/routing_trace.hpp): records of int32 layer, int32 k, k int32, k float32.  A route
record has 0 <= layer; a negative layer is a tag: -1 commit [window, n_positions, n_accepted, phase] (version 1: three
values, the phase in a separate -5 tag), -2 format [version], -3 request [id], -4 GPU-owned flat ids, -6
host-resident-at-boot flat ids.  Inside a window the route records are layer-major (each layer: positions 0..n-1);
position 0 is the committed input, positions 1..n_accepted the accepted drafts, the rest were rejected.  Phase 0 =
prompt tokens read through windows (serve parts of at most --short-read tokens), 1 = decode.

The baseline policy models the runtime's host tier (src/core/expert_source.cpp):
  - boot: set_capacity takes the order greedily, skipping what does not fit; admitted experts start at score 1
  - a layer's distinct host misses are loaded together (materialize_batch): evict while over the cap, the lowest score
    outside the current layer, ties to the lowest flat index (evict_one); +1 per load
  - +1 per distinct host expert per window-layer (blob/read_into, deduplicated by job_of)
  - every score x0.97 once per verify window (decay_scores)
The owned and boot sets are the trace's boot snapshot: replay traces recorded with --adapt-swaps 0.

  python n0sim.py TRACE --pack PACK_DIR --cap-gib 2 4 6 [--policy baseline]
"""
from __future__ import annotations

import argparse
import heapq
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "tools"))
from make_profile import iter_records  # noqa: E402

FORMAT_VERSIONS = (1, 2)
T_COMMIT, T_FORMAT, T_REQUEST, T_OWNED, T_PHASE_V1, T_BOOT = -1, -2, -3, -4, -5, -6


class Window:
    __slots__ = ("wid", "n_pos", "n_acc", "routes", "phase", "request")

    def __init__(self, wid, n_pos, n_acc, routes, phase, request):
        self.wid, self.n_pos, self.n_acc, self.routes, self.phase, self.request = wid, n_pos, n_acc, routes, phase, request

    def committed(self, layer):
        return self.routes[layer][: self.n_acc + 1]

    def rejected(self, layer):
        return self.routes[layer][self.n_acc + 1:]


class Trace:
    def __init__(self):
        self.version = None
        self.windows: list[Window] = []
        self.owned: set = set()
        self.boot: list = []
        self.dangling = 0   # route records after the last commit (a cut-off run)


def key_of(flat, n_expert):
    return divmod(flat, n_expert)


def load(path, n_expert):
    t = Trace()
    pending: dict = {}
    phase = request = None
    for layer, ids in iter_records(path):
        if layer >= 0:
            pending.setdefault(layer, []).append(ids)
        elif layer == T_FORMAT:
            t.version = ids[0]
            if t.version not in FORMAT_VERSIONS:
                raise ValueError(f"{path}: trace format version {t.version}, this reader handles {FORMAT_VERSIONS}")
        elif layer == T_COMMIT:
            wid, n_pos, n_acc = ids[:3]
            if len(ids) > 3:
                phase = ids[3]
            for l, pos in pending.items():
                if len(pos) != n_pos:
                    raise ValueError(f"{path}: window {wid} layer {l} has {len(pos)} positions, the commit says {n_pos}")
            t.windows.append(Window(wid, n_pos, n_acc, pending, phase, request))
            pending = {}
        elif layer == T_REQUEST:
            request = ids[0]
        elif layer == T_PHASE_V1:
            phase = ids[0]
        elif layer == T_OWNED:
            t.owned |= {key_of(i, n_expert) for i in ids}
        elif layer == T_BOOT:
            t.boot += [key_of(i, n_expert) for i in ids]
    if t.version is None:
        raise ValueError(f"{path}: no format record (version) - a trace from before #85")
    t.dangling = sum(len(v) for v in pending.values())
    return t


def pack_sizes(pack_dir):
    """Bytes of one expert per layer (blob_bytes, what the host tier counts) and n_expert, from the pack's index."""
    import iq_pack
    header, rows = iq_pack.read_index(__import__("pathlib").Path(pack_dir) / "native_experts.txt")
    n_expert = int(header[0].split("n_expert ")[1].split(",")[0])
    return [int(rows[str(l)][4]) for l in range(len(rows))], n_expert


# ------------------------------------------------------------------ policies


class Baseline:
    """The runtime's decayed LFU.  Scores are raw / scale (the decay is a uniform scale, so ranking by raw is ranking
    by score); per-layer heaps of (raw, flat index, version) with lazy deletion pick the victim in O(layers log n)."""

    name = "baseline"

    def __init__(self, n_expert, decay=0.97):
        self.ne, self.decay = n_expert, decay
        self.raw: dict = {}
        self.ver: dict = {}
        self.heaps: dict = {}
        self.scale = 1.0

    def score(self, key):
        return self.raw.get(key, 0.0) * self.scale

    def _push(self, key):
        v = self.ver.get(key, 0) + 1
        self.ver[key] = v
        heapq.heappush(self.heaps.setdefault(key[0], []), (self.raw[key], key[0] * self.ne + key[1], v, key))

    def add(self, key):
        self.raw[key] = self.raw.get(key, 0.0) + 1.0 / self.scale
        self._push(key)

    def on_boot(self, key):
        self.raw[key] = 1.0 / self.scale
        self._push(key)

    def on_load(self, key, w, layer):
        self.add(key)

    def on_access(self, key, w, layer):
        self.add(key)

    def on_window_end(self, w):
        self.scale *= self.decay
        if self.scale < 1e-150:   # renormalise before the float runs out
            self.raw = {k: v * self.scale for k, v in self.raw.items()}
            self.scale = 1.0
            for h in self.heaps.values():
                h[:] = [(self.raw[k], i, v, k) for _, i, v, k in h]
                heapq.heapify(h)

    def victim(self, resident, avoid_layer):
        best = None
        for layer, h in self.heaps.items():
            if layer == avoid_layer:
                continue
            while h and (h[0][3] not in resident or self.ver.get(h[0][3]) != h[0][2]):
                heapq.heappop(h)
            if h and (best is None or h[0][:2] < best[:2]):
                best = h[0]
        return best[3] if best else None


POLICIES = {"baseline": Baseline}


# ------------------------------------------------------------------ the replay


class Stats:
    def __init__(self, policy):
        self.policy = policy
        self.resident: set = set()
        self.boot_resident: set = set()
        self.evicted: list = []
        self.loads = self.load_bytes = self.hits = self.gpu = self.over_cap = 0
        self.rejected_only_loads = 0
        self.emitted = self.rounds = 0

    def score(self, key):
        return self.policy.score(key)

    @property
    def loads_per_token(self):
        return self.loads / max(self.emitted, 1)

    @property
    def loads_per_round(self):
        return self.loads / max(self.rounds, 1)

    @property
    def bytes_per_token(self):
        return self.load_bytes / max(self.emitted, 1)

    @property
    def pollution_share(self):
        """#88: the share of loads (each one an admission) that only a rejected draft position routed."""
        return self.rejected_only_loads / self.loads if self.loads else 0.0


def simulate(trace, sizes, cap_bytes, boot_order, owned, n_expert, policy="baseline", phases=None, **params):
    """Replay `trace` and return a Stats.  `sizes[layer]` = bytes of one expert; `boot_order` = flat ids or
    (layer, expert) keys in the runtime's host-tier order; `owned` = GPU-owned keys; `params` go to the policy."""
    pol = POLICIES[policy](n_expert, **params)
    s = Stats(pol)
    resident, used = s.resident, 0
    for x in boot_order:
        key = x if isinstance(x, tuple) else key_of(x, n_expert)
        if key in owned or key in resident:
            continue
        b = sizes[key[0]]
        if used + b > cap_bytes:
            continue
        used += b
        resident.add(key)
        pol.on_boot(key)
    s.boot_resident = set(resident)
    for w in trace.windows:
        if phases is not None and w.phase not in phases:
            continue
        for layer in sorted(w.routes):
            distinct = dict.fromkeys(e for ids in w.routes[layer] for e in ids if e >= 0)
            host = []
            for e in distinct:
                key = (layer, e)
                if key in owned:
                    s.gpu += 1
                else:
                    host.append(key)
            committed = {e for ids in w.committed(layer) for e in ids}
            for key in host:
                if key in resident:
                    s.hits += 1
                    continue
                if key[1] not in committed:
                    s.rejected_only_loads += 1   # #88: loaded (and admitted) for a rejected draft only
                b = sizes[layer]
                while cap_bytes and used + b > cap_bytes:
                    v = pol.victim(resident, layer)
                    if v is None:
                        break
                    resident.discard(v)
                    used -= sizes[v[0]]
                    s.evicted.append(v)
                if used + b > cap_bytes:
                    s.over_cap += 1
                resident.add(key)
                used += b
                s.loads += 1
                s.load_bytes += b
                pol.on_load(key, w, layer)
            for key in host:
                pol.on_access(key, w, layer)
        pol.on_window_end(w)
        s.emitted += w.n_acc + 1
        s.rounds += 1
    return s


def locality(trace, ks=(1, 2, 4, 5, 8, 16), phases=None):
    """#88: mean over committed tokens t (with history) and layers of |R_t & U(R_{t-1..t-k})| / |R_t|, per k.
    Only committed positions (0..n_accepted) count; rejected drafts never enter the history."""
    last: dict = {}            # layer -> {expert: index of the last committed token that routed it}
    tick: dict = {}            # layer -> committed tokens seen
    hit = {k: 0.0 for k in ks}
    n = 0
    for w in trace.windows:
        if phases is not None and w.phase not in phases:
            continue
        for layer in w.routes:
            seen = last.setdefault(layer, {})
            for ids in w.committed(layer):
                t = tick.get(layer, 0)
                tick[layer] = t + 1
                r = {e for e in ids if e >= 0}
                if t > 0 and r:
                    gaps = [t - seen[e] for e in r if e in seen]
                    for k in ks:
                        hit[k] += sum(1 for g in gaps if g <= k) / len(r)
                    n += 1
                for e in r:
                    seen[e] = t
    return {k: hit[k] / n if n else 0.0 for k in ks}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("trace")
    ap.add_argument("--pack", required=True, help="the pack directory (native_experts.txt: blob_bytes per layer)")
    ap.add_argument("--cap-gib", type=float, nargs="+", default=[6.0])
    ap.add_argument("--policy", nargs="+", default=["baseline"])
    a = ap.parse_args(argv)
    sizes, ne = pack_sizes(a.pack)
    t = load(a.trace, n_expert=ne)
    print(f"trace {a.trace}: {len(t.windows)} windows, owned {len(t.owned)}, boot {len(t.boot)}, "
          f"dangling {t.dangling}")
    for cap in a.cap_gib:
        for pol in a.policy:
            s = simulate(t, sizes, int(cap * 2 ** 30), t.boot, t.owned, ne, policy=pol)
            print(f"cap {cap:5.1f} GiB  {pol:10s}  loads/round {s.loads_per_round:7.3f}  loads/token "
                  f"{s.loads_per_token:6.3f}  MB/token {s.bytes_per_token / 1e6:6.2f}  hits {s.hits}  "
                  f"gpu {s.gpu}  over_cap {s.over_cap}  pollution {s.pollution_share:.3f}")


if __name__ == "__main__":
    main()
