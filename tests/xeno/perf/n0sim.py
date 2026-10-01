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
        self.orphans = 0    # uncommitted route records before a later window (a cancelled window, the per-token loop)


def key_of(flat, n_expert):
    return divmod(flat, n_expert)


def load(path, n_expert):
    t = Trace()
    pending: dict = {}
    phase = request = None
    last_layer = -1
    for layer, ids in iter_records(path):
        if layer >= 0:
            if layer < last_layer and pending:   # a new window began with no commit for the last: drop the orphans
                t.orphans += sum(len(v) for v in pending.values())
                pending = {}
            last_layer = layer
            pending.setdefault(layer, []).append(ids)
            continue
        if layer == T_FORMAT:
            t.version = ids[0]
            if t.version not in FORMAT_VERSIONS:
                raise ValueError(f"{path}: trace format version {t.version}, this reader handles {FORMAT_VERSIONS}")
        elif layer == T_COMMIT:
            last_layer = -1
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

    def on_evict(self, key):
        pass   # lazy: the heap entry goes stale and is dropped when it reaches the top

    def victim(self, resident, avoid_layer, skip=frozenset()):
        """The lowest-scoring resident key outside `avoid_layer` and `skip` (ties: lowest flat index); if `skip` leaves
        nothing, the lowest one ignoring it."""
        best = None
        for layer, h in self.heaps.items():
            if layer == avoid_layer:
                continue
            held = []
            while h:
                top = h[0]
                if top[3] not in resident or self.ver.get(top[3]) != top[2]:
                    heapq.heappop(h)
                elif top[3] in skip:
                    held.append(heapq.heappop(h))
                else:
                    break
            if h and (best is None or h[0][:2] < best[:2]):
                best = h[0]
            for e in held:
                heapq.heappush(h, e)
        if best is None and skip:   # Baseline's own pick, not a subclass's override (which would pass skip again)
            return Baseline.victim(self, resident, avoid_layer)
        return best[3] if best else None


class _RecentWindow:
    """#89: the union, per layer, of the experts the last k committed tokens routed (Apple's sliding window, with
    experts in place of neurons).  Updated at each window's end, so a window never protects its own positions."""

    def __init__(self, k):
        from collections import deque
        self.k = k
        self.hist: dict = {}
        self.deque = deque

    def update(self, w):
        if self.k <= 0:
            return
        for layer in w.routes:
            h = self.hist.setdefault(layer, self.deque(maxlen=self.k))
            for ids in w.committed(layer):
                h.append({e for e in ids if e >= 0})
        self._prot = None

    def protected(self):
        if getattr(self, "_prot", None) is None:   # cached until the next update
            self._prot = frozenset((layer, e) for layer, h in self.hist.items() for s in h for e in s)
        return self._prot


class WindowLFU(Baseline):
    """#89: a protected recent-tokens window, the runtime's decayed LFU for everything else.  k = 0 is the baseline."""

    name = "window_lfu"

    def __init__(self, n_expert, k=4, decay=0.97):
        super().__init__(n_expert, decay)
        self.win = _RecentWindow(k)

    def on_window_end(self, w):
        super().on_window_end(w)
        self.win.update(w)

    def victim(self, resident, avoid_layer, skip=frozenset()):
        return super().victim(resident, avoid_layer, self.win.protected())


class WindowLRU(Baseline):
    """#89: window-k alone - protect the recent-tokens window, evict the least recently used of the rest."""

    name = "window"

    def __init__(self, n_expert, k=4, decay=0.97):
        super().__init__(n_expert, decay)
        from collections import OrderedDict
        self.win = _RecentWindow(k)
        self.lru = OrderedDict()

    def _touch(self, key):
        self.lru[key] = None
        self.lru.move_to_end(key)

    def on_boot(self, key):
        self._touch(key)

    def on_load(self, key, w, layer):
        self._touch(key)

    def on_access(self, key, w, layer):
        self._touch(key)

    def on_evict(self, key):
        self.lru.pop(key, None)

    def on_window_end(self, w):
        self.win.update(w)

    def victim(self, resident, avoid_layer, skip=frozenset()):
        prot = self.win.protected()
        fallback = None
        for key in self.lru:
            if key[0] == avoid_layer:
                continue
            if key not in prot:
                return key
            if fallback is None:
                fallback = key
        return fallback


class WTinyLFU(Baseline):
    """#89 reference design: a small LRU window segment (`window_frac` of the cache) in front of an LRU main segment;
    when the window overflows, its oldest entry enters the main segment if there is room, or if it is used more often
    (decayed count) than the main segment's oldest entry, which is then evicted - otherwise the candidate is evicted."""

    name = "wtinylfu"

    def __init__(self, n_expert, window_frac=0.01, decay=0.97):
        super().__init__(n_expert, decay)
        from collections import OrderedDict
        self.frac = window_frac
        self.window, self.main = OrderedDict(), OrderedDict()
        self.wbytes = self.mbytes = 0
        self.sizes, self.wcap, self.mcap = None, 0, 0

    def bind(self, sizes, cap_bytes):
        self.sizes = sizes
        self.wcap = int(cap_bytes * self.frac)
        self.mcap = cap_bytes - self.wcap

    def _freq(self, key):
        return self.raw.get(key, 0.0)

    def _put(self, seg, key):
        if key in self.window:
            self.window.move_to_end(key)
        elif key in self.main:
            self.main.move_to_end(key)
        elif seg == "w":
            self.window[key] = None
            self.wbytes += self.sizes[key[0]]
        else:
            self.main[key] = None
            self.mbytes += self.sizes[key[0]]

    def on_boot(self, key):
        self.raw[key] = 1.0 / self.scale
        self._put("m", key)

    def on_load(self, key, w, layer):
        self.raw[key] = self.raw.get(key, 0.0) + 1.0 / self.scale
        self._put("w", key)

    def on_access(self, key, w, layer):
        self.raw[key] = self.raw.get(key, 0.0) + 1.0 / self.scale
        self._put("w", key)

    def on_evict(self, key):
        if self.window.pop(key, 0) is None:
            self.wbytes -= self.sizes[key[0]]
        elif self.main.pop(key, 0) is None:
            self.mbytes -= self.sizes[key[0]]

    def on_window_end(self, w):
        self.scale *= self.decay
        if self.scale < 1e-150:
            self.raw = {k: v * self.scale for k, v in self.raw.items()}
            self.scale = 1.0

    @staticmethod
    def _oldest(seg, avoid_layer):
        return next((k for k in seg if k[0] != avoid_layer), None)

    def _promote(self, key):
        del self.window[key]
        self.wbytes -= self.sizes[key[0]]
        self.main[key] = None
        self.mbytes += self.sizes[key[0]]

    def victim(self, resident, avoid_layer, skip=frozenset()):
        while True:
            cand = self._oldest(self.window, avoid_layer)
            if cand is not None and self.wbytes > self.wcap:
                if self.mbytes + self.sizes[cand[0]] <= self.mcap:
                    self._promote(cand)
                    continue
                v = self._oldest(self.main, avoid_layer)
                if v is None or self._freq(cand) > self._freq(v):
                    self._promote(cand)
                    if v is not None:
                        return v
                    continue
                return cand
            v = self._oldest(self.main, avoid_layer)
            return v if v is not None else cand


class SpecAware(Baseline):
    """#90: only committed positions raise scores.  An expert loaded for a rejected draft position alone goes on
    probation - no score, first to evict - and leaves probation when a committed position uses it."""

    name = "spec"

    def __init__(self, n_expert, decay=0.97):
        super().__init__(n_expert, decay)
        self.probation: set = set()
        self._cw = (None, None, frozenset())

    def _committed(self, w, layer):
        if self._cw[0] is not w or self._cw[1] != layer:
            self._cw = (w, layer, frozenset(e for ids in w.committed(layer) for e in ids))
        return self._cw[2]

    def on_load(self, key, w, layer):
        if key[1] in self._committed(w, layer):
            self.add(key)
        else:
            self.probation.add(key)

    def on_access(self, key, w, layer):
        if key[1] in self._committed(w, layer):
            self.add(key)
            self.probation.discard(key)

    def on_evict(self, key):
        self.probation.discard(key)

    def victim(self, resident, avoid_layer, skip=frozenset()):
        cands = [k for k in self.probation if k in resident and k[0] != avoid_layer]
        if cands:
            return min(cands, key=lambda k: k[0] * self.ne + k[1])
        return super().victim(resident, avoid_layer)


class _PositionCounts:
    """Per (window, layer): how many positions route each expert, and which experts a rejected position routes."""

    def __init__(self):
        self._w = self._layer = None
        self.count, self.rejected = {}, frozenset()

    def of(self, w, layer):
        if self._w is not w or self._layer != layer:
            self._w, self._layer, self.count = w, layer, {}
            for ids in w.routes[layer]:
                for e in set(ids):
                    self.count[e] = self.count.get(e, 0) + 1
            self.rejected = frozenset(e for ids in w.rejected(layer) for e in ids)
        return self


class PerPosition(Baseline):
    """Follow-up to #90: rejected drafts' experts are reused soon, so weight them up instead of down - an access scores
    one per position that routes the expert in the window-layer, not one per distinct expert."""

    name = "per_position"

    def __init__(self, n_expert, decay=0.97):
        super().__init__(n_expert, decay)
        self.pc = _PositionCounts()

    def on_access(self, key, w, layer):
        n = self.pc.of(w, layer).count.get(key[1], 1)
        self.raw[key] = self.raw.get(key, 0.0) + n / self.scale
        self._push(key)


class DraftBonus(Baseline):
    """Follow-up to #90: the baseline's +1 per access, plus `bonus` when a rejected draft position routes the expert."""

    name = "draft_bonus"

    def __init__(self, n_expert, bonus=1.0, decay=0.97):
        super().__init__(n_expert, decay)
        self.bonus = bonus
        self.pc = _PositionCounts()

    def on_access(self, key, w, layer):
        extra = self.bonus if key[1] in self.pc.of(w, layer).rejected else 0.0
        self.raw[key] = self.raw.get(key, 0.0) + (1.0 + extra) / self.scale
        self._push(key)


POLICIES = {p.name: p for p in (Baseline, WindowLFU, WindowLRU, WTinyLFU, SpecAware, PerPosition, DraftBonus)}


# ------------------------------------------------------------------ the replay


class Stats:
    def __init__(self, policy):
        self.policy = policy
        self.resident: set = set()
        self.boot_resident: set = set()
        self.evicted: list = []
        self.loads = self.load_bytes = self.hits = self.gpu = self.over_cap = 0
        self.rejected_only_loads = 0
        self.seed_loads = self.seed_bytes = 0
        self.start_hits = self.start_loads = 0
        self.emitted = self.rounds = 0
        self.reload_distances: list = []   # #89: windows from an eviction to the same expert's next load
        self._share_sum = 0.0              # #89: the protected share of the resident bytes, summed per window

    @property
    def window_share(self):
        """#89: the mean (over windows) share of the resident bytes a recent-tokens window protects."""
        return self._share_sum / self.rounds if self.rounds else 0.0

    @property
    def start_hit_rate(self):
        """#90: host hits / (hits + loads) over the first decode windows of each request."""
        n = self.start_hits + self.start_loads
        return self.start_hits / n if n else 0.0

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


def simulate(trace, sizes, cap_bytes, boot_order, owned, n_expert, policy="baseline", phases=None, seed_k=None,
             start_windows=8, **params):
    """Replay `trace` and return a Stats.  `sizes[layer]` = bytes of one expert; `boot_order` = flat ids or
    (layer, expert) keys in the runtime's host-tier order; `owned` = GPU-owned keys; `params` go to the policy.
    `phases`: replay only these phases (windows of other phases still feed the seed).  `seed_k` (#90): before a
    request's first decode window, load the experts its last `seed_k` prompt-window positions routed.
    `start_windows`: the first decode windows of each request counted into start_hits / start_loads."""
    pol = POLICIES[policy](n_expert, **params)
    if hasattr(pol, "bind"):
        pol.bind(sizes, cap_bytes)
    s = Stats(pol)
    resident, used = s.resident, 0
    evicted_at, wi = {}, 0

    def make_room(b, avoid_layer):
        nonlocal used
        while cap_bytes and used + b > cap_bytes:
            v = pol.victim(resident, avoid_layer)
            if v is None:
                break
            resident.discard(v)
            used -= sizes[v[0]]
            pol.on_evict(v)
            s.evicted.append(v)
            evicted_at[v] = wi
        if used + b > cap_bytes:
            s.over_cap += 1
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
    request, prompt_tail, decode_seen = object(), {}, 0
    for w in trace.windows:
        if w.request != request:              # a new request: its prompt tail and its start counters begin again
            request, prompt_tail, decode_seen = w.request, {}, 0
        if w.phase == 0:
            for layer in w.routes:
                tail = prompt_tail.setdefault(layer, [])
                tail.extend(w.committed(layer))
                if seed_k is not None:
                    del tail[:-seed_k]
        if phases is not None and w.phase not in phases:
            continue
        if w.phase != 0 and decode_seen == 0 and seed_k and prompt_tail:
            for layer in sorted(prompt_tail):     # #90: the prompt-tail seed, before the first decode window
                for e in dict.fromkeys(e for ids in prompt_tail[layer] for e in ids if e >= 0):
                    key = (layer, e)
                    if key in owned or key in resident:
                        continue
                    b = sizes[layer]
                    make_room(b, None)
                    resident.add(key)
                    used += b
                    s.seed_loads += 1
                    s.seed_bytes += b
                    pol.on_boot(key)
        counting = w.phase != 0 and decode_seen < start_windows
        if w.phase != 0:
            decode_seen += 1
        hits0, loads0 = s.hits, s.loads
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
                make_room(b, layer)
                if key in evicted_at:
                    s.reload_distances.append(wi - evicted_at.pop(key))
                resident.add(key)
                used += b
                s.loads += 1
                s.load_bytes += b
                pol.on_load(key, w, layer)
            for key in host:
                pol.on_access(key, w, layer)
        if counting:
            s.start_hits += s.hits - hits0
            s.start_loads += s.loads - loads0
        pol.on_window_end(w)
        if hasattr(pol, "win") and resident:
            prot = pol.win.protected()
            s._share_sum += sum(sizes[k[0]] for k in resident if k in prot) / sum(sizes[k[0]] for k in resident)
        wi += 1
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
