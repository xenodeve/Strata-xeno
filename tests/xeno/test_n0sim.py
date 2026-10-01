"""#87/#88: the N0 simulator reads #85's commit-aware --dump-routing traces and replays them through a model of the
runtime's host tier.  Each test builds a tiny trace with a known answer: the baseline must reproduce the runtime's
decayed LFU (expert_source.cpp: +1 per host load, +1 per distinct host expert per window-layer access, x0.97 per
verify window, evict the lowest score outside the current layer, ties to the lowest index, boot fill at score 1)."""
import os
import struct
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "perf"))
import n0sim  # noqa: E402

NE = 8          # experts per layer in these toy traces
SIZE = 100      # bytes per expert, every layer


def rec(layer, ids):
    return struct.pack(f"<ii{len(ids)}i{len(ids)}f", layer, len(ids), *ids, *([0.0] * len(ids)))


def window(wid, layers_positions, n_acc, phase=1, v1=False):
    """layers_positions: {layer: [ids of position 0, ids of position 1, ...]} - written layer-major.  Format 2's
    commit carries the phase; `v1` writes format 1's three-value commit."""
    out = b""
    n_pos = len(next(iter(layers_positions.values())))
    for layer in sorted(layers_positions):
        for ids in layers_positions[layer]:
            out += rec(layer, ids)
    return out + rec(-1, [wid, n_pos, n_acc] if v1 else [wid, n_pos, n_acc, phase])


_TMP = {}


@pytest.fixture(autouse=True)
def _tmp(tmp_path):
    _TMP["dir"], _TMP["n"] = tmp_path, 0


def trace(*chunks, version=2):
    _TMP["n"] += 1
    p = os.path.join(_TMP["dir"], f"t{_TMP['n']}.bin")
    open(p, "wb").write(rec(-2, [version]) + b"".join(chunks))
    return p


def sim(path, cap_bytes, boot=(), owned=(), policy="baseline", **params):
    t = n0sim.load(path, n_expert=NE)
    return n0sim.simulate(t, sizes=[SIZE] * 3, cap_bytes=cap_bytes, boot_order=list(boot), owned=set(owned),
                          n_expert=NE, policy=policy, **params)


# ------------------------------------------------------------------ reader


def test_reader_rebuilds_windows_positions_and_acceptance():
    p = trace(window(0, {0: [[1, 2], [3, 4], [5, 6]], 1: [[1], [2], [3]]}, 1))
    t = n0sim.load(p, n_expert=NE)
    assert len(t.windows) == 1
    w = t.windows[0]
    assert (w.n_pos, w.n_acc) == (3, 1)
    assert w.routes[0] == [(1, 2), (3, 4), (5, 6)]
    assert w.routes[1] == [(1,), (2,), (3,)]
    assert w.committed(0) == [(1, 2), (3, 4)]          # position 0 and the one accepted draft
    assert w.rejected(0) == [(5, 6)]


def test_reader_refuses_an_unknown_format_version():
    with pytest.raises(ValueError, match="version"):
        n0sim.load(trace(window(0, {0: [[1]]}, 0), version=99), n_expert=NE)


def test_reader_refuses_a_layer_whose_position_count_disagrees_with_the_commit():
    bad = rec(0, [1]) + rec(0, [2]) + rec(1, [3]) + rec(-1, [0, 2, 0])   # layer 1 has 1 position, the commit says 2
    with pytest.raises(ValueError, match="positions"):
        n0sim.load(trace(bad), n_expert=NE)


def test_reader_keeps_owned_boot_request_and_phase():
    owned = rec(-4, [0 * NE + 1, 1 * NE + 7])
    boot = rec(-6, [0 * NE + 2])
    p = trace(owned + boot + rec(-3, [5]) + window(0, {0: [[1]]}, 0, phase=0) + window(1, {0: [[2]]}, 0, phase=1))
    t = n0sim.load(p, n_expert=NE)
    assert t.owned == {(0, 1), (1, 7)}
    assert t.boot == [(0, 2)]
    assert [w.request for w in t.windows] == [5, 5]
    assert [w.phase for w in t.windows] == [0, 1]


def test_reader_still_reads_format_1_with_its_separate_phase_tag():
    p = trace(rec(-3, [5]) + rec(-5, [0]) + window(0, {0: [[1]]}, 0, v1=True) + rec(-5, [1]) +
              window(1, {0: [[2]]}, 0, v1=True), version=1)
    t = n0sim.load(p, n_expert=NE)
    assert [w.request for w in t.windows] == [5, 5]
    assert [w.phase for w in t.windows] == [0, 1]


# ------------------------------------------------------------------ baseline = the runtime's policy


def test_boot_fill_takes_the_order_until_the_cap_and_misses_count_bytes():
    # cap 300 B = 3 experts; boot puts (0,1) and (0,2) in.  Window routes 1, 2, 3 at layer 0: one load (expert 3).
    p = trace(window(0, {0: [[1, 2, 3]]}, 0))
    s = sim(p, 300, boot=[1, 2])
    assert s.loads == 1 and s.load_bytes == SIZE
    assert s.hits == 2


def test_boot_fill_skips_what_does_not_fit_and_keeps_going():
    # cap 100 B, sizes 100 per expert: the first fits, the rest do not
    p = trace(window(0, {0: [[1]]}, 0))
    s = sim(p, 100, boot=[1, 2, 3])
    assert s.boot_resident == {(0, 1)}


def test_eviction_takes_the_lowest_decayed_score_outside_the_current_layer():
    # cap 200 B.  Boot: L1e1 (index 9) and L0e5 (index 5), both score 1.
    # w0: layer 0 routes 5 (hit: 5 -> 2).  decay: L0e5 1.94, L1e1 0.97.
    # w1: layer 0 routes 6 (miss): victim must be outside layer 0 -> L1e1 (0.97), not L0e5.
    p = trace(window(0, {0: [[5]]}, 0), window(1, {0: [[6]]}, 0))
    s = sim(p, 200, boot=[NE + 1, 5])
    assert s.loads == 1
    assert s.evicted == [(1, 1)]
    assert s.resident == {(0, 5), (0, 6)}


def test_score_matches_a_hand_computed_case():
    # boot (0,1) at 1.0; w0 accesses it (+1 = 2.0) then decays (x0.97); w1 accesses it (+1), decays again
    p = trace(window(0, {0: [[1]]}, 0), window(1, {0: [[1]]}, 0))
    s = sim(p, 100, boot=[1])
    assert s.score((0, 1)) == pytest.approx(((1.0 + 1.0) * 0.97 + 1.0) * 0.97, abs=1e-6)


def test_an_expert_routed_by_several_positions_scores_once_per_window_layer():
    # the runtime calls blob() once per distinct expert per window-layer (job_of dedup)
    p = trace(window(0, {0: [[1], [1], [1]]}, 2))
    s = sim(p, 100, boot=[1])
    assert s.score((0, 1)) == pytest.approx((1.0 + 1.0) * 0.97, abs=1e-6)


def test_a_loaded_expert_scores_its_load_and_its_access():
    # a miss is +1 at load (materialize) and +1 at the window's access (blob): 2.0 before the decay
    p = trace(window(0, {0: [[3]]}, 0))
    s = sim(p, 100)
    assert s.score((0, 3)) == pytest.approx(2.0 * 0.97, abs=1e-6)


def test_gpu_owned_experts_never_touch_the_host_tier():
    p = trace(window(0, {0: [[1, 2]]}, 0))
    s = sim(p, 100, owned={(0, 1)})
    assert s.loads == 1 and s.hits == 0 and s.gpu == 1


def test_a_cache_smaller_than_one_layers_routed_bytes_is_reported_not_clamped():
    # three distinct host experts in one layer, room for one: the runtime runs over the cap
    p = trace(window(0, {0: [[1, 2, 3]]}, 0))
    s = sim(p, 100)
    assert s.over_cap > 0


def test_locality_is_one_when_every_committed_token_repeats_the_last():
    # #88: committed positions route the same set every token -> overlap(R_t, R_{t-1}) = 1 for every k
    w = [window(i, {0: [[1, 2], [1, 2]], 1: [[3], [3]]}, 1) for i in range(3)]
    t = n0sim.load(trace(*w), n_expert=NE)
    loc = n0sim.locality(t, ks=(1, 4))
    assert loc[1] == pytest.approx(1.0)
    assert loc[4] == pytest.approx(1.0)


def test_locality_is_zero_when_no_committed_token_repeats_and_ignores_rejected_positions():
    # committed tokens route 1, 2, 3, 4 (never repeating); the rejected position routes 1 again, which must not count
    w0 = window(0, {0: [[1], [2], [1]]}, 1)      # positions 0 and 1 committed, position 2 rejected
    w1 = window(1, {0: [[3], [4], [3]]}, 1)
    t = n0sim.load(trace(w0, w1), n_expert=NE)
    loc = n0sim.locality(t, ks=(1, 8))
    assert loc[1] == pytest.approx(0.0)
    assert loc[8] == pytest.approx(0.0)


def test_locality_counts_partial_overlap_per_token():
    # committed sequence at layer 0: {1,2}, then {2,3}: overlap of the second with the first = 1/2
    t = n0sim.load(trace(window(0, {0: [[1, 2], [2, 3]]}, 1)), n_expert=NE)
    assert n0sim.locality(t, ks=(1,))[1] == pytest.approx(0.5)


def test_pollution_counts_loads_and_admissions_routed_only_by_rejected_positions():
    # w0: position 0 (committed) routes 1; position 1 (rejected) routes 2 and 1.  Cold cache: loads 1 and 2.
    # expert 2 was routed ONLY by a rejected position -> 1 polluting load of 2.
    p = trace(window(0, {0: [[1], [2, 1]]}, 0))
    s = sim(p, 1000)
    assert s.loads == 2
    assert s.rejected_only_loads == 1
    assert s.pollution_share == pytest.approx(0.5)


def test_with_every_draft_accepted_there_is_no_pollution():
    p = trace(window(0, {0: [[1], [2]]}, 1))
    s = sim(p, 1000)
    assert s.rejected_only_loads == 0 and s.pollution_share == 0.0


def test_loads_per_emitted_token_counts_committed_positions():
    # two windows: 3 positions with 1 accepted (emits 2), 2 positions with 1 accepted (emits 2)
    p = trace(window(0, {0: [[1], [2], [3]]}, 1), window(1, {0: [[4], [5]]}, 1))
    s = sim(p, 1000)
    assert s.emitted == 4
    assert s.loads == 5
    assert s.loads_per_token == pytest.approx(5 / 4)
    assert s.loads_per_round == pytest.approx(5 / 2)


# ------------------------------------------------------------------ #89 window policies


def _recent_vs_frequent():
    # cap 2 experts.  w0-w3: layer 0 routes 1 (a frequent expert, score ~4.6).  w4: layer 0 routes 2 (a recent one).
    # w5: layer 1 routes 6 -> one eviction among layer 0's two.  The baseline drops the RECENT expert 2 (lower
    # score); a recent-tokens window of k=1 protects 2 (the last committed token at layer 0 routed it) and drops 1.
    ws = [window(i, {0: [[1]]}, 0) for i in range(4)] + [window(4, {0: [[2]]}, 0), window(5, {1: [[6]]}, 0)]
    return trace(*ws)


def test_baseline_drops_the_recent_expert_in_the_recent_vs_frequent_case():
    assert sim(_recent_vs_frequent(), 200).evicted == [(0, 2)]


def test_window_k_never_evicts_an_expert_inside_its_window_while_the_cache_fits_it():
    assert sim(_recent_vs_frequent(), 200, policy="window", k=1).evicted == [(0, 1)]
    assert sim(_recent_vs_frequent(), 200, policy="window_lfu", k=1).evicted == [(0, 1)]


def test_window_k_has_no_miss_after_warm_up_on_a_full_overlap_trace():
    ws = [window(i, {0: [[1, 2]], 1: [[3]]}, 0) for i in range(6)]
    s = sim(trace(*ws), 300, policy="window", k=2)
    assert s.loads == 3      # the first window's three cold loads, nothing after


def _pseudo_random_trace(seed=7, n=40):
    import random
    r = random.Random(seed)
    ws = []
    for i in range(n):
        n_pos = r.randint(1, 4)
        routes = {layer: [[r.randrange(NE) for _ in range(2)] for _ in range(n_pos)] for layer in (0, 1)}
        ws.append(window(i, routes, r.randint(0, n_pos - 1)))
    return trace(*ws)


def test_window_lfu_with_an_empty_window_is_exactly_the_baseline():
    p = _pseudo_random_trace()
    a, b = sim(p, 500), sim(p, 500, policy="window_lfu", k=0)
    assert (a.loads, a.hits, a.evicted) == (b.loads, b.hits, b.evicted)


def test_wtinylfu_keeps_a_frequent_expert_against_a_stream_of_one_offs():
    # (0,1) is used every window; every window also routes a new one-off expert, at layer 1 or 2 by turns (with one
    # other layer, the runtime's rule - never evict the layer being loaded - would leave (0,1) as the only victim).
    ws = [window(i, {0: [[1]], 1 + i % 2: [[i]]}, 0) for i in range(8)]
    s = sim(trace(*ws), 200, policy="wtinylfu", window_frac=0.5)
    assert (0, 1) not in s.evicted
    assert s.loads == 1 + 8   # (0,1) once, then every one-off


# ------------------------------------------------------------------ #90 speculation-aware admission, prompt seed


def test_spec_aware_equals_the_baseline_when_every_draft_is_accepted():
    import random
    r = random.Random(3)
    ws = []
    for i in range(30):
        n_pos = r.randint(1, 4)
        routes = {layer: [[r.randrange(NE) for _ in range(2)] for _ in range(n_pos)] for layer in (0, 1)}
        ws.append(window(i, routes, n_pos - 1))
    p = trace(*ws)
    a, b = sim(p, 400), sim(p, 400, policy="spec")
    assert (a.loads, a.hits, a.evicted) == (b.loads, b.hits, b.evicted)


def test_spec_aware_evicts_a_rejected_only_expert_before_a_committed_one():
    # cap 2.  w0: committed position routes layer-0 expert 1; the rejected draft routes layer-0 expert 2.
    # w1: layer 1 routes 6 -> one eviction.  Both have the same baseline score, so the baseline drops the lower index
    # (0,1) - the committed one; speculation-aware admission put (0,2) on probation and drops it first.
    p = trace(window(0, {0: [[1], [2]]}, 0), window(1, {1: [[6]]}, 0))
    assert sim(p, 200).evicted == [(0, 1)]
    assert sim(p, 200, policy="spec").evicted == [(0, 2)]


def test_prompt_tail_seed_loads_the_last_prompt_routes_before_decode():
    # request 0: a prompt window (phase 0) routes 3; the first decode window routes 3.  Seeded, the decode window hits.
    p = trace(rec(-3, [0]) + window(0, {0: [[3]]}, 0, phase=0) + window(1, {0: [[3]]}, 0, phase=1))
    plain = sim(p, 300, phases=(1,))
    seeded = sim(p, 300, phases=(1,), seed_k=4)
    assert plain.loads == 1 and seeded.loads == 0
    assert seeded.seed_loads == 1


def test_hit_rate_after_a_request_start_counts_the_first_decode_windows_of_each_request():
    # request 0: two decode windows, the first misses (load), the second hits
    p = trace(rec(-3, [0]) + window(0, {0: [[3]]}, 0) + window(1, {0: [[3]]}, 0))
    s = sim(p, 300, start_windows=2)
    assert s.start_hits == 1 and s.start_loads == 1
    assert s.start_hit_rate == pytest.approx(0.5)


def test_reader_drops_uncommitted_records_when_a_new_window_starts():
    # scrutinize: a cancelled window (or the per-token loop) leaves route records with no commit; the next window
    # starts again at layer 0.  The orphans are dropped and counted, not folded into the next window.
    orphan = rec(0, [7]) + rec(0, [7]) + rec(1, [7])            # two positions at layer 0, one at layer 1, no commit
    p = trace(orphan + window(0, {0: [[1], [2]], 1: [[3], [4]]}, 1))
    t = n0sim.load(p, n_expert=NE)
    assert len(t.windows) == 1
    assert t.windows[0].routes[0] == [(1,), (2,)]
    assert t.orphans == 3
