"""#34: the lazy-exact-refill simulator (tests/xeno/perf/lazy_refill_sim.py): how many of the borrowed tail's experts
decode actually routes to, window by window.  Expected values are worked out by hand."""
import os
import struct
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "perf"))
import lazy_refill_sim as lr  # noqa: E402


def trace_bytes(windows):
    """windows: [[(layer, [ids of the window's one token])...], ...]; K = len(ids).  `--dump-routing` format 2 (#93):
    a format tag, one record per token (int32 layer, int32 k, k int32 ids, k float32 weights), a commit per window."""
    def rec(layer, ids):
        return struct.pack(f"<ii{len(ids)}i{len(ids)}f", layer, len(ids), *ids, *([0.0] * len(ids)))
    out = rec(-2, [2])
    for n, w in enumerate(windows):
        for layer, ids in w:
            out += rec(layer, ids)
        out += rec(-1, [n, 1, 0, 1])
    return out


def test_first_touches_of_the_tail_are_counted_once(tmp_path):
    p = tmp_path / "t.trace"
    # window 1: layer 0 routes experts 5 and 6, layer 1 routes 5;  window 2: layer 0 routes 5 and 7 (5 is a repeat)
    p.write_bytes(trace_bytes([[(0, [5, 6]), (1, [5, -1])], [(0, [5, 7]), (1, [8, -1])]]))
    tail = {(0, 5), (0, 7), (1, 8)}
    curve = lr.first_touches(str(p), tail)
    # per window: the tail experts touched for the first time in it
    assert curve == [1, 2]            # window 1: (0,5); window 2: (0,7) and (1,8)


def test_stalls_count_the_layers_of_a_window_that_meet_a_new_tail_expert(tmp_path):
    p = tmp_path / "t.trace"
    p.write_bytes(trace_bytes([[(0, [5, 6]), (1, [5, -1])], [(0, [5, 7]), (1, [8, -1])]]))
    tail = {(0, 5), (0, 7), (1, 8)}
    # a stall is one (window, layer) with at least one first touch: its misses are read together
    assert lr.stall_layers(str(p), tail) == [1, 2]


def test_tail_is_the_profile_range_the_prompt_path_borrows():
    ranked = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5)]
    assert lr.tail(ranked, keep_from=3, slots=5) == {(1, 4), (2, 5)}
