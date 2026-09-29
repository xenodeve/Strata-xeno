"""#23: the topic-switch benchmark's analysis (tests/xeno/perf/topic_switch.py).  Expected values are worked out by
hand from the synthetic windows below."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "perf"))
import topic_switch as ts  # noqa: E402

PID = 3


def inst(name, t_ms, a, b):
    return {"ph": "i", "s": "t", "pid": PID, "tid": 1, "ts": t_ms * 1000.0, "name": name, "args": {"a": a, "b": b}}


def span(name, t0, t1, a=-1, b=-1):
    return {"ph": "X", "pid": PID, "tid": 1, "ts": t0 * 1000.0, "dur": (t1 - t0) * 1000.0, "name": name,
            "args": {"a": a, "b": b}}


def trace(tmp_path, windows, requests):
    """windows: [(t_ms, primary, cpu, g4070, pcie)] per window (not cumulative); requests: [(t0, t1, tokens)]"""
    ev = [{"ph": "M", "pid": PID, "tid": 1, "name": "thread_name", "args": {"name": "main"}}]
    cum = [0, 0, 0, 0]
    for t, pr, cpu, g, pc in windows:
        cum = [cum[0] + pr, cum[1] + cpu, cum[2] + g, cum[3] + pc]
        ev += [inst("tiers primary/cpu", t, cum[0], cum[1]), inst("tiers 4070/pcie", t, cum[2], cum[3])]
    for t0, t1, n in requests:
        ev.append(span("request", t0, t1, 100, n))
    p = tmp_path / "t.json"
    p.write_text("[\n" + "".join(json.dumps(e) + ",\n" for e in ev), encoding="utf-8")
    return str(p)


def test_windows_are_deltas_of_the_cumulative_counters(tmp_path):
    w = ts.windows(ts.load(trace(tmp_path, [(1, 6, 4, 0, 0), (2, 8, 1, 1, 0)], [(0, 3, 2)])))
    assert [(x.primary, x.cpu, x.g4070, x.pcie) for x in w] == [(6, 4, 0, 0), (8, 1, 1, 0)]
    assert w[0].primary_share == pytest.approx(0.6)
    assert w[1].primary_share == pytest.approx(0.8)


def test_windows_are_split_by_request(tmp_path):
    wins = [(1, 9, 1, 0, 0), (2, 9, 1, 0, 0), (11, 2, 8, 0, 0), (12, 5, 5, 0, 0)]
    segs = ts.segments(ts.load(trace(tmp_path, wins, [(0, 5, 2), (10, 15, 2)])))
    assert [len(s.windows) for s in segs] == [2, 2]
    assert segs[1].windows[0].primary_share == pytest.approx(0.2)


def test_recovery_counts_windows_until_the_primary_share_is_back(tmp_path):
    # a Thai segment after code: 0.2, 0.3, 0.5, then steady at 0.8 for 12 windows.  Steady = the median of the
    # last half = 0.8; back within 0.05 when the mean of the next 4 windows reaches 0.75: from window 3 on
    # (window 2's next four: 0.5, 0.8, 0.8, 0.8 -> mean 0.725 < 0.75)
    shares = [0.2, 0.3, 0.5] + [0.8] * 12
    assert ts.recovery_windows(shares, within=0.05, span=4) == 3
    # a segment that starts above its steady state (0.1, the median of its last half) needs no recovery
    assert ts.recovery_windows([0.9] * 8 + [0.1] * 8, within=0.05, span=4) == 0
    # too short for one span: -1
    assert ts.recovery_windows([0.5, 0.6], within=0.05, span=4) == -1


def test_recovery_of_a_flat_series_is_zero():
    assert ts.recovery_windows([0.7] * 10, within=0.05, span=4) == 0
