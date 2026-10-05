"""#178: the prompt report of timeline.py on a synthetic two-lane wave trace.
  python -m pytest tests/xeno/perf/test_timeline.py -q"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import timeline as T  # noqa: E402


def ev(tid, name, ts_ms, dur_ms, a=-1, b=-1):
    return {"ph": "X", "pid": 1, "tid": tid, "ts": ts_ms * 1000.0, "dur": dur_ms * 1000.0, "name": name,
            "args": {"a": a, "b": b}}


def lane(tid, name):
    return {"ph": "M", "pid": 1, "tid": tid, "name": "thread_name", "args": {"name": name}}


def wave_trace():
    """Two wave lanes reading at once: lane 1 on "main", lane 2 on "prompt wave lane 1"; each with its own CUDA0
    compute lane, and the split issuer feeding the 4070's copy engine."""
    return [
        lane(1, "main"), lane(2, "prompt wave lane 1"), lane(3, "prefill split issuer"),
        lane(10, "gpu0 compute (prefill)"), lane(11, "gpu0 compute (prefill, lane 2)"),
        lane(12, "gpu1 copy engine (prefill split)"),
        ev(1, "prefill run", 0, 100, 4096), ev(2, "prefill run", 0, 100, 4096),
        ev(10, "gdn", 0, 30, 0), ev(10, "wait host", 30, 70, 0),
        ev(11, "combine", 0, 60, 0), ev(11, "gather", 60, 40, 0),
        # the issuer issues expert 7 of layer 0, the copy runs 2 ms later; then expert 9
        ev(3, "split issue host", 10, 1, 0, 7), ev(12, "copy staged", 13, 2, 0, 7),
        ev(3, "split issue host", 20, 1, 0, 9), ev(12, "copy staged", 30, 2, 0, 9),
    ]


def test_each_run_reports_its_own_compute_lane():
    runs = T.prefill_report(T.Timeline(wave_trace()))
    assert len(runs) == 2
    assert {frozenset(r["phase_ms"]) for r in runs} == {frozenset({"gdn", "wait host"}), frozenset({"combine", "gather"})}


def test_split_copies_are_matched_to_their_issue():
    r = T.prefill_report(T.Timeline(wave_trace()))[0]
    assert "unmatched" not in r["copy_gaps"], r["copy_gaps"]
    assert r["issue_to_start_ms"]["n"] == 2, r["issue_to_start_ms"]
