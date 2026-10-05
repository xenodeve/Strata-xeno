"""#33: the pipeline timeline analyzer (tests/xeno/perf/timeline.py).  Every expected number below is worked out by
hand from the synthetic trace, not recomputed the way the analyzer does it: an analyzer that attributes a copy
engine's idle time to the wrong cause would send the next optimisation at the wrong stage (#31 spent ~12
experiments finding that out by hand)."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "perf"))
import timeline  # noqa: E402

PID = 7
LANES = {1: "main", 2: "gpu0 copy engine (prefill)", 3: "gpu0 compute (prefill)", 4: "prefill copy issuer",
         5: "pool worker 0", 6: "pool worker 1", 7: "pool worker 2", 8: "gpu1 4070 experts"}


def X(tid, name, t0_ms, t1_ms, a=-1, b=-1):
    e = {"ph": "X", "pid": PID, "tid": tid, "ts": t0_ms * 1000.0, "dur": (t1_ms - t0_ms) * 1000.0, "name": name}
    if a != -1 or b != -1:
        e["args"] = {"a": a, "b": b}
    return e


def names():
    return [{"ph": "M", "pid": PID, "tid": t, "name": "thread_name", "args": {"name": n}} for t, n in LANES.items()]


def write(tmp_path, events, closed=False):
    p = tmp_path / "t.json"
    body = "".join(json.dumps(e) + ",\n" for e in names() + events)
    p.write_text("[\n" + (body.rstrip(",\n") + "\n]" if closed else body), encoding="utf-8")
    return str(p)


def test_load_reads_the_open_array_the_engine_writes_and_a_closed_one(tmp_path):
    ev = [X(1, "a", 0, 1), X(1, "b", 1, 2)]
    for closed in (False, True):
        t = timeline.load(write(tmp_path, ev, closed))
        assert [s.name for s in t.lane("main")] == ["a", "b"]
        assert t.lane("main")[1].t0 == pytest.approx(1.0)   # milliseconds


PREFILL = [
    X(1, "prefill run", 0, 100, 8192, 0),
    # entry 0: issued at once, copies 1..11
    X(1, "copy issue", 0.0, 0.1, 0, 0), X(2, "copy staged", 1, 11, 0, 0),
    # entry 1: issued at 5 (while entry 0 is still copying) but starts only at 20: 9 ms waiting on the device
    X(1, "copy issue", 5.0, 5.1, 1, 0), X(2, "copy pinned", 20, 30, 1, 0),
    # entry 2: the host is in the router sync 28..40 and issues only at 40: 10 ms not yet issued (router sync),
    # 0.1 ms inside the issue call, 0.1 ms issued but not started
    X(1, "router sync", 28, 40, 1, 2), X(1, "copy issue", 40.0, 40.1, 2, 1), X(2, "copy pinned", 40.2, 50, 2, 1),
    # the compute lane's phases, charged to layers
    X(3, "qsa attn", 0, 10, 0), X(3, "wait copy", 10, 15, 0), X(3, "gemm down", 15, 18, 1),
]


def test_prefill_copy_engine_idle_is_split_by_cause(tmp_path):
    r = timeline.prefill_report(timeline.load(write(tmp_path, PREFILL)))[0]
    assert r["wall_ms"] == pytest.approx(100.0)
    assert r["copy_busy_ms"] == pytest.approx(10 + 10 + 9.8)
    assert r["copies"] == 3
    gaps = r["copy_gaps"]
    assert gaps["issued, not started (device)"] == pytest.approx(9.0 + 0.1)
    assert gaps["not issued yet (host)"] == pytest.approx(10.0)
    assert gaps["in the issue call"] == pytest.approx(0.1)
    # what the issuing thread was doing while the copy was not issued yet: its innermost span
    assert r["host_late_cause"] == {"router sync": pytest.approx(10.0)}
    assert r["copies_by_kind"] == {"copy staged": 1, "copy pinned": 2}


def test_prefill_compute_phases_are_budgeted_per_layer(tmp_path):
    r = timeline.prefill_report(timeline.load(write(tmp_path, PREFILL)))[0]
    assert r["phase_ms"] == {"qsa attn": pytest.approx(10.0), "wait copy": pytest.approx(5.0),
                             "gemm down": pytest.approx(3.0)}
    assert r["layer_phase_ms"][0] == {"qsa attn": pytest.approx(10.0), "wait copy": pytest.approx(5.0)}
    assert r["layer_phase_ms"][1] == {"gemm down": pytest.approx(3.0)}


def test_prefill_issuer_thread_waits_are_the_host_late_cause(tmp_path):
    ev = [X(1, "prefill run", 0, 50),
          X(4, "copy issue", 0, 0.1, 0, 0), X(2, "copy pinned", 0.1, 10, 0, 0),
          # the issuer waits for the compute to free a ring slot 10..20, then issues entry 1
          X(4, "issuer wait ring slot", 10, 20, 1), X(4, "copy issue", 20, 20.1, 1, 0), X(2, "copy pinned", 20.1, 30, 1, 0)]
    r = timeline.prefill_report(timeline.load(write(tmp_path, ev)))[0]
    assert r["copy_gaps"]["not issued yet (host)"] == pytest.approx(10.0)
    assert r["host_late_cause"] == {"issuer wait ring slot": pytest.approx(10.0)}


def test_prefill_top_gaps_name_the_layer_and_the_cause(tmp_path):
    r = timeline.prefill_report(timeline.load(write(tmp_path, PREFILL)))[0]
    top = r["top_copy_gaps"][0]
    assert top["entry"] == 2 and top["layer"] == 1
    assert top["ms"] == pytest.approx(10.2)
    assert top["cause"] == "router sync"


DECODE = [
    # round 1: 0..20
    X(1, "decode round", 0, 20, 100, 4),
    X(1, "verify window", 0, 15, 4, 100),
    X(1, "verify stage", 0, 1, 4), X(1, "verify launch", 1, 2, 4),
    X(1, "wait gpu", 2, 5, 0), X(1, "cpu experts", 5, 9, 0),
    X(1, "cpu pool", 5.5, 8.5, 0, 12),
    X(1, "wait gpu", 9, 11, 1), X(1, "cpu experts", 11, 14, 1),
    X(1, "verify tail", 14, 15, 4),
    X(1, "verify commit", 15, 16, 2), X(1, "mtp draft", 16, 19, 4, 1),
    # round 2: 20..30, with 1 ms nothing covers (unaccounted)
    X(1, "decode round", 20, 30, 102, 4),
    X(1, "verify window", 20, 27, 4, 102),
    X(1, "wait gpu", 20, 23, 0), X(1, "cpu experts", 23, 27, 0),
    X(1, "verify commit", 27, 28, 2), X(1, "mtp draft", 28, 29, 4, 1),
    # workers during round 1's layer 0 CPU experts (5..9): wake then drain; worker 2 is the straggler
    X(5, "wake", 5.5, 5.6, 0), X(5, "drain", 5.6, 7.0, 0),
    X(6, "wake", 5.5, 5.7, 1), X(6, "drain", 5.7, 7.2, 1),
    X(7, "wake", 5.5, 6.5, 2), X(7, "drain", 6.5, 8.5, 2),
    X(8, "4070 experts", 5.2, 6.0, 1, 3),
]


def test_decode_round_budget_is_per_round_and_names_the_unaccounted(tmp_path):
    r = timeline.decode_report(timeline.load(write(tmp_path, DECODE)))
    assert r["rounds"] == 2
    assert r["round_ms"] == pytest.approx(15.0)
    per = r["per_round_ms"]
    assert per["verify window"] == pytest.approx((15 + 7) / 2)
    assert per["wait gpu"] == pytest.approx((3 + 2 + 3) / 2)
    assert per["cpu experts"] == pytest.approx((4 + 3 + 4) / 2)
    assert per["cpu pool"] == pytest.approx(3 / 2)
    assert per["mtp draft"] == pytest.approx((3 + 1) / 2)
    # round 1: 20 - (15 + 1 + 3) = 1; round 2: 10 - (7 + 1 + 1) = 1
    assert r["unaccounted_ms"] == pytest.approx(1.0)


def test_decode_layers_rank_the_slowest_cpu_experts(tmp_path):
    r = timeline.decode_report(timeline.load(write(tmp_path, DECODE)))
    assert r["layer_ms"][0] == {"wait gpu": pytest.approx(3.0), "cpu experts": pytest.approx(4.0)}
    assert r["layer_ms"][1] == {"wait gpu": pytest.approx(2.0), "cpu experts": pytest.approx(3.0)}


def test_decode_pool_workers_show_the_straggler_and_the_wake_up(tmp_path):
    w = timeline.decode_report(timeline.load(write(tmp_path, DECODE)))["workers"]
    assert w["batches"] == 1
    assert w["straggler_ms"] == pytest.approx(8.5 - 7.0)   # last drain end - first drain end
    assert w["wake_mean_ms"] == pytest.approx((0.1 + 0.2 + 1.0) / 3)
    assert w["wake_max_ms"] == pytest.approx(1.0)
    assert w["slowest_worker"] == "pool worker 2"


def test_lane_utilisation_is_the_union_over_the_lane_span(tmp_path):
    u = timeline.utilisation(timeline.load(write(tmp_path, DECODE)))
    assert u["gpu1 4070 experts"]["busy_ms"] == pytest.approx(0.8)


def test_perfetto_export_is_a_closed_json_array(tmp_path):
    src = write(tmp_path, PREFILL)
    out = tmp_path / "p.json"
    timeline.export_perfetto(src, str(out))
    data = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(data, list) and len(data) == len(names()) + len(PREFILL)


def test_thread_lanes_are_grouped_and_split_by_innermost_span(tmp_path):
    lanes = {11: "prefill stager 0", 12: "prefill stager 1"}
    ev = [{"ph": "M", "pid": PID, "tid": t, "name": "thread_name", "args": {"name": n}} for t, n in lanes.items()]
    ev += [X(11, "stager wait buffer", 0, 9), X(11, "stager memcpy", 9, 10),
           X(12, "stager wait buffer", 0, 8), X(12, "stager memcpy", 8, 10)]
    g = timeline.thread_report(timeline.load(write(tmp_path, ev)))["prefill stager *"]
    assert g["lanes"] == 2
    assert g["ms"] == {"stager wait buffer": pytest.approx(17.0), "stager memcpy": pytest.approx(3.0)}


def test_copy_durations_are_split_by_the_compute_phase_they_overlap(tmp_path):
    # a copy that stretches while a long kernel runs (issuer mode, #31: 161 copies of 10.8 ms during qsa attn) must
    # show up against that phase; the copy's midpoint picks the phase
    r = timeline.prefill_report(timeline.load(write(tmp_path, PREFILL)))[0]
    assert r["copy_by_phase"]["qsa attn"] == {"n": 1, "ms": pytest.approx(10.0), "p50": pytest.approx(10.0)}
    assert r["copy_by_phase"]["(none)"]["n"] == 2
    assert r["copy_by_phase"]["(none)"]["ms"] == pytest.approx(10 + 9.8)


def test_server_file_merges_and_the_request_is_split_from_http_to_first_token(tmp_path):
    engine = write(tmp_path, [X(1, "request", 100, 400, 50, 10), X(1, "prompt read (batched)", 101, 200, 0, 49)])
    sv = [{"ph": "M", "pid": 9, "tid": 1, "name": "thread_name", "args": {"name": "server Thread-1"}},
          {**X(1, "http request", 90, 410), "pid": 9}, {**X(1, "template+tokenize", 91, 95, 50), "pid": 9},
          {**X(1, "queue wait", 95, 98), "pid": 9}, {**X(1, "engine request", 98, 405, 50, 10), "pid": 9},
          {"ph": "i", "s": "t", "pid": 9, "tid": 1, "ts": 210_000.0, "name": "first token"}]
    (tmp_path / "t.json.server.json").write_text("[\n" + "".join(json.dumps(e) + ",\n" for e in sv), encoding="utf-8")
    r = timeline.server_report(timeline.load(engine))[0]   # load() picks up <file>.server.json beside it
    assert r["http_ms"] == pytest.approx(320.0)
    assert r["template+tokenize_ms"] == pytest.approx(4.0)
    assert r["queue wait_ms"] == pytest.approx(3.0)
    assert r["engine request_ms"] == pytest.approx(307.0)
    assert r["ttft_ms"] == pytest.approx(120.0)            # first token - http start
    assert r["engine_side_ms"] == pytest.approx(300.0)     # the engine's own "request" inside it
    assert r["pipe_ms"] == pytest.approx(7.0)              # engine request - engine side: pipes and parsing


def test_a_copy_is_matched_to_the_issue_of_its_own_layer(tmp_path):
    # the per-layer path numbers entries from 0 in every layer: a backlogged engine can start layer 5's entry 3
    # after layer 6's entry 3 was issued; matching on the entry alone took layer 6's issue (code-review of #33)
    ev = [X(1, "prefill run", 0, 100),
          X(1, "copy issue", 0, 0.1, 2, 5), X(2, "copy pinned", 0.1, 10, 2, 5),
          X(1, "copy issue", 1, 1.1, 3, 5),
          X(1, "copy issue", 30, 30.1, 3, 6),
          X(2, "copy pinned", 40, 50, 3, 5)]
    r = timeline.prefill_report(timeline.load(write(tmp_path, ev)))[0]
    # layer 5's entry 3 was issued at 1.1, so its whole 10..40 gap waited on the device
    assert r["copy_gaps"]["issued, not started (device)"] == pytest.approx(30.0)
    assert r["copy_gaps"]["not issued yet (host)"] == pytest.approx(0.0)


def test_copy_engine_busy_is_reported_per_layer(tmp_path):
    # spec: "per layer ... copy-engine busy"; layer 0 has the copies 1..11 and 20..30, layer 1 has 40.2..50
    r = timeline.prefill_report(timeline.load(write(tmp_path, PREFILL)))[0]
    assert r["layer_copy_ms"] == {0: pytest.approx(20.0), 1: pytest.approx(9.8)}


def test_every_gpu_lane_is_split_by_phase(tmp_path):
    # #32: a lane added later (the 4070's split stream) must be broken down like the standard ones, or its waits read
    # as work: "wait copies" 0-5 and 7-8 is 6 ms of waiting, "products" 5-7 is 2 ms of work
    lanes = {21: "gpu1 compute (prefill split)"}
    ev = [{"ph": "M", "pid": PID, "tid": t, "name": "thread_name", "args": {"name": n}} for t, n in lanes.items()]
    ev += [X(21, "wait copies", 0, 5, a=3), X(21, "products", 5, 7, a=3), X(21, "wait copies", 7, 8, a=3)]
    r = timeline.gpu_phase_report(timeline.load(write(tmp_path, ev)))
    assert r["gpu1 compute (prefill split)"] == {"wait copies": pytest.approx(6.0), "products": pytest.approx(2.0)}


def test_one_layer_is_laid_out_across_every_lane(tmp_path):
    # #32: who waits on whom in one layer - each lane's spans of that layer as a window (first start, last end), its
    # busy time and count, ordered by start.  The copies of layer 20 trickle over 10 -> 121 ms but are busy 5 ms.
    lanes = {21: "gpu1 compute (prefill split)", 22: "gpu1 copy engine (prefill split)"}
    ev = [{"ph": "M", "pid": PID, "tid": t, "name": "thread_name", "args": {"name": n}} for t, n in lanes.items()]
    ev += [X(22, "copy staged", 10, 12, a=20), X(22, "copy staged", 50, 52, a=20), X(22, "copy staged", 120, 121, a=20),
           X(22, "copy staged", 130, 131, a=21),
           X(3, "gather", 100, 160, a=20),
           X(21, "wait copies", 105, 125, a=20), X(21, "products", 125, 140, a=20)]
    w = timeline.layer_window(timeline.load(write(tmp_path, ev)), 20)
    assert [(r["lane"], r["name"]) for r in w] == [
        ("gpu1 copy engine (prefill split)", "copy staged"), ("gpu0 compute (prefill)", "gather"),
        ("gpu1 compute (prefill split)", "wait copies"), ("gpu1 compute (prefill split)", "products")]
    assert (w[0]["t0"], w[0]["t1"], w[0]["busy"], w[0]["n"]) == (pytest.approx(10), pytest.approx(121), pytest.approx(5), 3)


def wave_trace():
    """#178: two wave lanes reading at once (lane 1 on "main", lane 2 on "prompt wave lane 2", the engine's names), each with its own CUDA0
    compute lane, and the split issuer feeding the 4070's copy engine; its own lanes, not LANES."""
    lanes = {1: "main", 2: "prompt wave lane 2", 3: "prefill split issuer", 10: "gpu0 compute (prefill)",
             11: "gpu0 compute (prefill, lane 2)", 12: "gpu1 copy engine (prefill split)"}
    meta = [{"ph": "M", "pid": PID, "tid": t, "name": "thread_name", "args": {"name": n}} for t, n in lanes.items()]
    return timeline.Timeline(meta + [
        X(1, "prefill run", 0, 100, 4096), X(2, "prefill run", 0, 100, 4096),
        X(10, "gdn", 0, 30, 0), X(10, "wait host", 30, 100, 0),
        X(11, "combine", 0, 60, 0), X(11, "gather", 60, 100, 0),
        # the split issuer issues expert 7 of layer 0 and its copy starts 2 ms later; then expert 9 (layer, expert)
        X(3, "split issue host", 10, 11, 0, 7), X(12, "copy staged", 13, 15, 0, 7),
        X(3, "split issue host", 20, 21, 0, 9), X(12, "copy staged", 30, 32, 0, 9),
    ])


def test_each_wave_lane_run_reports_its_own_compute_lane():
    runs = timeline.prefill_report(wave_trace())
    assert {frozenset(r["phase_ms"]) for r in runs} == {frozenset({"gdn", "wait host"}), frozenset({"combine", "gather"})}


def test_split_copies_are_matched_to_the_split_issuer():
    r = timeline.prefill_report(wave_trace())[0]
    assert "unmatched" not in r["copy_gaps"], r["copy_gaps"]
    # issue end -> copy start: 13 - 11 = 2 ms and 30 - 21 = 9 ms
    assert r["issue_to_start_ms"]["n"] == 2 and r["issue_to_start_ms"]["max"] == pytest.approx(9.0), r["issue_to_start_ms"]
    assert r["layer_copy_ms"] == {0: pytest.approx(4.0)}   # both copies on layer 0, keyed by layer, not by expert

