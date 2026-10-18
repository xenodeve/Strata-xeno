"""#44: decode_paths splits each decode layer's CPU-expert stage into its paths (the host's plan, the CPU pool, the
4070's GPU span) from one STRATA_TIMELINE run, so it shows which path the layer actually waited for."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "perf"))
import decode_paths as dp  # noqa: E402


def _x(name, ts, dur):
    return {"ph": "X", "name": name, "ts": ts, "dur": dur}


def test_a_layer_whose_pool_ends_last_waits_for_the_cpu():
    ev = [_x("decode round", 0, 100), _x("cpu experts", 10, 50), _x("dispatch plan", 10, 5), _x("cpu pool", 20, 30),
          _x("4070 experts", 12, 20)]
    r = dp.breakdown(ev)
    assert r["layers"] == 1 and r["rounds"] == 1
    assert r["ms"]["pool"] == 0.030 and r["ms"]["4070 GPU busy"] == 0.020
    assert r["4070 after pool"] == 0.0


def test_a_layer_whose_4070_ends_after_the_pool_is_counted():
    ev = [_x("decode round", 0, 100), _x("cpu experts", 10, 50), _x("dispatch plan", 10, 5), _x("cpu pool", 20, 10),
          _x("4070 experts", 12, 30)]
    assert dp.breakdown(ev)["4070 after pool"] == 1.0


def test_the_4070_sync_is_split_by_whether_its_gpu_had_already_finished():
    # layer 1: the GPU ended at 20, the host waited from 30 for 1 us; layer 2: the GPU ends at 140, the wait starts at
    # 100 and lasts 50, so 40 of it is the GPU and 10 the wake-up after it
    ev = [_x("decode round", 0, 200), _x("4070 experts", 5, 15), _x("4070 wait", 30, 1),
          _x("4070 experts", 90, 50), _x("4070 wait", 100, 50)]
    s = dp.sync_split(ev)
    assert s["done"] == {"n": 1, "ms_per_round": 0.001}
    assert s["running"] == {"n": 1, "ms_per_round": 0.05, "gpu_left_us": 40.0, "overshoot_us": 10.0}


def _xa(name, ts, dur, a, tid=1):
    return {"ph": "X", "name": name, "ts": ts, "dur": dur, "tid": tid, "args": {"a": a, "b": 0}}


def test_the_window_graph_4070_is_split_into_late_start_and_straggling_compute():
    # #227 window graph, one round, two layers.  Layer 3: plan done at 15, the segment runs 40..90, the CPU part ends at
    # 60, the host waits 60..95 -> the 4070 started 25 late and ran 30 past the CPU.  Layer 4: the segment ends at 140,
    # before the CPU part ends at 150; the host's wait (150..152) is all readback
    ev = [_x("decode round", 0, 300),
          _xa("cpu experts", 10, 90, 0), _xa("plan 4070 launch", 12, 3, 3), _xa("4070 finish", 60, 38, 3),
          _xa("4070 wait", 60, 35, 3), _xa("4070 segment", 40, 50, 3, tid=9),
          _xa("cpu experts", 105, 50, 0), _xa("plan 4070 launch", 107, 3, 4), _xa("4070 finish", 150, 4, 4),
          _xa("4070 wait", 150, 2, 4), _xa("4070 segment", 112, 28, 4, tid=9)]
    w = dp.window_split(ev)
    assert w["layers"] == 2 and w["rounds"] == 1
    assert w["4070 after cpu"] == 0.5
    assert w["ms"]["4070 straggle past cpu"] == 0.030      # layer 3 only: 90 - 60
    assert w["ms"]["4070 wait"] == 0.037
    assert w["ms"]["wait with the 4070 done"] == 0.007     # layer 3's 5 after its end at 90, layer 4's 2
    assert w["ms"]["plan -> segment start"] == 0.027       # 25 + 2
    assert w["ms"]["segment"] == 0.078
    # a shorter segment helps only where the 4070 straggled: layer 3's 30, capped at d; layer 4 gives nothing
    assert w["shave ms"] == {5: 0.005, 10: 0.01, 20: 0.02, 40: 0.03}


def test_a_window_graph_layer_without_a_segment_is_skipped():
    ev = [_x("decode round", 0, 100), _xa("cpu experts", 10, 50, 0), _xa("plan 4070 launch", 12, 3, 3),
          _xa("4070 finish", 40, 5, 3), _xa("4070 wait", 40, 4, 3)]
    assert dp.window_split(ev)["layers"] == 0


def test_layers_without_a_4070_share_are_skipped():
    ev = [_x("decode round", 0, 100), _x("cpu experts", 10, 50), _x("dispatch plan", 10, 5), _x("cpu pool", 20, 10),
          _x("4070 experts", 70, 5)]
    assert dp.breakdown(ev)["layers"] == 0


def test_the_round_edge_charges_each_gpu_idle_gap_to_the_host_spans_it_overlaps():
    ev = [_x("decode round", 0, 300),
          _x("verify graph", 0, 100), _x("commit graph", 110, 10), _x("mtp round", 130, 20), _x("verify graph", 170, 130),
          _x("verify commit", 100, 25), _x("mtp draft", 125, 35), _x("adapt join", 160, 5), _x("verify stage", 165, 4)]
    e = dp.edge(ev)
    assert e["gpu busy"] == {"verify graph": 0.23, "commit graph": 0.01, "mtp round": 0.02}
    assert e["gpu idle"] == 0.04
    assert e["idle during"] == {"verify commit": 0.015, "mtp draft": 0.015, "adapt join": 0.005, "verify stage": 0.004,
                                "(no host span)": 0.001}


def test_a_gap_goes_to_the_innermost_host_span():
    # the GPU idles 10..30; "verify stage" covers it, and "ple gather" inside it covers 15..25
    ev = [_x("decode round", 0, 100), _x("verify graph", 0, 10), _x("verify graph", 30, 10),
          _x("verify stage", 10, 20), _x("ple gather", 15, 10)]
    assert dp.edge(ev)["idle during"] == {"verify stage": 0.01, "ple gather": 0.01}
