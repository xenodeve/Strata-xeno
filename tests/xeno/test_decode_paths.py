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


def test_layers_without_a_4070_share_are_skipped():
    ev = [_x("decode round", 0, 100), _x("cpu experts", 10, 50), _x("dispatch plan", 10, 5), _x("cpu pool", 20, 10),
          _x("4070 experts", 70, 5)]
    assert dp.breakdown(ev)["layers"] == 0
