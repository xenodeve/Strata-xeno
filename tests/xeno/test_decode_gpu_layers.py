"""#44 D2: decode_gpu_layers splits the primary GPU's decode kernels into the regions of a layer, delimited by the
kernels that mark them: doorbell_publish (routes out to the host), the first wait_flag (the host's plan), the second
wait_flag (the CPU and 4070 partials) and merge_mapped (the partials in)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "perf"))
import decode_gpu_layers as dg  # noqa: E402


def _layer(t, gu=100, w2=5):
    # (start, end, name) in ns: trunk, publish, shared expert, plan wait, routed experts, partial wait, merge
    seq = [("gr_down_multi_kernel", 20), ("route", 3), ("doorbell_publish_kernel", 10), ("native_mmvq_multi_kernel", 7),
           ("wait_flag_ge_kernel", 2), ("copy_i32_from_mapped_kernel", 5), ("native_gu_q2_kernel", gu),
           ("wait_flag_ge_kernel", w2), ("merge_mapped_kernel", 18)]
    out = []
    for name, d in seq:
        out.append((t, t + d, name))
        t += d + 1   # 1 ns gap after every kernel
    return out, t


def test_a_layer_is_split_into_its_regions():
    k, t = _layer(0)
    k2, _ = _layer(t)
    r = dg.split(k + k2)
    assert r["layers"] == 2
    # the first layer's trunk (before its doorbell) is not attributed: nothing precedes it
    assert r["ns"]["routed experts (5060)"] == 2 * (5 + 1 + 100 + 1)
    assert r["ns"]["wait partials"] == 2 * (5 + 1)
    assert r["ns"]["merge partials"] == (18 + 1) + 18   # the capture's last kernel has no gap after it
    assert r["ns"]["shared expert"] == 2 * (7 + 1)
    assert r["ns"]["wait plan"] == 2 * (2 + 1)
    assert r["ns"]["publish routes"] == 2 * (10 + 1)
    # the second layer's trunk: the first layer ends with merge; then gr_down and route run before the doorbell
    assert r["ns"]["trunk"] == (20 + 1) + (3 + 1)
    assert r["idle"]["trunk"] == 2 and r["idle"]["merge partials"] == 1


def test_the_kernels_of_each_region_are_listed():
    k, t = _layer(0)
    k2, _ = _layer(t)
    r = dg.split(k + k2)
    assert r["kernels"]["routed experts (5060)"]["native_gu_q2_kernel"] == 200
    assert r["kernels"]["trunk"]["gr_down_multi_kernel"] == 20
