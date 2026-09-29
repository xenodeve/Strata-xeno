"""#32: the 2-GPU prefill simulator (tests/xeno/perf/prefill_route_sim.py).  Every expected value is worked out by
hand from the tiny synthetic trace below, not recomputed the way the simulator does it."""
import os
import struct
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "perf"))
import prefill_route_sim as sim  # noqa: E402

MAGIC = 0x52505453


def record(pos0, layer, T, K, experts, ids):
    """experts: [(rows, resident_here, owned_by_peer)]"""
    out = struct.pack("<6i", MAGIC, pos0, layer, T, K, len(experts))
    for rows, here, peer in experts:
        out += struct.pack("<3i", rows, here, peer)
    return out + struct.pack(f"<{len(ids)}i", *ids)


# 4 tokens, K = 2, 4 experts.  Layer 0: tokens 0,1 use experts 0,1; tokens 2,3 use experts 2,3.
# Expert 0 is resident on the 5060, expert 3 is owned by the 4070, experts 1 and 2 are in host RAM.
L0 = record(0, 0, 4, 2, [(2, 1, 0), (2, 0, 0), (2, 0, 0), (2, 0, 1)], [0, 1, 0, 1, 2, 3, 2, 3])
# Layer 1: every token uses experts 0 and 1, both in host RAM; experts 2, 3 unused.
L1 = record(0, 1, 4, 2, [(4, 0, 0), (4, 0, 0), (0, 0, 0), (0, 0, 0)], [0, 1] * 4)


@pytest.fixture
def trace(tmp_path):
    p = tmp_path / "route.bin"
    p.write_bytes(L0 + L1)
    return str(p)


def test_reader_returns_one_layer_per_record(trace):
    layers = sim.read_trace(trace)
    assert [(x.layer, x.T, x.K) for x in layers] == [(0, 4, 2), (1, 4, 2)]
    assert layers[0].rows == [2, 2, 2, 2]
    assert layers[0].here == [True, False, False, False]
    assert layers[0].peer == [False, False, False, True]
    assert layers[1].ids[:4] == [0, 1, 0, 1]


def test_whole_5060_streams_every_routed_expert_it_does_not_hold(trace):
    layers = sim.read_trace(trace)
    p = sim.Params(blob_mb=1.0, x4_gbs=1.0, x16_gbs=4.0, h_bytes=1000, relay_bytes_per_value=4,
                   ms_per_row_5060=0.5, speed_4070=2.0)
    r0 = sim.whole_5060(layers[0], p)
    # experts 1, 2 from host over x4 (2 MB); expert 3 is the peer's: 4070 -> host (x16) -> 5060 (x4)
    assert r0.x4_mb == pytest.approx(3.0)
    assert r0.x16_mb == pytest.approx(1.0)
    # stream: 3 MB at 1 GB/s = 3 ms on x4 (+ 0.25 ms on x16 for the peer read, serial with it) = 3.25 ms;
    # compute: 8 rows x 0.5 ms = 4 ms; overlapped -> max
    assert r0.ms == pytest.approx(4.0)
    r1 = sim.whole_5060(layers[1], p)
    assert r1.x4_mb == pytest.approx(2.0)
    assert r1.ms == pytest.approx(4.0)            # max(2 ms stream, 4 ms compute)


def test_whole_4070_pays_the_relay_and_the_5060s_residents(trace):
    layers = sim.read_trace(trace)
    p = sim.Params(blob_mb=1.0, x4_gbs=1.0, x16_gbs=4.0, h_bytes=1000, relay_bytes_per_value=4,
                   ms_per_row_5060=0.5, speed_4070=2.0)
    r = sim.whole_4070(layers[0], p)
    # experts 1, 2 from host over x16; expert 0 lives only on the 5060: 5060 -> host (x4) -> 4070 (x16); the relay
    # crosses x16 too
    assert r.x16_mb == pytest.approx(3.0 + 0.032)
    # relay: T x H x 4 bytes = 16,000 B each way, over both links (no P2P): 0.016 MB x 2 directions
    assert r.relay_mb == pytest.approx(0.032)
    assert r.x4_mb == pytest.approx(1.0 + 0.032)
    # x4 time 1.032 ms, x16 time (3 + 0.032) / 4 = 0.758 ms -> links serial per card chain: stream 1.032 + 0.758;
    # compute 8 rows x 0.5 / 2.0 = 2 ms; finish = max(stream, compute) = 2.0
    assert r.ms == pytest.approx(max(1.032 + 0.758, 2.0))


def test_token_split_counts_the_unique_experts_each_card_needs(trace):
    layers = sim.read_trace(trace)
    # half the tokens (0, 1) to the 5060, half (2, 3) to the 4070: the 5060 needs experts {0, 1}, the 4070 {2, 3}
    u5060, u4070 = sim.split_experts(layers[0], share_4070=0.5)
    assert u5060 == {0, 1} and u4070 == {2, 3}
    # layer 1: both halves need {0, 1}: the duplication trap
    u5060, u4070 = sim.split_experts(layers[1], share_4070=0.5)
    assert u5060 == {0, 1} and u4070 == {0, 1}


def test_report_totals_every_policy(trace):
    layers = sim.read_trace(trace)
    p = sim.Params(blob_mb=1.0, x4_gbs=1.0, x16_gbs=4.0, h_bytes=1000, relay_bytes_per_value=4,
                   ms_per_row_5060=0.5, speed_4070=2.0)
    t = sim.totals(layers, p)
    assert set(t) >= {"whole_5060", "whole_4070", "split_25", "split_50", "best_per_layer"}
    assert t["whole_5060"]["ms"] == pytest.approx(8.0)
    # the per-layer best never loses to a fixed policy
    assert t["best_per_layer"]["ms"] <= min(t[k]["ms"] for k in ("whole_5060", "whole_4070", "split_25", "split_50"))


def test_expert_split_keeps_each_expert_where_its_weights_are(trace):
    # #32's shape: the 5060 computes the experts it holds (no stream), the 4070 every other routed expert (host ones
    # over x16, its own for free); only the tokens that route to a 4070 expert are relayed.  Layer 0: expert 0 stays;
    # 1, 2 over x16; 3 is the 4070's; every token routes to a 4070 expert, so the relay is all 4 tokens.
    layers = sim.read_trace(trace)
    p = sim.Params(blob_mb=1.0, x4_gbs=1.0, x16_gbs=4.0, h_bytes=1000, relay_bytes_per_value=4,
                   ms_per_row_5060=0.5, speed_4070=2.0)
    r = sim.expert_split(layers[0], p)
    assert r.x4_mb == pytest.approx(0.032)
    assert r.x16_mb == pytest.approx(2.0 + 0.032)
    assert (r.rows_5060, r.rows_4070) == (2, 6)
    # max(0.032 / 1, 2.032 / 4, 2 x 0.5, 6 x 0.5 / 2) = 1.5
    assert r.ms == pytest.approx(1.5)
