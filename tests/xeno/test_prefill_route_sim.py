"""#32: the 2-GPU prefill simulator (tests/xeno/perf/prefill_route_sim.py).  Every expected value is worked out by
hand from the tiny synthetic trace below, not recomputed the way the simulator does it.

Exactness sets the data movement: moe_combine is a per-token fmaf chain over k = 0..9, so a token's combine runs on
the card that holds all K of its expert rows.  A policy moves expert rows (T*K granular), never partial sums."""
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

# 1 MB experts, x4 1 GB/s, x16 4 GB/s, H = 1000 (a row of f32 = 4,000 B), a q8 activation row = 100 B,
# 0.5 ms per expert row on the 5060, the 4070 twice as fast
P = sim.Params(blob_mb=1.0, x4_gbs=1.0, x16_gbs=4.0, h=1000, act_row_bytes=100, ms_per_row_5060=0.5, speed_4070=2.0)


@pytest.fixture
def layers(tmp_path):
    p = tmp_path / "route.bin"
    p.write_bytes(L0 + L1)
    return sim.read_trace(str(p))


def test_reader_returns_one_layer_per_record(layers):
    assert [(x.layer, x.T, x.K) for x in layers] == [(0, 4, 2), (1, 4, 2)]
    assert layers[0].rows == [2, 2, 2, 2]
    assert layers[0].here == [True, False, False, False]
    assert layers[0].peer == [False, False, False, True]
    assert layers[1].ids[:4] == [0, 1, 0, 1]


def test_whole_5060_streams_every_routed_expert_it_does_not_hold(layers):
    r0 = sim.whole_5060(layers[0], P)
    # experts 1, 2 from host over x4 (2 MB); expert 3 is the peer's: 4070 -> host (x16) -> 5060 (x4)
    assert r0.x4_mb == pytest.approx(3.0)
    assert r0.x16_mb == pytest.approx(1.0)
    assert r0.ms == pytest.approx(4.0)           # max(3 ms x4, 0.25 ms x16, 8 rows x 0.5 ms)
    assert sim.whole_5060(layers[1], P).ms == pytest.approx(4.0)


def test_whole_4070_ships_activations_shared_and_the_5060s_experts_out_and_the_layer_output_back(layers):
    r = sim.whole_4070(layers[0], P)
    # out: q8 activations 4 x 100 B, the shared-expert output 4 x 4,000 B, expert 0 (1 MB, only on the 5060);
    # back: the combined output 4 x 4,000 B.  Every byte crosses both links (no P2P).
    # x16 also carries the host experts 1, 2 (2 MB); expert 3 is already on the 4070.
    assert r.x4_mb == pytest.approx(1.0 + 0.0004 + 0.016 + 0.016)
    assert r.x16_mb == pytest.approx(2.0 + 1.0 + 0.0004 + 0.016 + 0.016)
    assert (r.rows_5060, r.rows_4070) == (0, 8)
    assert r.ms == pytest.approx(2.0)            # max(1.032, 3.032 / 4, 8 x 0.5 / 2)


def test_expert_split_keeps_the_5060s_experts_there_and_ships_their_rows(layers):
    r = sim.expert_split(layers[0], P)
    # the 5060 computes expert 0 (2 rows) and ships those rows (2 x 4,000 B) to the 4070, where every token's combine
    # runs; the 4070 computes experts 1, 2 (host, over x16) and 3 (its own); activations, shared and output as above
    assert r.x4_mb == pytest.approx(0.008 + 0.0004 + 0.016 + 0.016)
    assert r.x16_mb == pytest.approx(2.0 + 0.008 + 0.0004 + 0.016 + 0.016)
    assert (r.rows_5060, r.rows_4070) == (2, 6)
    assert r.ms == pytest.approx(1.5)            # max(0.04, 0.51, 2 x 0.5, 6 x 0.5 / 2)


def test_token_split_moves_only_the_4070_tokens(layers):
    # tokens 2, 3 to the 4070: it needs experts {2, 3} (2 from host, 3 its own), the 5060 {0, 1} (1 from host);
    # the 4070 combines its own tokens: activations, shared and output for 2 tokens
    r = sim.split(layers[0], P, 0.5)
    assert r.x4_mb == pytest.approx(1.0 + 0.0002 + 0.008 + 0.008)
    assert r.x16_mb == pytest.approx(1.0 + 0.0002 + 0.008 + 0.008)
    assert (r.rows_5060, r.rows_4070) == (4, 4)
    assert r.ms == pytest.approx(2.0)


def test_split_experts_shows_the_duplication_trap(layers):
    assert sim.split_experts(layers[0], share_4070=0.5) == ({0, 1}, {2, 3})
    assert sim.split_experts(layers[1], share_4070=0.5) == ({0, 1}, {0, 1})


def test_report_totals_every_policy(layers):
    t = sim.totals(layers, P)
    assert set(t) >= {"whole_5060", "whole_4070", "expert_split", "split_25", "split_50", "best_per_layer"}
    assert t["whole_5060"]["ms"] == pytest.approx(8.0)
    fixed = ("whole_5060", "whole_4070", "expert_split", "split_25", "split_50")
    assert t["best_per_layer"]["ms"] <= min(t[k]["ms"] for k in fixed)
