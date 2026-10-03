"""#93: route_tools.records reads `--dump-routing` format 2 (include/strata/core/routing_trace.hpp: int32 layer, int32
k, k int32 ids, k float32 weights; a negative layer is a tag) and yields what its readers (prefetch_sim, lazy_refill_sim,
counts) consumed from the retired int16 `--route-trace`: one (layer, n_tok, k, ids) per window and layer, the window's
tokens in order.  A commit tag (-1) closes a window; other tags are skipped; uncommitted records still count, as the
int16 trace kept them."""
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent / "perf"))

import route_tools


def rec(layer, ids, weights=None):
    w = weights if weights is not None else [0.0] * len(ids)
    return struct.pack(f"<ii{len(ids)}i{len(ids)}f", layer, len(ids), *ids, *w)


class RecordsFormat2(unittest.TestCase):
    def read(self, chunks):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.bin"
            p.write_bytes(b"".join(chunks))
            return list(route_tools.records(p))

    def test_windows_and_layers(self):
        got = self.read([
            rec(-2, [2]), rec(-3, [7]), rec(-4, [5]), rec(-6, [9]),          # format, request, owned, boot
            rec(0, [1, 2]), rec(0, [3, 4]), rec(1, [5, 6]), rec(1, [7, 8]),  # window 0: two tokens, two layers
            rec(-1, [0, 2, 1, 1]),                                           # commit
            rec(0, [9, 10]), rec(1, [11, 12]),                               # window 1: one token
            rec(-1, [1, 1, 0, 1]),
        ])
        self.assertEqual([(l, nt, k, list(ids)) for l, nt, k, ids in got],
                         [(0, 2, 2, [1, 2, 3, 4]), (1, 2, 2, [5, 6, 7, 8]), (0, 1, 2, [9, 10]), (1, 1, 2, [11, 12])])

    def test_counts_one_window_per_layer_zero(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.bin"
            p.write_bytes(b"".join([rec(-2, [2]), rec(0, [1, 2]), rec(1, [3, 4]), rec(-1, [0, 1, 0, 1]),
                                    rec(0, [1, 5]), rec(-1, [1, 1, 0, 1])]))
            c, windows = route_tools.counts([p])
        self.assertEqual(windows, 2)
        self.assertEqual(c[(0, 1)], 2)
        self.assertEqual(c[(1, 3)], 1)

    def test_an_uncommitted_tail_still_counts(self):
        got = self.read([rec(-2, [2]), rec(0, [1, 2]), rec(1, [3, 4])])   # a run cut before its commit
        self.assertEqual([(l, nt) for l, nt, _, _ in got], [(0, 1), (1, 1)])


if __name__ == "__main__":
    unittest.main()
