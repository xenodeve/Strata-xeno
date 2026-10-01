"""#85: a --dump-routing trace with tag records (negative layer: format, window commit, request, owned set, phase)
must rank exactly as the same trace without them.  make_profile.py reads only 0 <= layer < N_LAYER and advances by
each record's own k, so the tags are skipped; this pins that, byte layout as include/strata/core/routing_trace.hpp
writes it (int32 layer, int32 k, k int32, k float32)."""
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))   # runnable from the repo root too

import make_profile


def rec(layer, ids, weights=None):
    w = weights if weights is not None else [0.0] * len(ids)
    return struct.pack(f"<ii{len(ids)}i{len(ids)}f", layer, len(ids), *ids, *w)


ROUTES = [rec(0, [1, 2, 3]), rec(1, [4, 5, 6]), rec(47, [511, 0, 7], [0.5, 0.3, 0.2]), rec(0, [1, 9, 3])]
TAGS = [rec(-2, [1]), rec(-3, [42]), rec(-4, [0 * 512 + 1, 47 * 512 + 511]), rec(-5, [1]), rec(-1, [0, 4, 2])]


class MakeProfileSkipsTags(unittest.TestCase):
    def counts(self, chunks):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.bin"
            p.write_bytes(b"".join(chunks))
            return dict(make_profile.read_trace(p))

    def test_tags_change_no_count(self):
        plain = self.counts(ROUTES)
        tagged = self.counts([TAGS[0], TAGS[1], TAGS[2], TAGS[3]] + ROUTES[:2] + [TAGS[4]] + ROUTES[2:] + [TAGS[4]])
        self.assertEqual(plain, tagged)
        self.assertEqual(plain[(0, 1)], 2)
        self.assertEqual(plain[(47, 511)], 1)

    def test_no_tag_value_becomes_a_pair(self):
        tagged = self.counts(TAGS)
        self.assertEqual(tagged, {})


if __name__ == "__main__":
    unittest.main()
