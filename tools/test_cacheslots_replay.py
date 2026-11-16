"""tools/test_cacheslots_replay.py - the slot-rule replay (#321), synthetic streams, no engine.

    python -m pytest tools/test_cacheslots_replay.py -q
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cacheslots_replay as cr  # noqa: E402
from serve.server import slot_trace  # noqa: E402

HEAD = list(range(600))


def turn(user, t, step=1200):
    return HEAD + [user * 1_000_000 + i for i in range(step * (t + 1))]


def rec(ids, user=None, pinned=None):
    who = None if user is None else {"key": "owner", "device": f"addr:10.0.0.{user}", "addr": f"10.0.0.{user}"}
    return {"ids": ids, "who": who, "route": "anthropic", "pinned": pinned}


def alternating(turns=3):
    return [rec(turn(u, t), u) for t in range(turns) for u in (1, 2)]


class Replay(unittest.TestCase):
    def test_two_users_alternating(self):
        records = alternating()
        fam = cr.replay(records, "family")
        self.assertEqual(fam["switches"], 0)                    # one family, one slot
        self.assertEqual((fam["evictions"], fam["regrets"]), (5, 4))
        self.assertEqual(fam["read"], 1800 + 1200 + 2400 + 2400 + 3600 + 3600)   # each re-reads past the head
        for rule in ("identity", "continuation"):
            res = cr.replay(records, rule)
            self.assertEqual((res["switches"], res["evictions"], res["regrets"]), (5, 0, 0), rule)
            self.assertEqual(res["read"], 2 * 1800 + 4 * 1200, rule)   # each turn reads only its new 1,200
            self.assertEqual([r["how"] for r in res["rows"]], ["free"] * 2 + ["continues"] * 4)

    def test_a_pinned_request_goes_to_its_slot(self):
        records = [rec(turn(1, 0), 1, pinned=3), rec(turn(1, 1), 1, pinned=3)]
        for rule in cr.RULES:
            self.assertEqual([r["slot"] for r in cr.replay(records, rule)["rows"]], [3, 3], rule)

    def test_regret_tokens_are_the_evicted_prompt(self):
        records = [rec(turn(u, 0), u) for u in (1, 2, 3)] + [rec(turn(1, 1), 1)]
        res = cr.replay(records, "family", slots=1)
        self.assertEqual((res["evictions"], res["regrets"], res["regret_tokens"]), (3, 1, 1800))

    def test_live_conversations_per_sender(self):
        other = [7] * 600
        records = [rec(turn(1, 0), 1), rec(other + [1] * 100, 1), rec(turn(1, 1), 1), rec(other + [1] * 200, 1),
                   rec(turn(2, 0), 2)]
        live = cr.live_conversations(records)
        self.assertEqual(live["owner|addr:10.0.0.1"]["conversations"], 2)
        self.assertEqual(live["owner|addr:10.0.0.1"]["live_max"], 2)      # at request 1 both are open
        self.assertEqual(live["owner|addr:10.0.0.2"], {"conversations": 1, "live_max": 0, "live_median": 0})

    def test_a_server_trace_round_trips(self):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, True)
        for name in ("t.jsonl", "t.jsonl.gz"):
            path = str(d / name)
            for r in alternating(2):
                slot_trace(path, {"max_tokens": 8}, r["ids"], r["who"], "anthropic", {"cache_slots": 4})
            back = cr.read_trace(path)
            self.assertEqual([b["ids"] for b in back], [r["ids"] for r in alternating(2)])
            out = cr.report(back)
            self.assertIn("family", out)
            self.assertIn("owner|addr:10.0.0.2", out)

    def test_main_needs_a_trace_or_bench(self):
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            cr.main([])


if __name__ == "__main__":
    unittest.main()
