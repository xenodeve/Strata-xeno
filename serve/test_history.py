"""serve/test_history.py - the request history on disk (xeno UI S3): a summary line per request kept forever, a
detail file per request under a size cap, oldest deleted first.

    python -m unittest serve.test_history -v
"""
from __future__ import annotations

import gzip
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.history import HistoryStore  # noqa: E402


def rec(i, t=None, **kw):
    return {"id": f"r{i:04d}", "time": t if t is not None else 1_790_000_000.0 + i, "finish": "stop", **kw}


class Summaries(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_summary_goes_to_the_month_file(self):
        h = HistoryStore(self.dir)
        h.append(rec(1, t=time.mktime((2026, 10, 1, 12, 0, 0, 0, 0, -1))))
        h.append(rec(2, t=time.mktime((2026, 11, 2, 12, 0, 0, 0, 0, -1))))
        self.assertEqual(sorted(p.name for p in self.dir.glob("requests-*.jsonl")),
                         ["requests-2026-10.jsonl", "requests-2026-11.jsonl"])
        line = (self.dir / "requests-2026-10.jsonl").read_text(encoding="utf-8").strip()
        self.assertEqual(json.loads(line)["id"], "r0001")

    def test_page_is_newest_first_and_survives_a_restart(self):
        h = HistoryStore(self.dir)
        for i in range(25):
            h.append(rec(i))
        h2 = HistoryStore(self.dir)                         # a new process reads what the old one wrote
        page = h2.page(0, 10)
        self.assertEqual([r["id"] for r in page["items"]][:2], ["r0024", "r0023"])
        self.assertEqual(len(page["items"]), 10)
        self.assertEqual(page["total"], 25)
        self.assertEqual([r["id"] for r in h2.page(2, 10)["items"]], [f"r{i:04d}" for i in (4, 3, 2, 1, 0)])

    def test_a_torn_last_line_does_not_hide_the_rest(self):
        h = HistoryStore(self.dir)
        h.append(rec(1))
        f = next(self.dir.glob("requests-*.jsonl"))
        with open(f, "a", encoding="utf-8") as fh:
            fh.write('{"id": "r0002", "tim')                # a crash mid-write
        self.assertEqual([r["id"] for r in HistoryStore(self.dir).page(0, 10)["items"]], ["r0001"])

    def test_off_writes_nothing(self):
        h = HistoryStore(self.dir, enabled=False)
        h.append(rec(1))
        h.write_detail("r0001", {"rounds": []})
        self.assertEqual(list(self.dir.iterdir()), [])
        self.assertEqual(h.page(0, 10)["items"], [])


class Details(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_detail_round_trips_gzipped(self):
        h = HistoryStore(self.dir)
        h.write_detail("r0001", {"rounds": [1, 2, 3]})
        raw = (self.dir / "detail" / "r0001.json.gz").read_bytes()
        self.assertEqual(json.loads(gzip.decompress(raw)), {"rounds": [1, 2, 3]})
        self.assertEqual(h.detail("r0001"), {"rounds": [1, 2, 3]})

    def test_missing_detail_says_deleted_when_the_summary_exists(self):
        h = HistoryStore(self.dir)
        h.append(rec(1))
        self.assertIsNone(h.detail("r0001"))
        self.assertTrue(h.has_summary("r0001"))
        self.assertFalse(h.has_summary("nope"))

    def test_cap_deletes_oldest_first_and_keeps_summaries(self):
        h = HistoryStore(self.dir, detail_cap_bytes=3000)
        import base64
        import random
        blob = {"x": base64.b64encode(random.Random(1).randbytes(900)).decode()}  # ~900 B gzipped, ten of them > cap
        for i in range(10):
            h.append(rec(i))
            h.write_detail(f"r{i:04d}", blob)
            time.sleep(0.01)                                 # mtime order = write order
        left = sorted(p.stem.split(".")[0] for p in (self.dir / "detail").glob("*.json.gz"))
        self.assertTrue(left and len(left) < 10)
        self.assertIn("r0009", left)                         # the newest survives
        self.assertNotIn("r0000", left)
        self.assertLessEqual(sum(p.stat().st_size for p in (self.dir / "detail").glob("*.json.gz")), 3000)
        self.assertEqual(HistoryStore(self.dir).page(0, 50)["total"], 10)    # every summary stays

    def test_ids_cannot_leave_the_folder(self):
        h = HistoryStore(self.dir)
        with self.assertRaises(ValueError):
            h.write_detail("../evil", {})
        self.assertIsNone(h.detail("..\\..\\evil"))


class Meta(unittest.TestCase):
    def test_meta_keeps_numbers_names_and_200_chars_not_the_prompt(self):
        from serve.history import request_meta
        msgs = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "ok"},
                {"role": "user", "content": [{"type": "text", "text": "x" * 500}, {"type": "image_url"}]}]
        m = request_meta("anthropic", msgs, [{"name": "Read"}, {"function": {"name": "Bash"}}], "claude-cli/2.1 (x)")
        self.assertEqual(m["dialect"], "anthropic")
        self.assertEqual(m["client"], "claude-cli/2.1 (x)")
        self.assertEqual(m["tools"], ["Read", "Bash"])
        self.assertEqual(m["preview"], "x" * 200)
        self.assertNotIn("messages", m)
        self.assertTrue(m["id"].startswith("r") and len(m["id"]) >= 12)

    def test_meta_without_a_user_message_or_tools(self):
        from serve.history import request_meta
        m = request_meta("openai", [], None, None)
        self.assertEqual((m["preview"], m["tools"], m["client"]), ("", [], ""))

    def test_summary_adds_prefill_speed_from_what_was_read(self):
        from serve.history import summary_record
        s = summary_record({"id": "r1", "prompt_tokens": 1100, "reused": 100, "prompt_ms": 500.0})
        self.assertEqual(s["prefill_tok_s"], 2000.0)             # (1100 - 100 cached) / 0.5 s
        self.assertIsNone(summary_record({"id": "r1", "prompt_tokens": 10, "reused": 10, "prompt_ms": 5.0})["prefill_tok_s"])
        self.assertIsNone(summary_record({"id": "r1", "prompt_tokens": 10, "prompt_ms": None})["prefill_tok_s"])


class PrefillChunks(unittest.TestCase):
    """Prefill speed per chunk (max / min / mean) from the engine's PP lines: (position reached, ms since the read began)."""

    def test_chunks_from_cumulative_points(self):
        from serve.history import chunk_stats
        # 100 cached tokens, then chunks of 2048 (1 s), 2048 (2 s), 904 (0.5 s)
        s = chunk_stats([(2148, 1000.0), (4196, 3000.0), (5100, 3500.0)], start=100)
        self.assertEqual(s["chunks"], 3)
        self.assertEqual(s["items"], [[2048, 1000.0], [2048, 2000.0], [904, 500.0]])
        self.assertAlmostEqual(s["tok_s_max"], 2048.0)           # the 1 s chunk
        self.assertAlmostEqual(s["tok_s_min"], 1024.0)           # the 2 s chunk
        self.assertAlmostEqual(s["tok_s_mean"], 5000 / 3.5, places=1)    # all tokens over all time, not a mean of means

    def test_one_chunk_has_no_spread(self):
        from serve.history import chunk_stats
        s = chunk_stats([(600, 300.0)], start=0)
        self.assertEqual((s["chunks"], s["tok_s_max"], s["tok_s_min"], s["tok_s_mean"]), (1, 2000.0, 2000.0, 2000.0))

    def test_a_second_engine_call_restarts_the_clock_but_not_the_position(self):
        from serve.history import chunk_stats
        s = chunk_stats([(1000, 500.0), (1500, 100.0)], start=0)     # ms went back: a new read began at 1000
        self.assertEqual(s["items"], [[1000, 500.0], [500, 100.0]])

    def test_nothing_read_is_none(self):
        from serve.history import chunk_stats
        self.assertIsNone(chunk_stats([], start=0))
        self.assertIsNone(chunk_stats([(100, 0.0)], start=100))      # no tokens, no time: nothing to rate


class DecodeWindows(unittest.TestCase):
    """Decode speed over a sliding 16-token window: MTP lands tokens in bursts, so a per-token speed means nothing."""

    def test_steady_rate_has_no_spread(self):
        from serve.history import window_rates
        s = window_rates([i * 0.05 for i in range(101)])         # 20 tok/s, 101 tokens
        self.assertEqual((s["windows"], s["tok_s_min"], s["tok_s_max"], s["tok_s_mean"]), (85, 20.0, 20.0, 20.0))

    def test_bursts_do_not_make_an_instantaneous_peak(self):
        from serve.history import window_rates
        times, t = [], 0.0
        for _ in range(40):                                        # 3 tokens land together every 150 ms: 20 tok/s
            times += [t, t, t]
            t += 0.15
        s = window_rates(times)
        self.assertLess(s["tok_s_max"], 25.0)                      # the burst's own gap would read as infinity
        self.assertGreater(s["tok_s_min"], 15.0)

    def test_a_slow_stretch_shows_as_the_min(self):
        from serve.history import window_rates
        times = [i * 0.05 for i in range(50)]                      # 20 tok/s ...
        t0 = times[-1]
        times += [t0 + (i + 1) * 0.2 for i in range(50)]           # ... then 5 tok/s
        s = window_rates(times)
        self.assertEqual((s["tok_s_max"], s["tok_s_min"]), (20.0, 5.0))
        self.assertLess(s["series"][-1], s["series"][0])            # the trend over the request

    def test_series_is_downsampled(self):
        from serve.history import window_rates
        self.assertLessEqual(len(window_rates([i * 0.01 for i in range(5000)])["series"]), 120)

    def test_fewer_tokens_than_a_window_report_the_mean_only(self):
        from serve.history import window_rates
        s = window_rates([0.0, 0.1, 0.2, 0.3])                      # 3 intervals in 0.3 s
        self.assertEqual((s["windows"], s["tok_s_min"], s["tok_s_max"], s["tok_s_mean"]), (0, None, None, 10.0))
        self.assertIsNone(window_rates([0.0]))
        self.assertIsNone(window_rates([]))


if __name__ == "__main__":
    unittest.main()
