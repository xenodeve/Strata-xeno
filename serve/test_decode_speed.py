"""The decoding speed of an answer with tools (serve/server.py run_with_mcp): the final `timings` carry the speed of the decoding over every round, not that of the last round alone, and the page shows
that one.  Tokens over the time of the whole answer (which holds the tools' time and the reading of each next prompt) is not a decoding speed: an agent's answer showed 23 tok/s where the engine decoded at 50."""
from __future__ import annotations

import json
import sys
import unittest
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from serve.test_agent_chat import DONE, Fixture  # noqa: E402
from serve.test_harness import tool_call  # noqa: E402


class WithTools(Fixture):
    RATES = (50.0, 200.0)                                       # the engine's decoding speed in each round, tokens a second

    def engine_with_clock(self, rates):
        engine, calls = self.engine, [0]
        orig = engine.generate

        def generate(ids, max_new, sampling, cancel, embeddings=None):
            n = 0
            try:
                for t in orig(ids, max_new, sampling, cancel, embeddings):
                    n += 1
                    yield t
            finally:                                              # the server stops reading at the end token: the engine's figures are written when the generator is closed
                rate = rates[min(calls[0], len(rates) - 1)]
                calls[0] += 1
                engine.last = {"prompt_ms": 100.0, "decode_ms": n / rate * 1000, "generated": n, "reused": 0}
                self.counts.append(n)

        self.counts: list[int] = []
        engine.generate = generate

    def final_timings(self):
        last = None
        req = self.post("/v1/chat/completions", self.body())
        with urllib.request.urlopen(req, timeout=60) as r:
            for raw in r:
                ln = raw.decode().strip()
                if ln.startswith("data: {"):
                    c = json.loads(ln[6:])
                    if c.get("timings"):
                        last = c["timings"]
        return last

    def test_the_speed_is_that_of_every_round_not_of_the_last(self):
        self.start(tool_call("Read", file_path="a.txt"), DONE)
        self.engine_with_clock(self.RATES)
        t = self.final_timings()
        first, second = self.counts[:2]
        want = (first + second) / (first / self.RATES[0] + second / self.RATES[1])
        self.assertAlmostEqual(t["predicted_per_second"], want, delta=0.2)
        self.assertNotAlmostEqual(t["predicted_per_second"], self.RATES[1], delta=5)          # not the last round's alone
        self.assertEqual(t["rounds"], 2)
        self.assertEqual(t["predicted_n"], first + second)
        self.assertAlmostEqual(t["predicted_ms"], (first / self.RATES[0] + second / self.RATES[1]) * 1000, delta=1)

    def test_what_is_read_stays_the_last_rounds(self):
        self.start(tool_call("Read", file_path="a.txt"), DONE)
        self.engine_with_clock(self.RATES)
        t = self.final_timings()
        self.assertEqual((t["prompt_ms"], t["cache_n"]), (100.0, 0))                          # the reading of the prompt that was last is not summed with the others

    def test_an_answer_without_tools_is_as_it_was(self):
        self.start(DONE)
        self.engine_with_clock((80.0,))
        t = self.final_timings()
        self.assertNotIn("rounds", t)
        self.assertAlmostEqual(t["predicted_per_second"], 80.0, delta=0.2)

    def test_an_engine_with_no_clock_gives_no_speed_and_nothing_breaks(self):
        self.start(tool_call("Read", file_path="a.txt"), DONE)
        self.assertIsNone(self.final_timings())


if __name__ == "__main__":
    unittest.main()
