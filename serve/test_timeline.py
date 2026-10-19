"""serve/test_timeline.py - #33: the server's spans on the pipeline timeline.

    python -m unittest serve.test_timeline -v

A request through the HTTP layer must leave http request > template+tokenize, queue wait, engine request, and a
first token instant, on the engine's clock (time.perf_counter, QueryPerformanceCounter on Windows), in the file the
analyzer reads beside the engine's (tests/xeno/perf/timeline.py load()).
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import timeline  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def records(path: Path) -> list[dict]:
    timeline.flush()   # #248: the records are written by the timeline's writer thread
    text = path.read_text(encoding="utf-8").rstrip().rstrip(",")
    return json.loads(text + "\n]")


class ServerTimeline(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "run.json.server.json"
        timeline.configure(str(self.path))

    def tearDown(self):
        timeline.configure(None)
        self.dir.cleanup()

    def test_spans_are_on_the_perf_counter_clock_and_name_each_thread_once(self):
        t0 = time.perf_counter() * 1e6
        with timeline.span("outer", 3):
            timeline.instant("mark")
        timeline.complete("after", timeline.now_us(), timeline.now_us())
        t1 = time.perf_counter() * 1e6
        rs = records(self.path)
        xs = [r for r in rs if r["ph"] == "X"]
        self.assertEqual([r["name"] for r in xs], ["outer", "after"])
        self.assertEqual(xs[0]["args"], {"a": 3, "b": -1})
        self.assertTrue(t0 <= xs[0]["ts"] <= t1)
        self.assertEqual(sum(1 for r in rs if r.get("name") == "thread_name"), 1)
        self.assertEqual(sum(1 for r in rs if r["ph"] == "i"), 1)

    def test_a_record_does_not_wait_for_the_disk(self):
        # #248: on a slow drive (F:, a USB stick) every synchronous append held the request thread - 2.3 s before one
        # request's queue wait.  A record is handed to the writer thread; the caller returns at once.
        import builtins
        real_open = builtins.open

        def slow_open(*args, **kwargs):
            if str(args[0]) == str(self.path):
                time.sleep(0.4)
            return real_open(*args, **kwargs)

        builtins.open = slow_open
        try:
            t0 = time.perf_counter()
            for i in range(3):
                timeline.complete("r", timeline.now_us(), timeline.now_us(), i)
            took = time.perf_counter() - t0
        finally:
            builtins.open = real_open
        self.assertLess(took, 0.2)
        self.assertEqual([r["args"]["a"] for r in records(self.path) if r["ph"] == "X"], [0, 1, 2])

    def test_off_writes_nothing(self):
        timeline.configure(None)
        with timeline.span("x"):
            pass
        self.assertFalse(self.path.exists())

    def test_a_request_leaves_its_stages(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "</think>\n\nhello there", max_context=4096), tok,
                      ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        try:
            body = {"model": "m", "max_tokens": 20, "messages": [{"role": "user", "content": "hi"}]}
            req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}/v1/messages",
                                         data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                self.assertEqual(r.status, 200)
        finally:
            httpd.shutdown()
            httpd.server_close()
        rs = records(self.path)
        names = [r["name"] for r in rs if r["ph"] == "X"]
        for n in ("http request", "template+tokenize", "queue wait", "engine request"):
            self.assertIn(n, names)
        http = next(r for r in rs if r["name"] == "http request")
        first = next(r for r in rs if r["name"] == "first token")
        self.assertTrue(http["ts"] <= first["ts"] <= http["ts"] + http["dur"])


if __name__ == "__main__":
    unittest.main()
