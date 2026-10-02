"""An answer that uses the coding tools goes on when the page that asked is refreshed or closed (serve/runs.py): it is written to a buffer on its own thread, a page that comes back reads it again from
where it was, its permission questions are still there to answer, Stop is a request of its own, and a run that nobody looks at does not go on for ever."""
from __future__ import annotations

import json
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from serve import runs  # noqa: E402
from serve.test_agent_chat import DONE, Fixture  # noqa: E402
from serve.test_harness import tool_call  # noqa: E402


def gen(*chunks, hold: threading.Event | None = None):
    for c in chunks:
        yield c
    if hold is not None:
        hold.wait(5)


class TheStore(unittest.TestCase):
    def setUp(self):
        self.store = runs.RunStore()

    def launch(self, rid, chunks, cancel=None):
        return self.store.launch(rid, iter(chunks), cancel or threading.Event(), lambda e: {"error": {"message": str(e)}})

    def read(self, run, start=0, limit=50):
        out = []
        for ev in run.follow(start):
            if ev is not None:
                out.append(ev.decode())
            if len(out) >= limit:
                break
        return out

    def test_what_a_run_writes_can_be_read_whole_and_again_from_any_point(self):
        run = self.launch("run-aaaaaaaa", [{"n": 1}, {"n": 2}, {"n": 3}])
        whole = self.read(run)
        self.assertEqual(len(whole), 4)
        self.assertEqual(whole[-1], "data: [DONE]\n\n")
        self.assertEqual(self.read(run, 2), whole[2:])
        self.assertEqual(self.read(run, 4), [])
        self.assertEqual(self.read(run, 99), [])

    def test_a_reader_gets_events_as_they_come_and_ends_with_the_run(self):
        hold = threading.Event()
        run = self.launch("run-bbbbbbbb", gen({"n": 1}, hold=hold))
        got = []

        def collect():
            for ev in run.follow():
                if ev is not None:
                    got.append(ev.decode())

        t = threading.Thread(target=collect, daemon=True)
        t.start()
        time.sleep(0.4)
        self.assertEqual(len(got), 1)                                        # it is not made to wait for the end
        hold.set()
        t.join(5)
        self.assertEqual(got[-1], "data: [DONE]\n\n")

    def test_a_reader_that_leaves_does_not_stop_the_run(self):
        hold = threading.Event()
        run = self.launch("run-cccccccc", gen({"n": 1}, {"n": 2}, hold=hold))
        self.read(run, limit=1)                                              # looks at the first event and goes away
        self.assertEqual(run.viewers, 0)
        self.assertFalse(run.done)
        hold.set()
        deadline = time.time() + 5
        while not run.done and time.time() < deadline:
            time.sleep(0.05)
        self.assertTrue(run.done)
        self.assertEqual(len(self.read(run)), 3)

    def test_two_readers_read_the_same_run(self):
        run = self.launch("run-dddddddd", [{"n": i} for i in range(5)])
        self.assertEqual(self.read(run), self.read(run))

    def test_an_id_is_used_once(self):
        self.assertIsNotNone(self.launch("run-eeeeeeee", [{"n": 1}]))
        self.assertIsNone(self.launch("run-eeeeeeee", [{"n": 2}]))

    def test_a_stream_that_breaks_says_so_and_ends(self):
        def broken():
            yield {"n": 1}
            raise ValueError("the engine said no")
        run = self.launch("run-ffffffff", broken())
        events = self.read(run)
        self.assertEqual(json.loads(events[1][6:])["error"]["message"], "the engine said no")
        self.assertEqual(events[-1], "data: [DONE]\n\n")
        self.assertTrue(run.done)

    def test_stop_sets_the_runs_cancel(self):
        cancel = threading.Event()
        self.launch("run-gggggggg", [{"n": 1}], cancel)
        self.assertTrue(self.store.cancel("run-gggggggg"))
        self.assertTrue(cancel.is_set())
        self.assertFalse(self.store.cancel("run-nothere"))
        self.assertFalse(self.store.cancel(5))

    def test_a_run_that_grows_too_large_is_stopped_with_a_message(self):
        old = runs.MAX_BYTES
        runs.MAX_BYTES = 300
        self.addCleanup(setattr, runs, "MAX_BYTES", old)
        cancel = threading.Event()
        run = self.launch("run-hhhhhhhh", [{"text": "x" * 100}] * 10, cancel)
        events = self.read(run)
        self.assertTrue(cancel.is_set())
        self.assertIn("too large", events[-1] + events[-2])

    def test_one_that_nobody_has_looked_at_for_long_is_stopped_and_an_old_finished_one_is_forgotten(self):
        hold = threading.Event()
        cancel = threading.Event()
        run = self.launch("run-iiiiiiii", gen({"n": 1}, hold=hold), cancel)
        done = self.launch("run-jjjjjjjj", [{"n": 1}])
        deadline = time.time() + 5
        while not done.done and time.time() < deadline:
            time.sleep(0.05)
        self.store.sweep(time.time() + 60)                                    # a minute: not long
        self.assertFalse(cancel.is_set())
        self.assertIsNotNone(self.store.get("run-jjjjjjjj"))
        self.store.sweep(time.time() + runs.IDLE_S + 5)
        self.assertTrue(cancel.is_set())
        self.store.sweep(time.time() + runs.KEEP_S + 5)
        self.assertIsNone(self.store.get("run-jjjjjjjj"))
        hold.set()

    def test_a_run_that_is_being_read_is_not_idle(self):
        hold = threading.Event()
        cancel = threading.Event()
        run = self.launch("run-kkkkkkkk", gen({"n": 1}, hold=hold), cancel)
        t = threading.Thread(target=lambda: self.read(run), daemon=True)
        t.start()
        time.sleep(0.3)
        self.store.sweep(time.time() + runs.IDLE_S + 5)
        self.assertFalse(cancel.is_set())
        hold.set()
        t.join(5)

    def test_only_so_many_finished_runs_are_kept(self):
        for i in range(runs.MAX_RUNS + 4):
            r = self.launch(f"run-many-{i:04d}", [{"n": i}])
            deadline = time.time() + 5
            while not r.done and time.time() < deadline:
                time.sleep(0.01)
        self.store._prune()
        self.assertLessEqual(len(self.store.runs), runs.MAX_RUNS)
        self.assertIsNotNone(self.store.get(f"run-many-{runs.MAX_RUNS + 3:04d}"))       # the newest stay


RID = "run-test-0001"


class OverHttp(Fixture):
    """The chat endpoint with a run id: the page goes away, another reads on, Stop and the guards."""

    def stream(self, url_path, headers=None, on_event=None, stop_after=None):
        """Read an SSE endpoint; returns (events as parsed data, how many data lines were read).  `stop_after`: close the connection after that many data lines (the page going away)."""
        req = urllib.request.Request(self.base_url + url_path, headers=headers or {})
        out, count = [], 0
        with urllib.request.urlopen(req, timeout=30) as r:
            for raw in r:
                ln = raw.decode().strip()
                if not ln.startswith("data:"):
                    continue
                count += 1
                if ln != "data: [DONE]":
                    out.append(json.loads(ln[5:]))
                if on_event:
                    on_event(out[-1] if ln != "data: [DONE]" else None)
                if stop_after is not None and count >= stop_after:
                    break
        return out, count

    def chat_post(self, body, stop_after=None, on_event=None):
        req = self.post("/v1/chat/completions", body)
        out, count = [], 0
        with urllib.request.urlopen(req, timeout=30) as r:
            for raw in r:
                ln = raw.decode().strip()
                if not ln.startswith("data:"):
                    continue
                count += 1
                if ln != "data: [DONE]":
                    j = json.loads(ln[5:])
                    out.append(j)
                    if on_event and "strata_mcp" in j:
                        on_event(j["strata_mcp"])
                if stop_after is not None and count >= stop_after:
                    break
        return out, count

    def run_body(self, rid=RID, **kw):
        return self.body(agent={"run": rid, **kw.pop("agent", {})}, **kw)

    def texts(self, chunks):
        return "".join((c.get("choices") or [{}])[0].get("delta", {}).get("content") or "" for c in chunks if c.get("choices"))

    def events(self, chunks):
        return [c["strata_mcp"] for c in chunks if "strata_mcp" in c]

    def test_a_page_that_goes_away_does_not_stop_the_answer_and_a_page_that_comes_back_reads_the_rest(self):
        self.start(tool_call("Read", file_path="a.txt"), DONE)
        first, n = self.chat_post(self.run_body(), stop_after=3)                          # the page reads three events and is refreshed
        self.assertEqual(n, 3)
        rest, m = self.stream(f"/agent/run?id={RID}&from={n}")                            # the page that comes back
        whole, total = self.stream(f"/agent/run?id={RID}&from=0")
        self.assertEqual(n + m, total)
        self.assertEqual(first + rest, whole)                                              # what it read before and after is the whole of it
        self.assertTrue(self.texts(whole).strip().endswith("Done."))
        names = [e["event"] for e in self.events(whole)]
        self.assertIn("call", names)
        self.assertIn("result", names)
        self.assertTrue([e for e in self.events(whole) if e["event"] == "result"][0]["ok"])

    def test_a_permission_question_is_still_there_to_answer_after_the_page_came_back(self):
        outside = str(self.outside / "b.txt")
        self.start(tool_call("Read", file_path=outside), DONE)
        seen = []

        def until_asked(e):
            if e["event"] == "permission":
                seen.append(e)

        # the first page reads until the question and goes away without answering
        req = self.post("/v1/chat/completions", self.run_body())
        n = 0
        with urllib.request.urlopen(req, timeout=30) as r:
            for raw in r:
                ln = raw.decode().strip()
                if not ln.startswith("data:"):
                    continue
                n += 1
                j = json.loads(ln[5:]) if ln != "data: [DONE]" else {}
                if j.get("strata_mcp", {}).get("event") == "permission":
                    seen.append(j["strata_mcp"])
                    break
        self.assertEqual(len(seen), 1)
        time.sleep(0.5)
        # a page that comes back reads from the start, sees the same question, answers it, and the run goes on to its end
        answered = []

        def answer(e):
            if e and e.get("strata_mcp", {}).get("event") == "permission" and not answered:
                answered.append(e["strata_mcp"]["id"])
                with urllib.request.urlopen(self.post("/agent/permission", {"id": answered[0], "decision": "deny"}), timeout=10):
                    pass

        whole, _ = self.stream(f"/agent/run?id={RID}&from=0", on_event=answer)
        self.assertEqual(answered, [seen[0]["id"]])
        self.assertTrue(self.texts(whole).strip().endswith("Done."))
        self.assertFalse([e for e in self.events(whole) if e["event"] == "result"][0]["ok"])      # the user said no
        done = [e for e in self.events(whole) if e["event"] == "answered"]
        self.assertEqual([(e["id"], e["answer"]) for e in done], [(seen[0]["id"], "deny")])       # so that a page that reads the run again later does not show the question as waiting

    def test_stop_ends_a_run_that_is_waiting_for_the_user(self):
        outside = str(self.outside / "b.txt")
        self.start(tool_call("Read", file_path=outside), DONE)
        asked = threading.Event()
        out = {}

        def reader():
            out["chunks"], _ = self.chat_post(self.run_body(), on_event=lambda e: e["event"] == "permission" and asked.set())

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        self.assertTrue(asked.wait(10))
        with urllib.request.urlopen(self.post("/agent/cancel", {"id": RID}), timeout=10) as r:
            self.assertEqual(json.loads(r.read()), {"ok": True, "found": True})
        t.join(15)
        self.assertFalse(t.is_alive())                                                       # the run ended, and with it the stream
        res = [e for e in self.events(out["chunks"]) if e["event"] == "result"]
        self.assertTrue(res == [] or not res[0]["ok"])

    def test_stop_of_a_run_that_is_not_there_says_so(self):
        self.start(DONE)
        with urllib.request.urlopen(self.post("/agent/cancel", {"id": "run-nothere-1"}), timeout=10) as r:
            self.assertEqual(json.loads(r.read()), {"ok": True, "found": False})

    def test_what_is_not_there_is_not_found_and_an_id_is_used_once(self):
        self.start(DONE, DONE)
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(self.base_url + "/agent/run?id=run-none-0001&from=0", timeout=10)
        self.assertEqual(cm.exception.code, 404)
        self.chat_post(self.run_body())
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.chat_post(self.run_body())
        self.assertEqual(cm.exception.code, 409)

    def test_the_routes_check_what_they_get_and_who_asks(self):
        self.start(DONE)
        self.chat_post(self.run_body())
        for path, code in ((f"/agent/run?id={RID}&from=x", 400), (f"/agent/run?id={RID}&from=0", 403)):
            hdr = {"Origin": "http://evil.example"} if code == 403 else {}
            with self.assertRaises(urllib.error.HTTPError) as cm:
                urllib.request.urlopen(urllib.request.Request(self.base_url + path, headers=hdr), timeout=10)
            self.assertEqual(cm.exception.code, code, path)
        for body, hdr, code in (({"id": 5}, {}, 400), ([], {}, 400), ({"id": RID}, {"Origin": "http://evil.example"}, 403)):
            with self.assertRaises(urllib.error.HTTPError) as cm:
                urllib.request.urlopen(self.post("/agent/cancel", body, hdr), timeout=10)
            self.assertEqual(cm.exception.code, code, body)

    def test_a_request_with_no_run_id_streams_as_before_and_leaves_nothing_to_read(self):
        self.start(DONE)
        chunks, _ = self.chat_post(self.body())
        self.assertTrue(self.texts(chunks).strip().endswith("Done."))
        self.assertEqual(self.svc.runs.runs, {})

    def test_a_bad_run_id_is_not_used(self):
        self.start(DONE)
        for rid in ("short", "has space in it", "x" * 65, "a/b/c/d/e/f"):
            self.chat_post(self.run_body(rid) if False else self.body(agent={"run": rid}))
        self.assertEqual(self.svc.runs.runs, {})

    def test_a_run_that_is_over_can_still_be_read_by_a_page_that_was_away(self):
        self.start(DONE)
        self.chat_post(self.run_body(), stop_after=1)
        time.sleep(1.0)
        self.assertTrue(self.svc.runs.get(RID).done)
        whole, _ = self.stream(f"/agent/run?id={RID}&from=0")
        self.assertTrue(self.texts(whole).strip().endswith("Done."))


if __name__ == "__main__":
    unittest.main()
