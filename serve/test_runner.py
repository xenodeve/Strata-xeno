"""serve/test_runner.py - the runner program `python -m serve.runner` (#310 part 2, task 6): Strata Desktop's background
part on the engineer's own PC.  A real server (serve(svc, port=0), team mode, a scripted model) and a real
`serve.runner` on a thread, whose files (runner.json, device.json, checkpoints/) live in a temp folder.

    python -m pytest serve/test_runner.py -q
"""
from __future__ import annotations

import contextlib
import http.client
import io
import json
import os
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import runner as runner_mod, runner_hub  # noqa: E402
from serve.identity import OWNER  # noqa: E402
from serve.server import serve  # noqa: E402
from serve.test_harness import tool_call  # noqa: E402
from serve.test_identity_http import DEV1, ROOT  # noqa: E402
from serve.test_runner_http import DEV0, DEV3  # noqa: E402
from serve import server as server_mod  # noqa: E402
from serve.runner_fake import FakeRunner  # noqa: E402
from serve.test_runner_routing import DONE, Fixture, coding  # noqa: E402

FAKE_MCP = str(ROOT / "serve" / "mcp_fake_server.py")


def wait_for(cond, timeout=15.0, step=0.05):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        got = cond()
        if got:
            return got
        time.sleep(step)
    return cond()


class Base(Fixture):
    def setUp(self):
        super().setUp()
        self.loops: list = []
        self.said: list[str] = []
        p = mock.patch.object(runner_mod, "_say", self.said.append)
        p.start()
        self.addCleanup(p.stop)
        self.proj = self.dir / "proj"
        self.proj.mkdir()

    def runner(self, key, device=DEV1, cfg=None, machine="ann-pc", name="r1"):
        """A real runner (Executor + Client + RunnerLoop) for `key` on a thread; its files in a temp folder."""
        d = self.dir / name
        d.mkdir()
        (d / "runner.json").write_text(json.dumps(cfg or {}), encoding="utf-8")
        home = d / "home"
        home.mkdir()
        ex = runner_mod.Executor(home, d / "runner.json", d)
        client = runner_mod.Client(f"http://127.0.0.1:{self.port}", key, {"id": device, "name": machine})
        stop = threading.Event()
        loop = runner_mod.RunnerLoop(client, ex, stop=stop)
        loop.dir = d
        codes = []
        t = threading.Thread(target=lambda: codes.append(loop.run()), daemon=True, name="test-runner")
        t.start()
        loop.codes, loop.thread = codes, t

        def end():
            stop.set()
            t.join(10)
            ex.close()
        self.addCleanup(end)
        self.assertTrue(wait_for(lambda: loop.id is not None), self.said)
        return loop

    def stream(self, key, sa, device=DEV1, answer=None, mcp=False):
        """POST a streamed chat on a thread -> (thread, events); `answer(event)` is called for each permission event;
        `mcp`: the MCP tools are switched on (strata_mcp)."""
        events: list = []
        body = {"model": "m", "stream": True, "max_tokens": 1000, "strata_agent": sa,
                "messages": [{"role": "user", "content": "do it"}]}
        if mcp:
            body["strata_mcp"] = True

        def read():
            c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
            try:
                c.request("POST", "/v1/chat/completions", body=json.dumps(body).encode(), headers={
                    "Content-Type": "application/json", "Authorization": "Bearer " + key, "X-Strata-Device": device})
                r = c.getresponse()
                events.append(r.status)
                for raw in r:
                    ln = raw.decode().strip()
                    if ln.startswith("data: {"):
                        ev = json.loads(ln[6:]).get("strata_mcp")
                        if ev:
                            events.append(ev)
                            if ev.get("event") == "permission" and answer is not None:
                                threading.Thread(target=answer, args=(ev,), daemon=True).start()
            finally:
                c.close()
        t = threading.Thread(target=read, daemon=True)
        t.start()
        return t, events

    def run_chat(self, key, sa, **kw):
        t, events = self.stream(key, sa, **kw)
        t.join(60)
        self.assertFalse(t.is_alive(), events)
        self.assertEqual(events[0], 200, events)
        return events[1:]


class EndToEnd(Base):
    def test_end_to_end_write_with_approval(self):
        outside = self.dir / "outside"
        outside.mkdir()
        target = outside / "new.txt"
        self.start_routing(tool_call("Write", file_path=str(target), content="hello from my pc"), DONE)
        loop = self.runner(self.ann)
        answered = []

        def allow(ev):
            answered.append(self.req("POST", "/agent/permission", {"id": ev["id"], "decision": "allow"},
                                     key=self.ann, device=DEV1))
        events = self.run_chat(self.ann, {"cwd": str(self.proj), "session": "chat-1", "runner": loop.id}, answer=allow)
        self.assertEqual(answered, [(200, {"ok": True})], events)
        self.assertEqual(target.read_text(encoding="utf-8").strip(), "hello from my pc")
        call = next(e for e in events if e["event"] == "call")
        self.assertEqual((call["name"], call["server"], call["machine"]), ("Write", "my_agent", "my"))
        perm = next(e for e in events if e["event"] == "permission")
        self.assertIn("outside the project folder", json.dumps(perm))
        result = next(e for e in events if e["event"] == "result")
        self.assertIn("created successfully", result["text"])
        self.assertEqual(self.server_calls, [])                 # the server's own coding tools never ran

    def test_runner_page_routes(self):
        self.start_routing()
        (self.proj / "hello.txt").write_text("hi\n", encoding="utf-8")
        loop = self.runner(self.ann)
        code, body = self.req("GET", f"/agent/folders?runner={loop.id}&path={self.dir}", key=self.ann, device=DEV1)
        self.assertEqual(code, 200, body)
        self.assertIn("proj", [d["name"] for d in body["dirs"]])
        code, body = self.req("GET", f"/agent/files?runner={loop.id}&path={self.proj}&q=hel", key=self.ann, device=DEV1)
        self.assertEqual(code, 200, body)
        self.assertIn("hello.txt", json.dumps(body["files"]))
        r = self.svc.runner_hub.get_for(loop.id, SimpleNamespace(name="ann"))
        self.assertTrue(wait_for(lambda: not r._out, 5))       # the poll's ack let the page jobs go

    def test_page_job_uses_the_runners_home_checkpoints_and_memory_on(self):
        self.start_routing()
        loop = self.runner(self.ann)
        seen = []
        real = runner_mod.page_routes.answer

        def answer(path, query, body, **kw):
            seen.append(kw)
            return real(path, query, body, **kw)
        with mock.patch.object(runner_mod.page_routes, "answer", answer):
            self.cfg_path.write_text(json.dumps({"import": {"memory": {"on": ["claude:memory"]}}}), encoding="utf-8")
            code, body = self.req("GET", f"/agent/memory?runner={loop.id}&path={self.proj}", key=self.ann, device=DEV1)
        self.assertEqual(code, 200, body)
        self.assertEqual(seen[0]["home"], str(loop.dir / "home"))
        self.assertEqual(seen[0]["memory_on"], ["claude:memory"])
        self.assertIs(seen[0]["checkpoints"], loop.executor.checkpoints)
        self.assertEqual(Path(loop.executor.checkpoints.root), loop.dir / "checkpoints")

    def test_runner_mcp_group(self):
        self.start_routing(tool_call("my_fake__echo", text="hi from the PC"), DONE)
        loop = self.runner(self.ann, cfg={"mcp_servers": {"fake": {"command": sys.executable, "args": [FAKE_MCP]}}})

        def offered():
            code, body = self.req("GET", f"/mcp?runner={loop.id}", key=self.ann, device=DEV1)
            return code == 200 and any(t["name"] == "my_fake__echo" for s in body["servers"] for t in s["tools"])
        self.assertTrue(wait_for(offered, 20), self.said)        # the runner said hello again once the server was ready
        events = self.run_chat(self.ann, {"cwd": str(self.proj), "session": "chat-1", "runner": loop.id}, mcp=True)
        call = next(e for e in events if e["event"] == "call")
        self.assertEqual((call["name"], call["server"], call.get("machine")), ("my_fake__echo", "my_fake", "my"))
        result = next(e for e in events if e["event"] == "result")
        self.assertIn("hi from the PC", result["text"])

    def test_rewind_on_runner(self):
        a = self.proj / "a.txt"
        a.write_bytes(b"old\n")
        self.start_routing(tool_call("Read", file_path=str(a)), tool_call("Write", file_path=str(a), content="new"), DONE)
        loop = self.runner(self.ann)
        self.run_chat(self.ann, {"cwd": str(self.proj), "session": "chat-1", "checkpoint": "cp1", "runner": loop.id})
        self.assertEqual(a.read_text(encoding="utf-8").strip(), "new")
        self.assertTrue((loop.dir / "checkpoints" / "chat-1").is_dir())     # kept on the runner's PC
        code, body = self.req("POST", "/agent/rewind", {"session": "chat-1", "checkpoint": "cp1", "apply": True,
                                                        "runner": loop.id}, key=self.ann, device=DEV1)
        self.assertEqual(code, 200, body)
        self.assertEqual(a.read_bytes(), b"old\n")


class Connection(Base):
    def test_rehello_after_404(self):
        self.start_routing()
        loop = self.runner(self.ann)
        first, channel = loop.id, loop.client.channel
        self.svc.runner_hub.close()                              # the server stops (held polls end 503) ...
        self.httpd.shutdown()
        self.httpd.server_close()
        self.svc.runner_hub = runner_hub.RunnerHub()             # ... and starts again on the same port: no runners
        self.httpd = serve(self.svc, port=self.port)
        self.assertTrue(wait_for(lambda: self.svc.runner_hub.get_for(first, SimpleNamespace(name="ann")), 20), self.said)
        self.assertEqual(loop.id, first)                         # the same id, so the page's stored id keeps working
        self.assertNotEqual(loop.client.channel, channel)

    def test_rename_says_hello_again(self):
        self.start_routing()
        loop = self.runner(self.ann)
        code, body = self.req("POST", "/keys", {"action": "rename", "name": "ann", "to": "anna"}, key="k", device=DEV0)
        self.assertEqual(code, 200, body)
        new = runner_hub.runner_id("anna", DEV1)
        self.assertTrue(wait_for(lambda: loop.id == new, 10), self.said)
        self.assertTrue(any(new in line for line in self.said), self.said)          # the new id is printed

    def test_reply_after_a_rename_reaches_the_job(self):
        """A job running when the key is renamed: the poll held under the old id ends at the rename (#315: not after
        its whole wait), the runner says hello again, and the job's result reaches the page."""
        self.start_routing()
        loop = self.runner(self.ann)
        old, go, inside = loop.id, threading.Event(), threading.Event()
        new = runner_hub.runner_id("anna", DEV1)
        r = self.svc.runner_hub.get_for(old, SimpleNamespace(name="ann"))
        real = runner_mod.page_routes.answer

        def answer(*a, **kw):
            inside.set()
            go.wait(20)
            return real(*a, **kw)
        got = []
        with mock.patch.object(runner_mod.page_routes, "answer", answer):
            t = threading.Thread(target=lambda: got.append(self.req(
                "GET", f"/agent/folders?runner={old}&path={self.dir}", key=self.ann, device=DEV1)))
            t.start()
            self.assertTrue(inside.wait(10))
            self.assertTrue(wait_for(lambda: r._held, 5, 0.005))  # the runner's next poll is held on the server
            t0 = time.monotonic()
            code, body = self.req("POST", "/keys", {"action": "rename", "name": "ann", "to": "anna"}, key="k",
                                  device=DEV0)
            self.assertEqual(code, 200, body)
            self.assertTrue(wait_for(lambda: loop.id == new, 5, 0.005), self.said)
            self.assertLess(time.monotonic() - t0, 1.0)         # the held poll's wait (POLL_MAX_S, 2 s here) is not
            go.set()                                            # waited out
            t.join(20)
        self.assertEqual(got[0][0], 200, got)
        self.assertIn("proj", [d["name"] for d in got[0][1]["dirs"]])
        self.assertEqual(loop.codes, [])                        # the runner goes on

    def test_a_reply_met_by_404_goes_again_once_the_hello_is_answered(self):
        """A reply sent under the old id meets 404; it goes again under the new id as soon as the poll thread's hello
        is answered (#315: it looked every 0.1 s, so up to 100 ms late; here the look is 10 s, so only the hello can wake it)."""
        calls = []

        class Fake:
            channel, url = "c1", "http://server"

            def reply(self, rid, instance, msg):
                calls.append((rid, time.monotonic()))
                if rid == "r-old":
                    raise runner_mod.Gone("there is no such runner")

            def hello(self, info):
                return {"runner": "r-new", "who": "anna", "channel": "c1", "poll_s": 2}

            def on_hello(self, body):
                return False
        ex = SimpleNamespace(groups=lambda: [], skills=lambda: [], info=lambda: {}, starting=lambda: False)
        loop = runner_mod.RunnerLoop(Fake(), ex)
        loop.id, loop.poll_s = "r-old", 2.0
        loop.connected.set()
        answered = []

        def hello_later():
            wait_for(lambda: calls, 2, 0.001)
            time.sleep(0.02)
            loop._hello()
            answered.append(time.monotonic())
        threading.Thread(target=hello_later, daemon=True).start()
        with mock.patch.object(runner_mod, "STOP_LOOK_S", 10.0):   # only the hello can wake it within the bound
            self.assertIs(loop._post({"job": "j1", "rid": 1, "result": {}}), True)
        self.assertEqual([c[0] for c in calls], ["r-old", "r-new"])
        wait_for(lambda: answered, 2, 0.001)
        self.assertLess(calls[1][1] - answered[0], 1.0)

    # ---------------------------------------------------------------------------------------- #313: kept connections
    def hello_info(self, instance="inst-keep-0001"):
        return {"instance": instance, "machine": "ann-pc", "platform": "win32", "shell": None, "home": "C:\\Users\\ann",
                "groups": [], "skills": []}

    def count_connects(self):
        """TCP connects made by the runner's threads (its poll thread and its job threads), from now on."""
        made, real = [], http.client.HTTPConnection.connect

        def connect(conn):
            if threading.current_thread().name in ("test-runner", "runner-job", "keep-test"):
                made.append(threading.current_thread().name)
            return real(conn)
        p = mock.patch.object(http.client.HTTPConnection, "connect", connect)
        p.start()
        self.addCleanup(p.stop)
        return made

    def test_polls_and_replies_keep_their_connections(self):
        """Five page jobs: their polls and replies go over the connections the runner already has - no new TCP
        connection per message (it was two per job: the reply's and the next poll's)."""
        self.start_routing()
        (self.proj / "hello.txt").write_text("hi\n", encoding="utf-8")
        loop = self.runner(self.ann)
        code, body = self.req("GET", f"/agent/folders?runner={loop.id}&path={self.dir}", key=self.ann, device=DEV1)
        self.assertEqual(code, 200, body)                       # the first reply opened the reply connection
        made = self.count_connects()
        for _ in range(5):
            code, body = self.req("GET", f"/agent/folders?runner={loop.id}&path={self.dir}", key=self.ann, device=DEV1)
            self.assertEqual(code, 200, body)
        self.assertEqual(made, [])

    def test_a_kept_connection_the_server_closed_is_tried_again_once(self):
        self.start_routing()
        made = self.count_connects()
        c = runner_mod.Client(f"http://127.0.0.1:{self.port}", self.ann, {"id": DEV1, "name": "ann-pc"})
        self.addCleanup(c.close)
        got = []

        def go():
            got.append(c.hello(self.hello_info())["runner"])
            time.sleep(0.6)                                     # the server closes the idle connection meanwhile
            got.append(c.hello(self.hello_info())["runner"])
        with mock.patch.object(server_mod, "RUNNER_IDLE_S", 0.2):
            t = threading.Thread(target=go, name="keep-test")
            t.start()
            t.join(10)
        self.assertEqual(len(got), 2, self.said)
        self.assertEqual(got[0], got[1])
        self.assertEqual(len(made), 2)                          # the first, then one again after the server's close

    def test_a_server_that_closes_every_connection_still_works(self):
        """A server from before #313 answers every route HTTP/1.0 with Connection: close: the runner keeps nothing."""
        self.start_routing()
        made = self.count_connects()
        c = runner_mod.Client(f"http://127.0.0.1:{self.port}", self.ann, {"id": DEV1, "name": "ann-pc"})
        self.addCleanup(c.close)
        got = []

        def go():
            body = c.hello(self.hello_info())
            c.on_hello(body)
            got.append(c.next(body["runner"], "inst-keep-0001", 0))
            got.append(c.hello(self.hello_info())["runner"])
        with mock.patch.object(server_mod, "RUNNER_KEEP", set()):
            t = threading.Thread(target=go, name="keep-test")
            t.start()
            t.join(10)
        self.assertEqual(len(got), 2, self.said)
        self.assertEqual(len(made), 3)                          # one connection per request, as before
        self.assertEqual((c._kept, c._poll_kept), ([], None))

    def test_abort_poll_cuts_a_held_poll_on_a_kept_connection(self):
        self.start_routing()
        c = runner_mod.Client(f"http://127.0.0.1:{self.port}", self.ann, {"id": DEV1, "name": "ann-pc"})
        self.addCleanup(c.close)
        body = c.hello(self.hello_info())
        c.on_hello(body)
        rid = body["runner"]
        self.assertEqual(c.next(rid, "inst-keep-0001", 0), [])     # the poll's connection is kept from here
        r = self.svc.runner_hub.get_for(rid, SimpleNamespace(name="ann"))
        got = []
        with mock.patch.object(runner_hub, "POLL_MAX_S", 20.0):
            t = threading.Thread(target=lambda: got.append((c.next(rid, "inst-keep-0001", 20), time.monotonic())))
            t.start()
            self.assertTrue(wait_for(lambda: r._held, 5, 0.005))
            t0 = time.monotonic()
            c.abort_poll()
            t.join(10)
        self.assertEqual(got[0][0], [])
        self.assertLess(got[0][1] - t0, 1.0)
        self.assertEqual(c.next(rid, "inst-keep-0001", 5), [])     # and every later poll at once

    def test_replies_number_rid_and_answers_match_to(self):
        target = self.dir / "outside.txt"
        self.start_routing(tool_call("Write", file_path=str(target), content="x"), DONE)
        loop = self.runner(self.ann)
        sent, answers = [], []
        real_reply, real_dispatch = loop.client.reply, loop._dispatch

        def reply(rid, instance, msg):
            sent.append(msg)
            return real_reply(rid, instance, msg)

        def dispatch(msgs):
            answers.extend(m for m in msgs if "answer" in m)
            return real_dispatch(msgs)
        loop.client.reply, loop._dispatch = reply, dispatch

        def allow(ev):
            self.req("POST", "/agent/permission", {"id": ev["id"], "decision": "allow"}, key=self.ann, device=DEV1)
        self.run_chat(self.ann, {"cwd": str(self.proj), "session": "chat-1", "runner": loop.id}, answer=allow)
        self.assertTrue(target.exists())
        tool = [m for m in sent if "ask" in m or ("result" in m and "content" in m["result"])]
        self.assertEqual([(m["rid"], "ask" in m) for m in tool], [(1, True), (2, False)])   # numbered per job from 1
        self.assertEqual(len({m["job"] for m in tool}), 1)
        self.assertEqual([(a["to"], a["answer"]) for a in answers][:1], [(1, "allow")])     # the answer names its ask

    def test_duplicates_ignored_and_channel_change_resets(self):
        ex = SimpleNamespace(handled=[])
        loop = runner_mod.RunnerLoop(runner_mod.Client("http://127.0.0.1:1", "k", {"id": DEV1, "name": "pc"}), ex)
        started = []
        loop._start = lambda msg: started.append(msg["seq"])
        loop.client.on_hello({"runner": "r1", "who": "ann", "channel": "aaaa"})
        loop._dispatch([{"job": "j1", "kind": "page", "seq": 1}, {"job": "j2", "kind": "page", "seq": 2}])
        loop._dispatch([{"job": "j1", "kind": "page", "seq": 1}, {"job": "j2", "kind": "page", "seq": 2},
                        {"job": "j3", "kind": "page", "seq": 3}])       # at least once: the repeats are ignored
        self.assertEqual(started, [1, 2, 3])
        self.assertEqual(loop.client.ack, 3)
        link = runner_mod.JobLink("j9", lambda m: None)
        loop.links["j9"] = link
        self.assertFalse(loop.client.on_hello({"runner": "r1", "who": "ann", "channel": "aaaa"}))   # the same channel
        self.assertEqual(loop.client.ack, 3)                                                     # keeps its ack
        self.assertTrue(loop.client.on_hello({"runner": "r1", "who": "ann", "channel": "bbbb"}))
        loop._channel_changed()
        self.assertEqual((loop.client.ack, loop.client.handled), (0, set()))
        self.assertTrue(link.cancel.is_set())                   # the old registration's jobs ended on the server
        loop._dispatch([{"job": "j4", "kind": "page", "seq": 1}])
        self.assertEqual(started, [1, 2, 3, 1])

    def test_answer_reaches_the_ask_by_to(self):
        posted = []
        link = runner_mod.JobLink("j1", posted.append)
        got = []
        t = threading.Thread(target=lambda: got.append(link.ask({"tool": "Write"})))
        t.start()
        self.assertTrue(wait_for(lambda: posted, 2))
        self.assertEqual(posted[0], {"job": "j1", "rid": 1, "ask": {"tool": "Write"}})
        link.deliver({"job": "j1", "answer": "deny", "to": 7})  # another reply's answer: not this ask's
        link.deliver({"job": "j1", "answer": "allow", "to": 1})
        t.join(2)
        self.assertEqual(got, ["allow"])
        t = threading.Thread(target=lambda: got.append(link.question([{"question": "q?"}])))
        t.start()
        self.assertTrue(wait_for(lambda: len(posted) == 2, 2))
        link.cancel.set()
        t.join(2)
        self.assertEqual(got, ["allow", "cancelled"])


class ReviewFixes(Base):
    """Task 6 review: a key never leaks, a poll-thread bug is no exit, the plan's approval reaches the server's run, and a
    result the server refuses still ends the call."""

    def test_unexpected_error_in_a_poll_backs_off_without_the_key(self):
        self.start_routing()
        loop = self.runner(self.ann)
        real, calls = loop.client.next, []

        def boom(*a):
            calls.append(time.monotonic())
            if len(calls) <= 2:
                raise RuntimeError(f"broken while sending Bearer {self.ann}")
            return real(*a)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            loop.client.next = boom
            self.assertTrue(wait_for(lambda: len(calls) >= 4, 15), self.said)
        self.assertTrue(loop.thread.is_alive())                  # the runner goes on
        self.assertEqual(loop.codes, [])
        self.assertGreaterEqual(calls[1] - calls[0], 0.9)        # backed off: 1 s, then 2 s
        self.assertTrue(any("unexpected RuntimeError" in s for s in self.said), self.said)
        self.assertFalse(any(self.ann in s for s in self.said), self.said)
        self.assertNotIn(self.ann, err.getvalue())
        code, body = self.req("GET", f"/agent/folders?runner={loop.id}&path={self.dir}", key=self.ann, device=DEV1)
        self.assertEqual(code, 200, body)

    def test_a_header_error_never_shows_the_key(self):
        client = runner_mod.Client("http://127.0.0.1:9", "secret-key\n", {"id": DEV1, "name": "pc"})
        with self.assertRaises(runner_mod.NetworkError) as got:
            client.hello({})
        self.assertNotIn("secret-key", str(got.exception))

    def jobs_of(self, loop):
        """The tool jobs the runner receives, as they come."""
        seen = []
        real = loop._dispatch

        def dispatch(msgs):
            seen.extend(m for m in msgs if m.get("kind") == "tool")
            return real(msgs)
        loop._dispatch = dispatch
        return seen

    def plan_then_read(self, decision):
        a = self.proj / "a.txt"
        a.write_text("x\n", encoding="utf-8")
        self.start_routing(tool_call("ExitPlanMode", plan="read a.txt"), tool_call("Read", file_path=str(a)), DONE)
        loop = self.runner(self.ann)
        seen = self.jobs_of(loop)

        def answer(ev):
            self.req("POST", "/agent/permission", {"id": ev["id"], "decision": decision}, key=self.ann, device=DEV1)
        events = self.run_chat(self.ann, {"cwd": str(self.proj), "session": "chat-1", "mode": "plan", "runner": loop.id},
                               answer=answer)
        return [(m["tool"], m["mode"]) for m in seen], events

    def test_approved_plan_reaches_the_servers_run(self):
        jobs, events = self.plan_then_read("allow")
        self.assertEqual(jobs, [("ExitPlanMode", "plan"), ("Read", None)], events)     # the next job leaves plan mode
        self.assertTrue(any(e.get("event") == "mode" and e.get("mode") == "ask" for e in events), events)

    def test_denied_plan_keeps_plan_mode(self):
        jobs, events = self.plan_then_read("deny")
        self.assertEqual(jobs, [("ExitPlanMode", "plan"), ("Read", "plan")], events)

    def test_a_mode_emit_without_an_approval_changes_nothing(self):
        self.start_routing(tool_call("ExitPlanMode", plan="p"), tool_call("Read", file_path="a.txt"), DONE)
        seen = []

        def on_job(msg, reply):
            if msg.get("kind") == "run.start":
                reply({"result": {"run": "run-x", "cwd": msg.get("cwd"), "dirs": [], "shell": None, "platform": "win32",
                                  "git": False, "memory": [], "tools": ["Read", "ExitPlanMode"]}})
            elif msg.get("kind") == "tool":
                seen.append((msg["tool"], msg["mode"]))
                if msg["tool"] == "ExitPlanMode":               # a runner that skips the card
                    reply({"emit": {"event": "mode", "mode": "ask"}})
                reply({"result": {"content": [{"type": "text", "text": "ok"}]}})
        f = FakeRunner(self.port, self.ann, DEV1, [coding(["Read", "ExitPlanMode"])], on_job, machine="ann-pc")
        self.fakes.append(f)
        self.run_chat(self.ann, {"cwd": str(self.proj), "session": "chat-1", "mode": "plan", "runner": f.id})
        self.assertEqual(seen, [("ExitPlanMode", "plan"), ("Read", "plan")])

    def refused_result(self, patch):
        big = self.proj / "big.txt"
        big.write_text(("y" * 100 + "\n") * 60, encoding="utf-8")       # Read cuts a single line at 2000
        self.start_routing(tool_call("Read", file_path=str(big)), DONE)
        loop = self.runner(self.ann)
        with patch:
            t0 = time.monotonic()
            events = self.run_chat(self.ann, {"cwd": str(self.proj), "session": "chat-1", "runner": loop.id})
        result = next(e for e in events if e["event"] == "result")
        self.assertIn("too large to send", result["text"])     # the call ended with text, not a silent wait
        self.assertLess(time.monotonic() - t0, 20)

    def test_a_result_the_server_refuses_ends_the_call(self):
        self.refused_result(mock.patch.object(server_mod, "RUNNER_REPLY_MAX", 3000))     # the server's 413

    def test_a_result_over_the_limit_is_not_sent(self):
        self.refused_result(mock.patch.object(runner_mod, "REPLY_MAX", 3000))            # checked before sending


class Hello(unittest.TestCase):
    def test_hello_meets_server_limits(self):
        groups = [{"name": "agent", "kind": "coding", "status": "ready", "error": None,
                   "tools": [{"name": "Read", "description": "d" * 200_000, "inputSchema": "not an object"}]},
                  {"name": "x" * 65, "kind": "mcp", "status": "ready", "tools": []},
                  {"name": "fine", "kind": "mcp", "status": "s" * 50, "error": "e" * 5000,
                   "tools": [{"name": "t" * 200}, *({"name": f"t{i}"} for i in range(1100))]}]
        groups += [{"name": f"g{i}", "kind": "mcp", "status": "ready", "tools": []} for i in range(80)]
        skills = [{"name": "s" * 300, "description": ""}, {"name": "ok", "description": "d" * 20_000, "harness": "a" * 90},
                  *({"name": f"k{i}", "description": ""} for i in range(2100))]
        fit_g, fit_s = runner_mod.fit_groups(groups), runner_mod.fit_skills(skills)
        info = runner_mod.fit_info({"machine": "m" * 100, "platform": "win32", "shell": "bash", "home": "h" * 2000})
        body = {"protocol": 1, "instance": runner_mod.new_instance(), **info, "groups": fit_g, "skills": fit_s}
        got = runner_hub.check_hello(body)
        self.assertIsInstance(got, tuple, got)
        self.assertNotIn("x" * 65, [g["name"] for g in fit_g])
        self.assertEqual(len(fit_g), runner_hub.MAX_GROUPS)
        self.assertEqual(len(next(g for g in fit_g if g["name"] == "fine")["tools"]), runner_hub.MAX_TOOLS)
        self.assertEqual(len(fit_s), runner_hub.MAX_SKILLS)

    def test_executor_hello_is_accepted(self):
        with mock.patch.dict(os.environ, {}), tempfile_dir() as d:
            (d / "home").mkdir()
            ex = runner_mod.Executor(d / "home", d / "runner.json", d)
            try:
                body = {"protocol": 1, "instance": runner_mod.new_instance(), **ex.info(), "groups": ex.groups(),
                        "skills": ex.skills()}
                self.assertIsInstance(runner_hub.check_hello(body), tuple)
                coding = next(g for g in body["groups"] if g["kind"] == "coding")
                self.assertEqual(coding["name"], "agent")
                names = [t["name"] for t in coding["tools"]]
                self.assertIn("Read", names)
                self.assertFalse(set(names) & set(runner_hub.SERVER_TOOLS))     # no web, no Task on the runner
                self.assertTrue(all(runner_hub.plain_name_ok(n) for n in names))
            finally:
                ex.close()


class Device(unittest.TestCase):
    def test_load_device_is_made_once(self):
        with tempfile_dir() as d:
            dev = runner_mod.load_device(d)
            self.assertRegex(dev["id"], r"^[A-Za-z0-9_-]{22}$")
            self.assertTrue(dev["name"])
            self.assertEqual(runner_mod.load_device(d), dev)
            (d / "device.json").write_text("{broken", encoding="utf-8")
            again = runner_mod.load_device(d)
            self.assertRegex(again["id"], r"^[A-Za-z0-9_-]{22}$")
            self.assertEqual(runner_mod.load_device(d), again)

    def test_runner_dir(self):
        local = r"C:\Users\x\AppData\Local"
        with mock.patch.dict(os.environ, {"LOCALAPPDATA": local}):
            want = Path(local) / "StrataRunner" if os.name == "nt" else Path.home() / ".strata-runner"
            self.assertEqual(runner_mod.runner_dir(), want)


class SessionState(unittest.TestCase):
    """Todos, file reads and background shells are the chat's (`session`), not the request's run key."""

    def setUp(self):
        self._d = tempfile_dir()
        d = self._d.__enter__()
        self.addCleanup(self._d.__exit__, None, None, None)
        self.proj = d / "proj"
        self.proj.mkdir()
        (d / "home").mkdir()
        self.ex = runner_mod.Executor(d / "home", d / "runner.json", d)
        self.addCleanup(self.ex.close)

    def start(self, session):
        link = runner_mod.JobLink("j", lambda m: None)
        res = self.ex.handle({"kind": "run.start", "cwd": str(self.proj), "dirs": [], "mode": None, "allow": [],
                              "deny": [], "session": session, "checkpoint": None, "memory_on": []}, link)
        self.assertTrue(runner_hub.RUN_KEY.fullmatch(res["run"]), res)
        self.assertEqual(runner_hub.run_facts(res, "pc", 20, 60_000)["cwd"], os.path.realpath(self.proj))
        return res["run"]

    def tool(self, run, session, tool, **args):
        link = runner_mod.JobLink("j", lambda m: None)
        res = self.ex.handle({"kind": "tool", "group": "agent", "tool": tool, "arguments": args, "run": run,
                              "session": session, "mode": None, "helper": False, "vision": False, "timeout_s": 60}, link)
        return res["content"][0]["text"], res.get("isError")

    def test_background_shell_survives_into_the_next_message(self):
        if self.ex.agent.shell is None:
            self.skipTest("no shell on this PC")
        first = self.start("chat-1")
        text, err = self.tool(first, "chat-1", "Bash", command="echo from-the-first-message", run_in_background=True)
        self.assertFalse(err, text)
        bid = text.rsplit(" ", 1)[-1]
        second = self.start("chat-1")                            # the next message: another run key
        self.assertNotEqual(first, second)
        time.sleep(1.5)
        text, err = self.tool(second, "chat-1", "BashOutput", bash_id=bid)
        self.assertFalse(err, text)
        self.assertIn("from-the-first-message", text)
        other = self.start("chat-2")
        text, err = self.tool(other, "chat-2", "BashOutput", bash_id=bid)
        self.assertTrue(err, text)                               # another chat does not see it

    def test_reads_and_todos_are_the_chats(self):
        f = self.proj / "a.txt"
        f.write_text("x\n", encoding="utf-8")
        first = self.start("chat-1")
        self.assertFalse(self.tool(first, "chat-1", "Read", file_path=str(f))[1])
        second = self.start("chat-1")
        text, err = self.tool(second, "chat-1", "Edit", file_path=str(f), old_string="x", new_string="y")
        self.assertFalse(err, text)                              # the read of the first message counts
        self.tool(second, "chat-1", "TodoWrite", todos=[{"content": "a", "status": "pending", "activeForm": "a-ing"}])
        self.assertEqual(len(self.ex.agent.todos("chat-1")), 1)

    def test_unknown_run_is_text(self):
        text, err = self.tool("run-nope", "chat-1", "Read", file_path="a.txt")
        self.assertTrue(err)
        self.assertIn("send the message again", text)

    def test_helper_gets_its_mode_and_no_questions(self):
        run = self.start("chat-1")
        f = self.proj / "b.txt"
        link = runner_mod.JobLink("j", lambda m: None)
        res = self.ex.handle({"kind": "tool", "group": "agent", "tool": "Write", "arguments": {"file_path": str(f),
                              "content": "z"}, "run": run, "session": "chat-1", "mode": "plan", "helper": True,
                              "vision": False, "timeout_s": 60}, link)
        self.assertTrue(res["isError"], res)                     # plan mode: nothing is changed
        self.assertFalse(f.exists())
        self.assertIsNone(self.ex.runs[run][0].mode)             # the run's own policy is untouched


class NoWindows(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt", "CREATE_NO_WINDOW is Windows' flag")
    def test_children_start_without_a_window(self):
        flags = []
        real = subprocess.Popen

        class Recording(real):
            def __init__(self, args, *a, **kw):
                flags.append((args if isinstance(args, str) else " ".join(map(str, args))[:80], kw.get("creationflags", 0)))
                super().__init__(args, *a, **kw)
        with tempfile_dir() as d, mock.patch.object(subprocess, "Popen", Recording):
            (d / "home").mkdir()
            (d / "proj").mkdir()
            (d / "runner.json").write_text(json.dumps({"mcp_servers": {"fake": {"command": sys.executable,
                                                                                 "args": [FAKE_MCP]}}}), encoding="utf-8")
            ex = runner_mod.Executor(d / "home", d / "runner.json", d)
            try:
                self.assertTrue(wait_for(lambda: ex.hub.servers["fake"].status == "ready", 15))
                if ex.agent.shell is not None:
                    link = runner_mod.JobLink("j", lambda m: link.deliver({"to": m["rid"], "answer": "allow"}))  # the user allows
                    run = ex.handle({"kind": "run.start", "cwd": str(d / "proj"), "session": "s", "memory_on": []}, link)["run"]
                    call = lambda tool, **a: ex.handle({"kind": "tool", "group": "agent", "tool": tool,     # noqa: E731
                                                        "arguments": a, "run": run, "session": "s"}, link)
                    call("Bash", command="echo hi")
                    bid = call("Bash", command="sleep 30", run_in_background=True)["content"][0]["text"].rsplit(" ", 1)[-1]
                    call("KillShell", shell_id=bid)
            finally:
                ex.close()
        self.assertGreaterEqual(len(flags), 2, flags)
        for args, f in flags:
            self.assertTrue(f & subprocess.CREATE_NO_WINDOW, (args, f))


class Main(Base):
    def main(self, *argv, env=None, device=DEV1):
        """runner.main on a thread, its files in a temp folder -> (thread, codes, stop)."""
        d = self.dir / "main"
        d.mkdir(exist_ok=True)
        (d / "home").mkdir(exist_ok=True)
        codes, stop = [], threading.Event()
        environ = {"STRATA_HOME": str(d / "home"), **(env or {})}
        patches = [mock.patch.object(runner_mod, "runner_dir", lambda: d), mock.patch.dict(os.environ, environ)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        args = ["--server", f"http://127.0.0.1:{self.port}", "--device-id", device, *argv]
        t = threading.Thread(target=lambda: codes.append(runner_mod.main(args, stop=stop)), daemon=True)
        t.start()
        self.addCleanup(lambda: (stop.set(), t.join(10)))
        return t, codes, stop

    def test_runner_exits_on_401(self):
        self.start_routing()
        t, codes, _ = self.main(env={"STRATA_RUNNER_KEY": self.ann})
        rid = runner_hub.runner_id("ann", DEV1)
        self.assertTrue(wait_for(lambda: self.svc.runner_hub.get_for(rid, SimpleNamespace(name="ann"))), self.said)
        self.assertNotIn("STRATA_RUNNER_KEY", os.environ)        # MCP servers and commands never inherit the key
        connected = (f"[runner] connected to http://127.0.0.1:{self.port} as ann; this PC's runner id is {rid} - "
                     f"paste it in the web app under Settings > This browser's runner")
        self.assertIn(connected, self.said)
        t0 = time.monotonic()
        code, body = self.req("POST", "/keys", {"action": "revoke", "name": "ann"}, key="k", device=DEV0)
        self.assertEqual(code, 200, body)
        t.join(2 * runner_hub.POLL_MAX_S + 1)                    # within 2 polls
        self.assertEqual(codes, [3], self.said)
        self.assertLess(time.monotonic() - t0, 2 * runner_hub.POLL_MAX_S + 1)
        self.assertTrue(any("revoked" in s or "key" in s for s in self.said[-2:]), self.said)
        self.assertFalse(any(self.ann in s for s in self.said), self.said)          # the key is never printed

    def test_runner_waits_for_device_approval(self):
        self.start_routing()
        cat = self.svc.keys.create("cat", "guest")
        hellos = []
        real = runner_mod.Client.hello

        def hello(client, info):
            hellos.append(time.monotonic())
            return real(client, info)
        with mock.patch.object(runner_mod, "PENDING_RETRY_S", 0.3), mock.patch.object(runner_mod.Client, "hello", hello):
            t, codes, stop = self.main("--key", cat, device=DEV3)
            code = wait_for(lambda: self.svc.keys.device_code("cat", DEV3))
            self.assertTrue(code)
            self.assertTrue(wait_for(lambda: any(code in s for s in self.said)), self.said)
            time.sleep(1.2)
            self.assertEqual(sum(code in s for s in self.said), 1, self.said)      # the code is printed once
            gaps = [b - a for a, b in zip(hellos, hellos[1:])]
            self.assertTrue(gaps and min(gaps) >= 0.25, gaps)                        # it waits between tries: no spin
            self.assertTrue(self.svc.keys.decide_device("cat", DEV3, True, OWNER))
            rid = runner_hub.runner_id("cat", DEV3)
            self.assertTrue(wait_for(lambda: self.svc.runner_hub.get_for(rid, SimpleNamespace(name="cat"))), self.said)
            stop.set()
            t.join(10)
        self.assertEqual(codes, [0])
        self.assertFalse(any(cat in s for s in self.said))

    def test_pending_runner_says_hello_less_often_after_a_while(self):
        """PENDING_RETRY_S for the first PENDING_FAST_S of waiting, then PENDING_SLOW_S (#310 part 3 task 6: a short
        first interval, so the runner is online soon after the owner approves, without a hello every 2 s for ever)."""
        self.start_routing()
        cat = self.svc.keys.create("cat", "guest")
        hellos = []
        real = runner_mod.Client.hello

        def hello(client, info):
            hellos.append(time.monotonic())
            return real(client, info)
        with mock.patch.object(runner_mod, "PENDING_RETRY_S", 0.1), \
                mock.patch.object(runner_mod, "PENDING_FAST_S", 0.5), \
                mock.patch.object(runner_mod, "PENDING_SLOW_S", 0.6), \
                mock.patch.object(runner_mod.Client, "hello", hello):
            t, codes, stop = self.main("--key", cat, device=DEV3)
            self.assertTrue(wait_for(lambda: len(hellos) >= 9, timeout=10), hellos)
            stop.set()
            t.join(10)
        gaps = [b - a for a, b in zip(hellos, hellos[1:])]
        start = hellos[0]
        fast = [g for a, g in zip(hellos, gaps) if a - start < 0.3]
        slow = [g for a, g in zip(hellos, gaps) if a - start > 0.7]
        self.assertTrue(fast and max(fast) < 0.4, gaps)                  # every 0.1 s at first
        self.assertTrue(slow and min(slow) >= 0.55, gaps)                # then every 0.6 s
        self.assertTrue(any("Trying again every 0.1 s (every 0.6 s after" in s for s in self.said), self.said)

    def test_runner_exits_on_refused_device(self):
        self.start_routing()
        cat = self.svc.keys.create("cat", "guest")
        with mock.patch.object(runner_mod, "PENDING_RETRY_S", 0.3):
            t, codes, _ = self.main("--key", cat, device=DEV3)
            self.assertTrue(wait_for(lambda: self.svc.keys.device_code("cat", DEV3)))
            self.assertTrue(self.svc.keys.decide_device("cat", DEV3, False, OWNER))
            t.join(10)
        self.assertEqual(codes, [3], self.said)
        self.assertIn("refused", self.said[-1])

    def test_exit_with_a_command_running_stops_it(self):
        """Revoked while a Bash command runs on the PC: main exits 3, and the command does not finish."""
        self.start_routing(tool_call("Bash", command="echo s > started.txt; sleep 4; echo x > finished.txt"), DONE)
        if runner_mod.shell_mod.find_shell() is None:
            self.skipTest("no shell on this PC")
        t, codes, _ = self.main(env={"STRATA_RUNNER_KEY": self.ann})
        rid = runner_hub.runner_id("ann", DEV1)
        self.assertTrue(wait_for(lambda: self.svc.runner_hub.get_for(rid, SimpleNamespace(name="ann"))), self.said)

        def allow(ev):
            self.req("POST", "/agent/permission", {"id": ev["id"], "decision": "allow"}, key=self.ann, device=DEV1)
        chat, _events = self.stream(self.ann, {"cwd": str(self.proj), "session": "chat-1", "runner": rid}, answer=allow)
        self.assertTrue(wait_for(lambda: (self.proj / "started.txt").exists(), 20), self.said)
        code, body = self.req("POST", "/keys", {"action": "revoke", "name": "ann"}, key="k", device=DEV0)
        self.assertEqual(code, 200, body)
        t.join(2 * runner_hub.POLL_MAX_S + 6)
        self.assertEqual(codes, [3], self.said)
        time.sleep(5)
        self.assertFalse((self.proj / "finished.txt").exists())        # the command was stopped, not left running
        chat.join(30)

    def test_second_runner_on_the_pc_replaces_the_first(self):
        self.start_routing()
        first = self.runner(self.ann, name="r1")
        second = self.runner(self.ann, name="r2")
        self.assertEqual(second.id, first.id)
        self.assertTrue(wait_for(lambda: first.codes == [3], 2 * runner_hub.POLL_MAX_S + 3), self.said)
        self.assertTrue(any("newer runner" in s for s in self.said), self.said)
        self.assertEqual(second.codes, [])                       # the newer one goes on

    def test_key_with_a_line_end_works(self):
        self.start_routing()
        t, codes, stop = self.main(env={"STRATA_RUNNER_KEY": self.ann + "\r\n"})
        rid = runner_hub.runner_id("ann", DEV1)
        self.assertTrue(wait_for(lambda: self.svc.runner_hub.get_for(rid, SimpleNamespace(name="ann"))), self.said)
        stop.set()
        t.join(10)
        self.assertEqual(codes, [0])
        self.assertFalse(any(self.ann in s for s in self.said), self.said)

    def test_bad_key_exits_2_without_showing_it(self):
        self.start_routing()
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            t, codes, _ = self.main("--key", "abc\ndef-secret")
            t.join(10)
        self.assertEqual(codes, [2])
        self.assertFalse(any("secret" in s for s in self.said), self.said)
        self.assertNotIn("secret", err.getvalue())

    def test_no_key_is_an_error(self):
        self.start_routing()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("STRATA_RUNNER_KEY", None)
            t, codes, _ = self.main()
            t.join(5)
        self.assertEqual(codes, [2])


class tempfile_dir:
    """A temp folder as a Path (a context manager)."""

    def __enter__(self) -> Path:
        import tempfile
        self._t = tempfile.TemporaryDirectory()
        return Path(self._t.name)

    def __exit__(self, *exc):
        try:
            self._t.cleanup()
        except OSError:                                          # a process that just ended may still hold a file
            pass


if __name__ == "__main__":
    unittest.main()
