"""serve/test_runner_http.py - the runner channel over HTTP (#310 part 2, tasks 1-2): POST /runner/hello, GET
/runner/next, POST /runner/reply, GET /runners, and tool jobs with their calls back through `FakeRunner`, against the
mock engine in team mode and outside it.

    python -m pytest serve/test_runner_http.py -q
"""
from __future__ import annotations

import queue
import socket
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import agent_run, runner_hub  # noqa: E402
from serve.identity import OWNER  # noqa: E402
from serve.mcp import McpHub  # noqa: E402
from serve.permissions import Policy  # noqa: E402
from serve.runner_fake import FakeRunner  # noqa: E402
from serve.test_runner_hub import ctx  # noqa: E402
from serve.test_identity_http import DEV1, DEV2, REMOTE, Base  # noqa: E402

DEV0, DEV3 = "device-0000", "device-0003"
GROUPS = [{"name": "agent", "kind": "coding", "status": "ready", "error": None,
           "tools": [{"name": "Read", "description": "Reads a file.", "inputSchema": {"type": "object"}}]}]


def hello_body(instance="inst-aaaa", **over):
    body = {"protocol": 1, "instance": instance, "machine": "ann-pc", "platform": "win32", "shell": "Git Bash",
            "home": "C:\\Users\\ann", "groups": GROUPS, "skills": [{"name": "pdf", "description": "PDFs", "app": "claude"}]}
    body.update(over)
    return body


class RunnerBase(Base):
    def setUp(self):
        super().setUp()
        patch = mock.patch.object(runner_hub, "POLL_MAX_S", 2.0)
        patch.start()
        self.addCleanup(patch.stop)
        self.fakes = []

    def tearDown(self):
        for f in self.fakes:
            f.stop()
        self.svc.runner_hub.close()                           # held polls end before the server does
        super().tearDown()

    def team(self):
        """Team mode, every request from a remote PC: the owner (legacy key "k") on DEV0, guests ann and bob on DEV1."""
        self.start(api_key="k", peer_override=REMOTE)
        code, body = self.req("GET", "/props", key="k", device=DEV0)
        self.assertEqual((code, body["error"]["type"]), (403, "device_pending"))
        self.assertTrue(self.svc.keys.decide_device("owner", DEV0, True, OWNER))
        self.ann = self.approved("ann")
        self.bob = self.approved("bob")

    def hello(self, key, device=DEV1, **over):
        return self.req("POST", "/runner/hello", hello_body(**over), key=key, device=device)

    def poll(self, key, rid, instance="inst-aaaa", device=DEV1, wait=0):
        return self.req("GET", f"/runner/next?id={rid}&instance={instance}&wait={wait}", key=key, device=device)

    def reply(self, key, rid, job="j1", instance="inst-aaaa", device=DEV1, n=1, **more):
        return self.req("POST", "/runner/reply", {"id": rid, "instance": instance, "job": job, "rid": n, **more},
                        key=key, device=device)

    def runners(self, key, device=DEV1):
        code, body = self.req("GET", "/runners", key=key, device=device)
        self.assertEqual(code, 200, body)
        return body["runners"]


class Channel(RunnerBase):
    def test_hello_returns_id_and_who(self):
        self.team()
        code, body = self.hello(self.ann)
        self.assertEqual(code, 200, body)
        self.assertRegex(body.pop("channel"), r"^[0-9a-f]{16}$")
        self.assertEqual(body, {"runner": runner_hub.runner_id("ann", DEV1), "who": "ann", "protocol": 1, "poll_s": 2})
        code, got = self.poll(self.ann, body["runner"])
        self.assertEqual((code, got), (200, {"messages": []}))
        code, again = self.hello(self.ann)                          # same instance: the lists only
        self.assertEqual(again["runner"], body["runner"])

    def test_protocol_mismatch_is_409(self):
        self.team()
        code, body = self.req("POST", "/runner/hello", {"protocol": 2, "something": "new"}, key=self.ann, device=DEV1)
        self.assertEqual(code, 409)
        self.assertEqual(body["error"]["type"], "runner_protocol")
        self.assertEqual(body["error"]["server_protocol"], 1)
        self.assertEqual(body["error"]["message"], "update Strata Desktop (the server speaks runner protocol 1)")

    def test_hello_body_is_checked(self):
        self.team()
        for over in ({"instance": "short"}, {"instance": 5}, {"machine": "x" * 81}, {"groups": [{"name": "a", "kind": "x"}]},
                     {"groups": "no"}, {"skills": [5]}, {"shell": 3}, {"home": None}):
            code, body = self.hello(self.ann, **over)
            self.assertEqual(code, 400, (over, body))
        code, _ = self.req("POST", "/runner/hello", b"{}", key=self.ann, device=DEV1, headers={"Content-Type": "text/plain"})
        self.assertEqual(code, 415)
        code, body = self.req("POST", "/runner/hello", b" " * ((4 << 20) + 1), key=self.ann, device=DEV1)
        self.assertEqual(code, 413, body)
        self.assertEqual(self.runners(self.ann), [])

    def test_pending_device_runner_is_refused(self):
        self.team()
        code, body = self.hello(self.ann, device=DEV3)
        self.assertEqual((code, body["error"]["type"]), (403, "device_pending"))
        self.assertEqual(self.runners("k", DEV0), [])

    def test_poll_and_reply_for_another_identity_are_404(self):
        self.team()
        rid = self.hello(self.ann)[1]["runner"]
        r = self.svc.runner_hub.get_for(rid, SimpleNamespace(name="ann"))
        r.jobs["j1"] = SimpleNamespace(id="j1", t_picked=None, inbox=queue.Queue(), fail=lambda text: None)
        r.push({"job": "j1", "kind": "tool"})
        unknown = self.poll(self.bob, "r0000000000000000")
        code, body = self.poll(self.bob, rid)
        self.assertEqual((code, body), unknown)                     # another's id reads as one that is not there
        self.assertEqual(code, 404)
        code, body = self.poll(self.bob, rid, instance="inst-bbbb")
        self.assertEqual(code, 404)
        self.assertEqual(self.reply(self.bob, rid, result={"content": []})[0], 404)
        self.assertEqual(self.reply("k", rid, device=DEV0, result={"content": []})[0], 404)   # the owner neither
        self.assertTrue(r.jobs["j1"].inbox.empty())                 # nothing reached ann's PC's job
        code, body = self.poll(self.ann, rid)
        self.assertEqual((code, body), (200, {"messages": [{"job": "j1", "kind": "tool", "seq": 1}]}))
        self.assertEqual(self.reply(self.ann, rid, job="j9")[0], 404)   # unknown or finished job
        self.assertEqual(self.reply(self.ann, rid, result={"content": []}), (200, {"ok": True}))

    def test_replaced_instance_is_409(self):
        self.team()
        rid = self.hello(self.ann)[1]["runner"]
        held = []
        t = threading.Thread(target=lambda: held.append(self.poll(self.ann, rid, wait=2)))
        t.start()
        time.sleep(0.3)
        self.assertEqual(self.hello(self.ann, instance="inst-bbbb")[1]["runner"], rid)
        t.join()
        self.assertEqual(held[0][0], 409)                           # the held poll of the old process ends at once
        self.assertEqual(held[0][1]["error"]["type"], "runner_replaced")
        code, body = self.poll(self.ann, rid)
        self.assertEqual((code, body["error"]["type"]), (409, "runner_replaced"))
        self.assertEqual(self.reply(self.ann, rid)[0], 409)
        self.assertEqual(self.poll(self.ann, rid, instance="inst-bbbb")[0], 200)

    def test_wait_is_clamped(self):
        self.team()
        rid = self.hello(self.ann)[1]["runner"]
        with mock.patch.object(runner_hub, "POLL_MAX_S", 0.3):
            t0 = time.monotonic()
            self.assertEqual(self.poll(self.ann, rid, wait=600)[0], 200)
            self.assertLess(time.monotonic() - t0, 1.5)
        self.assertEqual(self.poll(self.ann, rid, wait="soon")[0], 400)
        self.assertEqual(self.poll(self.ann, rid, wait="nan")[0], 400)
        for ack in ("-1", "x", "1.5", "9" * 19):
            code, _ = self.req("GET", f"/runner/next?id={rid}&instance=inst-aaaa&wait=0&ack={ack}", key=self.ann, device=DEV1)
            self.assertEqual(code, 400, ack)

    def test_overlapping_polls_newest_wins(self):
        self.team()
        rid = self.hello(self.ann)[1]["runner"]
        r = self.svc.runner_hub.get_for(rid, SimpleNamespace(name="ann"))
        older, newer = [], []
        t1 = threading.Thread(target=lambda: older.append((self.poll(self.ann, rid, wait=2), time.monotonic())))
        t1.start()
        time.sleep(0.3)
        t0 = time.monotonic()
        t2 = threading.Thread(target=lambda: newer.append(self.poll(self.ann, rid, wait=2)))
        t2.start()
        t1.join()
        self.assertEqual(older[0][0], (200, {"messages": []}))
        self.assertLess(older[0][1] - t0, 1.0)
        r.push({"job": "j1", "answer": "allow"})
        t2.join()
        self.assertEqual(newer[0], (200, {"messages": [{"job": "j1", "answer": "allow", "seq": 1}]}))

    def test_hung_up_poll_leaves_message_queued(self):
        self.team()
        rid = self.hello(self.ann)[1]["runner"]
        r = self.svc.runner_hub.get_for(rid, SimpleNamespace(name="ann"))
        with mock.patch.object(runner_hub, "POLL_MAX_S", 10.0):
            sock = socket.create_connection(("127.0.0.1", self.port))
            sock.sendall((f"GET /runner/next?id={rid}&instance=inst-aaaa&wait=10 HTTP/1.0\r\nHost: 127.0.0.1\r\n"
                          f"Authorization: Bearer {self.ann}\r\nX-Strata-Device: {DEV1}\r\n\r\n").encode())
            deadline = time.monotonic() + 2
            while r._held == 0 and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertEqual(r._held, 1)                            # the poll is held
            sock.close()                                            # the runner's network drops
            t0 = time.monotonic()
            while r._held and time.monotonic() - t0 < 5:
                time.sleep(0.02)
            self.assertEqual(r._held, 0)
            self.assertLess(time.monotonic() - t0, 2.0)             # ended by the hang-up, not by the 10 s wait
            r.push({"job": "j1", "answer": "allow"})
            self.assertEqual(self.poll(self.ann, rid), (200, {"messages": [{"job": "j1", "answer": "allow", "seq": 1}]}))

    def test_runners_view_by_identity(self):
        self.team()
        ann_id = self.hello(self.ann)[1]["runner"]
        bob_id = self.hello(self.bob, machine="bob-pc")[1]["runner"]
        mine = {r["who"]: r for r in self.runners(self.ann)}
        self.assertEqual(mine["ann"]["id"], ann_id)
        self.assertEqual(mine["ann"]["machine"], "ann-pc")
        self.assertTrue(mine["ann"]["online"])
        self.assertEqual([g["name"] for g in mine["ann"]["groups"]], ["agent"])
        self.assertNotIn("instance", mine["ann"])
        self.assertEqual(mine["bob"], {"who": "bob", "online": True})
        full = {r["who"]: r for r in self.runners("k", DEV0)}
        self.assertEqual((full["ann"]["id"], full["bob"]["id"]), (ann_id, bob_id))
        self.assertEqual(full["bob"]["machine"], "bob-pc")

    def test_revoke_ends_runner(self):
        self.team()
        rid = self.hello(self.ann)[1]["runner"]
        held = []
        t = threading.Thread(target=lambda: held.append(self.poll(self.ann, rid, wait=2)))
        t.start()
        time.sleep(0.3)
        code, body = self.req("POST", "/keys", {"action": "revoke", "name": "ann"}, key="k", device=DEV0)
        self.assertEqual(code, 200, body)
        t.join()
        self.assertEqual(held[0][0], 401)                           # the held poll ends at once
        self.assertEqual([r["who"] for r in self.runners("k", DEV0)], [])
        self.assertEqual(self.poll(self.ann, rid)[0], 401)
        self.assertEqual(self.reply(self.ann, rid)[0], 401)

    def test_rename_keeps_runner(self):
        self.team()
        rid = self.hello(self.ann)[1]["runner"]
        code, body = self.req("POST", "/keys", {"action": "rename", "name": "ann", "to": "anna"}, key="k", device=DEV0)
        self.assertEqual(code, 200, body)
        self.assertEqual(self.poll(self.ann, rid)[0], 404)          # the id follows the name: the runner says hello again
        code, body = self.hello(self.ann)
        self.assertEqual((body["runner"], body["who"]), (runner_hub.runner_id("anna", DEV1), "anna"))
        self.assertEqual(len(self.runners(self.ann)), 1)

    def test_refused_or_cleared_device_ends_runner(self):
        self.team()
        rid = self.hello(self.ann)[1]["runner"]
        self.hello(self.ann, device=DEV2)                           # pending: no runner
        code, body = self.req("POST", "/devices", {"key": "ann", "device": DEV1, "approve": False}, key="k", device=DEV0)
        self.assertEqual(code, 200, body)
        self.assertIsNone(self.svc.runner_hub.get_for(rid, SimpleNamespace(name="ann")))
        rid_b = self.hello(self.bob)[1]["runner"]
        code, body = self.req("POST", "/devices", {"key": "bob", "device": DEV1, "approve": True}, key="k", device=DEV0)
        self.assertIsNotNone(self.svc.runner_hub.get_for(rid_b, SimpleNamespace(name="bob")))   # approve ends nothing
        code, body = self.req("POST", "/devices", {"key": "bob", "device": DEV1, "action": "clear"}, key="k", device=DEV0)
        self.assertEqual(code, 200, body)
        self.assertIsNone(self.svc.runner_hub.get_for(rid_b, SimpleNamespace(name="bob")))

    def test_demoted_owner_keeps_runner(self):
        self.team()
        carol = self.approved("carol", role="owner")
        rid = self.hello(carol)[1]["runner"]
        code, body = self.req("POST", "/keys", {"action": "role", "name": "carol", "role": "guest"}, key="k", device=DEV0)
        self.assertEqual(code, 200, body)
        self.assertEqual(self.poll(carol, rid)[0], 200)
        mine = [r for r in self.runners(carol) if r["who"] == "carol"]
        self.assertEqual(len(mine), 1)
        self.assertTrue(mine[0]["online"])

    def test_fake_runner_gets_messages_and_replies(self):
        self.team()
        seen = []

        def on_job(msg, reply):
            seen.append(msg)
            reply({"result": {"content": [{"type": "text", "text": "done"}], "isError": False}})

        fake = FakeRunner(self.port, self.ann, DEV1, GROUPS, on_job)
        self.fakes.append(fake)
        r = self.svc.runner_hub.get_for(fake.id, SimpleNamespace(name="ann"))
        self.assertIsNotNone(r)
        j = r.jobs["j1"] = SimpleNamespace(id="j1", t_picked=None, inbox=queue.Queue(), fail=lambda text: None)
        r.push({"job": "j1", "kind": "tool", "tool": "Read"})
        got = j.inbox.get(timeout=5)
        self.assertEqual(got["result"]["content"][0]["text"], "done")
        self.assertEqual(seen[0]["tool"], "Read")
        self.assertIsNotNone(j.t_picked)
        t0 = time.monotonic()
        fake.stop()
        self.assertLess(time.monotonic() - t0, 2.5)


def text(t, error=False):
    return {"content": [{"type": "text", "text": t}], "isError": error}


class Calls(RunnerBase):
    """Task 2: tool jobs over the channel, and the calls back during a job."""

    def fake(self, on_job, **kw):
        fake = FakeRunner(self.port, self.ann, DEV1, GROUPS, on_job, machine="ann-pc", **kw)
        self.fakes.append(fake)
        return fake

    def hub_for(self, rid):
        r = self.svc.runner_hub.get_for(rid, SimpleNamespace(name="ann"))
        return McpHub({}, builtins=r.servers())

    def call(self, hub, c=None, cancel=None, name="Read", arguments=None):
        progress = []
        r = hub.call(name, arguments or {"file_path": "a.txt"}, cancel or threading.Event(), c or ctx(),
                     progress=progress.append)
        return r, progress

    def test_tool_job_round_trip(self):
        self.team()
        seen = []

        def on_job(msg, reply):
            seen.append(msg)
            reply({"result": text("line 1"), "exec_ms": 3})
        fake = self.fake(on_job)
        r, progress = self.call(self.hub_for(fake.id))
        self.assertEqual((r["ok"], r["text"], r["server"], r["tool"]), (True, "line 1", "my_agent", "Read"))
        self.assertEqual(progress, ["sending", "running"])
        msg = dict(seen[0])
        self.assertIsInstance(msg.pop("job"), str)
        self.assertIsInstance(msg.pop("seq"), int)
        self.assertEqual(msg, {"kind": "tool", "group": "agent", "tool": "Read", "arguments": {"file_path": "a.txt"},
                               "run": "run-1", "session": "s1", "mode": "ask", "helper": False, "vision": True,
                               "timeout_s": 0})

    def test_ask_is_served_by_ctx_ask(self):
        self.team()
        answers = queue.Queue()
        card = {"tool": "Bash", "arguments": {"command": "rm x"}, "why": "outside the project", "danger": False}

        def on_job(msg, reply):
            if "kind" in msg:
                reply({"ask": card})
                got = answers.get(timeout=5)
                reply({"result": text(f"ran after {got['answer']}")})
            elif "answer" in msg:
                answers.put(msg)
        fake = self.fake(on_job)
        c = ctx()
        r, _ = self.call(self.hub_for(fake.id), c)
        self.assertEqual(c.asked, [card])
        self.assertEqual(r["text"], "ran after allow")
        got = [m for m in fake.received if "answer" in m]
        self.assertEqual([m["answer"] for m in got], ["allow"])

    def test_question_and_emit_reach_ctx(self):
        self.team()
        answers = queue.Queue()
        qs = [{"question": "Which one?", "options": [{"label": "A"}, {"label": "B"}]}]

        def on_job(msg, reply):
            if "kind" in msg:
                reply({"emit": {"event": "todos", "todos": [{"content": "x", "status": "pending"}]}})
                reply({"question": qs})
                got = answers.get(timeout=5)
                reply({"result": text(f"picked {got['answer']['answers']['Which one?']}")})
            elif "answer" in msg:
                answers.put(msg)
        fake = self.fake(on_job)
        asked = []
        c = ctx(question=lambda q: asked.append(q) or {"answers": {"Which one?": "B"}})
        r, _ = self.call(self.hub_for(fake.id), c)
        self.assertEqual(r["text"], "picked B")
        self.assertEqual(asked, [qs])
        self.assertEqual(c.emitted, [{"event": "todos", "todos": [{"content": "x", "status": "pending"}]}])

    def test_cancel_reaches_runner(self):
        self.team()
        cancels = queue.Queue()
        honour = [True]

        def on_job(msg, reply):
            if "kind" in msg:
                cancels.get(timeout=5)
                if honour[0]:
                    reply({"result": text("stopped by the user", error=True)})
            elif msg.get("cancel"):
                cancels.put(msg)
        fake = self.fake(on_job)
        hub = self.hub_for(fake.id)
        cancel = threading.Event()
        threading.Timer(0.3, cancel.set).start()
        t0 = time.monotonic()
        r, progress = self.call(hub, cancel=cancel)
        self.assertEqual(r["text"], "error: stopped by the user")     # the runner's own result, within the grace
        self.assertLess(time.monotonic() - t0, 2.0)
        honour[0] = False                                             # a runner that never answers the cancel
        with mock.patch.object(runner_hub, "CANCEL_GRACE_S", 0.5):
            cancel = threading.Event()
            threading.Timer(0.3, cancel.set).start()
            t0 = time.monotonic()
            r, _ = self.call(hub, cancel=cancel)
        self.assertEqual(r["text"], "error: The request was cancelled.")
        self.assertLess(time.monotonic() - t0, 0.3 + 0.5 + 0.6)

    def test_offline_before_pickup(self):
        self.team()
        rid = self.hello(self.ann)[1]["runner"]                     # said hello, but nothing polls
        with mock.patch.object(runner_hub, "OFFLINE_S", 0.3):
            t0 = time.monotonic()
            r, progress = self.call(self.hub_for(rid))
            self.assertLess(time.monotonic() - t0, 1.3)
        self.assertEqual(r["text"], "error: your PC (ann-pc) is offline, so Read was not run. Tell the user; do not "
                                    "retry it until they say their PC is back.")
        self.assertEqual(progress, ["sending", "offline"])
        self.assertEqual(self.poll(self.ann, rid), (200, {"messages": []}))   # the job is never handed out later

    def test_offline_after_pickup_fails_with_text(self):
        self.team()
        picked = threading.Event()
        with mock.patch.object(runner_hub, "OFFLINE_S", 0.5):
            fake = self.fake(lambda msg, reply: picked.set())
            hub = self.hub_for(fake.id)
            out = []
            t = threading.Thread(target=lambda: out.append(self.call(hub)))
            t.start()
            self.assertTrue(picked.wait(5))
            fake.stop()                                             # the lid closes: no more polls, no reply
            t0 = time.monotonic()
            t.join(5)
            self.assertLess(time.monotonic() - t0, runner_hub.OFFLINE_S + 1)
        r, progress = out[0]
        self.assertEqual(r["text"], "error: your PC (ann-pc) went offline while Read was running; whether it finished "
                                    "is unknown. Tell the user, and check the state before you repeat it.")
        self.assertEqual(progress, ["sending", "running", "offline"])

    def test_long_approval_spans_many_polls(self):
        self.team()
        answers = queue.Queue()

        def on_job(msg, reply):
            if "kind" in msg:
                reply({"ask": {"tool": "Bash", "arguments": {"command": "make"}}})
                got = answers.get(timeout=10)
                reply({"result": text(f"answer {got['answer']}")})
            elif "answer" in msg:
                answers.put(msg)

        def slow_ask(card):
            time.sleep(3.0)                                         # the user takes their time
            return "allow"
        with mock.patch.object(runner_hub, "POLL_MAX_S", 0.5):
            fake = self.fake(on_job)
            r, _ = self.call(self.hub_for(fake.id), ctx(ask=slow_ask))
        self.assertEqual(r["text"], "answer allow")
        self.assertGreaterEqual(len(fake.polls), 5)                 # the wait was carried by many short polls
        self.assertLessEqual(max(fake.polls), 1.5, fake.polls)

    def test_two_chats_share_one_runner(self):
        self.team()
        second_done = threading.Event()

        def on_job(msg, reply):
            if "kind" not in msg:
                return
            if msg["session"] == "s1":
                second_done.wait(5)                                 # the first call is answered last
                reply({"result": text("for s1")})
            else:
                reply({"result": text("for s2")})
                second_done.set()
        fake = self.fake(on_job)
        hub = self.hub_for(fake.id)
        out, order = {}, []

        def run(s):
            out[s] = self.call(hub, ctx(session=s))[0]["text"]
            order.append(s)
        ts = [threading.Thread(target=run, args=(s,)) for s in ("s1", "s2")]
        for t in ts:
            t.start()
        for t in ts:
            t.join(10)
        self.assertEqual(out, {"s1": "for s1", "s2": "for s2"})
        self.assertEqual(order, ["s2", "s1"])

    def test_revoke_fails_open_job(self):
        self.team()
        picked = threading.Event()
        fake = self.fake(lambda msg, reply: picked.set())           # picks the job up and never answers
        hub = self.hub_for(fake.id)
        out = []
        t = threading.Thread(target=lambda: out.append(self.call(hub)))
        t.start()
        self.assertTrue(picked.wait(5))
        t0 = time.monotonic()
        code, body = self.req("POST", "/keys", {"action": "revoke", "name": "ann"}, key="k", device=DEV0)
        self.assertEqual(code, 200, body)
        t.join(5)
        self.assertLess(time.monotonic() - t0, 1.5)
        self.assertEqual(out[0][0]["text"], "error: the key of this chat was revoked, so Read was not finished.")
        fake.thread.join(5)
        self.assertEqual(fake.ended[0], 401)

    def test_dropped_poll_result_is_sent_again(self):
        """A poll result lost on the way back (the runner's connection died after the server wrote it): the job,
        the answer and the cancel each arrive on the next poll, and each runs once."""
        self.team()
        answers, cancels, ready = queue.Queue(), queue.Queue(), threading.Event()
        fake = None

        def on_job(msg, reply):
            if msg.get("kind") and msg["arguments"].get("step") == "job":
                reply({"result": text("job ran")})
            elif msg.get("kind") and msg["arguments"].get("step") == "answer":
                fake.drop = 1                                       # the answer's poll result is lost
                reply({"ask": {"tool": "Bash"}})
                reply({"result": text(f"answer {answers.get(timeout=5)['answer']}")})
            elif msg.get("kind"):
                fake.drop = 1                                       # the cancel's poll result is lost
                ready.set()
                cancels.get(timeout=5)
                reply({"result": text("cancel seen", error=True)})
            elif "answer" in msg:
                answers.put(msg)
            elif msg.get("cancel"):
                cancels.put(msg)
        fake = self.fake(on_job)
        hub = self.hub_for(fake.id)
        fake.drop = 1                                               # the job's poll result is lost
        self.assertEqual(self.call(hub, arguments={"step": "job"})[0]["text"], "job ran")
        self.assertEqual(self.call(hub, arguments={"step": "answer"})[0]["text"], "answer allow")
        cancel = threading.Event()
        threading.Thread(target=lambda: ready.wait(5) and cancel.set()).start()
        self.assertEqual(self.call(hub, cancel=cancel, arguments={"step": "cancel"})[0]["text"], "error: cancel seen")
        self.assertEqual(fake.drop, 0)                              # all three were dropped once
        kinds = [("kind" in m, "answer" in m, "cancel" in m) for m in fake.received]
        self.assertEqual(kinds.count((True, False, False)), 3)      # each job ran once
        self.assertEqual(kinds.count((False, True, False)), 1)
        self.assertEqual(kinds.count((False, False, True)), 1)

    def test_lost_ack_runs_once(self):
        """The runner got the job but its ack did not arrive: the server sends it again, and the runner ignores it."""
        self.team()
        runs = []

        def on_job(msg, reply):
            if "kind" in msg:
                runs.append(msg["seq"])
                time.sleep(0.3)                                     # still running when the job comes again
                reply({"result": text("ran")})
        fake = self.fake(on_job)
        fake.lose_ack = 1
        self.assertEqual(self.call(self.hub_for(fake.id))[0]["text"], "ran")
        self.assertEqual(len(runs), 1)
        self.assertGreaterEqual(fake.repeats, 1)                    # it did come again

    def real_run(self):
        """A real AgentRun (its Broker, its events): the card the page would show."""
        run = agent_run.AgentRun(Policy(cwd=None), session="s1", broker=agent_run.Broker(), goal="g", side=None,
                                 cancel=threading.Event(), who="ann")
        return run

    def wait_card(self, run, kind):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if any(s["kind"] == kind for s in list(run.broker.pending.values())):
                return
            time.sleep(0.02)
        self.fail(f"no {kind} card opened")

    def test_offline_while_card_open_closes_card(self):
        """R2-1: the PC goes offline while its approval card is open: the card closes (a refusal, with the reason, for
        the page) and the call ends within OFFLINE_S with the offline text - not after the user's 600 s."""
        self.team()
        run = self.real_run()
        with mock.patch.object(runner_hub, "OFFLINE_S", 0.5):
            fake = self.fake(lambda msg, reply: reply({"ask": {"tool": "Bash", "arguments": {"command": "make"}}})
                             if "kind" in msg else None)
            hub = self.hub_for(fake.id)
            out = []
            t = threading.Thread(target=lambda: out.append(self.call(hub, run.ctx)))
            t.start()
            self.wait_card(run, "permission")
            fake.stop()                                             # the lid closes with the card open
            t0 = time.monotonic()
            t.join(5)
            self.assertLess(time.monotonic() - t0, runner_hub.OFFLINE_S + 1)
        r, progress = out[0]
        self.assertTrue(r["text"].startswith("error: your PC (ann-pc) went offline while Read was running"), r["text"])
        self.assertEqual(progress[-1], "offline")
        self.assertEqual(run.broker.pending, {})
        closed = [e for e in run.drain() if e["event"] == "answered"]
        self.assertEqual([(e["answer"], e["reason"]) for e in closed], [("deny", "offline")])

    def test_replaced_while_question_open_closes_card(self):
        self.team()
        run = self.real_run()
        fake = self.fake(lambda msg, reply: reply({"question": [{"question": "Which?", "options": [{"label": "A"}]}]})
                         if "kind" in msg else None)
        hub = self.hub_for(fake.id)
        out = []
        t = threading.Thread(target=lambda: out.append(self.call(hub, run.ctx)))
        t.start()
        self.wait_card(run, "question")
        t0 = time.monotonic()
        self.fake(lambda msg, reply: None)                          # a second runner process on the same PC
        t.join(5)
        self.assertLess(time.monotonic() - t0, 1.5)
        self.assertEqual(out[0][0]["text"], "error: your PC's runner restarted while Read was running; whether it "
                                            "finished is unknown.")
        self.assertEqual(run.broker.pending, {})
        closed = [e for e in run.drain() if e["event"] == "question_answered"]
        self.assertEqual([(e["answers"], e["reason"]) for e in closed], [(None, "replaced")])

    def test_retried_ask_opens_one_card(self):
        """I-2: a reply sent again (its answer was lost) is applied once, and each answer names the ask it answers."""
        self.team()
        answers = queue.Queue()

        def on_job(msg, reply):
            if "kind" in msg:
                for n, card in ((1, {"tool": "Bash", "n": 1}), (1, {"tool": "Bash", "n": 1}), (2, {"tool": "Edit", "n": 2})):
                    self.assertEqual(reply({"rid": n, "ask": card})[0], 200)   # the retry is answered 200 too
                got = [answers.get(timeout=5), answers.get(timeout=5)]
                reply({"rid": 3, "result": text(" ".join(f"{a['to']}:{a['answer']}" for a in got))})
            elif "answer" in msg:
                answers.put(msg)
        fake = self.fake(on_job)
        c = ctx(ask=lambda card: c.asked.append(card) or ("allow" if card["n"] == 1 else "deny"))
        r, _ = self.call(self.hub_for(fake.id), c)
        self.assertEqual([a["n"] for a in c.asked], [1, 2])         # one card per ask, not per post
        self.assertEqual(r["text"], "1:allow 2:deny")
        self.assertEqual(len([m for m in fake.received if "answer" in m]), 2)

    def test_bad_reply_shapes_are_refused_or_dropped(self):
        """I-4 / M-1: an ask that is no object is refused, a question that is no list gets no answer, an emit that is
        no object and a key the server keeps for itself ("failed") are dropped."""
        self.team()
        answers = queue.Queue()

        def on_job(msg, reply):
            if "kind" in msg:
                reply({"ask": "rm -rf /"})
                reply({"question": "which?"})
                reply({"emit": ["not", "an", "event"]})
                reply({"failed": "fake end"})
                got = [answers.get(timeout=5), answers.get(timeout=5)]
                reply({"result": text(repr([(a["to"], a["answer"]) for a in got]))})
            elif "answer" in msg:
                answers.put(msg)
        fake = self.fake(on_job)
        c = ctx()
        r, _ = self.call(self.hub_for(fake.id), c)
        self.assertEqual(r["text"], "[(1, 'deny'), (2, {'answers': None})]")
        self.assertEqual((c.asked, c.emitted), ([], []))

    def test_reply_needs_rid(self):
        self.team()
        rid = self.hello(self.ann)[1]["runner"]
        r = self.svc.runner_hub.get_for(rid, SimpleNamespace(name="ann"))
        r.jobs["j1"] = SimpleNamespace(id="j1", t_picked=None, inbox=queue.Queue(), fail=lambda text: None)
        for n in (None, 0, "1", 1.5, True):
            code, _ = self.req("POST", "/runner/reply", {"id": rid, "instance": "inst-aaaa", "job": "j1", "rid": n,
                                                         "emit": {}}, key=self.ann, device=DEV1)
            self.assertEqual(code, 400, n)
        self.assertTrue(r.jobs["j1"].inbox.empty())


class OutsideTeamMode(RunnerBase):
    def test_outside_team_mode_runner_is_owner(self):
        self.start(api_key="k", team=False, peer_override=REMOTE)
        code, body = self.hello("k")
        self.assertEqual(code, 200, body)
        self.assertEqual(body["who"], "owner")
        self.assertEqual(self.poll("k", body["runner"])[0], 200)
        self.assertEqual(self.hello(None)[0], 401)
        self.assertEqual(self.poll(None, body["runner"])[0], 401)
        self.assertEqual(self.req("GET", "/runners")[0], 401)
        self.assertFalse(self.keys_path.exists())

    def test_no_key_remote_is_refused(self):
        self.start(team=False, peer_override=REMOTE)
        code, body = self.hello(None)
        self.assertEqual(code, 403, body)
        self.assertEqual(self.req("GET", "/runners")[0], 403)
        self.svc.peer_override = None                               # this PC itself: the owner, as may_edit says
        self.assertEqual(self.hello(None)[0], 200)

    def test_two_devices_outside_team_mode_get_two_ids(self):
        self.start(api_key="k", team=False, peer_override=REMOTE)
        a = self.hello("k", device=DEV1)[1]["runner"]
        b = self.hello("k", device=DEV2, instance="inst-bbbb")[1]["runner"]
        self.assertNotEqual(a, b)
        self.assertEqual(self.poll("k", a, device=DEV1)[0], 200)
        self.assertEqual(self.poll("k", b, instance="inst-bbbb", device=DEV2)[0], 200)
        self.assertEqual(len(self.runners("k")), 2)


if __name__ == "__main__":
    unittest.main()
