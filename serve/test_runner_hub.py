"""serve/test_runner_hub.py - the runner channel's server side without HTTP (#310 part 2, tasks 1-2): runner ids,
long-poll, presence, a second instance, the identity a runner belongs to; jobs, seq / ack, the groups, the spans.

    python -m pytest serve/test_runner_hub.py -q
"""
from __future__ import annotations

import queue
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import runner_hub  # noqa: E402
from serve.identity import OWNER, Identity  # noqa: E402

ANN = Identity("ann", "guest")
BOB = Identity("bob", "guest")


def job(jid, fail=None):
    """The job side as Task 2 fills it, duck-typed: `t_picked`, `inbox`, `fail(text)`."""
    return SimpleNamespace(id=jid, t_picked=None, inbox=queue.Queue(), fail=fail or (lambda text: None))


INFO = {"machine": "ann-pc", "platform": "win32", "shell": "Git Bash", "home": "C:\\Users\\ann"}


class Hub(unittest.TestCase):
    def setUp(self):
        self.hub = runner_hub.RunnerHub()
        self.addCleanup(self.hub.close)

    def hello(self, who="ann", device="device-0001", instance="inst-aaaa", groups=None):
        return self.hub.hello(who, device, instance, dict(INFO), groups or [], [])

    def test_runner_id_is_stable(self):
        a = runner_hub.runner_id("ann", "device-0001")
        self.assertEqual(a, runner_hub.runner_id("ann", "device-0001"))
        self.assertNotEqual(a, runner_hub.runner_id("bob", "device-0001"))
        self.assertNotEqual(a, runner_hub.runner_id("ann", "device-0002"))
        self.assertRegex(a, r"^r[0-9a-f]{16}$")
        self.assertEqual(self.hello().id, a)

    def test_take_returns_at_wait_with_nothing(self):
        r = self.hello()
        t0 = time.monotonic()
        self.assertEqual(r.take(0.2), [])
        self.assertTrue(0.2 <= time.monotonic() - t0 < 0.5, time.monotonic() - t0)

    def test_online_window(self):
        with mock.patch.object(runner_hub, "OFFLINE_S", 0.3):
            r = self.hello()
            self.assertTrue(r.online())
            time.sleep(0.4)
            self.assertFalse(r.online())
            held = threading.Thread(target=r.take, args=(1.0,))
            held.start()
            time.sleep(0.5)                                     # past OFFLINE_S since hello, but a poll is held
            self.assertTrue(r.online())
            held.join()
            self.assertTrue(r.online())                         # the poll's end counts as a sign of life
            self.assertEqual(r.public(False), {"who": "ann", "online": True})

    def test_get_for_matches_who_exactly(self):
        r = self.hello()
        self.assertIs(self.hub.get_for(r.id, ANN), r)
        self.assertIsNone(self.hub.get_for(r.id, BOB))
        self.assertIsNone(self.hub.get_for(r.id, OWNER))      # nobody runs commands on another person's PC
        self.assertIsNone(self.hub.get_for("r0000000000000000", ANN))
        self.assertIsNone(self.hub.get_for(None, ANN))
        # the view is another matter: the owner sees every runner in full, a guest others' only as who/online
        mine = self.hub.hello("bob", "device-0002", "inst-bbbb", dict(INFO), [], [])
        self.assertEqual({v["who"] for v in self.hub.view(OWNER)}, {"ann", "bob"})
        self.assertTrue(all("machine" in v for v in self.hub.view(OWNER)))
        bob_view = {v["who"]: v for v in self.hub.view(BOB)}
        self.assertEqual(bob_view["ann"], {"who": "ann", "online": True})
        self.assertEqual(bob_view["bob"]["id"], mine.id)
        self.assertNotIn("instance", bob_view["bob"])

    def test_same_instance_updates_lists_only(self):
        r = self.hello()
        again = self.hello(groups=[{"name": "agent", "kind": "coding", "status": "ready", "error": None, "tools": []}])
        self.assertIs(again, r)
        self.assertEqual([g["name"] for g in r.groups], ["agent"])
        self.assertIsNone(r.ended)

    def test_second_instance_replaces_first(self):
        old = self.hello(instance="inst-aaaa")
        done = []
        held = threading.Thread(target=lambda: done.append(old.take(5.0)))
        held.start()
        time.sleep(0.1)
        t0 = time.monotonic()
        new = self.hello(instance="inst-bbbb")
        held.join()
        self.assertLess(time.monotonic() - t0, 1.0)             # the old process's held poll is woken at once
        self.assertEqual(new.id, old.id)
        self.assertIsNot(new, old)
        self.assertEqual(old.ended, "replaced")                 # the old instance is refused from now on
        self.assertEqual(old.take(1.0), [])
        self.assertIs(self.hub.get_for(old.id, ANN), new)
        self.assertEqual(new.instance, "inst-bbbb")
        self.assertIsNone(new.ended)

    def test_replaced_instance_fails_its_open_jobs(self):
        old = self.hello(instance="inst-aaaa")
        failed = []
        old.jobs["j1"] = job("j1", fail=failed.append)
        old.push({"job": "j1", "kind": "tool"})
        self.hello(instance="inst-bbbb")
        self.assertEqual(failed, ["your PC's runner restarted"])
        self.assertEqual(old.jobs, {})

    def test_take_marks_jobs_picked(self):
        r = self.hello()
        j = r.jobs["j1"] = job("j1")
        r.push({"job": "j1", "kind": "tool"})
        r.push({"job": "j1", "answer": "allow"})               # a message for an open job: not a pick-up again
        got = r.take(1.0)
        self.assertEqual([m["job"] for m in got], ["j1", "j1"])
        self.assertIsNotNone(j.t_picked)
        self.assertFalse(r.deliver({"job": "nope"}))            # unknown or finished: the route answers 404
        self.assertTrue(r.deliver({"job": "j1", "result": {"content": []}}))
        self.assertEqual(j.inbox.get_nowait(), {"job": "j1", "result": {"content": []}})

    def test_move_rekeys_to_the_new_name(self):
        r = self.hello()
        old_id = r.id
        self.assertEqual(self.hub.move("ann", "anna"), 1)
        self.assertIsNone(self.hub.get_for(old_id, ANN))
        self.assertEqual(r.id, runner_hub.runner_id("anna", "device-0001"))
        self.assertIs(self.hub.get_for(r.id, Identity("anna", "guest")), r)
        self.assertIs(self.hub.hello("anna", "device-0001", "inst-aaaa", dict(INFO), [], []), r)
        self.assertIsNone(r.ended)                              # a rename keeps the runner and its jobs going

    def test_end_all_and_end_device(self):
        a1 = self.hello(device="device-0001")
        a2 = self.hello(device="device-0002")
        b = self.hub.hello("bob", "device-0001", "inst-bbbb", dict(INFO), [], [])
        self.assertEqual(self.hub.end_device("ann", "device-0002"), 1)
        self.assertEqual(a2.ended, "device")
        self.assertIsNone(self.hub.get_for(a2.id, ANN))
        self.assertEqual(self.hub.end_all("ann"), 1)
        self.assertEqual(a1.ended, "revoked")
        self.assertEqual([v["who"] for v in self.hub.view(OWNER)], ["bob"])
        self.assertIsNone(b.ended)

    def test_newest_poll_wins(self):
        r = self.hello()
        older, newer = [], []
        t1 = threading.Thread(target=lambda: older.append((r.take(5.0), time.monotonic())))
        t1.start()
        time.sleep(0.1)
        t0 = time.monotonic()
        t2 = threading.Thread(target=lambda: newer.append(r.take(5.0)))
        t2.start()
        t1.join(2.0)
        self.assertFalse(t1.is_alive())
        self.assertEqual(older[0][0], [])                       # the stale poll ends at once, empty
        self.assertLess(older[0][1] - t0, 0.5)
        r.push({"job": "j1", "answer": "allow"})
        t2.join(2.0)
        self.assertEqual(newer, [[{"job": "j1", "answer": "allow", "seq": 1}]])   # the message reaches the live poll

    def test_hung_up_poll_takes_nothing(self):
        with mock.patch.object(runner_hub, "OFFLINE_S", 0.3):
            r = self.hello()
            gone = threading.Event()
            got = []
            t = threading.Thread(target=lambda: got.append(r.take(5.0, gone=gone)))
            t.start()
            time.sleep(0.4)
            t0 = time.monotonic()
            gone.set()
            t.join(2.0)
            self.assertFalse(t.is_alive())
            self.assertLess(time.monotonic() - t0, 0.5)
            self.assertEqual(got, [[]])
            self.assertFalse(r.online())                        # a hang-up is no sign of life
            r.push({"job": "j1", "answer": "allow"})
            gone2 = threading.Event()
            gone2.set()
            self.assertEqual(r.take(1.0, gone=gone2), [])       # a poll already hung up takes nothing either
            self.assertEqual(r.take(0.0), [{"job": "j1", "answer": "allow", "seq": 1}])   # the message stayed queued

    def test_close_wakes_held_polls(self):
        r = self.hello()
        got = []
        held = threading.Thread(target=lambda: got.append(r.take(10.0)))
        held.start()
        time.sleep(0.1)
        t0 = time.monotonic()
        self.hub.close()
        held.join(2.0)
        self.assertFalse(held.is_alive())
        self.assertLess(time.monotonic() - t0, 1.0)
        self.assertEqual(got, [[]])
        self.assertEqual(r.ended, "closed")


CODING = {"name": "agent", "kind": "coding", "status": "ready", "error": None,
          "tools": [{"name": "Read", "description": "Reads a file.", "inputSchema": {"type": "object"}}]}


def ctx(session="s1", **over):
    """A stub of the request's AgentContext: what RunnerGroup.call reads and calls."""
    c = SimpleNamespace(session=session, run_key="run-1", policy=SimpleNamespace(mode="ask"), in_helper=False,
                        vision=True, asked=[], emitted=[], ask=None, question=None, emit=None)
    c.ask = over.pop("ask", lambda card: c.asked.append(card) or "allow")
    c.question = over.pop("question", lambda qs: {"answers": None})
    c.emit = over.pop("emit", c.emitted.append)
    for k, v in over.items():
        setattr(c, k, v)
    return c


class Jobs(unittest.TestCase):
    """Task 2: jobs, their delivery (seq / ack) and the groups, without HTTP."""

    def setUp(self):
        self.hub = runner_hub.RunnerHub()
        self.addCleanup(self.hub.close)

    def hello(self, groups=None, instance="inst-aaaa"):
        return self.hub.hello("ann", "device-0001", instance, dict(INFO), groups or [CODING], [])

    def test_messages_are_sent_again_until_acked(self):
        r = self.hello()
        r.push({"job": "j1", "answer": "allow"})
        r.push({"job": "j1", "cancel": True})
        first = r.take(0.0)
        self.assertEqual([m["seq"] for m in first], [1, 2])
        self.assertEqual(r.take(0.0), first)                    # the answer was lost on the way: sent again
        self.assertEqual(r.take(0.0, ack=1), first[1:])         # what the runner confirmed leaves the outbox
        t0 = time.monotonic()
        self.assertEqual(r.take(0.3, ack=2), [])                # all confirmed: the poll waits again
        self.assertGreaterEqual(time.monotonic() - t0, 0.25)
        r.push({"job": "j1", "answer": "deny"})
        self.assertEqual(r.take(0.0, ack=99), [{"job": "j1", "answer": "deny", "seq": 3}])   # an ack from another channel
        self.assertEqual(r.take(0.0, ack=2), [{"job": "j1", "answer": "deny", "seq": 3}])

    def test_channel_names_the_registration(self):
        r = self.hello()
        self.assertRegex(r.channel, r"^[0-9a-f]{16}$")
        self.assertEqual(self.hello().channel, r.channel)       # the same process says hello again: same channel
        new = self.hello(instance="inst-bbbb")                  # a new registration: seq starts again, so a new channel
        self.assertNotEqual(new.channel, r.channel)
        new.push({"job": "x"})
        self.assertEqual(new.take(0.0)[0]["seq"], 1)

    def test_submit_and_withdraw_before_pickup(self):
        r = self.hello()
        job = r.submit("tool", {"tool": "Read"})
        self.assertIs(r.jobs[job.id], job)
        self.assertIsNone(job.t_picked)
        self.assertIsInstance(job.t_queued, float)
        self.assertTrue(r.withdraw(job))                        # never handed out: taken back
        self.assertEqual(r.take(0.0), [])
        self.assertNotIn(job.id, r.jobs)
        job2 = r.submit("tool", {"tool": "Read"})
        self.assertEqual(r.take(0.0)[0]["job"], job2.id)
        self.assertIsNotNone(job2.t_picked)
        self.assertFalse(r.withdraw(job2))                      # a poll took it: the runner may be running it
        self.assertIn(job2.id, r.jobs)

    def test_finished_job_keeps_only_its_cancel(self):
        r = self.hello()
        job = r.submit("tool", {"tool": "Bash"})
        r.take(0.0)
        r.push({"job": job.id, "answer": "allow"})
        r.push({"job": job.id, "cancel": True})
        r.finish(job)
        self.assertEqual(r.take(0.0), [{"job": job.id, "cancel": True, "seq": 3}])   # a lost cancel is sent again
        self.assertFalse(r.deliver({"job": job.id, "result": {}}))

    def test_submit_to_an_ended_runner_fails_at_once(self):
        r = self.hello()
        self.hub.end_all("ann")
        job = r.submit("tool", {"tool": "Read"})
        self.assertEqual(job.inbox.get_nowait(), {"job": job.id, "failed": runner_hub.REVOKED})
        self.assertEqual(r.jobs, {})

    def test_request_result_offline_and_timeout(self):
        r = self.hello()

        def answer():
            msg = r.take(2.0)[0]
            r.deliver({"job": msg["job"], "result": {"folders": ["C:\\work"]}})
        t = threading.Thread(target=answer)
        t.start()
        self.assertEqual(r.request("page", {"route": "folders"}), {"folders": ["C:\\work"]})
        t.join()
        got = r.request("page", {"route": "folders"}, timeout=0.3)
        self.assertEqual(got["error"], "timeout")
        self.assertIn("ann-pc", got["message"])
        self.assertEqual(r.jobs, {})
        with mock.patch.object(runner_hub, "OFFLINE_S", 0.0):
            got = r.request("page", {"route": "folders"}, timeout=5)
        self.assertEqual(got["error"], "offline")
        self.assertEqual(r.take(0.0, ack=99), [])               # the withdrawn page job is never handed out

    def test_servers_are_named_for_the_server_side(self):
        groups = [CODING, {"name": "skills", "kind": "skills", "status": "ready", "error": None, "tools": []},
                  {"name": "github", "kind": "mcp", "status": "ready", "error": None, "tools": []},
                  {"name": "agent", "kind": "mcp", "status": "failed", "error": "no npx", "tools": []},
                  dict(CODING, name="second")]                  # a second coding group is not offered
        r = self.hello(groups)
        s = r.servers()
        self.assertEqual(list(s), ["my_agent", "my_skills", "my_github", "my_agent_2"])
        coding, mcp = s["my_agent"], s["my_github"]
        self.assertEqual((coding.name, coding.kind, coding.machine, coding.local), ("my_agent", "runner", "my", "agent"))
        self.assertTrue(coding.wants_progress and coding.wants_context and coding.plain_names and coding.hidden)
        self.assertTrue(mcp.wants_progress)
        self.assertFalse(getattr(mcp, "wants_context", False) or getattr(mcp, "plain_names", False)
                         or getattr(mcp, "hidden", False))
        self.assertEqual((s["my_agent_2"].status, s["my_agent_2"].error), ("failed", "no npx"))
        self.assertEqual(coding.tools[0]["name"], "Read")
        coding.start()
        coding.close()

    def test_revoke_fails_the_call_with_its_text(self):
        r = self.hello()
        group = r.servers()["my_agent"]
        out = []
        t = threading.Thread(target=lambda: out.append(group.call("Read", {}, 0, threading.Event(), ctx())))
        t.start()
        self.assertEqual(r.take(2.0)[0]["tool"], "Read")
        t0 = time.monotonic()
        self.hub.end_all("ann")
        t.join(2.0)
        self.assertLess(time.monotonic() - t0, 1.0)
        self.assertEqual(out[0], {"content": [{"type": "text", "text": "the key of this chat was revoked, so Read was "
                                                                        "not finished."}], "isError": True})

    def test_spans_written(self):
        import json
        import tempfile
        from serve import timeline
        path = str(Path(tempfile.mkdtemp()) / "tl.json")
        timeline.configure(path)
        self.addCleanup(timeline.configure, None)
        r = self.hello()
        group = r.servers()["my_agent"]
        out = []
        t = threading.Thread(target=lambda: out.append(group.call("Read", {}, 0, threading.Event(), ctx())))
        t.start()
        msg = r.take(2.0)[0]
        time.sleep(0.05)
        r.deliver({"job": msg["job"], "result": {"content": [{"type": "text", "text": "12345"}]}, "exec_ms": 7})
        t.join(2.0)
        self.assertEqual(out[0]["content"][0]["text"], "12345")
        timeline.configure(None)
        recs = [json.loads(line.rstrip().rstrip(",")) for line in open(path, encoding="utf-8") if line.startswith("{")]
        spans = [x for x in recs if x.get("ph") == "X"]
        self.assertEqual(sorted(x["name"] for x in spans), ["runner.call", "runner.call wait"])
        call = next(x for x in spans if x["name"] == "runner.call")
        wait = next(x for x in spans if x["name"] == "runner.call wait")
        self.assertEqual(call["args"], {"a": 7, "b": 5})        # exec_ms, characters
        self.assertEqual(call["ts"], wait["ts"])                # both start when the job was queued
        self.assertLess(wait["dur"], call["dur"])               # picked up before the result came
        self.assertGreaterEqual(call["dur"], 40_000)            # us: the 50 ms between pick-up and result


class HubProgress(unittest.TestCase):
    """McpHub.call's `progress`: only a server that asks for it gets it."""

    class Plain:
        name, kind, status, error, info = "plain", "builtin", "ready", None, {}
        tools = [{"name": "echo", "description": "", "inputSchema": {"type": "object"}}]

        def call(self, tool, arguments, timeout, cancel):          # no progress parameter: passing one would raise
            return {"content": [{"type": "text", "text": f"echo {arguments.get('x')}"}], "isError": False}

        def start(self):
            pass

        def close(self):
            pass

    class Wants(Plain):
        name, wants_progress = "wants", True
        tools = [{"name": "say", "description": "", "inputSchema": {"type": "object"}}]

        def call(self, tool, arguments, timeout, cancel, progress=None):
            progress("running")
            return {"content": [{"type": "text", "text": "said"}], "isError": False}

    def test_hub_call_without_progress_unchanged(self):
        from serve.mcp import McpHub
        hub = McpHub({}, builtins={"plain": self.Plain(), "wants": self.Wants()})
        before = hub.call("plain__echo", {"x": 1})
        seen = []
        after = hub.call("plain__echo", {"x": 1}, None, None, progress=seen.append)
        before.pop("ms"), after.pop("ms")
        self.assertEqual(before, after)
        self.assertEqual(before, {"server": "plain", "tool": "echo", "ok": True, "text": "echo 1", "chars": 6,
                                  "truncated": False})
        self.assertEqual(seen, [])
        self.assertEqual(hub.call("wants__say", {}, progress=seen.append)["text"], "said")
        self.assertEqual(seen, ["running"])


if __name__ == "__main__":
    unittest.main()
