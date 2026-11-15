"""serve/test_runner_http.py - the runner channel over HTTP (#310 part 2, task 1): POST /runner/hello, GET
/runner/next, POST /runner/reply, GET /runners, against the mock engine in team mode and outside it.

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
from serve import runner_hub  # noqa: E402
from serve.identity import OWNER  # noqa: E402
from serve.runner_fake import FakeRunner  # noqa: E402
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

    def reply(self, key, rid, job="j1", instance="inst-aaaa", device=DEV1, **more):
        return self.req("POST", "/runner/reply", {"id": rid, "instance": instance, "job": job, **more},
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
        self.assertEqual((code, body), (200, {"messages": [{"job": "j1", "kind": "tool"}]}))
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
        self.assertEqual(newer[0], (200, {"messages": [{"job": "j1", "answer": "allow"}]}))

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
            self.assertEqual(self.poll(self.ann, rid), (200, {"messages": [{"job": "j1", "answer": "allow"}]}))

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
