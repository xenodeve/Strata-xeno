"""serve/test_runner_hub.py - the runner channel's server side without HTTP (#310 part 2, task 1): runner ids,
long-poll, presence, a second instance, the identity a runner belongs to.

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
        self.assertEqual(newer, [[{"job": "j1", "answer": "allow"}]])   # the message reaches the live poll

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
            self.assertEqual(r.take(0.0), [{"job": "j1", "answer": "allow"}])   # the message stayed queued

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


if __name__ == "__main__":
    unittest.main()
