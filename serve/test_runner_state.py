"""serve/test_runner_state.py - the runner's hooks for Strata Desktop (#310 part 3): the device label the Client sends
(Task 2), the states RunnerLoop reports to `on_state` (Task 3, a scripted fake client: no HTTP) and the Client's
`abort_poll` (Task 3, a socket server on port 0 that holds the poll).

    python -m pytest serve/test_runner_state.py -q
"""
from __future__ import annotations

import socket
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import runner as runner_mod  # noqa: E402

URL = "http://127.0.0.1:9"
DEV = {"id": "x" * 22, "name": "pc"}
KEY = "k-SECRET-0123"


class FakeClient:
    """The parts of runner.Client that RunnerLoop uses, answering from a script: each item is returned, or raised when it
    is an exception.  `next` past the end of its script sets `stop` and returns no messages."""

    url = "http://srv.example"

    def __init__(self, hellos, nexts, stop):
        self.hellos, self.nexts, self.stop = list(hellos), list(nexts), stop
        self.channel, self.ack, self.handled = None, 0, set()

    def scrub(self, text):
        return text.replace(KEY, "***")

    def on_hello(self, body):
        return False

    @staticmethod
    def _take(items):
        x = items.pop(0)
        if isinstance(x, Exception):
            raise x
        return x

    def hello(self, info):
        return self._take(self.hellos)

    def next(self, rid, instance, wait):
        if not self.nexts:
            self.stop.set()
            return []
        return self._take(self.nexts)


class FakeExecutor:
    def groups(self):
        return []

    def skills(self):
        return []

    def info(self):
        return {"shell": "bash"}

    def starting(self):
        return False


ONLINE = {"runner": "r1", "who": "owner", "channel": "c1"}


def script():
    """Pending, then a hello, a poll that fails, one that works, then the key refused (the brief's order)."""
    return [runner_mod.Pending("K7QZ"), ONLINE], [runner_mod.NetworkError("down " + KEY), [], runner_mod.Refused("key", "")]


class States(unittest.TestCase):
    def setUp(self):
        self.said: list[str] = []
        for p in (mock.patch.object(runner_mod, "_say", self.said.append),
                  mock.patch.object(runner_mod, "PENDING_RETRY_S", 0.01),
                  mock.patch.object(runner_mod, "RETRY_MAX_S", 0.01)):
            p.start()
            self.addCleanup(p.stop)

    def run_loop(self, hellos, nexts, **kw):
        stop = threading.Event()
        loop = runner_mod.RunnerLoop(FakeClient(hellos, nexts, stop), FakeExecutor(), stop=stop, **kw)
        return loop.run()

    def test_states_in_order(self):
        seen = []
        code = self.run_loop(*script(), on_state=lambda ev, data: seen.append((ev, data)))
        self.assertEqual(code, runner_mod.EXIT_REFUSED)
        self.assertEqual(seen, [("pending", {"code": "K7QZ"}),
                                ("online", {"id": "r1", "who": "owner"}),
                                ("offline", {"message": "down ***"}),           # scrubbed: the key never shows
                                ("online", {"id": "r1", "who": "owner"}),
                                ("refused", {"kind": "key", "message": ""})])

    def test_pending_after_offline_reports_pending_again(self):
        """pending -> offline -> pending with the same code: the tray must leave "offline" (the code is printed once)."""
        seen = []
        hellos = [runner_mod.Pending("K7QZ"), runner_mod.NetworkError("down"), runner_mod.Pending("K7QZ"), ONLINE]
        code = self.run_loop(hellos, [runner_mod.Refused("device_refused", "no " + KEY)],
                             on_state=lambda ev, data: seen.append((ev, data)))
        self.assertEqual(code, runner_mod.EXIT_REFUSED)
        self.assertEqual([ev for ev, _ in seen], ["pending", "offline", "pending", "online", "refused"])
        self.assertEqual(seen[-1], ("refused", {"kind": "device_refused", "message": "no ***"}))
        self.assertEqual(sum("K7QZ" in s for s in self.said), 1, self.said)

    def test_stop_reports_stopped(self):
        seen = []
        code = self.run_loop([ONLINE], [[]], on_state=lambda ev, data: seen.append((ev, data)))
        self.assertEqual(code, 0)
        self.assertEqual(seen, [("online", {"id": "r1", "who": "owner"}), ("stopped", {})])

    def test_callback_error_is_ignored(self):
        calls = []

        def broken(ev, data):
            calls.append(ev)
            raise RuntimeError("tray gone " + KEY)
        code = self.run_loop(*script(), on_state=broken)
        self.assertEqual(code, runner_mod.EXIT_REFUSED)                  # the loop went on to the end of the script
        self.assertEqual(calls, ["pending", "online", "offline", "online", "refused"])
        self.assertTrue(any("RuntimeError" in s for s in self.said), self.said)          # logged
        self.assertFalse(any(KEY in s for s in self.said), self.said)                   # with no text of the error

    def test_no_callback_unchanged(self):
        """The lines of this script as `python -m serve.runner` printed them before on_state existed (recorded from
        the code at 9e2a8104)."""
        before = [
            "[runner] this PC waits for the owner's approval: the owner sees it with the code K7QZ. Trying again every "
            "0.01 s.",
            "[runner] connected to http://srv.example as owner; this PC's runner id is r1 - paste it in the web app "
            "under Settings > This browser's runner",
            "[runner] the server cannot be reached (down ***); trying again",
            "[runner] the server answers again",
            "[runner] stopped: the key was refused (wrong or revoked)",
        ]
        self.assertEqual(self.run_loop(*script(), on_state=None), runner_mod.EXIT_REFUSED)
        self.assertEqual(self.said, before)
        self.said.clear()
        self.assertEqual(self.run_loop(*script()), runner_mod.EXIT_REFUSED)
        self.assertEqual(self.said, before)


class HeldPoll:
    """A socket server on 127.0.0.1:0 that reads a request and never answers: a held GET /runner/next."""

    def __init__(self):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen()
        self.port = self.sock.getsockname()[1]
        self.conns: list[socket.socket] = []
        self.got = threading.Event()
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                c, _ = self.sock.accept()
                self.conns.append(c)
                if c.recv(65536):
                    self.got.set()
            except OSError:
                return

    def close(self):
        for c in self.conns:
            c.close()
        self.sock.close()


class AbortPoll(unittest.TestCase):
    """Strata Desktop's stop must not wait out a held poll (25 s): `abort_poll` ends it now, as an empty poll."""

    def setUp(self):
        self.srv = HeldPoll()
        self.addCleanup(self.srv.close)
        self.client = runner_mod.Client(f"http://127.0.0.1:{self.srv.port}", KEY, DEV)

    def test_abort_ends_a_held_poll_at_once(self):
        got = []
        t = threading.Thread(target=lambda: got.append(self.client.next("r1", "inst-aaaa", 25.0)), daemon=True)
        t.start()
        self.assertTrue(self.srv.got.wait(5))                   # the poll is held by the server
        time.sleep(0.2)
        t0 = time.monotonic()
        self.client.abort_poll()
        t.join(3)
        self.assertFalse(t.is_alive())
        self.assertLess(time.monotonic() - t0, 1)
        self.assertEqual(got, [[]])                             # an empty poll, not a network error

    def test_polls_after_abort_end_at_once(self):
        self.client.abort_poll()
        t0 = time.monotonic()
        self.assertEqual(self.client.next("r1", "inst-aaaa", 25.0), [])
        self.assertLess(time.monotonic() - t0, 1)
        time.sleep(0.2)
        self.assertFalse(self.srv.got.is_set())                 # nothing was sent

    def test_abort_leaves_replies_alone(self):
        """Cancelled jobs still post their results: only the poll is cut."""
        self.client.abort_poll()
        with mock.patch.object(runner_mod.Client, "_request", return_value=(200, {"ok": True})) as req:
            self.client.reply("r1", "inst-aaaa", {"job": "j", "rid": 1, "result": {}})
        self.assertEqual(req.call_count, 1)

    def test_abort_while_connecting(self):
        """An abort that lands after the poll was registered but before its socket connected: the poll still ends."""
        real = runner_mod.http.client.HTTPConnection.connect

        def connect(conn):
            self.client.abort_poll()                           # lands in the middle of the connect
            real(conn)
        with mock.patch.object(runner_mod.http.client.HTTPConnection, "connect", connect):
            t0 = time.monotonic()
            self.assertEqual(self.client.next("r1", "inst-aaaa", 25.0), [])
        self.assertLess(time.monotonic() - t0, 3)


class Label(unittest.TestCase):
    def test_label_default_unchanged(self):
        c = runner_mod.Client(URL, "k", DEV)
        self.assertEqual(c._headers["X-Strata-Device-Name"], "Strata runner on pc")
        self.assertEqual(c._headers["X-Strata-Device"], "x" * 22)

    def test_label_given(self):
        c = runner_mod.Client(URL, "k", DEV, label="Strata Desktop on pc")
        self.assertEqual(c._headers["X-Strata-Device-Name"], "Strata Desktop on pc")
        self.assertEqual(c._headers["X-Strata-Device"], "x" * 22)

    def test_device_label(self):
        self.assertEqual(runner_mod.device_label(DEV), "Strata runner on pc")
        self.assertEqual(runner_mod.device_label(DEV, "Strata Desktop"), "Strata Desktop on pc")
        self.assertEqual(runner_mod.device_label({"id": "x" * 22, "name": "เครื่อง-1"}, "Strata Desktop"),
                         "Strata Desktop on " + "?" * len("เครื่อง") + "-1")         # printable ASCII only: what a header can carry
        self.assertEqual(runner_mod.device_label({"id": "x" * 22}, "A"), "A on " + runner_mod.machine_name())

    def test_label_is_made_header_safe(self):
        c = runner_mod.Client(URL, "k", DEV, label="Strata Desktop on เครื่อง")
        self.assertEqual(c._headers["X-Strata-Device-Name"], "Strata Desktop on " + "?" * len("เครื่อง"))


if __name__ == "__main__":
    unittest.main()
