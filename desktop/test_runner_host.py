"""desktop/test_runner_host.py - the runner in a thread of the app (#310 part 3 task 3): it connects to a real server,
and `stop` ends its threads, its jobs and their processes, and closes the executor once, within its bound.

The server is serve.server.serve(svc, port=0) with the mock engine and an api_key, outside team mode (as
serve/test_runner_http.py's OutsideTeamMode builds it).  It needs jinja2 (serve.frontend): the global python has it,
desktop\\.venv does not, so those tests are skipped there.  The poll is held for the server's real POLL_MAX_S (25 s), so
the stop tests show that stop does not wait one out.

    python -m pytest desktop/test_runner_host.py -q
"""
from __future__ import annotations

import ctypes
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import pytest

from desktop.runner_host import RunnerHost
from serve import runner as runner_mod, runner_hub

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "owner-key-SECRET-0123"
DEVICE = {"id": "AbCdEfGhIjKlMnOpQrStUv", "name": "ann-pc"}       # 22 characters, as load_device makes
LABEL = runner_mod.device_label(DEVICE, "Strata Desktop")
THREADS = ("desktop-runner", "runner-poll", "runner-job")
FAKE_MCP = str(ROOT / "serve" / "mcp_fake_server.py")


def wait_for(cond, timeout=10.0, step=0.05):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        got = cond()
        if got:
            return got
        time.sleep(step)
    return cond()


def runner_threads() -> list[str]:
    return [t.name for t in threading.enumerate() if t.name in THREADS and t.is_alive()]


def alive(pid: int) -> bool:
    """Whether process `pid` still runs (Windows: its exit code is STILL_ACTIVE)."""
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = ctypes.c_void_p
    k32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    k32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    k32.CloseHandle.argtypes = [ctypes.c_void_p]
    h = k32.OpenProcess(0x1000, False, pid)                     # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    try:
        code = ctypes.c_uint32()
        return bool(k32.GetExitCodeProcess(h, ctypes.byref(code))) and code.value == 259
    finally:
        k32.CloseHandle(h)


class FakeExecutor:
    """groups / skills / info / starting for the hello; `handle` holds a job until it is cancelled; `close` counted."""

    def __init__(self, shell="Git Bash"):
        self.closed = 0
        self.shell = shell
        self.handling = threading.Event()

    def groups(self):
        return []

    def skills(self):
        return []

    def info(self):
        return runner_mod.fit_info({"machine": "ann-pc", "platform": "win32", "shell": self.shell, "home": "C:\\h"})

    def starting(self):
        return False

    def handle(self, msg, link):
        self.handling.set()
        link.cancel.wait(30)
        return runner_mod._err("cancelled")

    def close(self):
        self.closed += 1


class SlowJobExecutor(FakeExecutor):
    """A job that goes on 0.3 s after its cancel; `close` records which job threads were still alive."""

    def __init__(self):
        super().__init__()
        self.handled = threading.Event()
        self.in_handle = False
        self.closed_during_handle = False
        self.jobs_at_close: list[str] | None = None

    def handle(self, msg, link):
        self.in_handle = True
        self.handling.set()
        link.cancel.wait(30)
        time.sleep(0.3)                                         # ignores the cancel for a while
        self.in_handle = False
        self.handled.set()
        return runner_mod._err("cancelled")

    def close(self):
        self.closed_during_handle = self.closed_during_handle or self.in_handle
        self.jobs_at_close = [t.name for t in threading.enumerate() if t.name == "runner-job" and t.is_alive()]
        super().close()


class Base(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._cleanup, tmp)
        self.dir = Path(tmp.name)
        self.folder = self.dir / "runner"
        self.folder.mkdir()
        (self.dir / "home").mkdir()
        self.said: list[str] = []
        for p in (mock.patch.object(runner_mod, "_say", self.said.append),
                  mock.patch.dict(os.environ, {"STRATA_HOME": str(self.dir / "home")})):
            p.start()
            self.addCleanup(p.stop)
        self.hosts: list[RunnerHost] = []
        self.url = "http://127.0.0.1:9"                         # nothing listens: tests without a server

    @staticmethod
    def _cleanup(tmp):
        try:
            tmp.cleanup()
        except OSError:                                         # a process that just ended may still hold a file
            pass

    def server(self):
        """The real server in-process, outside team mode, with an api_key: this PC's caller is the owner."""
        pytest.importorskip("jinja2")
        from serve.frontend import ChatTemplate
        from serve.identity import KeyStore
        from serve.server import ByteTokenizer, MockEngine, Service, serve

        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "</think>\n\nok", max_context=4096), tok,
                           ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cfg = self.dir / "strata-test.json"
        cfg.write_text("{}", encoding="utf-8")
        self.svc.api_key = TOKEN
        self.svc.config_path = str(cfg)
        self.svc.keys = KeyStore(str(self.dir / "strata-keys.json"), TOKEN)       # no keys file: not team mode
        httpd = serve(self.svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        self.addCleanup(self.svc.runner_hub.close)              # held polls end before the server does
        self.url = f"http://127.0.0.1:{httpd.server_address[1]}"

    def host(self, factory=None, folder=None):
        events: list[tuple[str, dict]] = []
        h = RunnerHost(self.url, TOKEN, DEVICE, LABEL, folder or self.folder,
                       lambda ev, data: events.append((ev, data)), executor_factory=factory)
        h.events = events                                       # type: ignore[attr-defined]
        self.hosts.append(h)
        self.addCleanup(h.stop, 5.0)
        return h

    def kinds(self, h):
        return [ev for ev, _ in h.events]


class Connect(Base):
    def test_connects_to_real_server(self):
        self.server()
        h = self.host()                                         # the real Executor, its files in a temp folder
        h.start()
        self.assertTrue(wait_for(lambda: "online" in self.kinds(h), 10), (h.events, self.said))
        rid = runner_hub.runner_id("owner", DEVICE["id"])
        self.assertEqual((h.runner_id, h.who), (rid, "owner"))
        self.assertEqual(self.kinds(h)[0], "connecting")
        online = dict(h.events)["online"]
        self.assertEqual((online["id"], online["who"]), (rid, "owner"))
        self.assertEqual(online["no_shell"], h.shell is None)
        self.assertEqual(h.shell, runner_mod.shell_mod.describe(runner_mod.shell_mod.find_shell())
                         if runner_mod.shell_mod.find_shell() else None)
        self.assertIn(rid, self.svc.runner_hub._runners)
        self.assertTrue(h.stop(5.0))
        self.assertEqual(runner_threads(), [])
        self.assertFalse(any(TOKEN in s for s in self.said), self.said)

    def test_device_and_label_reach_the_server(self):
        self.server()
        h = self.host(factory=FakeExecutor)
        h.start()
        self.assertTrue(wait_for(lambda: h.runner_id is not None, 10), (h.events, self.said))
        r = self.svc.runner_hub._runners[h.runner_id]
        self.assertEqual((r.device, r.info.get("machine")), (DEVICE["id"], "ann-pc"))
        self.assertEqual(h.runner_id, runner_hub.runner_id("owner", DEVICE["id"]))

    def test_no_shell(self):
        self.server()
        h = self.host(factory=lambda: FakeExecutor(shell=None))
        h.start()
        self.assertTrue(wait_for(lambda: "online" in self.kinds(h), 10), (h.events, self.said))
        self.assertIsNone(h.shell)
        self.assertTrue(dict(h.events)["online"]["no_shell"])


class Stop(Base):
    def test_stop_ends_threads_and_closes_executor(self):
        """RF 3: a job in flight and a poll held by the server (25 s): stop ends both within its bound."""
        self.server()
        ex = FakeExecutor()
        h = self.host(factory=lambda: ex)
        h.start()
        self.assertTrue(wait_for(lambda: "online" in self.kinds(h), 10), (h.events, self.said))
        h._loop._dispatch([{"job": "j1", "kind": "tool", "group": "agent", "tool": "Bash", "arguments": {}}])
        self.assertTrue(ex.handling.wait(5))
        self.assertIn("runner-job", runner_threads())
        time.sleep(0.5)                                         # the poll is held by the server now
        t0 = time.monotonic()
        self.assertTrue(h.stop(5.0))
        self.assertLess(time.monotonic() - t0, 5.0)
        self.assertEqual(runner_threads(), [])
        self.assertEqual(ex.closed, 1)
        n = len(h.events)
        self.assertTrue(h.stop(5.0))                            # twice: nothing more happens
        self.assertEqual(ex.closed, 1)
        time.sleep(0.3)
        self.assertEqual(len(h.events), n)                      # no event after stop returned
        self.assertNotIn("stopped", self.kinds(h))
        self.assertFalse(any("cannot be reached" in s for s in self.said), self.said)

    def test_close_waits_for_a_job_that_ignores_cancel(self):
        """A job that goes on 0.3 s after its cancel: the executor closes only after it ended (finish, then close)."""
        self.server()
        ex = SlowJobExecutor()
        h = self.host(factory=lambda: ex)
        h.start()
        self.assertTrue(wait_for(lambda: "online" in self.kinds(h), 10), (h.events, self.said))
        h._loop._dispatch([{"job": "j1", "kind": "tool", "group": "agent", "tool": "Bash", "arguments": {}}])
        self.assertTrue(ex.handling.wait(5))
        time.sleep(0.3)                                         # the poll is held by the server now
        self.assertTrue(h.stop(5.0))
        self.assertTrue(ex.handled.is_set())
        self.assertEqual(ex.closed, 1)
        self.assertEqual(ex.jobs_at_close, [])                  # no job thread was alive when the executor closed
        self.assertFalse(ex.closed_during_handle)

    def test_job_of_a_poll_answered_just_before_abort_is_waited_for(self):
        """A poll that brought a job just as stop cut the polls: that job is handed out after stop began, and stop
        still waits for its thread before the executor closes."""
        self.server()
        ex = SlowJobExecutor()
        h = self.host(factory=lambda: ex)
        real_next, started = runner_mod.Client.next, []

        def next_(client, rid, instance, wait):
            if not started:                                     # the first poll: answered as the abort comes
                started.append(rid)
                while not client.polls_ended:
                    time.sleep(0.01)
                return [{"job": "j-late", "kind": "tool", "group": "agent", "tool": "Bash", "arguments": {}}]
            return real_next(client, rid, instance, wait)
        real_start, handed_out = runner_mod.RunnerLoop._start, []

        def start_(loop, msg):
            handed_out.append(msg["job"])
            real_start(loop, msg)
        with mock.patch.object(runner_mod.Client, "next", next_), \
                mock.patch.object(runner_mod.RunnerLoop, "_start", start_):
            h.start()
            self.assertTrue(wait_for(lambda: started, 10), (h.events, self.said))
            self.assertTrue(h.stop(5.0))
        self.assertEqual(handed_out, ["j-late"])
        self.assertEqual(ex.closed, 1)
        self.assertEqual(ex.jobs_at_close, [])
        self.assertFalse(ex.closed_during_handle)
        self.assertEqual(runner_threads(), [])

    def test_stop_while_executor_still_building(self):
        """RF 3: the executor is still being built when stop comes: it is closed once the build ends, and no loop
        starts."""
        ex = FakeExecutor()

        def slow():
            time.sleep(1.0)
            return ex
        h = self.host(factory=slow)
        h.start()
        self.assertTrue(wait_for(lambda: "connecting" in self.kinds(h), 2))
        t0 = time.monotonic()
        self.assertTrue(h.stop(5.0))
        self.assertLess(time.monotonic() - t0, 3.0)
        self.assertEqual(ex.closed, 1)
        self.assertIsNone(h._loop)
        self.assertEqual(self.kinds(h), ["connecting"])
        self.assertEqual(runner_threads(), [])

    def test_stop_shorter_than_the_build(self):
        ex = FakeExecutor()

        def slow():
            time.sleep(1.0)
            return ex
        h = self.host(factory=slow)
        h.start()
        self.assertTrue(wait_for(lambda: "connecting" in self.kinds(h), 2))
        self.assertFalse(h.stop(0.2))                           # the thread is still building
        self.assertTrue(wait_for(lambda: ex.closed == 1, 3))    # closed once the build ends
        self.assertTrue(wait_for(lambda: runner_threads() == [], 3))
        self.assertIsNone(h._loop)
        self.assertEqual(ex.closed, 1)

    def test_stop_before_start_and_twice(self):
        h = self.host(factory=FakeExecutor)
        self.assertTrue(h.stop(5.0))
        self.assertTrue(h.stop(5.0))
        self.assertEqual(h.events, [])

    def test_stop_ends_children(self):
        """The real Executor: an MCP server from runner.json, a background command and a command in flight - none of
        their processes is left after stop (#314's job objects, Executor.close)."""
        self.server()
        (self.folder / "runner.json").write_text(json.dumps(
            {"mcp_servers": {"fake": {"command": sys.executable, "args": [FAKE_MCP]}}}), encoding="utf-8")
        h = self.host()
        h.start()
        self.assertTrue(wait_for(lambda: "online" in self.kinds(h), 10), (h.events, self.said))
        ex = h._executor
        mcp = ex.hub.servers["fake"]
        self.assertTrue(wait_for(lambda: getattr(mcp.transport, "proc", None) is not None
                                 and mcp.transport.proc.poll() is None, 20), mcp.status)
        pids = {"mcp": mcp.transport.proc.pid}
        if ex.agent.shell is not None:
            proj = self.dir / "proj"
            proj.mkdir()
            link = runner_mod.JobLink("j0", lambda m: True)
            run = ex.handle({"kind": "run.start", "cwd": str(proj), "dirs": [], "mode": None, "allow": ["Bash"], "deny": [],
                             "session": "chat-1", "checkpoint": None, "memory_on": []}, link)["run"]
            py = sys.executable.replace("\\", "/")

            def sleeper(name):
                return f"\"{py}\" -c \"import os,time; open('{name}','w').write(str(os.getpid())); time.sleep(60)\""

            def tool(job, **args):
                return {"job": job, "kind": "tool", "group": "agent", "tool": "Bash", "arguments": args, "run": run,
                        "session": "chat-1", "mode": None, "helper": False, "vision": False, "timeout_s": 120}
            res = ex.handle(tool("j-bg", command=sleeper("bg.pid"), run_in_background=True), link)
            self.assertFalse(res.get("isError"), res)
            h._loop._dispatch([tool("j-fg", command=sleeper("fg.pid"))])
            for name in ("bg", "fg"):
                f = proj / f"{name}.pid"
                self.assertTrue(wait_for(lambda: f.exists() and f.read_text().strip().isdigit(), 20), name)
                pids[name] = int(f.read_text().strip())
        for name, pid in pids.items():
            self.assertTrue(alive(pid), name)
        t0 = time.monotonic()
        self.assertTrue(h.stop(5.0))
        self.assertLess(time.monotonic() - t0, 5.0)
        self.assertEqual(runner_threads(), [])
        for name, pid in pids.items():
            self.assertTrue(wait_for(lambda: not alive(pid), 5), f"{name} {pid} still runs")


class Errors(Base):
    def test_bad_runner_json_is_error(self):
        (self.folder / "runner.json").write_text("[1]", encoding="utf-8")
        raised = []
        with mock.patch.object(threading, "excepthook", lambda a: raised.append(a)):
            h = self.host()
            h.start()
            self.assertTrue(wait_for(lambda: "error" in self.kinds(h), 10), h.events)
            self.assertTrue(wait_for(lambda: runner_threads() == [], 5))
        self.assertEqual(raised, [])
        self.assertEqual(self.kinds(h), ["connecting", "error"])
        message = dict(h.events)["error"]["message"]
        self.assertIn("must be a JSON object", message)
        self.assertFalse(message.startswith("[runner]"), message)
        self.assertIsNone(h._loop)
        self.assertTrue(h.stop(5.0))

    def test_any_build_error_is_error(self):
        def broken():
            raise OSError("disk gone " + TOKEN)
        h = self.host(factory=broken)
        h.start()
        self.assertTrue(wait_for(lambda: "error" in self.kinds(h), 5), h.events)
        message = dict(h.events)["error"]["message"]
        self.assertIn("OSError", message)
        self.assertNotIn(TOKEN, message)                        # scrubbed, if a build error ever carried it

    def test_callback_error_does_not_end_the_host(self):
        ex = FakeExecutor()

        def factory():
            return ex
        calls = []

        def broken(ev, data):
            calls.append(ev)
            raise RuntimeError("tray gone")
        h = RunnerHost("http://127.0.0.1:9", TOKEN, DEVICE, LABEL, self.folder, broken, executor_factory=factory)
        self.addCleanup(h.stop, 5.0)
        with mock.patch.object(runner_mod, "RETRY_MAX_S", 0.05):
            h.start()
            self.assertTrue(wait_for(lambda: "offline" in calls, 10), (calls, self.said))   # nothing listens on :9
            self.assertIn("desktop-runner", runner_threads())
            self.assertTrue(h.stop(5.0))
        self.assertEqual(ex.closed, 1)

    def test_repr_hides_token(self):
        h = self.host(factory=FakeExecutor)
        self.assertNotIn(TOKEN, repr(h))
        self.assertNotIn(TOKEN, str(h))


class Replaced(Base):
    def test_refused_replaced_is_terminal(self):
        """RF 4: a second host with the same device on the same server takes the first one's place: the first is
        refused (runner_replaced), its thread ends and its executor is closed; the second goes on."""
        self.server()
        first_ex, second_ex = FakeExecutor(), FakeExecutor()
        first = self.host(factory=lambda: first_ex)
        first.start()
        self.assertTrue(wait_for(lambda: "online" in self.kinds(first), 10), (first.events, self.said))
        other = self.dir / "other"
        other.mkdir()
        second = self.host(factory=lambda: second_ex, folder=other)
        second.start()
        self.assertTrue(wait_for(lambda: "online" in self.kinds(second), 10), (second.events, self.said))
        self.assertTrue(wait_for(lambda: "refused" in self.kinds(first), 15), (first.events, self.said))
        self.assertEqual(dict(first.events)["refused"]["kind"], "runner_replaced")
        self.assertTrue(wait_for(lambda: not first._thread.is_alive(), 10))
        self.assertEqual(first_ex.closed, 1)
        self.assertEqual(second_ex.closed, 0)
        self.assertEqual(self.kinds(first)[-1], "refused")      # nothing after: refused ends the host
        self.assertTrue(second._thread.is_alive())
        self.assertTrue(first.stop(5.0))
        self.assertEqual(first_ex.closed, 1)


if __name__ == "__main__":
    unittest.main()
