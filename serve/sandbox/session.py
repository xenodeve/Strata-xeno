"""serve/sandbox/session.py - one chat's sandbox on one project folder: the copy, search and symbols over it, a box that may write only the copy, the output budget, and what was tested.

Nothing starts until the first sandbox call, so a chat that never uses it costs nothing, and all of it goes with the chat. Code the model wrote runs only inside a box, and only when the run config says so: real Python on the
developer's PC is not approved yet (the design's D1), so `real_python` is off unless it is set, and with it off nothing here starts a box. A throwaway VM (`exec_bridge.throwaway`) stays as it was. A box that cannot start stops the
run and says why, and there is no way round that (I4). Reading and changing the copy need no box: they run in this process.

A box has its limits for its whole life, so a run is held by its own timeout and the box lasts a chat; what a run leaves behind ends with it. The base is tested in a box of its own, so nothing that runs in the copy can reach what a
change is measured against. That run happens once for one state of the project and one command."""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from serve import checkpoints
from serve.sandbox import testrun
from serve.sandbox.exec_bridge import throwaway
from serve.sandbox.isolation import ROOT_ENV, Isolation, IsolationError, Policy, RunResult, _remove_tree, process_gone
from serve.sandbox.output import RESULT_BYTES, TOUCHED, TURN_BYTES, Output, conflict_lines
from serve.sandbox.search import Search
from serve.sandbox.symbols import Symbols
from serve.sandbox.workspace import Workspace, WorkspaceError

BOX_SECONDS = 3600.0                   # a box's whole life, wall clock and CPU, past which the session starts another: a run is held by the two limits below
RUN_SECONDS = 120.0                    # one `sandbox_run`
TEST_SECONDS = 900.0                   # one run of the tests
TEST_WAIT = 90.0                       # a call waits this long, and past it answers that the tests are still running (design case 24)
BOX_MEMORY = 2 << 30                   # the box's whole job (design case 51)
KEEP_SECONDS = 86400.0                 # a copy a dead server left is removed after this long (design case 36)
MAX_LOG = 16 * 1024 * 1024             # bytes of one chat's record: past it the diffs are left out and the commands still go in
MAX_LOGGED_DIFF = 256 * 1024
PYTEST_MARKS = ("pytest.ini", "tox.ini", "conftest.py", "setup.cfg", "pyproject.toml")    # a file that says the folder has pytest tests
STOPPED_BY = {"timeout": "เกินเวลา", "time": "box หมดเวลา", "output": "พิมพ์เกินเพดาน", "disk": "ใช้ดิสก์เกินเพดาน", "stop": "ผู้ใช้กดหยุด", "kill": "ปิดแล้ว"}
NO_REAL_PYTHON = ("real Python is not approved on this PC (design D1), so sandbox_run and sandbox_test are off. Reading, searching, editing and the diff still work on the copy. "
                  "The switch is \"sandbox\": {\"real_python\": true} in the run config, and it is the developer's to set.")


class Refused(Exception):
    """A clear no: the text goes to the model as the result."""


def _number(value, default, whole: bool = True):
    good = isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0
    return (int(value) if whole else float(value)) if good else default


def settings(cfg) -> dict:
    """The run config's `sandbox`. Off unless it says so; anything odd is ignored. `on`: the route exists. `real_python`: code may run in a box (D1). The rest have their defaults."""
    s = cfg.get("sandbox") if isinstance(cfg, dict) and isinstance(cfg.get("sandbox"), dict) else {}
    command = s.get("test_command")
    return {"on": s.get("on") is True, "real_python": s.get("real_python") is True, "test_command": command if isinstance(command, str) and command.strip() else None,
            "result_bytes": _number(s.get("result_bytes"), RESULT_BYTES), "turn_bytes": _number(s.get("turn_bytes"), TURN_BYTES),
            "run_seconds": _number(s.get("run_seconds"), RUN_SECONDS, False), "test_seconds": _number(s.get("test_seconds"), TEST_SECONDS, False), "test_wait_seconds": _number(s.get("test_wait_seconds"), TEST_WAIT, False)}


def scratch_root(root=None) -> Path:
    """Where the copies go. On Windows that has to be under the box's scratch root, the one tree whose ACLs the box may change."""
    if os.name == "nt":
        from serve.sandbox.isolation import Scratch
        return Scratch(root).make("workspaces")
    return Path(root or os.environ.get(ROOT_ENV) or os.path.join(tempfile.gettempdir(), "strata-sandbox"))


def sweep(root=None, keep: float = KEEP_SECONDS) -> None:
    """Removes the copies a server that ended without closing them left behind (design case 36). The state that would resume one lives in that server's memory, so a copy like that can only go.
    A copy is named with its server's pid, and only a dead server's go: every server on this PC shares the folder, and a copy a live chat uses can sit untouched for days, since an edit inside it moves no date this
    can see. A name with no pid, from before that, goes by its age."""
    try:
        for entry in os.scandir(scratch_root(root)):
            parts = entry.name.split("-", 2)
            if parts[0] != "ws":
                continue
            if len(parts) == 3 and parts[1].isdigit():
                if int(parts[1]) != os.getpid() and process_gone(int(parts[1])):
                    _remove_tree(entry.path)
                continue
            newest = max((os.path.getmtime(p) for p in (entry.path, os.path.join(entry.path, "copy"), os.path.join(entry.path, "base.pack")) if os.path.exists(p)), default=0.0)
            if time.time() - newest > keep:
                _remove_tree(entry.path)
    except (OSError, IsolationError):
        pass


def log_folder() -> Path:
    return checkpoints.default_dir().parent / "sandbox-log"


def _log_file(log_dir, chat: str) -> Path:
    return Path(log_dir) / (("".join(c for c in chat if c.isalnum() or c in "-_")[:80] or "default") + ".jsonl")


def forget_log(log_dir, chat: str) -> None:
    try:
        _log_file(log_dir, chat).unlink(missing_ok=True)
    except OSError:
        pass


@dataclass
class TestDone:
    """One finished `sandbox_test`: `lines` are the result's body."""
    exit_code: int | None
    seconds: float
    lines: list[str]
    verdict: str


class SandboxSession:
    def __init__(self, chat: str, project: str, conf: dict, root=None, log_dir=None, start=Isolation.start):
        self.chat, self.project, self.conf, self.root, self.log_dir = chat, project, conf, root, log_dir
        self.start = start                     # how a box is made
        self.lock = threading.RLock()          # one thing at a time on the workspace; a test run holds it only between its runs
        self.ws: Workspace | None = None
        self.search: Search | None = None
        self.symbols: Symbols | None = None
        self.out: Output | None = None
        self.command: str | None = None        # the last test command, used when a call names none
        self.closed = False
        self._box = None
        self._turn: tuple | None = None
        self._base_runs: dict[tuple, testrun.Outcome | None] = {}             # (command, base id) -> the run on the base
        self._pool: ThreadPoolExecutor | None = None
        self._test = None                      # the test run in progress or not yet told, a Future
        self._say: Callable | None = None      # how the running test tells the page its step, while a call is waiting for it
        self._cancel: threading.Event | None = None

    # -------------------------------------------------------------------------------------------- start and end
    def open(self, emit=None) -> None:
        with self.lock:
            if self.closed:
                raise Refused("this sandbox was closed with its chat")
            if self.ws is not None:
                return
            if emit:
                emit({"event": "sandbox", "step": "copying"})
            try:
                self.ws = Workspace(self.project, scratch_root(self.root))
            except (WorkspaceError, IsolationError, OSError) as e:
                raise Refused(f"the sandbox could not start: {e}") from e
            self.search = Search(self.ws)
            self.symbols = Symbols(self.ws, self.search)
            self.out = Output(self.ws, self.conf["result_bytes"], self.conf["turn_bytes"])

    def begin(self, ctx) -> None:
        """A call of the model's. A new round of its tool calls is a new turn for the output budget."""
        with self.lock:
            if self._turn is None or self._turn[0] is not ctx or self._turn[1] != ctx.round:
                self.out.new_turn()
            self._turn = (ctx, ctx.round)

    def close(self) -> None:
        """Ends what runs and removes the copy. Whatever is still running finds the workspace closed."""
        with self.lock:
            self.closed = True
            box, self._box = self._box, None
            ws, pool = self.ws, self._pool
            self._pool = None
        if box is not None:
            box.kill()
        if pool is not None:
            pool.shutdown(wait=False)
        if ws is not None:
            ws.close()

    def busy(self) -> bool:
        return self._test is not None and not self._test.done()

    # -------------------------------------------------------------------------------------------- freshness and the result's text
    def fresh(self) -> list[str]:
        """Brings the copy up to the real project before a command (design 4.4, cases 32-34), and says what the model has to know of it. Not while tests run: the tree under them must stand still."""
        if self.busy():
            return []
        with self.lock:
            try:
                report = self.ws.sync()
            except WorkspaceError as e:
                raise Refused(f"the real project could not be read: {e}") from e
        lines = []
        if report["head"]:
            old, new = (report["head"]["from"] or "none")[:10], (report["head"]["to"] or "none")[:10]
            lines.append(f"[branch หรือ commit ของจริงเปลี่ยน: {old} -> {new} การแก้ในสำเนายังอยู่]")
        if report["conflicts"]:
            lines.append(conflict_lines(report["conflicts"]))
        lines += [f"[ตามของจริงไม่ได้: {p}]" for p in report["skipped"]]
        return lines

    def render(self, body, exit_code=None, seconds=None, touched: bool = False) -> str:
        with self.lock:
            return self.out.render(body, exit_code, seconds, len(self.ws.changed()) if touched else None)

    def summary(self) -> str:
        """Where this chat's sandbox stands, for after the context was squeezed (design case 39, Q6)."""
        with self.lock:
            rows = self.ws.changed()
            last = self.ws.tested[-1] if self.ws.tested else None
        lines = [TOUCHED.format(n=len(rows)) if rows else "ไม่มีการแก้ในสำเนา"]
        if last:
            lines.append(f"test ล่าสุด: {last['command']} -> {last['summary']}")
        return "\n".join(lines)

    # -------------------------------------------------------------------------------------------- the box
    def may_run(self) -> None:
        if not (self.conf["real_python"] or throwaway()):
            raise Refused(NO_REAL_PYTHON)

    def _policy(self, folder: str) -> Policy:
        return Policy(readwrite=[folder], denied=[self.ws.project], memory_bytes=BOX_MEMORY, seconds=BOX_SECONDS, disk_bytes=2 * self.ws.quota)

    def _confirm(self, box, folder: str) -> None:
        """The short check at the start of a box (design case 49, I10): a file of the real project has to be out of its reach, or nothing runs."""
        known = next(iter(self.ws.base), None)
        if known is None or not box.isolated:                                 # nothing to try it on, or a throwaway VM with no wall to confirm
            return
        program = f"import sys\ntry:\n    open({os.path.join(self.ws.project, *known.split('/'))!r}, 'rb').close()\nexcept OSError:\n    sys.exit(0)\nsys.exit(3)\n"
        try:
            seen = box.run(["python", "-c", program], timeout=30, cwd=folder)
        except IsolationError as e:
            raise Refused(f"the box could not run its first check, so nothing ran (design I4): {e}") from e
        if seen.exit_code != 0:
            raise Refused(f"the box's first check did not hold (exit {seen.exit_code}), so nothing ran (design I10)")

    def _start_box(self, folder: str):
        try:
            box = self.start(self._policy(folder), root=self.root)
        except IsolationError as e:
            raise Refused(f"the box could not start, so nothing ran (design I4): {e}") from e
        try:
            self._confirm(box, folder)
        except Refused:
            box.kill()
            raise
        return box

    def _copy_box(self):
        with self.lock:
            if self._box is None:
                self._box = self._start_box(self.ws.copy)
            return self._box

    def _in_box(self, box, argv, stdin, timeout: float, folder: str, stopped: Callable[[], bool]) -> RunResult:
        """One run, ended by the user's Stop as well as by its time. What it leaves behind ends with it (case 30): a process left to change the copy would change what the next run measures."""
        done = threading.Event()

        def watch() -> None:
            while not done.wait(0.1):
                if stopped():
                    box.stop()
                    return

        threading.Thread(target=watch, daemon=True, name="sandbox-stop").start()
        try:
            return box.run(argv, stdin, timeout, cwd=folder)
        finally:
            done.set()
            box.stop()

    def run_in_copy(self, argv: list[str], stdin, timeout: float, stopped: Callable[[], bool]) -> RunResult:
        """A run in the copy. A box past its time or stopped by its limits is replaced once, as can happen at any point of a long chat."""
        for last in (False, True):
            box = self._copy_box()
            try:
                return self._in_box(box, argv, stdin, timeout, self.ws.copy, stopped)
            except IsolationError as e:
                with self.lock:
                    if self._box is box:
                        self._box = None
                box.kill()
                if last:
                    raise Refused(f"the box stopped, so nothing ran (design I4): {e}") from e
        raise AssertionError("unreachable")

    # -------------------------------------------------------------------------------------------- tests
    def default_command(self) -> str | None:
        """The command for a call that names none: the last one, the run config's, or pytest when the project shows it has tests of that kind."""
        if self.command or self.conf["test_command"]:
            return self.command or self.conf["test_command"]
        paths = self.ws.base
        if any(name in paths for name in PYTEST_MARKS) or any(p.startswith(("tests/", "test/")) or p.rpartition("/")[2].startswith("test_") for p in paths):
            return "python -m pytest -q"
        return None

    def _tell(self, step: str) -> None:
        say = self._say
        if say:
            say({"event": "sandbox", "step": step})

    def _stopped(self) -> bool:
        cancel = self._cancel
        return cancel is not None and cancel.is_set()

    @staticmethod
    def _outcome(kind: str, result: RunResult) -> testrun.Outcome:
        text = result.stdout.decode("utf-8", "replace") + result.stderr.decode("utf-8", "replace")
        return testrun.parse(kind, result.exit_code, text, STOPPED_BY.get(result.killed, result.killed) if result.killed else None)

    def _on_base(self, argv: list[str], kind: str, timeout: float) -> testrun.Outcome | None:
        """The same command on the base, in a box of its own that sees nothing else. None: it could not run there, which the result says."""
        target = os.path.join(self.ws.root, "base")
        box = None
        try:
            os.makedirs(target, exist_ok=True)
            box = self._start_box(target)                                     # before the files, which take the grant from the folder: nothing then has to walk them to give it
            with self.lock:
                self.ws.export_base(target)
            return self._outcome(kind, self._in_box(box, argv, None, timeout, target, self._stopped))
        except (Refused, IsolationError, WorkspaceError, OSError):
            return None
        finally:
            if box is not None:
                box.kill()
            _remove_tree(target)

    def _run_tests(self, command: str, argv: list[str], kind: str, simulated: str | None) -> TestDone:
        began = time.monotonic()
        timeout = self.conf["test_seconds"]
        self._tell("testing")
        with self.lock:
            key = (tuple(argv), self.ws.base_id())
            nothing_changed = not self.ws.changed()
        now = self._outcome(kind, self.run_in_copy(argv, None, timeout, self._stopped))
        if nothing_changed:                                                   # the copy is the base, and this was its run
            self._base_runs[key] = now
        if key not in self._base_runs and not self._stopped():
            self._tell("comparing")
            self._base_runs[key] = self._on_base(argv, kind, timeout)
        base = self._base_runs.get(key)
        result = testrun.compare(now, base)
        if (result.fresh or result.verdict == "broke") and not self._stopped():         # design case 23: a test that fails once is run once more before the change is blamed
            self._tell("retrying")
            flaky = self._run_again(command, argv, kind, result.fresh, timeout)
            if flaky:
                result = testrun.compare(now, base, flaky)
        with self.lock:
            record = self.ws.tested_record(command, now.exit_code if now.exit_code is not None else -1, result.line, simulated)
            record["verdict"] = result.verdict
            paths = [r["path"] for r in self.ws.changed()]
            self.note("test", command=command, verdict=result.verdict, line=result.line, delta=record["delta"], diff=self._diff_for_log())
        passed = result.verdict in testrun.PASSES
        tests = testrun.touches_tests(paths)
        lines = [result.line, *result.rows,
                 *([f"[จำลอง: {simulated}]"] if simulated else []),
                 *([f"[รันไม่ได้ใน box: {now.missing}]"] if now.missing and result.verdict != "passed" else []),
                 *([f"[การแก้นี้แก้ไฟล์ test ด้วย: {', '.join(tests[:4])}]"] if tests and passed else []),
                 *([f"[ผ่านแค่คำสั่งนี้: {command}]"] if passed else [])]
        return TestDone(now.exit_code, time.monotonic() - began, lines, result.verdict)

    def _run_again(self, command: str, argv: list[str], kind: str, fresh: list[str], timeout: float) -> set[str]:
        """What passed when run a second time. A command that cannot name its tests runs whole."""
        again = (testrun.again(kind, fresh) if fresh else None) or argv
        second = self._outcome(kind, self.run_in_copy(again, None, timeout, self._stopped))
        if second.killed:
            return set()
        if fresh:
            return {n for n in fresh if n not in second.failed} if second.read else set()
        return {command} if second.exit_code == 0 else set()

    def test(self, command: str | None, simulated: str | None, emit, cancel: threading.Event | None) -> TestDone | None:
        """Runs the tests on the copy and against the base, or goes on waiting for the run in progress. None: it is still running after the wait (design case 24), and the next call picks it up."""
        with self.lock:
            if self._test is None:
                command = command or self.default_command()
                if not command:
                    raise Refused("no test command is known for this project: say which with `command`")
                try:
                    argv, kind = testrun.prepared(testrun.split_command(command))
                except ValueError as e:
                    raise Refused(str(e)) from e
                self.command = command
                self._pool = self._pool or ThreadPoolExecutor(1, thread_name_prefix="sandbox-test")
                self._say, self._cancel = emit, cancel
                self._test = self._pool.submit(self._run_tests, command, argv, kind, simulated)
            job = self._test
            self._say, self._cancel = emit, cancel
        try:
            done = job.result(self.conf["test_wait_seconds"])
        except FutureTimeout:
            with self.lock:
                self._say = self._cancel = None                               # the call ends and the run goes on: a later Stop of another request is not this run's
            return None
        except (Refused, WorkspaceError, IsolationError, OSError) as e:
            with self.lock:
                self._test = None
            raise Refused(str(e)) from e
        with self.lock:
            self._test = None
            self._say = self._cancel = None
        return done

    # -------------------------------------------------------------------------------------------- the record of the chat (design 4.8)
    def _diff_for_log(self) -> str | None:
        try:
            text = self.ws.diff()
        except (OSError, WorkspaceError):
            return None
        return text if len(text) <= MAX_LOGGED_DIFF else None

    def note(self, kind: str, **fields) -> None:
        """One line of the record the user can read back: every command and diff, per chat. It outlives the copy and goes when the chat is deleted."""
        if self.log_dir is None:
            return
        path = _log_file(self.log_dir, self.chat)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.stat().st_size > MAX_LOG:
                fields.pop("diff", None)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"time": time.time(), "project": self.project, "kind": kind, **fields}, ensure_ascii=False) + "\n")
        except OSError:
            pass                                                              # a record that cannot be written must not stop the work
