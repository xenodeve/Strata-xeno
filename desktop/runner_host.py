"""The runner inside Strata Desktop (#310 part 3): serve.runner's Executor and RunnerLoop on one thread of the app,
`desktop-runner`, with a stop that ends its jobs, their processes and the executor within a bound.

    host = RunnerHost(url, token, device, label, runner_folder, on_state)
    host.start()          # "connecting", then the loop's events ("online", "pending", ...), or "error"
    host.stop(5.0)        # True when the thread and every job thread ended

The thread builds `Executor(harness.home_dir(), runner_folder / "runner.json", runner_folder)` (its MCP servers start
here), then runs `RunnerLoop.run` itself: there is no `runner-poll` thread; the loop's `runner-job` threads run under
it.  A build that fails (a SystemExit for a bad runner.json, or any exception) is reported as "error" and the thread
ends.  When the server refuses the runner, the thread lets its jobs end and closes the executor, as
`python -m serve.runner` does.

`stop`: sets the stop event, cuts the poll the server holds (`Client.abort_poll`), lets the jobs end
(`RunnerLoop.finish`: cancelled, their process trees ended by #314's job objects), joins the thread, then closes the
executor (background commands and MCP servers end) - all within one deadline.  The executor is closed exactly once,
by whichever of `stop` and the thread comes last to it; a stop that comes while the executor is still being built
leaves the closing to the thread, once the build ends.  No event is delivered once `stop` has begun, so a late event
of an old host never overwrites the state of the next one.

This class never prints (serve.runner's `_say` lines go where Task 4 routes them) and holds the token only inside
serve.runner.Client; its repr shows the server's address only.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable

log = logging.getLogger("strata.desktop.runner")

FINISH_S = 5.0                  # after a refusal: how long the jobs get to end before the executor closes


class RunnerHost:
    def __init__(self, url: str, token: str, device: dict, label: str, runner_folder: Path,
                 on_state: Callable[[str, dict], None], executor_factory: Callable[[], object] | None = None):
        self.url, self.device, self.label = url, dict(device), label
        self.runner_folder = Path(runner_folder)
        self._token = token
        self._on_state = on_state
        self._factory = executor_factory or self._executor_default
        self.runner_id: str | None = None
        self.who: str | None = None
        self.shell: str | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()                   # _executor, _loop, _client, _closed
        self._emit_lock = threading.RLock()             # an event is delivered whole, or not at all once stop began
        self._stopping = False
        self._executor = None
        self._loop = None
        self._client = None
        self._closed = False
        self._thread: threading.Thread | None = None

    def __repr__(self) -> str:
        return f"RunnerHost({self.url!r}, runner_id={self.runner_id!r})"

    __str__ = __repr__

    def _executor_default(self):
        from serve import harness
        from serve.runner import Executor
        return Executor(harness.home_dir(), self.runner_folder / "runner.json", self.runner_folder)

    # -------------------------------------------------------------------------------------------- events
    def _emit(self, event: str, data: dict) -> None:
        with self._emit_lock:
            if self._stopping:
                return
            if event == "online":
                self.runner_id, self.who = data.get("id"), data.get("who")
                data = {**data, "no_shell": self.shell is None}
            try:
                self._on_state(event, data)
            except Exception as e:  # noqa: BLE001 - the tray's trouble must not end the runner
                log.warning("the state callback raised %s on %r", type(e).__name__, event)

    def _scrub(self, text: str) -> str:
        return text.replace(self._token, "***") if self._token else text

    # -------------------------------------------------------------------------------------------- the thread
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="desktop-runner")
        self._thread.start()

    def _run(self) -> None:
        from serve.runner import Client, RunnerLoop

        self._emit("connecting", {})
        try:
            ex = self._factory()
        except (SystemExit, Exception) as e:  # noqa: BLE001 - a bad runner.json is a SystemExit from Executor
            text = str(e)
            text = text[len("[runner] "):] if text.startswith("[runner] ") else text
            if not isinstance(e, SystemExit):
                text = f"{type(e).__name__}: {text}"
            self._emit("error", {"message": self._scrub(text)})
            return
        with self._lock:
            self._executor = ex
            stopping = self._stop.is_set()
        if stopping:                                    # stop came while the executor was built: no loop starts
            self._close_executor()
            return
        loop = None
        try:
            info = ex.info()
            self.shell = info.get("shell") if isinstance(info, dict) else None
            client = Client(self.url, self._token, self.device, label=self.label)
            loop = RunnerLoop(client, ex, stop=self._stop, on_state=self._emit)
            with self._lock:
                self._client, self._loop = client, loop
            loop.run()
        except Exception as e:  # noqa: BLE001 - e.g. a bad address: words for the tray, never a traceback
            self._emit("error", {"message": self._scrub(f"{type(e).__name__}: {e}")})
        finally:
            if loop is not None and not self._stop.is_set():
                loop.finish(FINISH_S)                   # refused: the jobs end before the executor closes
            self._close_executor()

    def _close_executor(self) -> None:
        with self._lock:
            ex, done = self._executor, self._closed
            if ex is not None:
                self._closed = True
        if ex is not None and not done:
            try:
                ex.close()
            except Exception as e:  # noqa: BLE001 - closing must go on to the end
                log.warning("closing the runner's executor raised %s", type(e).__name__)

    # -------------------------------------------------------------------------------------------- stop
    def stop(self, timeout: float = 5.0) -> bool:
        """Stop the runner: its poll, its jobs (and their processes), its thread and its executor, within `timeout`
        seconds.  True when the thread and every job thread ended.  Safe before start, during the build, and twice."""
        end = time.monotonic() + timeout
        got = self._emit_lock.acquire(timeout=max(0.0, timeout))      # an event being delivered ends first (bounded:
        self._stopping = True                                          # a callback that blocks cannot hold stop)
        if got:
            self._emit_lock.release()
        self._stop.set()
        with self._lock:
            client, loop = self._client, self._loop
        if client is not None:
            client.abort_poll()
        if loop is not None:
            loop.finish(max(0.0, end - time.monotonic()))
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(max(0.0, end - time.monotonic()))
        with self._lock:
            building = self._executor is None and t is not None and t.is_alive()
        if not building:                                # else the thread closes it once the build ends
            self._close_executor()
        thread_ended = t is None or not t.is_alive()
        if loop is None:
            return thread_ended
        with loop._lock:
            jobs = list(loop._threads)
        return thread_ended and not any(j.is_alive() for j in jobs)
