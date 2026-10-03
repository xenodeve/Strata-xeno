"""serve/runs.py - an agent's answer that goes on when the page goes away (issue #99 follow-up).

A request that uses the coding tools can run for many minutes.  If the page that asked for it is refreshed or closed, the answer used to stop with a network error.  Now such a request (one that
names a run id in `strata_agent.run`) runs on its own thread and writes what it would stream into a buffer; the page that asked reads the buffer as it grows, and a page that comes back after a
refresh reads it again from where it was (GET /agent/run?id=...&from=N) and goes on from there, so the work, its permission questions and its answer are all still there.  The run ends when it is done,
when the user stops it (POST /agent/cancel), or when nobody has looked at it for a long time.  A run that is over is kept for a while so that a page that was away can still read its end.
"""
from __future__ import annotations

import collections
import json
import threading
import time
from typing import Callable, Iterator

KEEP_S = 30 * 60              # a finished run can be read again for this long
IDLE_S = 15 * 60              # a running one that nobody has looked at for this long is stopped
MAX_RUNS = 12
MAX_BYTES = 64 * 1024 * 1024  # of one run's buffered stream
BEAT_S = 5.0                  # how often a reader that is waiting is given a sign of life


class Run:
    def __init__(self, rid: str, cancel: threading.Event, owner=None):
        self.id, self.cancel, self.owner = rid, cancel, owner         # `owner`: the AgentRun whose steering queue what the user sends meanwhile goes to
        self.events: list[bytes] = []             # each a whole SSE event: b"data: {...}\n\n"
        self.size = 0
        self.done = False
        self.cond = threading.Condition()
        self.viewers = 0
        self.last_seen = time.time()
        self.ended_at: float | None = None

    def add(self, event: bytes, force: bool = False) -> bool:
        with self.cond:
            if not force and self.size + len(event) > MAX_BYTES:       # the closing events are always kept, however large the run
                return False
            self.events.append(event)
            self.size += len(event)
            self.cond.notify_all()
            return True

    def finish(self) -> None:
        with self.cond:
            self.done = True
            self.ended_at = time.time()
            self.cond.notify_all()

    def follow(self, start: int = 0) -> Iterator[bytes | None]:
        """The events from number `start` on, as they come; None now and then while nothing does (a sign of life for the connection); ends when the run is over and everything was given."""
        i = max(0, start)
        with self.cond:
            self.viewers += 1
        try:
            while True:
                with self.cond:
                    if i >= len(self.events) and not self.done:
                        self.cond.wait(BEAT_S)
                    batch = self.events[i:i + 64]
                    over = self.done and i + len(batch) >= len(self.events)
                    self.last_seen = time.time()
                if batch:
                    i += len(batch)
                    yield from batch
                elif over:
                    return
                else:
                    yield None
        finally:
            with self.cond:
                self.viewers -= 1
                self.last_seen = time.time()


class RunStore:
    def __init__(self):
        self.lock = threading.Lock()
        self.runs: collections.OrderedDict[str, Run] = collections.OrderedDict()
        self.watch = threading.Thread(target=self._watch, daemon=True)
        self.watch.start()

    def get(self, rid) -> Run | None:
        with self.lock:
            return self.runs.get(rid) if isinstance(rid, str) else None

    def launch(self, rid: str, chunks: Iterator, cancel: threading.Event, describe: Callable[[BaseException], dict | None], owner=None) -> Run | None:
        """Start a run that writes what `chunks` yields (dicts, None for a keep-alive) into a buffer; None when there is already a run with that id.
        `describe(exc)`: the error chunk to write when the stream breaks, or None to let it end quietly."""
        run = Run(rid, cancel, owner)
        with self.lock:
            if rid in self.runs:
                return None
            self.runs[rid] = run
        self._prune()
        threading.Thread(target=self._work, args=(run, chunks, describe), daemon=True).start()
        return run

    def _work(self, run: Run, chunks: Iterator, describe) -> None:
        try:
            for c in chunks:
                if c is None:
                    continue
                if not run.add(b"data: " + json.dumps(c, ensure_ascii=False).encode("utf-8") + b"\n\n"):
                    run.cancel.set()                                       # a stream of this size is not kept: stop it, and say so
                    run.add(b"data: " + json.dumps({"error": {"type": "server_error", "message": "the answer grew too large to be kept"}}).encode() + b"\n\n", force=True)
                    break
            run.add(b"data: [DONE]\n\n", force=True)
        except BaseException as e:  # noqa: BLE001 - whatever stops the stream is told to the reader like any error of it
            err = describe(e)
            if err is not None:
                run.add(b"data: " + json.dumps(err, ensure_ascii=False).encode("utf-8") + b"\n\n", force=True)
            run.add(b"data: [DONE]\n\n", force=True)
        finally:
            try:
                chunks.close()
            except Exception:  # noqa: BLE001
                pass
            run.finish()

    def cancel(self, rid) -> bool:
        run = self.get(rid)
        if run is None:
            return False
        run.cancel.set()
        return True

    def steer(self, rid, text: str) -> bool:
        """Hand what the user sent while the run works to its agent; False when there is no such run, or it is over."""
        run = self.get(rid)
        if run is None or run.done or run.owner is None or not hasattr(run.owner, "steer"):
            return False
        run.owner.steer(text)
        return True

    def _prune(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        with self.lock:
            for rid in [r for r, run in self.runs.items() if run.done and run.ended_at is not None and now - run.ended_at > KEEP_S]:
                del self.runs[rid]
            while len(self.runs) > MAX_RUNS:
                gone = next((r for r, run in self.runs.items() if run.done), None)
                if gone is None:
                    break
                del self.runs[gone]

    def _watch(self) -> None:
        while True:
            time.sleep(30)
            self.sweep(time.time())

    def sweep(self, now: float) -> None:
        """Stop the runs that nobody has looked at for a long time, and forget the ones that ended long ago."""
        with self.lock:
            runs = list(self.runs.values())
        for run in runs:
            if not run.done and run.viewers == 0 and now - run.last_seen > IDLE_S:
                run.cancel.set()                                       # nobody has been looking for a long time: it does not go on for ever
        self._prune(now)
