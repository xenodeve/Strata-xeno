"""serve/timeline.py - #33: the server's lanes on the pipeline timeline.

STRATA_TIMELINE=<file> makes the engine write <file> (include/strata/timeline.hpp) and this server write
<file>.server.json beside it: two processes appending to one file would interleave their lines.
tests/xeno/perf/timeline.py load() reads both.  The clock is time.perf_counter() in microseconds -
QueryPerformanceCounter on Windows, the timebase of the engine's std::chrono::steady_clock - so a request's HTTP,
template, queue and pipe time line up with the engine's spans.  Records are queued as they happen (a few per
request) and appended by a writer thread of their own (#248: on a slow drive - F:, a USB stick - an append on the
request thread held one request 2.3 s, while the engine's flush was writing to the same stick); off, every call
returns at once.  flush() waits until everything queued is written (tests, exit).
"""
from __future__ import annotations

import atexit
import json
import os
import threading
import time
from contextlib import contextmanager

_lock = threading.Lock()
_path: str | None = None
_opened = False
_tids: dict[int, int] = {}
# #248: (path, lines) queued for the writer thread; _queued / _written count batches
_qcv = threading.Condition()
_queue: list[tuple[str, list[str]]] = []
_queued = 0
_written = 0
_writer: threading.Thread | None = None


def _writer_loop() -> None:
    global _written
    while True:
        with _qcv:
            while not _queue:
                _qcv.wait()
            batch = _queue[:]
            _queue.clear()
        by_path: dict[str, list[str]] = {}
        for path, lines in batch:
            by_path.setdefault(path, []).extend(lines)
        for path, lines in by_path.items():
            try:
                with open(path, "a", encoding="utf-8") as f:
                    f.writelines(lines)
            except OSError:
                pass
        with _qcv:
            _written += len(batch)
            _qcv.notify_all()


def flush(timeout: float = 30.0) -> None:
    """Wait until every record queued so far is in its file."""
    deadline = time.monotonic() + timeout
    with _qcv:
        target = _queued
        while _written < target:
            left = deadline - time.monotonic()
            if left <= 0:
                return
            _qcv.wait(left)


atexit.register(flush)


def configure(path: str | None) -> None:
    """Where the server's records go (None: off).  main() calls it from $STRATA_TIMELINE."""
    global _path, _opened
    flush()   # the last file's records first
    with _lock:
        _path, _opened = path, False
        _tids.clear()


def enabled() -> bool:
    return _path is not None


def now_us() -> float:
    return time.perf_counter() * 1e6


def _write(rec: dict) -> None:
    global _opened
    with _lock:
        if _path is None:
            return
        ident = threading.get_ident()
        lines = []
        if not _opened:
            lines.append("[\n")
            lines.append(json.dumps({"ph": "M", "pid": os.getpid(), "tid": 0, "name": "process_name",
                                     "args": {"name": f"server {os.getpid()}"}}) + ",\n")
            _opened = True
        if ident not in _tids:
            _tids[ident] = len(_tids) + 1
            lines.append(json.dumps({"ph": "M", "pid": os.getpid(), "tid": _tids[ident], "name": "thread_name",
                                     "args": {"name": "server " + threading.current_thread().name}}) + ",\n")
        rec.update(pid=os.getpid(), tid=_tids[ident])
        lines.append(json.dumps(rec) + ",\n")
        path = _path
    global _queued, _writer
    with _qcv:   # #248: the writer thread appends them; the caller does not wait for the disk
        _queue.append((path, lines))
        _queued += 1
        if _writer is None:
            _writer = threading.Thread(target=_writer_loop, name="timeline writer", daemon=True)
            _writer.start()
        _qcv.notify_all()


def complete(name: str, t0_us: float, t1_us: float, a: int = -1, b: int = -1) -> None:
    if _path is None:
        return
    rec = {"ph": "X", "ts": t0_us, "dur": t1_us - t0_us, "name": name}
    if a != -1 or b != -1:
        rec["args"] = {"a": a, "b": b}
    _write(rec)


def instant(name: str, a: int = -1, b: int = -1) -> None:
    if _path is None:
        return
    rec = {"ph": "i", "s": "t", "ts": now_us(), "name": name}
    if a != -1 or b != -1:
        rec["args"] = {"a": a, "b": b}
    _write(rec)


@contextmanager
def span(name: str, a: int = -1, b: int = -1):
    if _path is None:
        yield
        return
    t0 = now_us()
    try:
        yield
    finally:
        complete(name, t0, now_us(), a, b)
