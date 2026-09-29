"""serve/timeline.py - #33: the server's lanes on the pipeline timeline.

STRATA_TIMELINE=<file> makes the engine write <file> (include/strata/timeline.hpp) and this server write
<file>.server.json beside it: two processes appending to one file would interleave their lines.
tests/xeno/perf/timeline.py load() reads both.  The clock is time.perf_counter() in microseconds -
QueryPerformanceCounter on Windows, the timebase of the engine's std::chrono::steady_clock - so a request's HTTP,
template, queue and pipe time line up with the engine's spans.  Records are written as they happen (a few per
request); off, every call returns at once.
"""
from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager

_lock = threading.Lock()
_path: str | None = None
_opened = False
_tids: dict[int, int] = {}


def configure(path: str | None) -> None:
    """Where the server's records go (None: off).  main() calls it from $STRATA_TIMELINE."""
    global _path, _opened
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
        with open(_path, "a", encoding="utf-8") as f:
            f.writelines(lines)


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
