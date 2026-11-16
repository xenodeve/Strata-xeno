"""One Strata Desktop per data folder (#310 part 3, RF 4): two would run two runners on one device, and they would
replace each other on the server forever.

`acquire(name)` takes the named mutex `name` (`Local\\StrataDesktop`, or `Local\\StrataDesktop-<sha8 of the folder>`
with `--data-dir`) and makes the auto-reset event `name + ".show"`; a second start gets None, calls
`signal_show(name)` and exits 0, and the first shows its window from the thread `desktop-show` (`Instance.listen`).
The event exists from `acquire` on, so a signal sent before `listen` starts is kept, not lost.  Off Windows there is no
named mutex: `acquire` always succeeds and `signal_show` is False.
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
from pathlib import Path
from typing import Callable

log = logging.getLogger("strata.desktop.single")

BASE = "Local\\StrataDesktop"
ERROR_ALREADY_EXISTS = 183
WAIT_OBJECT_0 = 0
INFINITE = 0xFFFFFFFF
EVENT_MODIFY_STATE = 0x0002

if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _k32.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
    _k32.CreateMutexW.restype = wintypes.HANDLE
    _k32.CreateEventW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
    _k32.CreateEventW.restype = wintypes.HANDLE
    _k32.OpenEventW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
    _k32.OpenEventW.restype = wintypes.HANDLE
    _k32.SetEvent.argtypes = [wintypes.HANDLE]
    _k32.SetEvent.restype = wintypes.BOOL
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]
    _k32.CloseHandle.restype = wintypes.BOOL
    _k32.WaitForMultipleObjects.argtypes = [wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE), wintypes.BOOL,
                                            wintypes.DWORD]
    _k32.WaitForMultipleObjects.restype = wintypes.DWORD


def instance_name(data_dir: Path, overridden: bool) -> str:
    """`Local\\StrataDesktop`, or with `--data-dir` `Local\\StrataDesktop-` + the first 8 hex digits of the sha256 of
    the folder's full path (case-folded: one folder however it is written)."""
    if not overridden:
        return BASE
    full = os.path.normcase(os.path.abspath(os.fspath(data_dir)))
    return BASE + "-" + hashlib.sha256(full.encode("utf-8")).hexdigest()[:8]


class Instance:
    """The mutex this process holds, its `.show` event, and the listener."""

    def __init__(self, name: str, mutex, show_event):
        self.name = name
        self._mutex, self._show = mutex, show_event
        self._stop = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"Instance({self.name!r})"

    def listen(self, on_show: Callable[[], None]) -> None:
        """Calls `on_show` (on the thread `desktop-show`) each time a second start signals.  An exception from it is
        logged and the listener goes on."""
        if os.name != "nt" or self._show is None:
            return
        with self._lock:
            if self._thread is not None:
                return
            self._stop = _k32.CreateEventW(None, True, False, None)
            if not self._stop:
                raise OSError(ctypes.get_last_error(), "CreateEventW failed")
            handles = (wintypes.HANDLE * 2)(self._show, self._stop)
            self._thread = threading.Thread(target=self._run, args=(handles, on_show), name="desktop-show",
                                            daemon=True)
            self._thread.start()

    def _run(self, handles, on_show) -> None:
        while True:
            r = _k32.WaitForMultipleObjects(2, handles, False, INFINITE)
            if r != WAIT_OBJECT_0:                 # the stop event, or the wait failed
                return
            try:
                on_show()
            except Exception as e:
                log.error("showing the window raised %s", type(e).__name__)

    def release(self) -> None:
        """Ends the listener, closes the event and the mutex (the next start then becomes the first).  Twice is fine."""
        if os.name != "nt":
            return
        with self._lock:
            thread, self._thread = self._thread, None
            stop, self._stop = self._stop, None
            show, self._show = self._show, None
            mutex, self._mutex = self._mutex, None
        if stop:
            _k32.SetEvent(stop)
        if thread is not None and thread is not threading.current_thread():
            thread.join(5.0)
        for h in (stop, show, mutex):
            if h:
                _k32.CloseHandle(h)


def acquire(name: str) -> Instance | None:
    """This process's instance, or None when another process holds `name`."""
    if os.name != "nt":
        return Instance(name, None, None)
    mutex = _k32.CreateMutexW(None, False, name)
    err = ctypes.get_last_error()
    if not mutex:
        raise OSError(err, "CreateMutexW failed")
    if err == ERROR_ALREADY_EXISTS:
        _k32.CloseHandle(mutex)
        return None
    show = _k32.CreateEventW(None, False, False, name + ".show")
    if not show:
        err = ctypes.get_last_error()
        _k32.CloseHandle(mutex)
        raise OSError(err, "CreateEventW failed")
    return Instance(name, mutex, show)


def signal_show(name: str) -> bool:
    """Asks the instance holding `name` to show its window.  False when there is none (or it has not made its event
    yet)."""
    if os.name != "nt":
        return False
    ev = _k32.OpenEventW(EVENT_MODIFY_STATE, False, name + ".show")
    if not ev:
        return False
    try:
        return bool(_k32.SetEvent(ev))
    finally:
        _k32.CloseHandle(ev)
