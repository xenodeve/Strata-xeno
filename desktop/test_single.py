"""desktop/test_single.py - one Strata Desktop per data folder (#310 part 3 task 4, RF 4): a named mutex, and the
second start asks the first to show its window through a named event.

    python -m pytest desktop/test_single.py -q
"""
from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path

import pytest

from desktop import single

pytestmark = pytest.mark.skipif(os.name != "nt", reason="named mutexes and events are Windows'")


def _name() -> str:
    return "Local\\StrataDesktopTest-" + uuid.uuid4().hex


def test_second_acquire_fails():
    name = _name()
    first = single.acquire(name)
    assert first is not None
    try:
        assert single.acquire(name) is None
    finally:
        first.release()
    again = single.acquire(name)
    assert again is not None
    again.release()


def test_show_event_reaches_first():
    name = _name()
    first = single.acquire(name)
    assert first is not None
    shown = threading.Event()
    try:
        first.listen(shown.set)
        assert any(t.name == "desktop-show" for t in threading.enumerate())
        assert single.signal_show(name) is True
        assert shown.wait(2.0)
        shown.clear()
        assert single.signal_show(name) is True                      # and again
        assert shown.wait(2.0)
    finally:
        first.release()
    time.sleep(0.05)
    assert not any(t.name == "desktop-show" for t in threading.enumerate())


def test_signal_before_listen_is_kept():
    """A second start that signals while the first has not started listening yet is not lost."""
    name = _name()
    first = single.acquire(name)
    shown = threading.Event()
    try:
        assert single.signal_show(name) is True
        first.listen(shown.set)
        assert shown.wait(2.0)
    finally:
        first.release()


def test_signal_without_first_is_false():
    assert single.signal_show(_name()) is False


def test_callback_error_does_not_end_the_listener():
    name = _name()
    first = single.acquire(name)
    calls = []

    def cb():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("once")

    try:
        first.listen(cb)
        single.signal_show(name)
        deadline = time.monotonic() + 2
        while len(calls) < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        single.signal_show(name)
        while len(calls) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(calls) == 2
    finally:
        first.release()


def test_release_twice_is_safe():
    inst = single.acquire(_name())
    inst.release()
    inst.release()


def test_instance_name(tmp_path: Path):
    assert single.instance_name(tmp_path, False) == "Local\\StrataDesktop"
    a = single.instance_name(tmp_path, True)
    assert a.startswith("Local\\StrataDesktop-") and len(a) == len("Local\\StrataDesktop-") + 8
    assert "\\" not in a[len("Local\\"):]
    assert single.instance_name(Path(str(tmp_path).upper()), True) == a        # one folder, however it is written
    assert single.instance_name(tmp_path / "sub" / "..", True) == a
    assert single.instance_name(tmp_path / "other", True) != a


def test_instance_name_resolves_a_junction(tmp_path: Path):
    """Two spellings of one folder through a junction are one instance (else two runners on one device)."""
    import _winapi
    target, link = tmp_path / "data", tmp_path / "link"
    target.mkdir()
    _winapi.CreateJunction(str(target), str(link))
    assert single.instance_name(link, True) == single.instance_name(target, True)


def test_mutex_of_an_elevated_first_is_refused_not_raised(monkeypatch):
    """A medium-integrity second start gets NULL + ERROR_ACCESS_DENIED for a name an elevated first made."""
    class K32:
        def CreateMutexW(self, *a):
            return 0

        def CloseHandle(self, h):
            raise AssertionError("nothing to close")

    monkeypatch.setattr(single, "_k32", K32())
    monkeypatch.setattr(single.ctypes, "get_last_error", lambda: single.ERROR_ACCESS_DENIED)
    assert single.acquire(_name()) is None
    monkeypatch.setattr(single.ctypes, "get_last_error", lambda: 87)               # anything else still raises
    with pytest.raises(OSError):
        single.acquire(_name())
