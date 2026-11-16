"""desktop/test_tray.py - the tray icon's adapter (#310 part 3 task 5) on a fake `pystray`: its tooltip, colour and
menu follow the state, a menu item never runs on the icon's thread, a notice is cut to what Windows takes, and stop
leaves no thread.  No icon is put on the notification area.

    python -m pytest desktop/test_tray.py -q
"""
from __future__ import annotations

import sys
import threading
import time
import types

import pytest

from desktop import state, tray, words
from desktop.state import TrayState


class FakeIcon:
    made: list["FakeIcon"] = []

    def __init__(self, name, icon=None, title=None, menu=None):
        self.name, self.icon, self.title, self.menu = name, icon, title, menu
        self.visible = False
        self.notices: list[tuple] = []
        self.menu_updates = 0
        self._stop = threading.Event()
        self.thread_name = None
        FakeIcon.made.append(self)

    def run(self, setup=None):                     # what pystray's win32 run does: setup, then the message loop
        self.thread_name = threading.current_thread().name
        if setup:
            setup(self)
        self._stop.wait(30)

    def stop(self):
        self._stop.set()

    def update_menu(self):
        self.menu_updates += 1

    def notify(self, message, title=None):
        self.notices.append((title, message))


class FakeItem:
    def __init__(self, text, action, default=False, visible=True):
        self.text, self.action, self.default, self._visible = text, action, default, visible

    @property
    def visible(self):
        return self._visible(self) if callable(self._visible) else self._visible


class FakeMenu:
    def __init__(self, *items):
        self.items = list(items)


@pytest.fixture(autouse=True)
def fake_pystray(monkeypatch):
    mod = types.ModuleType("pystray")
    mod.Icon, mod.Menu, mod.MenuItem = FakeIcon, FakeMenu, FakeItem
    monkeypatch.setitem(sys.modules, "pystray", mod)
    monkeypatch.setattr(tray, "tinted", lambda color, *a, **k: f"image:{color}")
    monkeypatch.setattr(words, "LANG", "en")
    FakeIcon.made = []
    yield


def make(calls=None):
    calls = calls if calls is not None else []
    done = threading.Event()

    def rec(name):
        def f():
            calls.append((name, threading.current_thread().name))
            done.set()
        return f
    t = tray.Tray(rec("open"), rec("switch"), rec("copy_code"), rec("quit"))
    return t, calls, done


def visible_texts(icon):
    return [i.text for i in icon.menu.items if i.visible]


def test_icon_follows_the_state():
    t, _, _ = make()
    t.start()
    icon = FakeIcon.made[0]
    try:
        assert icon.name == "StrataDesktop" and icon.icon == "image:grey"
        assert icon.title == words.t("tray_idle")
        assert visible_texts(icon) == ["Open", "Switch server", "Quit"]
        assert [i.text for i in icon.menu.items if i.default] == ["Open"]
        s = TrayState("pending", server="lab", code="K7QZ")
        t.update(s)
        assert icon.icon == "image:amber" and icon.title == state.tooltip(s)
        assert visible_texts(icon) == ["Open", "Copy code", "Switch server", "Quit"]
        assert icon.menu_updates >= 1
        t.update(TrayState("online", server="lab", who="ann"))
        assert icon.icon == "image:green" and visible_texts(icon) == ["Open", "Switch server", "Quit"]
        t.update(TrayState("refused", server="lab", refused="key"))
        assert icon.icon == "image:red"
    finally:
        t.stop()


def test_menu_actions_run_off_the_icon_thread():
    t, calls, done = make()
    t.start()
    icon = FakeIcon.made[0]
    try:
        quit_item = [i for i in icon.menu.items if i.text == "Quit"][0]
        quit_item.action(icon, quit_item)
        assert done.wait(5)
        assert calls == [("quit", "desktop-tray-action")]
    finally:
        t.stop()


def test_stop_joins_the_icon_thread():
    t, _, _ = make()
    t.start()
    icon = FakeIcon.made[0]
    deadline = time.monotonic() + 5
    while icon.thread_name is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert icon.thread_name == "desktop-tray" and icon.visible
    thread = t._thread
    assert thread.daemon
    t.stop()
    assert not thread.is_alive()
    t.stop()                                                   # twice is fine
    t.update(TrayState("online"))                              # after stop: nothing, no error
    t.notify("a", "b")
    assert icon.notices == []


def test_quit_from_the_menu_stops_the_icon():
    """The tray's Quit runs on desktop-tray-action, which may join desktop-tray."""
    t = None
    ended = threading.Event()

    def on_quit():
        t.stop()
        ended.set()
    t = tray.Tray(lambda: None, lambda: None, lambda: None, on_quit)
    t.start()
    t._ready.wait(5)
    icon = FakeIcon.made[0]
    item = [i for i in icon.menu.items if i.text == "Quit"][0]
    item.action(icon, item)
    assert ended.wait(5)
    assert not t._thread.is_alive()


def test_stop_before_start_is_nothing():
    t, _, _ = make()
    t.stop()
    t.start()                                                  # a stopped tray never starts
    assert FakeIcon.made == []


def test_notice_is_cut_to_what_windows_takes():
    t, _, _ = make()
    t.start()
    t._ready.wait(5)
    try:
        t.notify("T" * 100, "x" * 400)
        (title, text), = FakeIcon.made[0].notices
        assert len(title) == tray.TITLE_MOST and len(text) == tray.TEXT_MOST
    finally:
        t.stop()


def test_tinted_keeps_the_shape(monkeypatch):
    Image = pytest.importorskip("PIL.Image")
    monkeypatch.undo()                                         # the real tinted, the real PIL (no fake pystray needed)
    im = tray.tinted("green")
    assert im.size == (tray.ICON_SIZE, tray.ICON_SIZE) and im.mode == "RGBA"
    colors = {px[:3] for _, px in im.getcolors(maxcolors=1 << 16) if px[3] > 0}
    assert colors == {tray.COLORS["green"]}
    with Image.open(tray.ICON) as src:
        assert src.mode == "RGBA"
