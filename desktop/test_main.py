"""desktop/test_main.py - `python -m desktop`'s start (#310 part 3 task 5): the flags, the exit codes, a second start
handing over to the first (or saying it cannot), the selftest token taken out of the environment at once, and the
wiring of one label into the runner, the server test and the page.  Every window, the tray, the log's process-wide
hooks and the WebView2 check are replaced: nothing is shown, nothing is written outside a temp folder.

    python -m pytest desktop/test_main.py -q
"""
from __future__ import annotations

import os

import pytest

from desktop import __main__ as app
from desktop import applog, controller, gui, inject, single, tray, webview2, words

TOKEN = "sk-strata-main-SECRET-0123456789"


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    """The log's process-wide parts become no-ops, and serve.runner._say is restored after each test."""
    from serve import runner
    monkeypatch.setattr(words, "LANG", "en")
    monkeypatch.setattr(applog, "setup", lambda path: None)
    monkeypatch.setattr(applog, "install_hooks", lambda: None)
    monkeypatch.setattr(applog, "capture_stdio", lambda: None)
    monkeypatch.setattr(runner, "_say", runner._say)
    boxes = []
    monkeypatch.setattr(webview2, "message_box", boxes.append)
    monkeypatch.setattr(webview2, "preflight", lambda: None)
    monkeypatch.delenv(app.SELFTEST_ENV, raising=False)
    yield boxes
    applog.forget_secrets()


def test_server_is_for_the_selftest_only(tmp_path):
    with pytest.raises(SystemExit) as e:
        app.main(["--server", "http://127.0.0.1:8099", "--data-dir", str(tmp_path)])
    assert e.value.code == 2


def test_selftest_token_is_popped_at_once(tmp_path, monkeypatch):
    monkeypatch.setenv(app.SELFTEST_ENV, TOKEN)
    code = app.main(["--selftest", "--server", "http://127.0.0.1:8099", "--data-dir", str(tmp_path)])
    assert code == app.EXIT_SELFTEST_FAILED                    # task 6 fills the selftest in
    assert app.SELFTEST_ENV not in os.environ
    assert applog.scrub("x " + TOKEN) == "x ***"


def test_token_is_popped_on_a_normal_start_too(tmp_path, monkeypatch):
    monkeypatch.setenv(app.SELFTEST_ENV, TOKEN)
    monkeypatch.setattr(webview2, "preflight", lambda: 4)
    assert app.main(["--data-dir", str(tmp_path)]) == 4
    assert app.SELFTEST_ENV not in os.environ


def test_no_webview2_exits_4_before_the_mutex(tmp_path, monkeypatch):
    monkeypatch.setattr(webview2, "preflight", lambda: webview2.EXIT_MISSING)
    monkeypatch.setattr(single, "acquire", lambda name: pytest.fail("acquired"))
    assert app.main(["--data-dir", str(tmp_path)]) == 4


def test_second_start_shows_the_first(tmp_path, monkeypatch, quiet):
    seen = []
    monkeypatch.setattr(single, "acquire", lambda name: None)
    monkeypatch.setattr(single, "signal_show", lambda name: seen.append(name) or True)
    assert app.main(["--data-dir", str(tmp_path)]) == 0
    assert seen == [single.instance_name(tmp_path, True)] and quiet == []


def test_second_start_that_cannot_reach_the_first_says_so(tmp_path, monkeypatch, quiet):
    monkeypatch.setattr(single, "acquire", lambda name: None)
    monkeypatch.setattr(single, "signal_show", lambda name: False)
    assert app.main(["--data-dir", str(tmp_path)]) == 0
    assert quiet == [words.t("already_running")]


class FakeInstance:
    def __init__(self):
        self.released = 0
        self.listening = None

    def listen(self, on_show):
        self.listening = on_show

    def release(self):
        self.released += 1


class FakeGui:
    made: list["FakeGui"] = []

    def __init__(self, data, api, quitting=lambda: False, *, debug=False, servers_hidden=False):
        self.data, self.api, self.quitting, self.debug = data, api, quitting, debug
        self.calls = []
        FakeGui.made.append(self)

    def attach(self, api):
        self.api = api

    def run(self, start):
        start()
        return 0

    def __getattr__(self, name):                       # every WindowPort call is recorded
        def call(*a, **k):
            self.calls.append(name)
            return False
        return call


class FakeTray:
    made: list["FakeTray"] = []

    def __init__(self, on_open, on_switch, on_copy_code, on_quit):
        self.calls = []
        FakeTray.made.append(self)

    def start(self):
        self.calls.append("start")

    def update(self, s):
        pass

    def notify(self, title, text):
        pass

    def stop(self):
        self.calls.append("stop")


def test_wiring(tmp_path, monkeypatch):
    from serve import runner
    inst = FakeInstance()
    monkeypatch.setattr(single, "acquire", lambda name: inst)
    monkeypatch.setattr(gui, "Gui", FakeGui)
    monkeypatch.setattr(tray, "Tray", FakeTray)
    made, exits = [], []
    real = controller.Controller

    def ctl(*a, **k):
        c = real(*a, **k, exit_process=exits.append)
        made.append((a, c))
        return c
    monkeypatch.setattr(controller, "Controller", ctl)
    FakeGui.made, FakeTray.made = [], []

    assert app.main(["--data-dir", str(tmp_path), "--debug"]) == 0

    (args, c), = made
    store, device, label, folder = args[:4]
    assert folder == tmp_path / "runner" and (folder / "device.json").exists()     # never the real device
    assert label == inject.page_label(runner.device_label(device, "Strata Desktop"))
    g = FakeGui.made[0]
    assert g.debug is True and g.api._label == label and g.api._device == device and g.api._ctl is c
    assert c.quitting and c.wait_quit(1) and exits == []     # the loop ended: the same clean-up as Quit
    assert inst.listening == c.open and inst.released >= 1
    assert FakeTray.made[0].calls[:2] == ["start", "stop"]
    assert "show_servers" in g.calls and "destroy" in g.calls
