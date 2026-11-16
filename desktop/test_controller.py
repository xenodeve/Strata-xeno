"""desktop/test_controller.py - the order of Strata Desktop's steps (#310 part 3 task 5): connect waits for the runner's
hello before anything is written into the web view, Switch / Quit / Remove stop the runner and remove the token in
order (RF 3), a crash's leftover is cleared first at the next start, the device notice comes once per device.  Fake
window, tray, runner host and server test: no window, no tray icon, no process.

    python -m pytest desktop/test_controller.py -q
"""
from __future__ import annotations

import http.server
import json
import threading
import time

import pytest

from desktop import applog, controller, gui, inject, probe, state, words
from desktop.servers import Secret, ServerEntry

TOKEN_A = "sk-strata-ctl-A-SECRET-0123456789"
TOKEN_B = "sk-strata-ctl-B-SECRET-9876543210"
DEVICE = {"id": "AbCdEfGhIjKlMnOpQrStUv", "name": "ann-pc"}
LABEL = "Strata Desktop on ann-pc"
URL_A, URL_B = "http://127.0.0.1:8099", "https://lab.example.ts.net/strata"
ORIGIN_A, ORIGIN_B = "http://127.0.0.1:8099", "https://lab.example.ts.net"
RID_1, RID_2 = "r0123456789abcdef", "rfedcba9876543210"


@pytest.fixture(autouse=True)
def english(monkeypatch):
    monkeypatch.setattr(words, "LANG", "en")
    yield
    applog.forget_secrets()


# ------------------------------------------------------------------------------------------------ fakes
class Log:
    """One list of every call the fakes saw, in order, across the window, the tray, the hosts and the store."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.lock = threading.Lock()

    def add(self, *call):
        with self.lock:
            self.calls.append(call)

    def names(self, *keep) -> list[str]:
        with self.lock:
            return [c[0] for c in self.calls if not keep or c[0] in keep]

    def of(self, name) -> list[tuple]:
        with self.lock:
            return [c for c in self.calls if c[0] == name]


class FakeStore:
    def __init__(self, log: Log, rows=None, last=None, injected=None):
        self.log = log
        self.rows = rows if rows is not None else {"A": ServerEntry("A", URL_A, Secret(TOKEN_A)),
                                                   "B": ServerEntry("B", URL_B, Secret(TOKEN_B))}
        self.last, self.injected = last, list(injected or [])

    def entries(self):
        return list(self.rows.values())

    def get(self, name):
        return self.rows.get(name)

    def public(self):
        return [{"name": e.name, "url": e.url, "token": "set" if e.token.reveal() else "missing"}
                for e in self.rows.values()]

    def save(self, name, url, token, old_name=None):
        old = self.rows.pop(old_name, None) if old_name is not None else None
        tok = Secret(token) if token else old.token if old else Secret("")
        e = ServerEntry(name, url, tok)
        self.rows[name] = e
        if old_name is not None and self.last == old_name:
            self.last = name
        return e

    def remove(self, name):
        self.log.add("store.remove", name)
        if self.last == name:
            self.last = None
        return self.rows.pop(name, None) is not None

    def set_last(self, name):
        if name is not None and name not in self.rows:
            raise ValueError("There is no server with that name.")
        self.log.add("set_last", name)
        self.last = name

    def set_injected(self, origins):
        assert isinstance(origins, list)                         # P3-R3: always the whole list
        self.log.add("set_injected", list(origins))
        self.injected = list(origins)


class FakeWindow:
    def __init__(self, log: Log, inject_ok=True, clear_ok=True):
        self.log = log
        self.inject_ok, self.clear_ok = inject_ok, clear_ok
        self.block: threading.Event | None = None         # navigate_and_inject waits on it when set
        self.views: list[dict] = []

    def show_servers(self):
        self.log.add("show_servers")

    def hide_servers(self):
        self.log.add("hide_servers")

    def render(self, view):
        self.views.append(view)
        self.log.add("render", view)

    def navigate_and_inject(self, base, script, timeout=15.0):
        assert script != inject.clear_script(), "a clear goes through clear_storage (its local page, I-3)"
        self.log.add("inject", base, script, timeout)
        if self.block is not None:
            self.block.wait(10)
        return self.inject_ok

    def clear_storage(self, base, script, timeout=15.0):
        self.log.add("clear", base, script, timeout)
        if self.block is not None:
            self.block.wait(10)
        return self.clear_ok

    def open_app(self, base, server=None):
        self.log.add("open_app", base, server)

    def show_app(self):
        self.log.add("show_app")

    def hide_app(self):
        self.log.add("hide_app")

    def destroy(self):
        self.log.add("destroy")


class FakeTray:
    def __init__(self, log: Log):
        self.log = log
        self.states: list[state.TrayState] = []
        self.notices: list[tuple[str, str]] = []

    def update(self, s):
        self.states.append(s)
        self.log.add("tray.update", s.kind)

    def notify(self, title, text):
        self.notices.append((title, text))
        self.log.add("notify", title, text)

    def stop(self):
        self.log.add("tray.stop")


class FakeHost:
    def __init__(self, log: Log, url, token, device, label, folder, on_state):
        self.log, self.url, self.token, self.device, self.label = log, url, token, device, label
        self.on_state = on_state
        self.hang: threading.Event | None = None          # stop blocks on it when set
        self.stopped = False
        log.add("host.made", url)

    def start(self):
        self.log.add("host.start", self.url)

    def emit(self, event, data=None):
        self.on_state(event, data or {})

    def stop(self, timeout=5.0):
        self.log.add("host.stop", self.url, timeout)
        if self.hang is not None:
            self.hang.wait(30)
        self.stopped = True
        self.log.add("host.stopped", self.url)
        return True


def ok_probe(url, token, device, label, timeout=10.0):
    return probe.ProbeResult(True, model="gyro", role="owner", engine_running=True)


class Rig:
    def __init__(self, probe_fn=ok_probe, store_kw=None, window_kw=None, **ctl_kw):
        self.log = Log()
        self.store = FakeStore(self.log, **(store_kw or {}))
        self.window = FakeWindow(self.log, **(window_kw or {}))
        self.tray = FakeTray(self.log)
        self.hosts: list[FakeHost] = []
        self.probed: list[tuple] = []
        self.exits: list[int] = []
        self.copied: list[str] = []

        def make_host(*a):
            h = FakeHost(self.log, *a)
            self.hosts.append(h)
            return h

        def probe_rec(url, token, device, label, timeout=10.0):
            self.probed.append((url, token, device, label))
            return probe_fn(url, token, device, label)

        ctl_kw.setdefault("fetch_pending", lambda *a, **k: [])
        ctl_kw.setdefault("exit_process", self.exits.append)
        ctl_kw.setdefault("clipboard", lambda text: self.copied.append(text) or True)
        ctl_kw.setdefault("release", lambda: self.log.add("release"))
        self.ctl = controller.Controller(self.store, DEVICE, LABEL, "unused-runner-folder", self.window, self.tray,
                                         make_host=make_host, probe=probe_rec, **ctl_kw)

    def emit(self, event, data=None, host=-1):
        self.hosts[host].emit(event, data)
        assert self.ctl.wait_idle(5)

    def close(self):
        if not self.ctl.quitting:
            self.ctl.quit()


@pytest.fixture
def rig():
    made = []

    def make(**kw):
        r = Rig(**kw)
        made.append(r)
        return r
    yield make
    for r in made:
        for h in r.hosts:
            if h.hang is not None:
                h.hang.set()
        if r.window.block is not None:
            r.window.block.set()
        r.close()


def values_of(script: str) -> dict:
    """The {"set", "remove"} literal inside an inject_script."""
    start = script.index("var d=") + len("var d=")
    end = script.index(",s=window.localStorage")
    return json.loads(script[start:end])


# ------------------------------------------------------------------------------------------------ connect
def test_connect_waits_for_hello_then_injects(rig):
    r = rig()
    r.ctl.connect("A")
    assert r.log.names("host.made", "host.start") == ["host.made", "host.start"]
    assert r.log.of("inject") == [] and r.log.of("open_app") == []          # nothing before the runner is online
    r.emit("connecting")
    assert r.log.of("inject") == []
    r.emit("online", {"id": RID_1, "who": "ann"})
    (_, base, script, _), = r.log.of("inject")
    assert base == URL_A
    d = values_of(script)
    assert json.loads(d["set"]["strata.runner"]) == RID_1
    assert json.loads(d["set"]["strata.device"]) == DEVICE["id"]
    assert json.loads(d["set"]["strata.apikey"]) == TOKEN_A
    assert json.loads(d["set"]["strata.deviceName"]) == LABEL
    names = r.log.names("inject", "open_app", "hide_servers")
    assert names == ["inject", "open_app", "hide_servers"]
    assert r.log.of("open_app") == [("open_app", URL_A, "A")]
    assert r.store.injected == [ORIGIN_A]
    assert r.store.last == "A"
    assert r.ctl.state.kind == "online" and r.ctl.state.runner_id == RID_1
    assert r.tray.states[-1].kind == "online"


def test_injected_is_recorded_before_the_token_is_written(rig):
    r = rig()
    r.ctl.connect("A")
    r.emit("online", {"id": RID_1, "who": "ann"})
    names = r.log.names("set_injected", "inject")
    assert names.index("set_injected") < names.index("inject")


def test_one_label_everywhere(rig):
    r = rig()
    r.ctl.connect("A")
    r.emit("online", {"id": RID_1, "who": "ann"})
    assert r.probed[0][3] == LABEL and r.probed[0][2] == DEVICE
    assert r.hosts[0].label == LABEL and r.hosts[0].device == DEVICE
    assert json.loads(values_of(r.log.of("inject")[0][2])["set"]["strata.deviceName"]) == LABEL


def test_pending_shows_code_then_opens(rig):
    r = rig()
    r.ctl.connect("A")
    r.emit("pending", {"code": "K7QZ"})
    assert r.window.views[-1]["kind"] == "pending" and r.window.views[-1]["code"] == "K7QZ"
    assert "K7QZ" in r.window.views[-1]["text"]
    assert r.tray.states[-1].kind == "pending" and r.tray.states[-1].code == "K7QZ"
    assert r.log.of("inject") == [] and r.log.of("open_app") == []
    r.emit("online", {"id": RID_1, "who": "ann"})
    assert len(r.log.of("inject")) == 1 and len(r.log.of("open_app")) == 1


def test_runner_id_change_reinjects(rig):
    r = rig()
    r.ctl.connect("A")
    r.emit("online", {"id": RID_1, "who": "ann"})
    r.emit("offline", {"message": "x"})
    r.emit("online", {"id": RID_1, "who": "ann"})                              # the same runner: nothing
    assert len(r.log.of("inject")) == 1 and len(r.log.of("open_app")) == 1
    r.emit("online", {"id": RID_2, "who": "ann"})                              # a key rename: inject and reload
    injects = r.log.of("inject")
    assert len(injects) == 2
    assert json.loads(values_of(injects[1][2])["set"]["strata.runner"]) == RID_2
    assert len(r.log.of("open_app")) == 2


def test_inject_false_keeps_servers_window(rig):
    r = rig(window_kw={"inject_ok": False})
    r.ctl.connect("A")
    r.emit("online", {"id": RID_1, "who": "ann"})
    assert r.log.of("open_app") == [] and r.log.of("hide_servers") == []
    assert r.ctl.notice == words.t("window_not_prepared")
    assert r.window.views[-1]["notice"] == words.t("window_not_prepared")
    assert r.store.last is None
    assert r.store.injected == [ORIGIN_A]                # the script may have written part of it: Quit clears it


def test_refused_key_does_not_open(rig):
    r = rig(probe_fn=lambda *a: probe.ProbeResult(True, error=words.t("token_refused")))
    r.ctl.connect("A")
    assert r.hosts == [] and r.log.of("inject") == [] and r.log.of("open_app") == []
    assert r.ctl.notice == words.t("token_refused")
    assert r.window.views[-1]["notice"] == words.t("token_refused")
    assert r.ctl.state.kind == "idle" and r.ctl.state.server == "A"


def test_runner_refused_key_does_not_open(rig):
    r = rig()
    r.ctl.connect("A")
    r.emit("refused", {"kind": "key", "message": "401"})
    r.emit("online", {"id": RID_1, "who": "ann"})                              # an end state: ignored
    assert r.log.of("inject") == [] and r.log.of("open_app") == []
    assert r.tray.states[-1].kind == "refused" and r.tray.states[-1].refused == "key"
    assert r.window.views[-1]["can_open_without_runner"] is False


@pytest.mark.parametrize("probe_result", [probe.ProbeResult(False, error="unreachable words"),
                                          probe.ProbeResult(True, error="device refused words")])
def test_unreachable_or_device_refused_starts_no_runner(rig, probe_result, monkeypatch):
    monkeypatch.setitem(words.TABLE, "device_refused", {"en": "device refused words"})
    r = rig(probe_fn=lambda *a: probe_result)
    r.ctl.connect("A")
    assert r.hosts == []
    assert r.ctl.notice == probe_result.error


def test_pending_device_and_engine_down_still_start_the_runner(rig):
    r = rig(probe_fn=lambda *a: probe.ProbeResult(True, pending_code="K7QZ"))
    r.ctl.connect("A")
    assert len(r.hosts) == 1
    r2 = rig(probe_fn=lambda *a: probe.ProbeResult(True, engine_running=False, role="guest"))
    r2.ctl.connect("A")
    assert len(r2.hosts) == 1


def test_no_token_or_unknown_server(rig):
    r = rig(store_kw={"rows": {"A": ServerEntry("A", URL_A, Secret(""))}})
    r.ctl.connect("A")
    assert r.hosts == [] and r.probed == [] and r.ctl.notice == words.t("token_missing_test")
    r.ctl.connect("nope")
    assert r.hosts == [] and r.ctl.notice == words.t("no_such_server")


def test_protocol_offers_open_without_runner(rig):
    r = rig()
    r.ctl.connect("A")
    r.emit("refused", {"kind": "runner_protocol", "message": "the server speaks runner protocol 3"})
    assert r.window.views[-1]["can_open_without_runner"] is True
    assert r.log.of("inject") == []
    r.ctl.open_without_runner()
    (_, base, script, _), = r.log.of("inject")
    d = values_of(script)
    assert base == URL_A
    assert "strata.runner" in d["remove"] and "strata.runner" not in d["set"]
    assert json.loads(d["set"]["strata.apikey"]) == TOKEN_A
    assert r.log.of("open_app") == [("open_app", URL_A, "A")]
    assert r.store.injected == [ORIGIN_A]


def test_error_offers_open_without_runner(rig):
    r = rig()
    r.ctl.connect("A")
    r.emit("error", {"message": "runner.json: bad"})
    assert r.window.views[-1]["can_open_without_runner"] is True


def test_open_without_runner_with_nothing_chosen_does_nothing(rig):
    r = rig()
    r.ctl.open_without_runner()
    assert r.log.of("inject") == []


# ------------------------------------------------------------------------------------------------ the runner's thread
def test_on_state_returns_at_once(rig):
    """The host calls on_state on desktop-runner, holding its emit lock: the window work (which can block for 20 s)
    happens on another thread."""
    r = rig()
    r.ctl.connect("A")
    r.window.block = threading.Event()
    t0 = time.monotonic()
    r.hosts[0].emit("online", {"id": RID_1, "who": "ann"})
    r.hosts[0].emit("offline", {"message": "x"})
    took = time.monotonic() - t0
    assert took < 0.2, took
    deadline = time.monotonic() + 5
    while not r.log.of("inject") and time.monotonic() < deadline:
        time.sleep(0.01)
    assert r.log.of("inject")                                     # the worker is in navigate_and_inject, blocked
    assert threading.current_thread().name != "desktop-events"
    r.window.block.set()
    assert r.ctl.wait_idle(5)


def test_quit_during_an_inject_clears_after_it(rig):
    """A Quit that lands while the token is being written clears it after the write, never before (else the token
    would stay with no record)."""
    r = rig()
    r.ctl.connect("A")
    r.window.block = threading.Event()
    r.hosts[0].emit("online", {"id": RID_1, "who": "ann"})
    deadline = time.monotonic() + 5
    while not r.log.of("inject") and time.monotonic() < deadline:
        time.sleep(0.01)
    q = threading.Thread(target=r.ctl.quit, daemon=True)
    q.start()
    time.sleep(0.2)
    assert r.log.of("clear") == []                                # waits for the write to end
    r.window.block.set()
    q.join(5)
    names = r.log.names("inject", "clear", "open_app")
    assert names == ["inject", "clear"]                           # cleared after, and the app never opened
    assert r.store.injected == []


def test_late_event_of_an_old_runner_is_dropped(rig):
    r = rig()
    r.ctl.connect("A")
    old = r.hosts[0]
    r.ctl.connect("B")
    old.emit("online", {"id": RID_1, "who": "ann"})               # a late event of A: never injected into B
    assert r.ctl.wait_idle(5)
    assert r.log.of("inject") == []
    assert r.ctl.state.kind == "connecting" and r.ctl.state.server == "B"


# ------------------------------------------------------------------------------------------------ switch, remove
def test_switch_stops_old_runner_first(rig):
    r = rig()
    r.ctl.connect("A")
    r.emit("online", {"id": RID_1, "who": "ann"})
    r.ctl.connect("B")
    r.emit("online", {"id": RID_2, "who": "ann"})
    names = [(c[0], c[1] if len(c) > 1 else None) for c in r.log.calls
             if c[0] in ("host.stop", "host.stopped", "host.made", "clear", "inject")]
    assert names == [("host.made", URL_A), ("inject", URL_A), ("host.stop", URL_A), ("host.stopped", URL_A),
                     ("clear", URL_A), ("host.made", URL_B), ("inject", URL_B)]
    assert r.hosts[0].stopped and r.store.injected == [ORIGIN_B]
    assert r.log.of("clear")[0][3] == controller.CLEAR_S


def test_switch_menu_keeps_the_runner(rig):
    r = rig()
    r.ctl.connect("A")
    r.emit("online", {"id": RID_1, "who": "ann"})
    r.ctl.switch()
    assert r.log.names("hide_app", "show_servers")[-2:] == ["hide_app", "show_servers"]
    assert r.log.of("host.stop") == [] and r.log.of("clear") == []
    r.ctl.open()
    assert r.log.names()[-1] == "show_servers"                    # after Switch, Open shows the list


def test_open_shows_the_app_once_open(rig):
    r = rig()
    r.ctl.open()
    assert r.log.names()[-1] == "show_servers"
    r.ctl.connect("A")
    r.emit("online", {"id": RID_1, "who": "ann"})
    r.ctl.open()
    assert r.log.names()[-1] == "show_app"


def test_clear_false_keeps_injected_and_is_tried_again(rig):
    """P3-R1: the clear needs the server; when it cannot run the record stays, and the next clear tries again."""
    r = rig(window_kw={"clear_ok": False})
    r.ctl.connect("A")
    r.emit("online", {"id": RID_1, "who": "ann"})
    r.ctl.connect("B")                                          # A's clear fails
    assert r.store.injected == [ORIGIN_A]
    r.emit("online", {"id": RID_2, "who": "ann"})
    assert r.store.injected == [ORIGIN_A, ORIGIN_B]             # P3-R3: both listed
    r.window.clear_ok = True
    r.ctl.quit()
    cleared = [c[1] for c in r.log.of("clear")]
    assert cleared[-2:] == [URL_A, URL_B]                       # A tried again at Quit, then B
    assert r.store.injected == []


def test_failed_clear_then_another_inject_then_a_crash(rig):
    """I-1: server A is down, so the user switches to B; B is injected; the app crashes (no Quit).  The next start
    must clear A as well as B."""
    r = rig(window_kw={"clear_ok": False})
    r.ctl.connect("A")
    r.emit("online", {"id": RID_1, "who": "ann"})
    r.ctl.connect("B")                                          # A's clear fails: A is down
    r.emit("online", {"id": RID_2, "who": "ann"})               # B written
    on_disk = list(r.store.injected)                            # a crash here: what the next start reads
    assert on_disk == [ORIGIN_A, ORIGIN_B]
    r2 = rig(store_kw={"injected": on_disk})
    r2.ctl.start()
    assert [c[1] for c in r2.log.of("clear")] == [URL_A, URL_B]
    assert r2.store.injected == []


def test_quit_with_clear_false_keeps_injected(rig):
    r = rig(window_kw={"clear_ok": False})
    r.ctl.connect("A")
    r.emit("online", {"id": RID_1, "who": "ann"})
    r.ctl.quit()
    assert r.store.injected == [ORIGIN_A]
    assert r.log.of("set_injected")[-1] == ("set_injected", [ORIGIN_A])


def test_one_failed_clear_of_two_keeps_only_that_one(rig):
    r = rig(store_kw={"injected": [ORIGIN_A, ORIGIN_B]})
    r.window.clear_ok = True
    real = r.window.clear_storage
    r.window.clear_storage = lambda base, script, timeout=15.0: real(base, script, timeout) and base != URL_A
    r.ctl.start()
    assert r.store.injected == [ORIGIN_A]


def test_remove_open_server(rig):
    r = rig()
    r.ctl.connect("A")
    r.emit("online", {"id": RID_1, "who": "ann"})
    r.ctl.remove("A")
    names = r.log.names("host.stop", "clear", "set_injected", "store.remove", "show_servers")
    assert names[-5:] == ["host.stop", "clear", "set_injected", "store.remove", "show_servers"]
    assert r.store.injected == [] and "A" not in r.store.rows
    assert r.ctl.state.kind == "idle"


@pytest.mark.parametrize("told", [True, False])
def test_rename_then_remove_the_open_server(rig, told):
    """I-2: the open server renamed in the list, then removed: its runner stops and its token is cleared at the
    address it was opened at (path prefix kept) - whether or not the controller was told of the rename."""
    r = rig()
    r.ctl.connect("B")
    r.emit("online", {"id": RID_1, "who": "ann"})
    if told:
        api = gui.ServersApi(r.store, DEVICE, LABEL, r.ctl, probe=ok_probe)
        assert api.save("Lab", URL_B, None, "B") == {"ok": True}
    else:
        r.store.save("Lab", URL_B, None, "B")
    r.ctl.remove("Lab")
    assert r.hosts[0].stopped
    assert r.log.of("clear")[-1][1] == URL_B
    assert r.store.injected == [] and "Lab" not in r.store.rows
    assert r.ctl.state.kind == "idle"


def test_rename_follows_the_open_server(rig):
    r = rig()
    r.ctl.connect("B")
    r.emit("online", {"id": RID_1, "who": "ann"})
    r.ctl.saved("B", r.store.save("Lab", URL_B, None, "B"))
    r.ctl.remove("B")                                            # the old name no longer names it
    assert not r.hosts[0].stopped


def test_remove_a_server_whose_clear_failed_clears_it_first(rig):
    """M-3: its address (with its path prefix) is tried while the entry still gives it."""
    r = rig(store_kw={"injected": [ORIGIN_B]})
    r.ctl.remove("B")
    assert r.log.names("clear", "store.remove") == ["clear", "store.remove"]
    assert r.log.of("clear")[0][1] == URL_B and r.store.injected == []


def test_remove_a_listed_server_clears_only_its_origin(rig):
    """N-1: nothing open, A and B listed: removing B clears B alone."""
    r = rig(store_kw={"injected": [ORIGIN_A, ORIGIN_B]})
    r.ctl.remove("B")
    assert [c[1] for c in r.log.of("clear")] == [URL_B]
    assert r.store.injected == [ORIGIN_A]


def test_remove_a_listed_server_never_touches_the_open_one(rig):
    """N-1: A is open in the app window and B is listed: removing B leaves A's page, token and runner alone; B stays
    listed (its address kept) and is cleared when A is left."""
    r = rig(store_kw={"injected": [ORIGIN_B]}, window_kw={"clear_ok": False})
    r.ctl.connect("A")                                     # B's clear fails here (its server is down)
    r.window.clear_ok = True
    r.emit("online", {"id": RID_1, "who": "ann"})
    n = len(r.log.calls)
    r.ctl.remove("B")
    window_calls = [c[0] for c in r.log.calls[n:] if c[0] in ("clear", "inject", "hide_app", "open_app",
                                                              "show_servers", "host.stop")]
    assert window_calls == []
    assert not r.hosts[0].stopped and r.ctl.state.kind == "online"
    assert r.store.injected == [ORIGIN_B, ORIGIN_A]
    r.ctl.quit()
    assert [c[1] for c in r.log.of("clear")][-2:] == [URL_B, URL_A]     # B at its address, prefix kept
    assert r.store.injected == []


def test_remove_a_listed_server_on_the_connecting_origin_waits(rig):
    """B shares A's origin (one localStorage): while A is being connected B's clear would take A's token too."""
    rows = {"A": ServerEntry("A", URL_A, Secret(TOKEN_A)), "A2": ServerEntry("A2", URL_A + "/two", Secret(TOKEN_B))}
    r = rig(store_kw={"rows": rows, "injected": [ORIGIN_A]}, window_kw={"clear_ok": False})
    r.ctl.connect("A")
    r.window.clear_ok = True
    n = len(r.log.calls)
    r.ctl.remove("A2")
    assert [c for c in r.log.calls[n:] if c[0] == "clear"] == []
    assert not r.hosts[0].stopped


def test_two_entries_with_one_address_are_not_both_open(rig):
    rows = {"A": ServerEntry("A", URL_A, Secret(TOKEN_A)), "A2": ServerEntry("A2", URL_A, Secret(TOKEN_A))}
    r = rig(store_kw={"rows": rows})
    r.ctl.connect("A")
    r.emit("online", {"id": RID_1, "who": "ann"})
    r.ctl.remove("A2")
    assert not r.hosts[0].stopped and r.ctl.state.kind == "online"
    r.ctl.remove("A")
    assert r.hosts[0].stopped


def test_remove_other_server_leaves_the_open_one(rig):
    r = rig()
    r.ctl.connect("A")
    r.emit("online", {"id": RID_1, "who": "ann"})
    r.ctl.remove("B")
    assert r.log.of("host.stop") == [] and r.log.of("clear") == []
    assert "B" not in r.store.rows and r.ctl.state.kind == "online"


def test_remove_while_unreachable_then_restart_clears_from_the_origin(rig):
    r = rig(window_kw={"clear_ok": False})
    r.ctl.connect("B")
    r.emit("online", {"id": RID_1, "who": "ann"})
    r.ctl.remove("B")
    assert r.store.injected == [ORIGIN_B] and "B" not in r.store.rows
    r.ctl.quit()
    assert r.log.of("clear")[-1][1] == URL_B                     # this session still knows its address
    r2 = rig(store_kw={"rows": dict(r.store.rows), "injected": r.store.injected})
    r2.ctl.start()
    assert r2.log.calls[0][:2] == ("clear", ORIGIN_B)            # no entry names it: the origin alone
    assert r2.store.injected == []


# ------------------------------------------------------------------------------------------------ start
def test_crash_leftover_cleared_at_start(rig):
    r = rig(store_kw={"injected": [ORIGIN_B], "last": None})
    r.ctl.start()
    first = r.log.calls[0]
    assert first[0] == "clear" and first[1] == URL_B          # the entry with that origin: its path prefix kept
    assert first[2] == inject.clear_script()
    assert r.store.injected == []
    assert r.log.names().index("set_injected") < r.log.names().index("show_servers")


def test_crash_leftover_kept_when_the_server_is_down(rig):
    r = rig(store_kw={"injected": [ORIGIN_A]}, window_kw={"clear_ok": False})
    r.ctl.start()
    assert r.store.injected == [ORIGIN_A]
    assert "show_servers" in r.log.names()


def test_start_connects_to_the_last_server(rig):
    r = rig(store_kw={"last": "B"})
    r.ctl.start()
    assert r.log.names("show_servers", "host.made")[:2] == ["show_servers", "host.made"]
    assert r.hosts[0].url == URL_B


def test_start_without_last_only_shows_the_list(rig):
    r = rig()
    r.ctl.start()
    assert r.hosts == [] and r.log.names("show_servers") == ["show_servers"]


# ------------------------------------------------------------------------------------------------ quit
def test_quit_order(rig):
    r = rig()
    r.ctl.connect("A")
    r.emit("online", {"id": RID_1, "who": "ann"})
    n = len(r.log.calls)
    r.ctl.quit()
    after = [c[0] for c in r.log.calls[n:]
             if c[0] in ("host.stopped", "clear", "set_injected", "tray.stop", "destroy", "release")]
    assert after == ["host.stopped", "clear", "set_injected", "tray.stop", "destroy", "release"]
    assert r.log.of("set_injected")[-1] == ("set_injected", [])
    assert r.log.of("clear")[-1][3] == controller.CLEAR_S        # the 3 s cap
    assert r.ctl.wait_quit(1) and r.exits == []
    r.ctl.quit()                                                   # twice: nothing more
    assert r.log.names("tray.stop") == ["tray.stop"]


def test_session_end_quits_in_order_with_short_caps(rig):
    """I-2: Windows signs out or shuts down - the same Quit, with caps that fit in the few seconds Windows gives."""
    r = rig()
    r.ctl.connect("A")
    r.emit("online", {"id": RID_1, "who": "ann"})
    n = len(r.log.calls)
    t0 = time.monotonic()
    r.ctl.end_session()
    assert time.monotonic() - t0 < 2 and r.ctl.quitting and r.ctl.wait_quit(0)
    after = [c[0] for c in r.log.calls[n:]
             if c[0] in ("host.stopped", "clear", "set_injected", "tray.stop", "destroy", "release")]
    assert after == ["host.stopped", "clear", "set_injected", "tray.stop", "destroy", "release"]
    assert r.log.of("host.stop")[-1][2] == controller.SESSION_STOP_S
    assert r.log.of("clear")[-1][3] == controller.SESSION_CLEAR_S
    assert controller.SESSION_STOP_S + controller.SESSION_CLEAR_S < 4.0
    assert r.store.injected == [] and r.exits == []


def test_session_end_during_a_quit_waits_for_it(rig):
    r = rig()
    r.ctl.connect("A")
    r.hosts[0].hang = threading.Event()
    threading.Thread(target=r.ctl.quit, name="test-quit", daemon=True).start()
    while not r.ctl.quitting:
        time.sleep(0.01)
    threading.Timer(0.3, r.hosts[0].hang.set).start()
    t0 = time.monotonic()
    r.ctl.end_session()
    assert r.ctl.wait_quit(0) and 0.2 <= time.monotonic() - t0 < 3


def test_quit_ends_the_worker_and_the_watchdog(rig):
    r = rig()
    r.ctl.connect("A")
    r.emit("online", {"id": RID_1, "who": "ann"})
    watch = r.ctl._watch
    r.ctl.quit()
    r.ctl._worker.join(5)
    assert not r.ctl._worker.is_alive()
    assert watch is not None and not watch._thread.is_alive()


def test_quit_with_hung_stop_exits(rig):
    r = rig()
    r.ctl.connect("A")
    r.hosts[0].hang = threading.Event()
    t0 = time.monotonic()
    threading.Thread(target=r.ctl.quit, name="test-quit", daemon=True).start()
    while not r.exits and time.monotonic() - t0 < 12:
        time.sleep(0.05)
    took = time.monotonic() - t0
    assert r.exits == [0], took
    assert controller.QUIT_WATCHDOG_S - 0.5 <= took <= 10.5, took
    r.hosts[0].hang.set()


def test_watchdog_logs_the_hung_step(rig, caplog):
    r = rig(watchdog_s=0.2)
    r.ctl.connect("A")
    r.hosts[0].hang = threading.Event()
    with caplog.at_level("ERROR", logger="strata.desktop.controller"):
        threading.Thread(target=r.ctl.quit, daemon=True).start()
        deadline = time.monotonic() + 5
        while not r.exits and time.monotonic() < deadline:
            time.sleep(0.02)
    assert r.exits == [0]
    assert "hung at the runner" in caplog.text
    r.hosts[0].hang.set()


def test_quit_before_start_keeps_the_crash_record(rig):
    """Quit before start() (the web view failed, a tray Quit during start-up) must not drop the leftover's record."""
    r = rig(store_kw={"injected": [ORIGIN_A]}, window_kw={"clear_ok": False})
    r.ctl.quit()
    assert r.store.injected == [ORIGIN_A]
    r2 = rig(store_kw={"injected": [ORIGIN_A]})
    r2.ctl.quit()
    assert r2.log.of("clear")[0][1] == URL_A and r2.store.injected == []


def test_no_runner_starts_after_quit_during_a_connect(rig):
    """M-1: a Quit landing between the host being made and its start never sees a host that starts after it."""
    r = rig()
    started = threading.Event()

    def slow_start(host):
        def start():
            time.sleep(0.3)                                  # the gap a Quit could land in
            host.log.add("host.start", host.url)
            started.set()
        return start

    real_make = r.ctl._make_host

    def make(*a):
        h = real_make(*a)
        h.start = slow_start(h)
        threading.Thread(target=r.ctl.quit, daemon=True).start()     # Quit waits for the lock connect holds
        return h
    r.ctl._make_host = make
    r.ctl.connect("A")
    assert r.ctl.wait_quit(5)
    names = r.log.names("host.start", "host.stop")
    assert names == ["host.start", "host.stop"]                     # started before Quit stopped it, never after


def test_start_after_quit_does_nothing(rig):
    r = rig(store_kw={"injected": [ORIGIN_A], "last": "A"})
    r.ctl.quit()
    n = len(r.log.calls)
    r.ctl.start()
    assert r.log.calls[n:] == [] and r.hosts == []


def test_a_failing_quit_step_does_not_strand_the_rest(rig):
    r = rig()
    r.ctl.connect("A")
    r.emit("online", {"id": RID_1, "who": "ann"})

    def full_disk(origin):
        raise OSError(28, "No space left on device")
    r.store.set_injected = full_disk
    r.hosts[0].stop = lambda timeout=5.0: (_ for _ in ()).throw(RuntimeError("stop broke"))
    r.ctl.quit()
    assert r.log.names("tray.stop", "destroy", "release") == ["tray.stop", "destroy", "release"]
    assert r.ctl.wait_quit(1) and r.exits == []


def test_nothing_starts_after_quit(rig):
    r = rig()
    r.ctl.quit()
    r.ctl.connect("A")
    assert r.hosts == []


# ------------------------------------------------------------------------------------------------ copy code
def test_copy_code_only_while_pending(rig):
    r = rig()
    assert r.ctl.copy_code() is False and r.copied == []
    r.ctl.connect("A")
    r.emit("pending", {"code": "K7QZ<script>"})
    assert r.ctl.copy_code() is True and r.copied == ["K7QZscript"]


# ------------------------------------------------------------------------------------------------ the servers API
def test_api_never_returns_token(rig):
    """RF 1, with the real controller behind the servers window's API."""
    r = rig()
    api = gui.ServersApi(r.store, DEVICE, LABEL, r.ctl, probe=ok_probe)
    r.ctl.connect("A")
    r.emit("pending", {"code": "K7QZ"})
    r.emit("online", {"id": RID_1, "who": "ann"})
    out = [api.list(), api.warning(URL_A), api.save("C", "http://127.0.0.1:8100", TOKEN_B),
           api.save("C", "http://127.0.0.1:8100", None, "C"), api.test("A"), api.connect("B"), api.copy_code(),
           api.open_without_runner(), api.remove("C"), api.list()]
    for value in out:
        text = json.dumps(value)
        assert TOKEN_A not in text and TOKEN_B not in text, text
    assert repr(r.ctl).count("sk-") == 0 and "sk-" not in repr(r.ctl._open)


def test_list_carries_the_notice(rig):
    r = rig(probe_fn=lambda *a: probe.ProbeResult(False, error=words.t("unreachable")))
    api = gui.ServersApi(r.store, DEVICE, LABEL, r.ctl, probe=ok_probe)
    r.ctl.connect("A")
    assert api.list()["state"]["notice"] == words.t("unreachable")


# ------------------------------------------------------------------------------------------------ device notices
ROW = {"key": "bob", "device": "ZyXwVuTsRqPoNmLkJiHgFe", "label": "Strata on bob-pc", "addr": "100.64.0.7",
       "code": "Q2W8", "first_seen": 1}


def test_pending_notice_once_per_device():
    notes = []
    w = controller.PendingWatch(lambda: [dict(ROW)], lambda t, x: notes.append((t, x)), lambda: True)
    assert w.poll_once() == 1
    assert w.poll_once() == 0
    assert notes == [("A device asks to be let in",
                      "bob on Strata on bob-pc (100.64.0.7), code Q2W8. Open Strata Desktop, Settings > Keys, "
                      "to decide.")]


def test_pending_notice_errors_are_silent(caplog):
    def boom():
        raise OSError("refused " + TOKEN_A)
    w = controller.PendingWatch(boom, lambda t, x: None, lambda: True)
    with caplog.at_level("INFO"):
        assert w.poll_once() == 0
    assert TOKEN_A not in caplog.text and "OSError" in caplog.text


def test_pending_notice_cuts_and_cleans_server_text():
    notes = []
    row = {"key": "k\x07" + "x" * 300, "device": "d", "label": "l\nl", "addr": "a", "code": "<b>"}
    controller.PendingWatch(lambda: [row, {"key": "", "device": "x"}],
                            lambda t, x: notes.append((t, x)), lambda: True).poll_once()
    (title, text), = notes
    assert len(title) <= controller.NOTICE_TITLE_MOST and len(text) <= controller.NOTICE_TEXT_MOST
    assert "\x07" not in text and "\n" not in text and "<b>" not in text


def test_pending_watch_runs_while_online_and_ends_on_quit(rig):
    rows = []
    r = rig(fetch_pending=lambda *a, **k: list(rows), pending_every=0.05)
    r.ctl.connect("A")
    r.emit("pending", {"code": "K7QZ"})
    rows.append(dict(ROW))
    time.sleep(0.2)
    assert r.tray.notices == []                                   # not online yet: no question asked
    r.emit("online", {"id": RID_1, "who": "ann"})
    deadline = time.monotonic() + 5
    while not r.tray.notices and time.monotonic() < deadline:
        time.sleep(0.02)
    assert len(r.tray.notices) == 1
    time.sleep(0.2)
    assert len(r.tray.notices) == 1                               # the same row again: no second notice
    watch = r.ctl._watch
    r.ctl.quit()
    assert not watch._thread.is_alive()


class PendingServer(http.server.BaseHTTPRequestHandler):
    seen: list = []

    def do_GET(self):
        PendingServer.seen.append((self.path, dict(self.headers)))
        body = json.dumps({"devices": [ROW, "junk"]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def test_fetch_pending_sends_token_and_device_headers():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), PendingServer)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        PendingServer.seen = []
        rows = controller.fetch_pending(f"http://127.0.0.1:{srv.server_address[1]}/strata", TOKEN_A, DEVICE, LABEL)
    finally:
        srv.shutdown()
        srv.server_close()
    assert rows == [ROW]
    (path, headers), = PendingServer.seen
    assert path == "/strata/devices/pending"
    assert headers["Authorization"] == "Bearer " + TOKEN_A
    assert headers["X-Strata-Device"] == DEVICE["id"] and headers["X-Strata-Device-Name"] == LABEL


def test_clipboard_runs_clip_from_system32(monkeypatch):
    """A bare "clip.exe" is searched in the app's folder and the current directory before System32."""
    import os
    import subprocess
    seen = []

    class Done:
        returncode = 0

    monkeypatch.setattr(controller.os, "name", "nt")
    monkeypatch.setenv("SystemRoot", r"C:\WinRoot")
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: seen.append((argv, kw)) or Done())
    assert controller.copy_to_clipboard("K7QZ") is True
    (argv, kw), = seen
    assert argv == [os.path.join(r"C:\WinRoot", "System32", "clip.exe")]
    assert kw["creationflags"] == controller.NO_WINDOW and kw["input"] == b"K7QZ"
