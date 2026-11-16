"""desktop/test_gui.py - the parts of the windows that are not visual (#310 part 3 task 4): the servers page's text,
the Python API the servers page calls (no method returns a token, RF 1; nothing but its methods is exposed), and the
app window's prepare-then-open steps against a fake window (the origin check, `True` only).  No window is opened.

    python -m pytest desktop/test_gui.py -q
"""
from __future__ import annotations

import inspect
import json
import re
import threading
from dataclasses import dataclass

import pytest

from desktop import applog, gui, inject, probe, state, words
from desktop.dpapi import DpapiError
from desktop.servers import Secret, ServerEntry

TOKEN = "sk-strata-gui-SECRET-0123456789"
DEVICE = {"id": "AbCdEfGhIjKlMnOpQrStUv", "name": "ann-pc"}
LABEL = "Strata Desktop on ann-pc"


@pytest.fixture(autouse=True)
def english(monkeypatch):
    monkeypatch.setattr(words, "LANG", "en")
    yield
    applog.forget_secrets()


# ------------------------------------------------------------------------------------------------ the page
def test_servers_html_carries_the_words():
    html = gui.servers_html()
    m = re.search(r"const WORDS = (\{.*?\});", html, re.S)
    assert m, "the words were not filled in"
    table = json.loads(m.group(1))
    for key in gui.PAGE_WORDS:
        assert table[key] == words.t(key), key
    assert "/*WORDS*/" not in html
    assert "</script" not in m.group(1)


def test_servers_html_in_thai(monkeypatch):
    monkeypatch.setattr(words, "LANG", "th")
    html = gui.servers_html()
    table = json.loads(re.search(r"const WORDS = (\{.*?\});", html, re.S).group(1))
    assert table["button_connect"] == words.TABLE["button_connect"]["th"]
    assert 'lang="th"' in html


def test_servers_html_is_self_contained():
    html = gui.servers_html()
    assert not re.search(r"""<(script|link|img|iframe)[^>]+(src|href)\s*=\s*["']?(https?:)?//""", html, re.I)
    assert "@import" not in html
    assert "innerHTML" not in html                                  # server text goes in as textContent only
    assert "prefers-color-scheme: dark" in html
    assert "window.render" in html and "pywebviewready" in html


def test_page_words_exist():
    for key in gui.PAGE_WORDS:
        assert key in words.TABLE, key


def test_state_view():
    s = state.TrayState("pending", server="lab", code="K7QZ")
    v = gui.state_view(s)
    assert v["kind"] == "pending" and v["code"] == "K7QZ" and v["server"] == "lab"
    assert v["text"] == words.t("pending", server="lab", code="K7QZ")
    assert v["can_open_without_runner"] is False
    assert gui.state_view(state.TrayState("refused", server="lab", refused="runner_protocol",
                                          message="protocol 3"))["can_open_without_runner"] is True
    assert gui.state_view(state.TrayState("error", message="x"))["can_open_without_runner"] is True
    assert gui.state_view(state.TrayState("refused", refused="key"))["can_open_without_runner"] is False
    json.dumps(v)


# ------------------------------------------------------------------------------------------------ the API
class FakeStore:
    def __init__(self):
        self.rows = {"lab": ServerEntry("lab", "http://127.0.0.1:8099", Secret(TOKEN))}
        self.saved = []
        self.fail: Exception | None = None
        self.last = "lab"

    def public(self):
        return [{"name": e.name, "url": e.url, "token": "set"} for e in self.rows.values()]

    def get(self, name):
        return self.rows.get(name)

    def save(self, name, url, token, old_name=None):
        if self.fail:
            raise self.fail
        self.saved.append((name, url, token, old_name))
        return ServerEntry(name, url, Secret(token or ""))


class FakeController:
    def __init__(self):
        self.calls = []
        self.state = state.TrayState("online", server="lab", who="ann", runner_id="r0123456789abcdef")

    def connect(self, name):
        self.calls.append(("connect", name))

    def remove(self, name):
        self.calls.append(("remove", name))

    def open_without_runner(self):
        self.calls.append(("open_without_runner",))

    def copy_code(self):
        self.calls.append(("copy_code",))


def _api(probe_fn=None):
    store, ctl = FakeStore(), FakeController()
    seen = []

    def fake_probe(url, token, device, label, timeout=10.0):
        seen.append((url, token, device, label))
        return probe.ProbeResult(True, model="gyro", role="owner", engine_running=True)

    api = gui.ServersApi(store, DEVICE, LABEL, ctl, probe=probe_fn or fake_probe)
    return api, store, ctl, seen


def _exposed(obj) -> set[str]:
    """What pywebview's inject_pywebview lists for a js_api object (webview/util.py get_functions): public methods,
    and the public attributes of public objects, recursively."""
    found, seen = set(), set()

    def walk(o, base):
        if id(o) in seen:
            return
        seen.add(id(o))
        for name in dir(o):
            if name.startswith("_"):
                continue
            attr = getattr(o, name)
            full = f"{base}.{name}" if base else name
            if inspect.ismethod(attr) or inspect.isfunction(attr):
                found.add(full)
            elif inspect.isclass(attr) or (not callable(attr) and hasattr(attr, "__module__")):
                walk(attr, full)

    walk(obj, "")
    return found


def test_api_exposes_only_its_methods():
    api, *_ = _api()
    assert _exposed(api) == {"list", "warning", "save", "remove", "test", "connect", "open_without_runner",
                             "copy_code"}


def test_api_never_returns_token():
    api, store, ctl, seen = _api()
    out = [api.list(), api.warning("http://192.168.1.5:8091"), api.save("lab2", "http://h:1", TOKEN, None),
           api.test("lab"), api.connect("lab"), api.remove("lab"), api.open_without_runner(), api.copy_code(),
           api.test("nope")]
    store.fail = ValueError("bad")
    out.append(api.save("x", "y", TOKEN, None))
    for v in out:
        assert TOKEN not in json.dumps(v)
    assert seen and seen[0][1] == TOKEN                       # the probe got the token; the page did not


def test_list():
    api, store, ctl, _ = _api()
    got = api.list()
    assert got["servers"] == store.public()
    assert got["last"] == "lab"
    assert got["state"] == gui.state_view(ctl.state)


def test_warning():
    api, *_ = _api()
    assert api.warning("http://192.168.1.5:8091") == words.t("plain_http")
    assert api.warning("https://lab.example.com") is None
    assert api.warning(None) is None


def test_save_add_edit_and_keep_token():
    api, store, *_ = _api()
    assert api.save("lab2", "http://h:1", TOKEN, None) == {"ok": True}
    assert api.save("lab3", "http://h:1", "lab2", "lab2") == {"ok": True}         # a rename with a new token
    assert api.save("lab3", "http://h:2", None, "lab3") == {"ok": True}           # keep the stored token
    assert api.save("lab4", "http://h:2", "", "lab3") == {"ok": True}             # an empty field keeps it too
    assert store.saved == [("lab2", "http://h:1", TOKEN, None), ("lab3", "http://h:1", "lab2", "lab2"),
                           ("lab3", "http://h:2", None, "lab3"), ("lab4", "http://h:2", None, "lab3")]
    assert applog.scrub("x " + TOKEN) == "x ***"                                  # a saved token is blanked in the log


def test_save_errors_are_words():
    api, store, *_ = _api()
    store.fail = ValueError("A server with this name is already in the list.")
    assert api.save("lab", "http://h:1", None, None) == {"error": "A server with this name is already in the list."}
    store.fail = DpapiError("the token store needs Windows")
    assert api.save("lab", "http://h:1", TOKEN, None) == {"error": "the token store needs Windows"}
    store.fail = RuntimeError("boom " + TOKEN)
    assert api.save("lab", "http://h:1", TOKEN, None) == {"error": words.t("internal_error")}


def test_test_sends_the_device_and_label():
    api, store, ctl, seen = _api()
    got = api.test("lab")
    assert seen == [("http://127.0.0.1:8099", TOKEN, DEVICE, LABEL)]
    assert got["reachable"] is True and got["role"] == "owner" and got["model"] == "gyro"
    assert set(got) == {"reachable", "model", "role", "pending_code", "engine_running", "error"}


def test_test_without_token():
    api, store, ctl, seen = _api()
    store.rows["empty"] = ServerEntry("empty", "http://h:1", Secret(""))
    got = api.test("empty")
    assert got["reachable"] is False and got["error"] == words.t("token_missing_test")
    assert seen == []
    assert api.test("nope")["error"] == words.t("no_such_server")


def test_controller_calls():
    api, store, ctl, _ = _api()
    assert api.connect("lab") == {"ok": True}
    assert api.remove("lab") == {"ok": True}
    assert api.open_without_runner() == {"ok": True}
    assert api.copy_code() == {"ok": True}
    assert ctl.calls == [("connect", "lab"), ("remove", "lab"), ("open_without_runner",), ("copy_code",)]


def test_controller_error_is_words():
    api, store, ctl, _ = _api()

    def boom(name):
        raise RuntimeError("x " + TOKEN)

    ctl.connect = boom
    assert api.connect("lab") == {"error": words.t("internal_error")}


# ------------------------------------------------------------------------------------------------ the app window
class FakeEvent:
    def __init__(self):
        self._e = threading.Event()
        self.handlers = []

    def wait(self, timeout=None):
        return self._e.wait(timeout)

    def set(self):
        self._e.set()

    def clear(self):
        self._e.clear()

    def __iadd__(self, fn):
        self.handlers.append(fn)
        return self


@dataclass
class FakeEvents:
    loaded: FakeEvent
    closing: FakeEvent


class FakeWindow:
    """pywebview's Window as the Gui uses it: load_url clears `loaded`, then the page "loads" at `lands_on(url)`."""

    def __init__(self, lands_on=lambda url: url, result=True, loads=True):
        self.events = FakeEvents(FakeEvent(), FakeEvent())
        self.lands_on, self.result, self.loads = lands_on, result, loads
        self.url, self.calls, self.scripts = None, [], []

    def load_url(self, url):
        self.calls.append(("load_url", url))
        self.events.loaded.clear()
        if url == self.url:                     # the WebView2 control: the same Source again does not navigate
            return
        self.url = self.lands_on(url)
        if self.loads:
            self.events.loaded.set()

    def get_current_url(self):
        return self.url

    def evaluate_js(self, script):
        self.calls.append(("evaluate_js",))
        self.scripts.append(script)
        return self.result

    def set_title(self, title):
        self.calls.append(("set_title", title))

    def show(self):
        self.calls.append(("show",))

    def hide(self):
        self.calls.append(("hide",))


def _gui(window) -> gui.Gui:
    g = gui.Gui(data=None, api=None)
    g._app = window
    g._nav_guarded = True
    return g


def test_without_the_navigation_check_nothing_is_written_or_opened():
    w = FakeWindow()
    g = gui.Gui(data=None, api=None)
    g._app = w
    assert g.navigate_and_inject("http://h:1", "true", timeout=1) is False
    g.open_app("http://h:1")
    assert w.calls == []


def test_navigate_and_inject_true():
    w = FakeWindow()
    g = _gui(w)
    assert g.navigate_and_inject("http://Lab:8099/p", "true", timeout=2) is True
    assert w.calls[0][0] == "load_url" and w.calls[0][1].startswith("http://Lab:8099/p/health?")
    assert w.calls[1] == ("evaluate_js",)
    assert 'location.origin' in w.scripts[0] and '"http://lab:8099"' in w.scripts[0]
    assert 'location.pathname!=="/p/health"' in w.scripts[0]


def test_inject_twice_in_a_row_navigates_twice():
    """The control ignores a Source equal to the current one; each inject must still load (seen in the hidden-window
    check: the second inject at the same /health waited out its timeout)."""
    w = FakeWindow()
    g = _gui(w)
    assert g.navigate_and_inject("http://h:1", "true", timeout=1) is True
    assert g.navigate_and_inject("http://h:1", "true", timeout=1) is True
    urls = [c[1] for c in w.calls if c[0] == "load_url"]
    assert len(urls) == 2 and urls[0] != urls[1]


@pytest.mark.parametrize("result", [False, None, "true", 1, {"x": 1}])
def test_navigate_and_inject_only_true(result):
    assert _gui(FakeWindow(result=result)).navigate_and_inject("http://h:1", "x", timeout=2) is False


def test_no_inject_on_another_origin():
    w = FakeWindow(lands_on=lambda url: "https://login.example.com/")
    assert _gui(w).navigate_and_inject("http://h:1", "x", timeout=2) is False
    assert ("evaluate_js",) not in w.calls


def test_no_inject_on_an_error_page():
    w = FakeWindow(lands_on=lambda url: "chrome-error://chromewebdata/")
    assert _gui(w).navigate_and_inject("http://h:1", "x", timeout=2) is False
    assert ("evaluate_js",) not in w.calls


def test_no_inject_when_never_loaded():
    w = FakeWindow(loads=False)
    assert _gui(w).navigate_and_inject("http://h:1", "x", timeout=0.2) is False
    assert ("evaluate_js",) not in w.calls


def test_inject_error_is_false():
    w = FakeWindow()

    def boom(script):
        raise RuntimeError("js failed")

    w.evaluate_js = boom
    assert _gui(w).navigate_and_inject("http://h:1", "x", timeout=2) is False


def test_open_app_shows_on_loaded():
    w = FakeWindow(loads=False)
    g = _gui(w)
    g._hook_app(w)
    g.open_app("http://h:1/p", server="lab")
    assert ("set_title", "Strata - lab") in w.calls
    assert ("load_url", "http://h:1/p/") in w.calls
    assert ("show",) not in w.calls
    for fn in w.events.loaded.handlers:
        fn()
    assert w.calls[-1] == ("show",)
    for fn in w.events.loaded.handlers:                            # a later load (the page's own reload) shows nothing
        fn()
    assert w.calls.count(("show",)) == 1


def test_hide_app_cancels_a_pending_show():
    w = FakeWindow(loads=False)
    g = _gui(w)
    g._hook_app(w)
    g.open_app("http://h:1")
    g.hide_app()
    for fn in w.events.loaded.handlers:
        fn()
    assert ("show",) not in w.calls


def test_closing_hides_unless_quitting():
    quitting = [False]
    g = gui.Gui(data=None, api=None, quitting=lambda: quitting[0])
    w = FakeWindow()
    handler = g._closing_handler(w)
    assert handler() is False
    assert ("hide",) in w.calls
    quitting[0] = True
    assert handler() is not False


# ------------------------------------------------------------------------------------------------ links and navigation
REFUSED = [
    "file:///C:/Windows/System32/calc.exe",
    "FILE://host/share/x.exe",
    "\\\\host\\share\\x.exe",
    "//host/share/x.exe",
    "http:\\\\host\\share\\x.exe",
    "https://host\\..\\x",
    "ms-msdt:/id PCWDiagnostic /skip force",
    "search-ms:query=x&crumb=location:\\\\host\\share",
    "ms-settings:privacy",
    "javascript:alert(1)",
    "data:text/html,<b>x</b>",
    "vbscript:x",
    "mailto:a@b.c",
    "steam://run/1",
    "C:\\Windows\\System32\\calc.exe",
    "calc.exe",
    "https://user:pass@example.com/",
    "https://evil.com@example.com/",
    "http:example.com",
    "https:/example.com",
    "https://",
    "https://exa mple.com/",
    "https://example.com/\nx",
    "https://example.com:99999/",
    "",
    None,
    5,
    "https://example.com/" + "a" * 9000,
]
ALLOWED = ["https://example.com/", "http://127.0.0.1:8099/x?y=1#z", "HTTPS://Example.COM/a%20b", "http://[::1]:1/"]


@pytest.mark.parametrize("url", REFUSED)
def test_link_refused(url):
    assert gui.external_url_ok(url) is False


@pytest.mark.parametrize("url", ALLOWED)
def test_link_allowed(url):
    assert gui.external_url_ok(url) is True


def test_guard_passes_only_web_links_and_logs_no_url(caplog):
    opened = []
    open_link = gui.guard_links(lambda url, new=0, autoraise=True: opened.append(url) or True)
    for url in REFUSED:
        assert open_link(url) is False
    assert opened == []
    assert open_link("https://example.com/?q=secret-query") is True
    assert opened == ["https://example.com/?q=secret-query"]
    open_link("ms-msdt:/id PCWDiagnostic /param secret-query")
    assert "secret-query" not in caplog.text and "PCWDiagnostic" not in caplog.text
    assert "'ms-msdt'" in caplog.text


def test_install_link_guard_wraps_webbrowser_open(monkeypatch):
    """The wiring: pywebview's new-window handler calls `webbrowser.open`; after the guard, that refuses file: and
    passes https to what was there before."""
    import webbrowser
    opened = []
    monkeypatch.setattr(webbrowser, "open", lambda url, new=0, autoraise=True: opened.append(url) or True)
    gui.install_link_guard()
    gui.install_link_guard()                                    # once: not wrapped twice
    assert webbrowser.open("file:///C:/Windows/System32/calc.exe") is False
    assert webbrowser.open("\\\\host\\share\\x.exe") is False
    assert webbrowser.open_new_tab("ms-msdt:/id x") is False   # the module's other entry points go through it too
    assert webbrowser.open("https://example.com/") is True
    assert opened == ["https://example.com/"]


def test_pywebview_new_windows_go_through_webbrowser_open():
    """The guard sits where pywebview 6.2.1 sends a new-window request (only where pywebview is installed: the venv)."""
    import importlib.util
    spec = importlib.util.find_spec("webview")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("pywebview is not installed in this python")
    from pathlib import Path
    src = (Path(list(spec.submodule_search_locations)[0]) / "platforms" / "edgechromium.py").read_text(encoding="utf-8")
    body = src[src.index("def on_new_window_request"):]
    body = body[:body.index("\n    def ", 10)]
    assert "webbrowser.open(" in body
    assert "import webbrowser" in src


@pytest.mark.parametrize("uri, ok", [
    ("http://h:1/health?desktop=1", True),
    ("http://H:1/#/chat", True),
    ("blob:http://h:1/0f0e", True),
    ("about:blank", True),
    ("http://h:2/", False),
    ("https://h:1/", False),
    ("http://evil:1/", False),
    ("file:///C:/x", False),
    ("ms-msdt:/id x", False),
    ("javascript:alert(1)", False),
    ("data:text/html,x", False),
    ("blob:http://evil:1/0f0e", False),
    ("http://user@h:1/", False),
    ("about:srcdoc", False),
])
def test_navigation_ok(uri, ok):
    assert gui.navigation_ok(uri, "http://h:1") is ok


def test_navigation_without_a_server_goes_nowhere():
    assert gui.navigation_ok("http://h:1/", None) is False
    assert gui.navigation_ok("about:blank", None) is True


class NavArgs:
    def __init__(self, uri, user=False, redirected=False):
        self.Uri, self.IsUserInitiated, self.IsRedirected, self.Cancel = uri, user, redirected, False


class FakeControl:
    def __init__(self):
        self.NavigationStarting = FakeEvent()


class FakeNative:
    def __init__(self):
        self.webview = FakeControl()


def test_app_navigation_is_checked(monkeypatch):
    import webbrowser
    opened = []
    monkeypatch.setattr(webbrowser, "open", lambda url, new=0, autoraise=True: opened.append(url) or True)
    gui.install_link_guard()
    w = FakeWindow()
    w.native = FakeNative()
    g = gui.Gui(data=None, api=None)
    g._app = w
    assert g._guard_navigation(w) is True
    handler, = w.native.webview.NavigationStarting.handlers
    assert g.navigate_and_inject("http://h:1", "true", timeout=1) is True        # sets the allowed origin

    same = NavArgs("http://h:1/#/chat")
    handler(None, same)
    assert same.Cancel is False
    for uri in ("ms-msdt:/id x", "file:///C:/Windows/win.ini", "\\\\host\\share\\x"):
        a = NavArgs(uri, user=True)
        handler(None, a)
        assert a.Cancel is True
    scripted = NavArgs("https://elsewhere.example/")
    handler(None, scripted)
    assert scripted.Cancel is True
    redirected = NavArgs("https://elsewhere.example/", user=True, redirected=True)
    handler(None, redirected)
    assert redirected.Cancel is True
    clicked = NavArgs("https://elsewhere.example/doc", user=True)
    handler(None, clicked)
    assert clicked.Cancel is True
    assert opened == ["https://elsewhere.example/doc"]                          # only the clicked web link


def test_navigation_check_fails_closed():
    g = gui.Gui(data=None, api=None)

    class Broken:
        @property
        def Uri(self):
            raise RuntimeError("x")
        Cancel = False

    a = Broken()
    g._on_app_navigating(None, a)
    assert a.Cancel is True
    w = FakeWindow()
    w.native = None
    assert g._guard_navigation(w) is False
    assert g._nav_guarded is False
