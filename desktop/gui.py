"""Strata Desktop's two windows (#310 part 3).  The only module that imports `webview` (inside `Gui.run`); `pystray`
and `PIL` stay in `desktop/tray.py`, also inside functions.

- **servers** - the local page `desktop/ui/servers.html` (`html=`, its words filled in here), and the calls it makes:
  eight plain functions (`api_functions`, given to `window.expose`), never a `js_api` object - pywebview follows a
  dotted name from the page into any attribute of that object, `_` ones included.  Its own page is its only page: its
  first navigation (pywebview's NavigateToString of that page) is let through and every later one is cancelled
  (`_on_servers_navigating`), and every call is refused until that check is on.  No method returns a token.
- **app** - the server's web app, created hidden and with no `js_api`.  `navigate_and_inject` loads `<base>/health`,
  checks the page is on the server's origin, and runs the localStorage script there (desktop/inject.py);
  `open_app` then loads `<base>/` and shows the window once it has loaded.  `clear_storage` removes the token with no
  server at all: the window loads `<origin>/.strata-desktop/clear?n=<nonce>` and answers that one request itself
  (WebView2's WebResourceRequested, `_on_app_request`), so the page commits on the origin, with its localStorage.

The servers window takes no link or file dropped from outside (`AllowExternalDrop` off, set in pywebview's
`before_show`, on the UI thread before the first navigation).  The app window takes dropped files (#333: the chat's
attach-by-dropping, as in a browser): a file dropped on the chat reaches the page's drop handler; one dropped anywhere
else would navigate to a file: address, which `_on_app_navigating` cancels like any address off the server's origin.  Nothing in the app window reaches Windows but a plain http(s)
address: a new-window link goes through `install_link_guard` (pywebview hands it to `webbrowser.open`, which is
`os.startfile`), and a top-level navigation of the app window off the open server's origin is cancelled
(`_on_app_navigating`; a link the user clicked to another http(s) address opens in the system browser).

Closing either window hides it (the app stays in the tray) unless the controller is quitting.  When Windows closes them
(sign-out or shutdown, WM_QUERYENDSESSION; Task Manager), `_on_form_closing` runs Quit first (`on_session_end`, capped
at SESSION_END_S) and lets them go.  `webview.start` runs
with `gui="edgechromium"`, `private_mode=False` and `storage_path=<data>\\webview` (the WebView2 profile: chats survive a
restart); downloads are allowed (the web app saves files).  pywebview falls back to MSHTML without saying so when
Edge Chromium is missing: `desktop/webview2.py` checks before, and a fallback seen here gives the same box and exit 4.
"""
from __future__ import annotations

import base64
import dataclasses
import logging
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

from desktop import applog, inject, paths, probe, state, urlcheck, webview2, words
from desktop.dpapi import DpapiError

log = logging.getLogger("strata.desktop.gui")

UI = Path(__file__).resolve().parent / "ui"
SERVERS_HTML = UI / "servers.html"
SERVERS_SIZE, SERVERS_MIN = (440, 680), (320, 520)
APP_SIZE, APP_MIN = (1200, 820), (480, 400)

PAGE_WORDS = (
    "servers_title", "servers_empty", "label_name", "label_url", "label_token", "label_model", "token_set",
    "token_missing", "token_unreadable", "token_kept", "token_keep", "button_save", "button_test", "button_connect",
    "button_remove", "button_new", "button_open_without_runner", "menu_copy_code", "copied", "confirm_remove",
    "test_reachable", "role_owner", "role_guest", "engine_down", "pending", "internal_error", "servers_heading",
)


WEB_SCHEMES = ("http", "https")
URL_MOST = 8192

API_NAMES = ("list", "warning", "save", "remove", "test", "connect", "open_without_runner", "copy_code")
CLEAR_PATH = inject.CLEAR_PATH
LOCAL_PAGE = b"<!doctype html><meta charset=utf-8><meta name=strata-desktop content=clear><title>Strata Desktop</title>"
LOCAL_HEADERS = "Content-Type: text/html; charset=utf-8\r\nCache-Control: no-store"
SESSION_END_S = 4.0     # Windows ends the session: Quit gets this long (it shows "an app is preventing" after ~5 s)


def external_url_ok(url) -> bool:
    """True for a plain http(s) address that may go to the system browser: a host, no user name or password in it,
    no backslash (`http:\\\\host\\share` is a UNC path to ShellExecute), no space or control character, at most
    8192 characters.  Everything else - file:, a UNC path, ms-msdt:, search-ms:, javascript:, any other protocol
    handler - is refused."""
    if not isinstance(url, str) or not url or len(url) > URL_MOST:
        return False
    if "\\" in url or any(ord(c) <= 0x20 or ord(c) == 0x7F for c in url):
        return False
    try:
        u = urlsplit(url)
        host = u.hostname
        u.port
    except ValueError:
        return False
    return u.scheme.lower() in WEB_SCHEMES and bool(host) and "@" not in u.netloc and url[len(u.scheme):].startswith("://")


def navigation_ok(uri, allowed_origin: str | None) -> bool:
    """May the app window go to `uri`: about:blank, or a plain http(s) address (or a blob: of one) on
    `allowed_origin`, the open server's origin."""
    if uri == "about:blank":
        return True
    if not allowed_origin or not isinstance(uri, str):
        return False
    if uri[:5].lower() == "blob:":
        uri = uri[5:]
    if not external_url_ok(uri):
        return False
    try:
        return inject.origin(uri) == allowed_origin
    except ValueError:
        return False


def _scheme(url) -> str:
    """A URL's scheme for the log (never the URL: its query may hold anything)."""
    s = url.split(":", 1)[0] if isinstance(url, str) and ":" in url else ""
    s = "".join(c for c in s[:24] if c.isascii() and (c.isalnum() or c in "+-."))
    return repr(s or "(none)")


def guard_links(opener: Callable[..., bool]) -> Callable[..., bool]:
    """`opener` (webbrowser.open's signature) for addresses `external_url_ok` takes; anything else is dropped and
    logged by its scheme only."""
    def open_link(url, new=0, autoraise=True):
        if not external_url_ok(url):
            log.warning("a link with the scheme %s was not handed to Windows", _scheme(url))
            return False
        return opener(url, new, autoraise)
    open_link.strata_link_guard = True
    return open_link


def install_link_guard(opener: Callable[..., bool] | None = None) -> None:
    """Replaces `webbrowser.open` (the function pywebview's new-window handler calls) with `guard_links` of it, once.
    `opener` replaces the real opener under the guard (the hidden-window check records instead of opening)."""
    import webbrowser
    if getattr(webbrowser.open, "strata_link_guard", False) and opener is None:
        return
    webbrowser.open = guard_links(opener or webbrowser.open)


def _open_nothing(url, new=0, autoraise=True) -> bool:
    """The headless Gui's opener: a link is logged by its scheme and never handed to the system browser."""
    log.info("a link with the scheme %s was not opened: no window in the selftest", _scheme(url))
    return False


def _do_events() -> None:
    """One pass of the UI thread's message queue (WinForms, loaded by pywebview): what a call from another thread
    (`Invoke`, a script's answer, a navigation's events) needs while the UI thread waits in a handler."""
    import System.Windows.Forms as WinForms
    WinForms.Application.DoEvents()


def _drops(control, which: str, allow: bool) -> None:
    """Whether the window takes a link or a file dragged onto it from outside.  The servers window does not (WebView2
    would navigate to it); the app window does (#333), behind its navigation check."""
    try:
        control.AllowExternalDrop = allow
    except Exception as e:
        log.error("the %s window's drops from outside could not be set (%s)", which, type(e).__name__)


# #332: the web app's fonts (serve/ui/src/styles.css, the same OFL files and unicode ranges), inlined as data: fonts
# because the page is given to the web view as text (`html=`) and has no folder to load a file from.
LATIN = ("U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6, U+02DA, U+02DC, U+2000-206F, U+20AC, U+2122, U+2191, "
         "U+2193, U+2212, U+2215, U+FEFF, U+FFFD")
FONTS = (("Inter Variable", "100 900", "inter-latin-wght-normal.woff2", LATIN),
         ("Anuphan Variable", "100 700", "anuphan-thai-wght-normal.woff2", "U+0E01-0E5B, U+200C-200D, U+25CC"),
         ("Anuphan Variable", "100 700", "anuphan-latin-wght-normal.woff2", LATIN),
         ("JetBrains Mono Variable", "100 800", "jetbrains-mono-latin-wght-normal.woff2", LATIN))


def font_faces() -> str:
    """The @font-face rules of FONTS with each file as a data: URL; a file that is missing is left out (the page then
    falls back to the system font, as the web app's stack does)."""
    out = []
    for family, weight, name, ranges in FONTS:
        try:
            data = base64.b64encode((UI / "fonts" / name).read_bytes()).decode("ascii")
        except OSError:
            log.warning("font %s is missing; the window uses the system font", name)
            continue
        out.append(f'@font-face {{ font-family: "{family}"; font-style: normal; font-weight: {weight}; font-display: swap; '
                   f'src: url("data:font/woff2;base64,{data}") format("woff2-variations"); unicode-range: {ranges}; }}')
    return "\n  ".join(out)


def servers_html() -> str:
    """`ui/servers.html` with the words of the current language in place of `/*WORDS*/{}`, its `lang`, and the fonts in
    place of `/*FONTS*/`."""
    text = SERVERS_HTML.read_text(encoding="utf-8")
    table = {k: words.t(k) for k in PAGE_WORDS}
    return (text.replace("/*WORDS*/{}", inject.js_literal(table)).replace("__LANG__", words.LANG)
            .replace("/*FONTS*/", font_faces()))


def state_view(s: state.TrayState, notice: str = "") -> dict:
    """The runner's state as the servers page shows it: the TrayState's fields, the words for it, whether the page
    offers "Open without My PC" (an old server, or the runner could not start), and `notice`: what the last step
    could not do (the server test's words, "the window could not be prepared"), or ""."""
    return {**dataclasses.asdict(s), "text": state.describe(s), "notice": notice or "",
            "can_open_without_runner": s.kind == "error" or (s.kind == "refused" and s.refused == "runner_protocol")}


class ServersApi:
    """What the servers page calls (`window.pywebview.api.<method>`).  Every value it returns is JSON with no token
    (RF 1); a failure is `{"error": words}`, never an exception (pywebview would hand its text and traceback to the
    page).  It is never given to pywebview as a `js_api`: pywebview 6.2.1 resolves a call's name with a dotted
    `getattr` chain that does not stop at `_` (webview/util.py `js_bridge_call` -> `get_nested_attribute`), so a page
    could reach `_store.save` or, through `_ctl`, the whole process.  The window gets `api_functions(api)` instead.

    `controller` gives `state` (the current TrayState), `notice` (words or "") and `connect(name)`, `remove(name)`,
    `saved(old_name, entry)`, `open_without_runner()`, `copy_code() -> bool` (desktop/controller.py `Controller`)."""

    def __init__(self, store, device: dict, label: str, controller, probe=probe.test_server):
        self._store, self._device, self._label = store, device, label
        self._ctl, self._probe = controller, probe

    def __repr__(self) -> str:
        return "ServersApi()"

    def _safe(self, what: str, fn: Callable[[], object], default):
        try:
            return fn()
        except Exception as e:
            log.error("the servers window's %s raised %s", what, type(e).__name__, exc_info=True)
            return default

    def _fail(self) -> dict:
        return {"error": words.t("internal_error")}

    def list(self) -> dict:
        return self._safe("list", lambda: {"servers": self._store.public(), "last": self._store.last,
                                           "state": state_view(self._ctl.state, self._ctl.notice)}, self._fail())

    def warning(self, url) -> str | None:
        return self._safe("warning", lambda: urlcheck.plain_http_warning(url), None)

    def save(self, name, url, token=None, old_name=None) -> dict:
        """Adds a server (`old_name` None) or changes `old_name`; an empty or missing token keeps the stored one."""
        def do():
            tok = token if isinstance(token, str) and token else None
            if tok is not None:
                applog.add_secret(tok)
            try:
                entry = self._store.save(name, url, tok, old_name)
            except (ValueError, DpapiError) as e:     # their words name the field and never repeat the token
                return {"error": str(e)}
            self._ctl.saved(old_name, entry)          # a rename of the open server: the controller follows it
            return {"ok": True}
        return self._safe("save", do, self._fail())

    def remove(self, name) -> dict:
        def do():
            self._ctl.remove(name)
            return {"ok": True}
        return self._safe("remove", do, self._fail())

    def test(self, name) -> dict:
        """The server test (desktop/probe.py) with the stored token and this PC's device: reachable, model, role,
        pending_code, engine_running, error."""
        def do():
            entry = self._store.get(name)
            if entry is None:
                return dataclasses.asdict(probe.ProbeResult(False, error=words.t("no_such_server")))
            token = entry.token.reveal()
            if not token:
                return dataclasses.asdict(probe.ProbeResult(False, error=words.t("token_missing_test")))
            applog.add_secret(token)
            return dataclasses.asdict(self._probe(entry.url, token, self._device, self._label))
        return self._safe("test", do, self._fail())

    def connect(self, name) -> dict:
        def do():
            self._ctl.connect(name)
            return {"ok": True}
        return self._safe("connect", do, self._fail())

    def open_without_runner(self) -> dict:
        def do():
            self._ctl.open_without_runner()
            return {"ok": True}
        return self._safe("open_without_runner", do, self._fail())

    def copy_code(self) -> dict:
        def do():
            return {"ok": bool(self._ctl.copy_code())}
        return self._safe("copy_code", do, self._fail())


def api_functions(api: ServersApi, allowed: Callable[[], bool] = lambda: True) -> tuple:
    """`api`'s calls as plain functions named as the page calls them (API_NAMES), for `window.expose`: pywebview
    looks an exposed function up by its exact name (`window._functions`) and leaves `_js_api` None, so a dotted name
    from the page reaches nothing.  A call is refused - words, nothing done - while `allowed()` is False: the servers
    window's navigation check is not on, so the page in it may not be this app's."""
    def make(name: str):
        method = getattr(api, name)

        def call(*args):
            if not allowed():
                log.error("the servers window's %s was refused: its navigation check is not on", name)
                return {"error": words.t("internal_error")}
            return method(*args)
        call.__name__ = call.__qualname__ = name
        return call
    return tuple(make(n) for n in API_NAMES)


class Gui:
    """The two windows (desktop/controller.py's WindowPort).  `quitting()` tells the closing handlers whether the
    controller is quitting; `servers_hidden` keeps the servers window hidden at its creation (the hidden-window check);
    `headless` (the selftest) does that and more: no window is ever shown (every `show` is skipped, the app window's
    too) and no message box is raised - a failure is only logged and returned.  `on_session_end` runs Quit when Windows
    closes the windows (the controller's `end_session`); `pump` is one pass of the UI thread's messages (tests)."""

    def __init__(self, data: Path | None, api: ServersApi | None, quitting: Callable[[], bool] = lambda: False, *,
                 debug: bool = False, servers_hidden: bool = False, headless: bool = False,
                 on_session_end: Callable[[], None] | None = None, pump: Callable[[], None] | None = None):
        self._data, self._api, self._quitting = data, api, quitting
        self._debug, self._servers_hidden, self._headless = debug, servers_hidden or headless, headless
        self._on_session_end, self._pump = on_session_end, pump or _do_events
        self._servers = None
        self._app = None
        self._app_lock = threading.Lock()        # one navigation of the app window at a time
        self._flag_lock = threading.Lock()
        self._show_pending = False               # open_app: show the app window at its next `loaded`
        self._ending = False                     # destroy(): the closing handlers let the windows go
        self._destroyed = False
        self._failed = False
        self._loads = 0                          # navigate_and_inject's count: each load has its own URL
        self._app_origin: str | None = None      # the only origin the app window may navigate to
        self._nav_guarded = False                # _on_app_navigating is on the app window's control
        self._servers_guarded = False            # _on_servers_navigating is on the servers window's control
        self._servers_navigated = False          # its own page's navigation was seen: every later one is cancelled
        self._local_ready = threading.Event()    # _on_app_request is on the app window's CoreWebView2
        self._local_failed = False               # ... and it could not be put there: a clear needs the server
        self._local_page: str | None = None      # the one address _on_app_request answers now (a clear's load)
        self._session_thread: threading.Thread | None = None
        self._session_deadline = 0.0

    def __repr__(self) -> str:
        return "Gui()"

    def attach(self, api: ServersApi) -> None:
        """The servers window's API, given before `run` (desktop/__main__.py: the controller needs this Gui, and the
        API needs the controller)."""
        self._api = api

    # ------------------------------------------------------------------ the loop
    def run(self, start: Callable[[], None]) -> int:
        """Creates the windows and runs pywebview's loop on this (the main) thread; `start` runs on pywebview's own
        thread once the loop is up.  0 when the windows were closed; 4 (after the box) when Edge Chromium could not
        be used."""
        import webview

        os.environ.pop("PYWEBVIEW_GUI", None)            # it would override the forced gui
        for h in logging.getLogger("pywebview").handlers:
            h.setFormatter(applog.ScrubFormatter("[pywebview] %(message)s"))
        webview.settings["ALLOW_DOWNLOADS"] = True
        webview.settings["IGNORE_SSL_ERRORS"] = False    # with it, pywebview accepts any certificate: never
        # A window.open / target=_blank link goes to `webbrowser.open` (OPEN_EXTERNAL_LINKS_IN_BROWSER, on by default;
        # pywebview platforms/edgechromium.py on_new_window_request), which on Windows is os.startfile: any scheme
        # would reach ShellExecute.  Only plain http(s) addresses get through; headless, none (a browser is a window).
        install_link_guard(_open_nothing if self._headless else None)
        self._servers = webview.create_window(words.t("servers_title"), html=servers_html(),
                                              width=SERVERS_SIZE[0], height=SERVERS_SIZE[1], min_size=SERVERS_MIN,
                                              hidden=self._servers_hidden)
        if self._api is not None:                        # functions, never an object (see ServersApi)
            self._servers.expose(*api_functions(self._api, lambda: self._servers_guarded))
        self._app = webview.create_window("Strata", hidden=True, width=APP_SIZE[0], height=APP_SIZE[1],
                                          min_size=APP_MIN)
        for w in (self._servers, self._app):
            w.events.closing += self._closing_handler(w)
        # `before_show` runs on the UI thread right after the form is made, before WebView2 has started: before
        # either window's first navigation.
        self._servers.events.before_show += self._prepare_servers
        self._app.events.before_show += self._prepare_app
        self._hook_app(self._app)

        def begin():
            if webview.renderer != "edgechromium":
                log.error("the web view runs on %s, not Edge Chromium", webview.renderer)
                self._failed = True
                if not self._headless:
                    webview2.message_box(webview2.missing_message())
                self.destroy()
                return
            self._app.events.shown.wait(20)
            self._guard_navigation(self._app)
            start()

        try:
            webview.start(begin, gui="edgechromium", private_mode=False,
                          storage_path=str(paths.webview_dir(self._data)), debug=self._debug)
        except Exception as e:
            log.error("the web view could not start: %s", type(e).__name__, exc_info=True)
            if not self._headless:
                webview2.message_box(webview2.missing_message())
            return webview2.EXIT_MISSING
        return webview2.EXIT_MISSING if self._failed else 0

    def _closing_handler(self, window) -> Callable[[], bool]:
        def on_closing() -> bool:
            if self._ending or self._quitting():
                return True
            try:
                window.hide()
            except Exception as e:
                log.error("hiding a window raised %s", type(e).__name__)
            return False                              # cancels the close
        return on_closing

    # ------------------------------------------------------------------ before the first navigation (UI thread)
    def _prepare_servers(self, window) -> None:
        """No drop from outside, the navigation check (its own page only) and the session-end hook.  Without the
        check every call of the page is refused (`api_functions`): fail closed."""
        native = getattr(window, "native", None)
        control = getattr(native, "webview", None)
        _drops(control, "servers", False)
        try:
            control.NavigationStarting += self._on_servers_navigating
            self._servers_guarded = True
        except Exception as e:
            log.error("the servers window's navigation check could not be installed (%s): its calls are refused",
                      type(e).__name__)
        self._hook_closing(native)

    def _on_servers_navigating(self, sender, args) -> None:
        """The servers window shows its own page and nothing else: the first navigation is pywebview's
        NavigateToString of it (a `data:` URI; the only one seen in the hidden run), every later one - a dropped link
        or file, a script - is cancelled."""
        try:
            if not self._servers_navigated:
                self._servers_navigated = True
                return
            args.Cancel = True
            log.warning("the servers window was kept from going to a %s address", _scheme(str(args.Uri)))
        except Exception as e:
            try:
                args.Cancel = True
            except Exception:
                pass
            log.error("the servers window's navigation check raised %s; the navigation was cancelled",
                      type(e).__name__)

    def _prepare_app(self, window) -> None:
        """Files dropped from outside (#333), the session-end hook, and - once WebView2 has started - the local clear page."""
        native = getattr(window, "native", None)
        control = getattr(native, "webview", None)
        _drops(control, "app", True)
        try:
            control.CoreWebView2InitializationCompleted += self._on_app_core_ready
        except Exception as e:
            self._local_failed = True
            log.error("the app window's local page could not be hooked (%s): a clear needs the server",
                      type(e).__name__)
        self._hook_closing(native)

    def _on_app_core_ready(self, sender, args) -> None:
        """`_on_app_request` on the app window's CoreWebView2.  pywebview 6.2.1 has already asked for every request
        (`AddWebResourceRequestedFilter('*', All)` in on_webview_ready), so no filter of our own is needed."""
        try:
            if not bool(args.IsSuccess):
                raise RuntimeError("WebView2 did not start")
            sender.CoreWebView2.WebResourceRequested += self._on_app_request
        except Exception as e:
            self._local_failed = True
            log.error("the app window's local page could not be hooked (%s): a clear needs the server",
                      type(e).__name__)
            return
        self._local_ready.set()

    def _on_app_request(self, sender, args) -> None:
        """Answers the one address `clear_storage` armed (its nonce is new for each clear) with LOCAL_PAGE; every
        other request goes on to the network untouched."""
        want = self._local_page
        if want is None:
            return
        try:
            if str(args.Request.Uri) != want:
                return
            args.Response = self._respond(sender)
        except Exception as e:
            log.error("answering the local clear page raised %s", type(e).__name__)

    def _respond(self, core):
        from System import Array, Byte
        from System.IO import MemoryStream
        return core.Environment.CreateWebResourceResponse(MemoryStream(Array[Byte](LOCAL_PAGE)), 200, "OK",
                                                          LOCAL_HEADERS)

    # ------------------------------------------------------------------ Windows ends the session
    def _hook_closing(self, native) -> None:
        try:
            native.FormClosing += self._on_form_closing
        except Exception as e:
            log.error("a window's close hook could not be installed (%s): sign-out would not run Quit",
                      type(e).__name__)

    def _on_form_closing(self, sender, args) -> None:
        """WinForms' FormClosing, after pywebview's own handler (which asked `_closing_handler`, so a close outside
        Quit is cancelled by now).  The title bar's X and our own `destroy` (both UserClosing) are left as they are.
        Any other close - Windows signing out or shutting down (WM_QUERYENDSESSION: WindowsShutDown; a cancel there
        answers FALSE, "Strata Desktop is preventing sign-out"), Task Manager, Application.Exit - lets the window go
        and runs Quit first, once."""
        try:
            if self._ending:
                return
            reason = str(args.CloseReason)
            if reason == "UserClosing":
                return
            args.Cancel = False
            self._end_session(reason)
        except Exception as e:
            log.error("Windows closing the app: the close handler raised %s", type(e).__name__)

    def _end_session(self, reason: str) -> None:
        """Quit (`on_session_end`) on its own thread while this - the UI thread - keeps pumping messages: Quit's clear
        and its closing of the windows need this thread.  One Quit; every close waits on it (the other form's, a
        re-entrant one under the pump), all within one deadline, SESSION_END_S from the first."""
        if self._on_session_end is None:
            return
        with self._flag_lock:
            first = self._session_thread is None
            if first:
                self._session_thread = threading.Thread(target=self._run_session_end, name="desktop-session-end",
                                                        daemon=True)
                self._session_deadline = time.monotonic() + SESSION_END_S
            t = self._session_thread
        if first:
            log.warning("Windows closes the app (%s): Quit runs first", reason)
            t.start()
        while t.is_alive() and time.monotonic() < self._session_deadline:
            try:
                self._pump()
            except Exception as e:
                log.error("pumping the UI thread's messages raised %s", type(e).__name__)
                t.join(max(0.0, self._session_deadline - time.monotonic()))
                break
            t.join(0.02)

    def _run_session_end(self) -> None:
        try:
            self._on_session_end()
        except Exception as e:
            log.error("Quit at the session's end raised %s", type(e).__name__, exc_info=True)

    def _hook_app(self, window) -> None:
        window.events.loaded += self._on_app_loaded

    def _guard_navigation(self, window) -> bool:
        """Puts `_on_app_navigating` on the app window's WebView2 control (NavigationStarting).  Without it the app
        window is never used (`navigate_and_inject` is False, `open_app` does nothing): fail closed."""
        control = getattr(getattr(window, "native", None), "webview", None)
        try:
            control.NavigationStarting += self._on_app_navigating
        except Exception as e:
            log.error("the app window's navigation check could not be installed (%s): it stays unused",
                      type(e).__name__)
            return False
        self._nav_guarded = True
        return True

    def _on_app_navigating(self, sender, args) -> None:
        """The app window goes only to the open server's origin (and about:blank).  Anything else is cancelled; a
        link the user clicked to another http(s) address opens in the system browser (through the link guard)."""
        try:
            uri = str(args.Uri)
            if navigation_ok(uri, self._app_origin):
                return
            args.Cancel = True
            by_user = bool(args.IsUserInitiated) and not bool(args.IsRedirected)
            log.warning("the app window was kept from going to a %s address", _scheme(uri))
            if by_user and external_url_ok(uri):
                import webbrowser
                webbrowser.open(uri)
        except Exception as e:
            try:
                args.Cancel = True
            except Exception:
                pass
            log.error("the app window's navigation check raised %s; the navigation was cancelled", type(e).__name__)

    def _on_app_loaded(self) -> None:
        with self._flag_lock:
            show, self._show_pending = self._show_pending, False
        if show and not self._headless:
            try:
                self._app.show()
            except Exception as e:
                log.error("showing the app window raised %s", type(e).__name__)

    # ------------------------------------------------------------------ the WindowPort
    def show_servers(self) -> None:
        if not self._headless:
            self._servers.show()

    def hide_servers(self) -> None:
        self._servers.hide()

    def render(self, view: dict) -> None:
        """Hands the servers page a state (`state_view`) through its `window.render`."""
        try:
            self._servers.evaluate_js("window.render&&window.render(" + inject.js_literal(view) + ")")
        except Exception as e:
            log.error("rendering the state raised %s", type(e).__name__)

    def navigate_and_inject(self, base: str, script: str, timeout: float = 15.0) -> bool:
        """Loads `<base>/health` in the app window, and runs `script` there only when the page landed on `base`'s
        origin (scheme and host compared lower-cased), and the script itself checks `location.origin` again.  True only
        when the script evaluated to `true`; one deadline for the load.  The script (it holds the token) is never
        logged."""
        deadline = time.monotonic() + timeout
        try:
            want = inject.origin(base)
        except ValueError:
            return False
        if not self._nav_guarded:
            log.error("the app window has no navigation check: nothing written")
            return False
        with self._app_lock:
            self._app_origin = want
            self._loads += 1
            # A query that changes every time: the WebView2 control does not navigate when its Source is set to the
            # URL it already shows, so a second inject in a row would wait out its timeout (seen in the hidden-window
            # check).  The server routes /health on the path alone.
            url = f"{inject.health_url(base)}?desktop={self._loads}"
            return self._load_and_run(url, want, urlsplit(inject.health_url(base)).path, script, deadline, timeout)

    def clear_storage(self, base: str, script: str, timeout: float = 15.0) -> bool:
        """Runs `script` (the clear) on `base`'s origin with no server: the app window loads
        `<origin>/.strata-desktop/clear?n=<nonce>`, and `_on_app_request` answers that request itself (LOCAL_PAGE), so
        the page commits on that origin - its localStorage - without the network (hidden run: 32-63 ms on a closed
        port, an unresolvable host and https).  The answer is armed for this load only, and its nonce is new each
        time.  Before WebView2 has started it waits for the hook (within `timeout`); when the hook could not be put
        on, the clear goes to the server's /health as `navigate_and_inject` does (the server must answer)."""
        deadline = time.monotonic() + timeout
        try:
            want = inject.origin(base)
        except ValueError:
            return False
        if not self._nav_guarded:
            log.error("the app window has no navigation check: nothing cleared")
            return False
        if self._local_failed or not self._local_ready.wait(max(0.0, deadline - time.monotonic())):
            log.warning("the app window has no local clear page: the clear goes to the server")
            return self.navigate_and_inject(base, script, max(0.0, deadline - time.monotonic()))
        url = f"{want}{CLEAR_PATH}?n={secrets.token_hex(16)}"
        with self._app_lock:
            self._app_origin = want
            self._local_page = url
            try:
                return self._load_and_run(url, want, CLEAR_PATH, script, deadline, timeout)
            finally:
                self._local_page = None

    def _load_and_run(self, url: str, want: str, path: str, script: str, deadline: float, timeout: float) -> bool:
        """Under `_app_lock`: loads `url`, and runs `script` only when the page landed on `want` (and, in the page,
        on `path`); True only when it evaluated to `true`.  The script is never logged."""
        try:
            self._app.load_url(url)                            # clears `loaded` first (webview/window.py load_url)
            if not self._app.events.loaded.wait(max(0.0, deadline - time.monotonic())):
                log.info("the app window did not load %s%s in %g s", want, path, timeout)
                return False
            try:
                landed = inject.origin(self._app.get_current_url())
            except ValueError:
                landed = None
            if landed != want:
                log.info("the app window landed on another origin, not %s: nothing written", want)
                return False
            return self._app.evaluate_js(inject.guarded(script, want, path)) is True
        except Exception as e:
            log.error("preparing the app window raised %s", type(e).__name__)
            return False

    def open_app(self, base: str, server: str | None = None) -> None:
        """Loads `<base>/` in the app window, titled "Strata - <server>", and shows it when it has loaded."""
        if not self._nav_guarded:
            log.error("the app window has no navigation check: not opened")
            return
        with self._app_lock:
            self._app_origin = inject.origin(base)
            self._app.set_title(f"Strata - {server or self._app_origin}")
            with self._flag_lock:
                self._show_pending = True
            self._app.load_url(inject.app_url(base))

    def show_app(self) -> None:
        """The app window as it is (the tray's Open): no reload, so an answer being written goes on."""
        if not self._headless:
            self._app.show()

    def hide_app(self) -> None:
        with self._flag_lock:
            self._show_pending = False
        self._app.hide()

    def evaluate(self, which: str, script: str):
        """`script`'s value in the "app" or the "servers" window, or None when it raised (the selftest's reads; the
        script is never logged)."""
        window = self._app if which == "app" else self._servers
        try:
            return window.evaluate_js(script)
        except Exception as e:
            log.error("a script in the %s window raised %s", which, type(e).__name__)
            return None

    def wait_app_loaded(self, timeout: float) -> bool:
        """The app window's page has loaded (`open_app` and `navigate_and_inject` clear the event before they load)."""
        return bool(self._app.events.loaded.wait(timeout))

    def destroy(self) -> None:
        """Closes both windows (Quit); pywebview's loop then ends and `run` returns.  A second call does nothing."""
        if self._destroyed:
            return
        self._destroyed = self._ending = True
        for w in (self._app, self._servers):
            if w is None:
                continue
            try:
                w.destroy()
            except Exception as e:
                log.error("closing a window raised %s", type(e).__name__)
