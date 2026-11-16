"""Strata Desktop's two windows (#310 part 3).  The only module that imports `webview` (inside `Gui.run`); `pystray`
and `PIL` stay in `desktop/tray.py`, also inside functions.

- **servers** - the local page `desktop/ui/servers.html` (`html=`, its words filled in here), with the Python API
  `ServersApi` (`js_api`).  Nothing but this page ever loads in it, and no method of the API returns a token.
- **app** - the server's web app, created hidden and with no `js_api`.  `navigate_and_inject` loads `<base>/health`,
  checks the page is on the server's origin, and runs the localStorage script there (desktop/inject.py);
  `open_app` then loads `<base>/` and shows the window once it has loaded.

Nothing in the app window reaches Windows but a plain http(s) address: a new-window link goes through
`install_link_guard` (pywebview hands it to `webbrowser.open`, which is `os.startfile`), and a top-level navigation
of the app window off the open server's origin is cancelled (`_on_app_navigating`; a link the user clicked to another
http(s) address opens in the system browser).

Closing either window hides it (the app stays in the tray) unless the controller is quitting.  `webview.start` runs
with `gui="edgechromium"`, `private_mode=False` and `storage_path=<data>\\webview` (the WebView2 profile: chats survive a
restart); downloads are allowed (the web app saves files).  pywebview falls back to MSHTML without saying so when
Edge Chromium is missing: `desktop/webview2.py` checks before, and a fallback seen here gives the same box and exit 4.
"""
from __future__ import annotations

import dataclasses
import logging
import os
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
    "test_reachable", "role_owner", "role_guest", "engine_down", "pending", "internal_error",
)


WEB_SCHEMES = ("http", "https")
URL_MOST = 8192


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


def servers_html() -> str:
    """`ui/servers.html` with the words of the current language in place of `/*WORDS*/{}` and its `lang`."""
    text = SERVERS_HTML.read_text(encoding="utf-8")
    table = {k: words.t(k) for k in PAGE_WORDS}
    return text.replace("/*WORDS*/{}", inject.js_literal(table)).replace("__LANG__", words.LANG)


def state_view(s: state.TrayState) -> dict:
    """The runner's state as the servers page shows it: the TrayState's fields, the words for it, and whether the
    page offers "Open without My PC" (an old server, or the runner could not start)."""
    return {**dataclasses.asdict(s), "text": state.describe(s),
            "can_open_without_runner": s.kind == "error" or (s.kind == "refused" and s.refused == "runner_protocol")}


class ServersApi:
    """What the servers page calls (`window.pywebview.api.<method>`).  Every value it returns is JSON with no token
    (RF 1); a failure is `{"error": words}`, never an exception (pywebview would hand its text and traceback to the
    page).  Its state is kept in `_` attributes: pywebview lists every public attribute of a js_api object, and the
    public objects inside it, as callable from the page.

    `controller` gives `state` (the current TrayState) and `connect(name)`, `remove(name)`, `open_without_runner()`,
    `copy_code()` (desktop/controller.py, Task 5)."""

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
                                           "state": state_view(self._ctl.state)}, self._fail())

    def warning(self, url) -> str | None:
        return self._safe("warning", lambda: urlcheck.plain_http_warning(url), None)

    def save(self, name, url, token=None, old_name=None) -> dict:
        """Adds a server (`old_name` None) or changes `old_name`; an empty or missing token keeps the stored one."""
        def do():
            tok = token if isinstance(token, str) and token else None
            if tok is not None:
                applog.add_secret(tok)
            try:
                self._store.save(name, url, tok, old_name)
            except (ValueError, DpapiError) as e:     # their words name the field and never repeat the token
                return {"error": str(e)}
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
            self._ctl.copy_code()
            return {"ok": True}
        return self._safe("copy_code", do, self._fail())


class Gui:
    """The two windows (the controller's WindowPort, Task 5).  `quitting()` tells the closing handlers whether the
    controller is quitting; `servers_hidden` keeps the servers window hidden too (the selftest)."""

    def __init__(self, data: Path | None, api: ServersApi | None, quitting: Callable[[], bool] = lambda: False, *,
                 debug: bool = False, servers_hidden: bool = False):
        self._data, self._api, self._quitting = data, api, quitting
        self._debug, self._servers_hidden = debug, servers_hidden
        self._servers = None
        self._app = None
        self._app_lock = threading.Lock()        # one navigation of the app window at a time
        self._flag_lock = threading.Lock()
        self._show_pending = False               # open_app: show the app window at its next `loaded`
        self._ending = False                     # destroy(): the closing handlers let the windows go
        self._failed = False
        self._loads = 0                          # navigate_and_inject's count: each load has its own URL
        self._app_origin: str | None = None      # the only origin the app window may navigate to
        self._nav_guarded = False                # _on_app_navigating is on the app window's control

    def __repr__(self) -> str:
        return "Gui()"

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
        # would reach ShellExecute.  Only plain http(s) addresses get through.
        install_link_guard()
        self._servers = webview.create_window(words.t("servers_title"), html=servers_html(), js_api=self._api,
                                              width=SERVERS_SIZE[0], height=SERVERS_SIZE[1], min_size=SERVERS_MIN,
                                              hidden=self._servers_hidden)
        self._app = webview.create_window("Strata", hidden=True, width=APP_SIZE[0], height=APP_SIZE[1],
                                          min_size=APP_MIN)
        for w in (self._servers, self._app):
            w.events.closing += self._closing_handler(w)
        self._hook_app(self._app)

        def begin():
            if webview.renderer != "edgechromium":
                log.error("the web view runs on %s, not Edge Chromium", webview.renderer)
                self._failed = True
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
        if show:
            try:
                self._app.show()
            except Exception as e:
                log.error("showing the app window raised %s", type(e).__name__)

    # ------------------------------------------------------------------ the WindowPort
    def show_servers(self) -> None:
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
            try:
                self._app.load_url(url)                        # clears `loaded` first (webview/window.py load_url)
                if not self._app.events.loaded.wait(max(0.0, deadline - time.monotonic())):
                    log.info("the app window did not load %s/health in %g s", want, timeout)
                    return False
                try:
                    landed = inject.origin(self._app.get_current_url())
                except ValueError:
                    landed = None
                if landed != want:
                    log.info("the app window landed on another origin, not %s: nothing written", want)
                    return False
                path = urlsplit(inject.health_url(base)).path
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

    def hide_app(self) -> None:
        with self._flag_lock:
            self._show_pending = False
        self._app.hide()

    def destroy(self) -> None:
        """Closes both windows (Quit); pywebview's loop then ends and `run` returns."""
        self._ending = True
        for w in (self._app, self._servers):
            if w is None:
                continue
            try:
                w.destroy()
            except Exception as e:
                log.error("closing a window raised %s", type(e).__name__)
