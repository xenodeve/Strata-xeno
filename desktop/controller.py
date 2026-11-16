"""What orders Strata Desktop's steps (#310 part 3): connect, switch server, remove, quit, the clean-up a crash left,
the tray's state and the notice when a device asks to be let in.  No GUI import: the windows and the tray come in
through two ports (desktop/gui.py `Gui`, desktop/tray.py `Tray`), so the tests drive it with fakes.

A connect, in order (RF 3):
    stop the runner of the server open before (its jobs and processes end), remove the token from every web origin
    that may hold one -> test the server (desktop/probe.py) -> start a RunnerHost -> on its first `online`: write the
    token, the device and the runner id into the app window's storage (`navigate_and_inject`) -> load the app.
Nothing is written before the runner is online, except "Open without My PC".

Threads.  The runner host calls `on_state` on `desktop-runner`; it only puts the event on a queue and returns, and
the thread `desktop-events` does the window work (render, inject, open), which can block for seconds.  An event of a
host that is no longer the current one is dropped.  connect / remove / open_without_runner run one at a time (they
come from the servers window's js_api threads); quit does not wait for them: it marks the controller as quitting
first, and every later step of theirs checks that before it starts a runner or opens a window.  `desktop-pending`
asks the server for devices that wait for approval every 30 s while the runner is online.  Quit starts
`desktop-quit-watchdog`, which ends the process after 10 s if a step hangs, and logs the step.

The token in the web view's storage (rulings P3-R1, P3-R3): `servers.json` `injected` lists every origin that may
hold it - the one open, and each whose clear failed.  An origin is written there *before* the token is (a crash in
between leaves a record, not a token nobody knows of), and leaves the list only when its clear script answered True:
the clear runs on that server's own page, so it needs the server to answer, and a False keeps the origin for the next
try (the next switch, Quit, or the next start, which clears every listed origin first).  The address each origin was
opened at is remembered for the session, so a server behind a path prefix is cleared there even after its entry was
renamed or removed.
"""
from __future__ import annotations

import http.client
import json
import logging
import os
import queue
import ssl
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol
from urllib.parse import urlsplit

from desktop import applog, inject, probe, state, words
from desktop.runner_host import RunnerHost
from desktop.state import TrayState

log = logging.getLogger("strata.desktop.controller")

STOP_S = 5.0                    # a runner's stop: its jobs, their processes, its thread, its executor
CLEAR_S = 3.0                   # one clear of the token on one origin (Quit's cap, and every other clear's)
QUIT_WATCHDOG_S = 10.0          # Quit ends the process after this, whatever hangs
PENDING_EVERY_S = 30.0          # the devices that wait for approval: asked this often while online
PENDING_TIMEOUT_S = 10.0
PENDING_BODY_MOST = 1 << 20
NOTICE_TITLE_MOST = 63          # NOTIFYICONDATAW szInfoTitle: 64 WCHAR with the closing NUL
NOTICE_TEXT_MOST = 255          # szInfo: 256
FIELD_MOST = 80
CODE_MOST = 16
NO_WINDOW = 0x08000000          # CREATE_NO_WINDOW


class WindowPort(Protocol):
    def show_servers(self) -> None: ...
    def hide_servers(self) -> None: ...
    def render(self, view: dict) -> None: ...
    def navigate_and_inject(self, base: str, script: str, timeout: float = 15.0) -> bool: ...
    def open_app(self, base: str, server: str | None = None) -> None: ...
    def show_app(self) -> None: ...
    def hide_app(self) -> None: ...
    def destroy(self) -> None: ...


class TrayPort(Protocol):
    def update(self, s: TrayState) -> None: ...
    def notify(self, title: str, text: str) -> None: ...
    def stop(self) -> None: ...


@dataclass
class _Open:
    """The server chosen now: its name, address, origin and token, and the runner id the app window was opened with
    (None until it was)."""
    name: str
    base: str
    origin: str
    token: str
    opened_id: str | None = None
    opened: bool = False

    def __repr__(self) -> str:
        return f"_Open({self.name!r}, {self.base!r}, opened_id={self.opened_id!r})"


def _plain(v, most: int = FIELD_MOST) -> str:
    """Server text for a notice: printable characters only, cut."""
    s = v if isinstance(v, str) else "" if v is None else str(v)
    return "".join(c for c in s if c.isprintable()).strip()[:most]


def code_text(v) -> str:
    """A pending code as it may be copied: letters, digits and `-` only."""
    return "".join(c for c in str(v or "") if c.isascii() and (c.isalnum() or c == "-"))[:CODE_MOST]


def copy_to_clipboard(text: str) -> bool:
    """`text` (a pending code: ASCII) onto the Windows clipboard through `clip.exe`, with no window."""
    if os.name != "nt" or not text:
        return False
    try:
        r = subprocess.run(["clip.exe"], input=text.encode("ascii", "replace"), timeout=5,
                           creationflags=NO_WINDOW, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError) as e:
        log.warning("copying the code raised %s", type(e).__name__)
        return False


def fetch_pending(base: str, token: str, device: dict, label: str, timeout: float = PENDING_TIMEOUT_S) -> list[dict]:
    """GET `<base>/devices/pending` with the token and this PC's device headers (desktop/probe.py's): the devices
    that wait for approval and that this key may decide.  http.client: no redirect, no proxy.  Raises on any failure
    (the caller logs its type only)."""
    u = urlsplit(base)
    host, port = u.hostname, u.port
    if u.scheme not in ("http", "https") or not host:
        raise ValueError("not an http(s) address")
    headers = {**probe.device_headers(device, label), "Authorization": "Bearer " + token,
               "User-Agent": probe.USER_AGENT, "Accept": "application/json"}
    if u.scheme == "https":
        conn = http.client.HTTPSConnection(host, port, timeout=timeout, context=ssl.create_default_context())
    else:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request("GET", u.path.rstrip("/") + "/devices/pending", headers=headers)
        r = conn.getresponse()
        raw = r.read(PENDING_BODY_MOST)
    finally:
        conn.close()
    if r.status != 200:
        raise RuntimeError(f"the server answered {r.status}")
    body = json.loads(raw)
    rows = body.get("devices") if isinstance(body, dict) else None
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


class PendingWatch:
    """Thread `desktop-pending`: while `active()` (the runner is online), asks for the waiting devices at once and
    then every `every` seconds; each (key, device) not seen before gives one tray notice.  Errors are logged by type
    and the watch goes on.  Deciding stays in the web app (Settings > Keys)."""

    def __init__(self, fetch: Callable[[], list[dict]], notify: Callable[[str, str], None],
                 active: Callable[[], bool], every: float = PENDING_EVERY_S):
        self._fetch, self._notify, self._active, self._every = fetch, notify, active, every
        self._seen: set[tuple[str, str]] = set()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="desktop-pending", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        wait = 0.0
        while not self._stop.wait(wait):
            wait = self._every
            try:
                if self._active():
                    self.poll_once()
            except Exception as e:  # noqa: BLE001 - the watch outlives one bad answer
                log.warning("the device notice raised %s", type(e).__name__)

    def poll_once(self) -> int:
        """One question to the server; the count of notices given."""
        try:
            rows = self._fetch()
        except Exception as e:  # noqa: BLE001 - a notice is a convenience: never more than a log line
            log.info("asking for the waiting devices failed: %s", type(e).__name__)
            return 0
        given = 0
        for row in rows:
            key, dev = _plain(row.get("key")), _plain(row.get("device"), 64)
            if not key or not dev or (key, dev) in self._seen or self._stop.is_set():
                continue
            self._seen.add((key, dev))
            title = words.t("notice_device_title")[:NOTICE_TITLE_MOST]
            text = words.t("notice_device_text", key=key, label=_plain(row.get("label")) or "?",
                           addr=_plain(row.get("addr"), 64) or "?", code=code_text(row.get("code")) or "?")
            try:
                self._notify(title, text[:NOTICE_TEXT_MOST])
                given += 1
            except Exception as e:  # noqa: BLE001
                log.warning("the tray notice raised %s", type(e).__name__)
        return given

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout)


class Controller:
    def __init__(self, store, device: dict, label: str, runner_folder: Path, window: WindowPort, tray: TrayPort,
                 make_host: Callable[..., RunnerHost] = RunnerHost, probe=probe.test_server, *,
                 fetch_pending: Callable[..., list[dict]] = fetch_pending, pending_every: float = PENDING_EVERY_S,
                 clipboard: Callable[[str], bool] = copy_to_clipboard, release: Callable[[], None] | None = None,
                 exit_process: Callable[[int], None] | None = None, watchdog_s: float = QUIT_WATCHDOG_S):
        self._store, self._device, self._label = store, dict(device), label
        self._runner_folder = Path(runner_folder)
        self._window, self._tray = window, tray
        self._make_host, self._probe = make_host, probe
        self._fetch_pending, self._pending_every = fetch_pending, pending_every
        self._clipboard, self._release = clipboard, release
        self._exit = exit_process or (lambda code: os._exit(code))
        self._watchdog_s = watchdog_s

        self._mu = threading.Lock()              # the fields below; held only for bookkeeping, never across a wait
        self._op = threading.Lock()              # connect / remove / open_without_runner, one at a time
        self._storage = threading.Lock()         # an inject (record, write, open) and a clear never interleave
        self._state = TrayState("idle")
        self._notice = ""                        # what the last step could not do, in words (the servers window)
        self._open: _Open | None = None
        self._host = None
        self._ticket: object | None = None       # the current host's: its events carry it, a stale one is dropped
        self._watch: PendingWatch | None = None
        # An earlier run's (a crash): known from the first moment, so a Quit before start() still clears them or
        # keeps them listed.
        self._dirty: list[str] = list(store.injected)
        self._bases: dict[str, str] = {}         # origin -> the address it was opened at (this session)
        self._front = "servers"                  # what Open shows: "servers", or "app" once the app was opened
        self._quitting = False
        self._quit_step = ""
        self._quit_done = threading.Event()
        self._events: queue.Queue = queue.Queue()
        self._worker = threading.Thread(target=self._work, name="desktop-events", daemon=True)
        self._worker.start()

    def __repr__(self) -> str:
        return f"Controller(state={self._state.kind!r})"

    # ------------------------------------------------------------------ what the servers window and the tray read
    @property
    def state(self) -> TrayState:
        return self._state

    @property
    def notice(self) -> str:
        return self._notice

    @property
    def quitting(self) -> bool:
        return self._quitting

    def view(self) -> dict:
        from desktop.gui import state_view          # gui imports no GUI library at module level
        return state_view(self._state, self._notice)

    def _set(self, s: TrayState, notice: str | None = None) -> None:
        with self._mu:
            self._state = s
            if notice is not None:
                self._notice = notice
        self._show_state()

    def _show_state(self) -> None:
        if self._quitting:
            return
        try:
            self._tray.update(self._state)
        except Exception as e:  # noqa: BLE001
            log.warning("updating the tray raised %s", type(e).__name__)
        try:
            self._window.render(self.view())
        except Exception as e:  # noqa: BLE001
            log.warning("rendering the state raised %s", type(e).__name__)

    # ------------------------------------------------------------------ start
    def start(self) -> None:
        """The crash leftover first (a token an earlier run left in the web view), then the servers window, then the
        server opened last."""
        if self._quitting:
            return
        if self._dirty:                              # the crash leftover (seeded from servers.json at __init__)
            self._clear()
        self._window.show_servers()
        self._show_state()
        last = self._store.last
        if last and not self._quitting:
            self.connect(last)

    # ------------------------------------------------------------------ the token in the web view
    def _base_for(self, origin: str) -> str:
        """The address to load for `origin`: the one it was opened at this session, else a listed server's (its path
        prefix kept), else the origin itself."""
        with self._mu:
            known = self._bases.get(origin)
        if known:
            return known
        for e in self._store.entries():
            try:
                if inject.origin(e.url) == origin:
                    return e.url
            except ValueError:
                continue
        return origin

    def _mark(self, origin: str, base: str) -> None:
        """`origin` may hold a token from now on: recorded, with every origin not cleared yet, before the token is
        written."""
        with self._mu:
            if origin not in self._dirty:
                self._dirty.append(origin)
            self._bases[origin] = base
            listed = list(self._dirty)
        self._store.set_injected(listed)

    def _clear(self) -> None:
        """Removes the token from every origin that may hold one; the file then lists the origins whose clear failed
        (their server did not answer).  Never between an inject's record and its end (`_storage`)."""
        with self._storage:
            self._clear_locked()

    def _clear_locked(self) -> None:
        with self._mu:
            tried = list(self._dirty)
        failed = []
        for origin in tried:
            ok = False
            try:
                ok = self._window.navigate_and_inject(self._base_for(origin), inject.clear_script(), timeout=CLEAR_S)
            except Exception as e:  # noqa: BLE001
                log.warning("clearing the token raised %s", type(e).__name__)
            if not ok:
                failed.append(origin)
                log.warning("the token could not be removed from %s (its server did not answer); tried again later",
                            origin)
        with self._mu:
            self._dirty = [o for o in self._dirty if o not in tried or o in failed]
            listed = list(self._dirty)
        if tried:                                    # nothing tried: the record is left as it is
            self._store.set_injected(listed)

    # ------------------------------------------------------------------ the runner
    def _stop_runner(self) -> None:
        """Ends the current runner and the pending watch; later events of that host are dropped."""
        with self._mu:
            host, self._host, self._ticket = self._host, None, None
            watch, self._watch = self._watch, None
        if watch is not None:
            watch.stop()
        if host is not None:
            if not host.stop(STOP_S):
                log.warning("the runner did not end within %g s", STOP_S)

    def _on_state(self, ticket: object) -> Callable[[str, dict], None]:
        def on_state(event: str, data: dict) -> None:
            self._events.put((ticket, event, data))   # nothing else: this runs on desktop-runner
        return on_state

    def _work(self) -> None:
        while True:
            item = self._events.get()
            if item is None:
                return
            if isinstance(item, threading.Event):
                item.set()
                continue
            ticket, event, data = item
            try:
                self._handle(ticket, event, data)
            except Exception as e:  # noqa: BLE001 - the worker must outlive one bad event
                log.error("handling the runner's %r raised %s", event, type(e).__name__, exc_info=True)

    def wait_idle(self, timeout: float = 5.0) -> bool:
        """Every event queued so far was handled (tests, the selftest)."""
        done = threading.Event()
        self._events.put(done)
        return done.wait(timeout)

    def _current(self, ticket) -> bool:
        return ticket is not None and ticket is self._ticket and not self._quitting

    def _handle(self, ticket, event: str, data: dict) -> None:
        with self._mu:
            if not self._current(ticket):
                return
            before = self._state
            self._state = state.step(before, event, data)
            if self._state is before and event != "connecting":
                return                                   # an end state (refused, error) ignores it: so do we
            conn = self._open
            if event == "online":
                self._notice = ""
        self._show_state()
        if conn is None:
            return
        if event == "online":
            self._watch_pending(ticket, conn)
            rid = data.get("id") if isinstance(data, dict) else None
            if conn.opened and rid == conn.opened_id:
                return                                   # the same runner said hello again: nothing to do
            self._inject_and_open(ticket, conn, rid)
        elif event in ("refused", "error"):
            with self._mu:
                watch, self._watch = (self._watch, None) if ticket is self._ticket else (None, self._watch)
            if watch is not None:
                watch.stop()

    def _watch_pending(self, ticket, conn: _Open) -> None:
        with self._mu:
            if self._watch is not None or not self._current(ticket):
                return
            self._watch = PendingWatch(
                lambda: self._fetch_pending(conn.base, conn.token, self._device, self._label),
                self._notify, lambda: self._current(ticket) and self._state.kind == "online", self._pending_every)
            watch = self._watch
        watch.start()

    def _notify(self, title: str, text: str) -> None:
        if not self._quitting:
            self._tray.notify(title, text)

    # ------------------------------------------------------------------ the app window
    def _inject_and_open(self, ticket, conn: _Open, runner_id: str | None) -> bool:
        """Writes the token, the device, its name and the runner id (None: removes it) into `conn`'s origin, then
        loads the app.  False (and the words in the servers window) when the window could not be prepared."""
        try:
            values, remove = inject.storage_values(conn.token, runner_id, self._device, self._label)
        except ValueError as e:
            log.error("the values for the web app were refused: %s", e)
            self._set(self._state, words.t("window_not_prepared"))
            return False
        def moved_on() -> bool:
            return self._quitting or (ticket is not None and ticket is not self._ticket) or self._open is not conn

        with self._storage:          # a clear never runs between the mark and the open: it comes before or after
            with self._mu:
                if moved_on():
                    return False
            self._mark(conn.origin, conn.base)
            ok = self._window.navigate_and_inject(conn.base, inject.inject_script(values, remove))
            with self._mu:
                if moved_on():
                    return False                         # the clear that follows (marked: it is in its list) wins
                if ok:
                    conn.opened, conn.opened_id = True, runner_id
                    self._front, self._notice = "app", ""
            if ok:
                self._window.open_app(conn.base, conn.name)
                self._window.hide_servers()
        if not ok:
            log.info("the app window could not be prepared for %s", conn.origin)
            self._set(self._state, words.t("window_not_prepared"))
            return False
        try:
            self._store.set_last(conn.name)
        except ValueError:
            pass                                         # renamed or removed meanwhile
        return True

    # ------------------------------------------------------------------ what the user does
    def connect(self, name: str) -> None:
        with self._op:
            self._connect(name)

    def _connect(self, name: str) -> None:
        if self._quitting:
            return
        was_open = self._open is not None and self._open.opened
        with self._mu:
            self._open = None
            self._front = "servers"
        self._stop_runner()
        if was_open:
            self._window.hide_app()
        self._clear()

        entry = self._store.get(name)
        if entry is None:
            self._set(TrayState("idle"), words.t("no_such_server"))
            return
        token = entry.token.reveal()
        if not token:
            self._set(TrayState("idle", server=name), words.t("token_missing_test"))
            return
        applog.add_secret(token)
        try:
            origin = inject.origin(entry.url)
        except ValueError:
            self._set(TrayState("idle", server=name), words.t("bad_url"))
            return
        self._set(TrayState("connecting", server=name), "")
        r = self._probe(entry.url, token, self._device, self._label)
        if not r.reachable or r.error in (words.t("token_refused"), words.t("device_refused")):
            self._set(TrayState("idle", server=name), r.error or words.t("unreachable"))
            return
        conn, ticket = _Open(name, entry.url, origin, token), object()
        with self._mu:
            if self._quitting:
                return
            host = self._make_host(entry.url, token, self._device, self._label, self._runner_folder,
                                   self._on_state(ticket))
            self._host, self._ticket, self._open = host, ticket, conn
            host.start()                             # inside: a Quit cannot stop it before it started (it only
                                                     # starts a thread), so no runner thread begins after Quit

    def open_without_runner(self) -> None:
        """The app without My PC (an older server, or the runner could not start): `strata.runner` removed."""
        with self._op:
            with self._mu:
                conn = self._open
            if conn is None or self._quitting:
                return
            self._inject_and_open(None, conn, None)

    def switch(self) -> None:
        """The servers window in front of the app; the runner goes on until another server is chosen."""
        with self._mu:
            self._front = "servers"
        self._window.hide_app()
        self._window.show_servers()
        self._show_state()

    def open(self) -> None:
        """The tray's Open (and a second start): the app window when a server is open in it, else the servers
        window."""
        if self._quitting:
            return
        with self._mu:
            app = self._front == "app" and self._open is not None and self._open.opened
        if app:
            self._window.show_app()
        else:
            self._window.show_servers()

    def copy_code(self) -> bool:
        code = code_text(self._state.code) if self._state.kind == "pending" else ""
        return bool(code) and bool(self._clipboard(code))

    def saved(self, old_name: str | None, entry) -> None:
        """The servers window saved `entry`, renamed from `old_name`: the open server keeps its connection (to the
        address it was opened at) under its new name."""
        if old_name is None or entry is None:
            return
        with self._mu:
            if self._open is not None and self._open.name == old_name:
                self._open.name = entry.name

    def _is_open(self, name: str) -> bool:
        """`name` is the server open now: by its name, or by the address the connection was made to (a rename the
        controller was not told of)."""
        with self._mu:
            conn = self._open
        if conn is None:
            return False
        if conn.name == name:
            return True
        entry = self._store.get(name)
        return entry is not None and entry.url == conn.base

    def remove(self, name: str) -> None:
        with self._op:
            is_open = self._is_open(name)
            if is_open:
                with self._mu:
                    self._open = None
                    self._front = "servers"
                self._stop_runner()
                self._window.hide_app()
                self._clear()                            # before the entry goes: its address finds the origin
            else:
                entry = self._store.get(name)
                try:
                    dirty = entry is not None and inject.origin(entry.url) in self._dirty
                except ValueError:
                    dirty = False
                if dirty:                                # a clear that failed before: try it while the address
                    self._clear()                        # (and its path prefix) is still listed
            self._store.remove(name)
            if is_open:
                self._set(TrayState("idle"), "")
                self._window.show_servers()

    # ------------------------------------------------------------------ quit
    def quit(self) -> None:
        """In this order (RF 3): the runner (its jobs and processes), the token out of the web view, the tray, the
        windows, the single-instance mutex.  A watchdog ends the process after `watchdog_s` if a step hangs."""
        with self._mu:
            if self._quitting:
                return
            self._quitting = True
        threading.Thread(target=self._watchdog, name="desktop-quit-watchdog", daemon=True).start()
        steps = (("the runner", self._stop_runner), ("the app window", self._window.hide_app),
                 ("the token's clear", self._clear), ("the tray", self._tray.stop),
                 ("the windows", self._window.destroy), ("the mutex", self._release or (lambda: None)))
        try:
            for name, step in steps:                     # a step that raises never keeps the next from running
                self._quit_step = name
                try:
                    step()
                except Exception as e:  # noqa: BLE001
                    log.error("Quit: %s raised %s", name, type(e).__name__, exc_info=True)
            self._quit_step = "done"
        finally:
            self._events.put(None)
            self._quit_done.set()

    def wait_quit(self, timeout: float | None = None) -> bool:
        return self._quit_done.wait(timeout)

    def _watchdog(self) -> None:
        if self._quit_done.wait(self._watchdog_s):
            return
        log.error("Quit did not finish in %g s: it hung at %s; the process ends now", self._watchdog_s,
                  self._quit_step)
        for h in logging.getLogger().handlers:
            try:
                h.flush()
            except Exception:  # noqa: BLE001
                pass
        self._exit(0)
