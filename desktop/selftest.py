"""`StrataDesktop.exe --selftest --server URL --data-dir D` (#310 part 3): the program checks itself against a
server without showing anything - the windows are hidden, the tray is a recorder with no icon, no message box - and
writes what it found to `D\\selftest.json` (a windowed exe has no stdout).  Exit 0 when every step passed, 5 when one
failed.  The token comes from STRATA_DESKTOP_SELFTEST_TOKEN (desktop/__main__.py pops it at once).

It drives the app's own code, in the app's order: the server list (DPAPI), the server test, the Controller's connect
(the runner host in `D\\runner`, then on its first `online` the token, device and runner id written into the hidden app
window, then the app loaded), and the Controller's Quit.  Steps, each `{"step", "ok", "detail", "ms"}`:

    store    the entry saved and read back from a new ServerStore: the token round-trips through DPAPI, and is not
             in servers.json's text
    probe    the server test (role, model, a pending code); a refused token or device fails here
    pending  only when this PC waits for the owner: written at once, with "code", so a harness can approve it
    runner   the runner said hello (online) within 120 s: who, its runner id, this PC's device and label
    window   the app window: the inject answered true and the app was opened; on the page, the four values (the token
             compared inside the page, never read back), the origin, #root mounted, and 5 s after the load no
             [data-device-gate] (the waiting screen); the servers window holds none of the values.  "not checked"
             (ok null) with no window (the in-process test)
    runners  GET /runners with the token and this PC's device: its own runner is online, on this device, and its id is
             the one the page was given
    quit     Controller.quit(): every runner, job, pending-watch and event thread ended, no child process left (but
             the web view's own), the clear's read-back shows no strata.apikey on the server's origin, servers.json
             `injected` empty, and no tray icon (the selftest's tray is a recorder; pystray is never loaded)

`selftest.json` never holds the token: every value is scrubbed before it is written.
"""
from __future__ import annotations

import http.client
import json
import logging
import os
import ssl
import sys
import threading
import time
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

from desktop import applog, inject, paths, probe, words
from desktop.state import TrayState

log = logging.getLogger("strata.desktop.selftest")

EXIT_OK, EXIT_FAILED = 0, 5
RESULT_FILE = "selftest.json"
NAME = "selftest"                       # the server's name in the selftest's own list
RUNNER_WAIT_S = 120.0                   # for the runner's hello (the owner approves the device meanwhile)
OPEN_WAIT_S = 30.0                      # from online to the app window opened
LOAD_WAIT_S = 30.0                      # for the app's page to load, and again for #root to mount
GATE_WAIT_S = 5.0                       # after the load: the waiting screen would be up by then
QUIT_WAIT_S = 15.0
SETTLE_S = 5.0                          # for threads and child processes to end after Quit
WORKER_THREADS = ("desktop-runner", "runner-job", "runner-poll", "desktop-pending", "desktop-events", "desktop-tray")
WEBVIEW_PROCESS = "msedgewebview2.exe"  # the web view's own processes: they end with its windows
STORAGE_KEYS = ("strata.apikey", "strata.device", "strata.deviceName", "strata.runner")


# ------------------------------------------------------------------------------------------------ the result file
class Result:
    """selftest.json, rewritten whole (a .tmp, then os.replace) after every step, so a harness can read it while the
    selftest runs.  `secret` is replaced by *** in every value before anything is written."""

    def __init__(self, path: Path, secret: str | None, steps: list[dict] | None = None):
        self.path, self._secret = Path(path), secret or ""
        self.steps: list[dict] = list(steps or [])
        self._lock = threading.Lock()
        self._done, self._exit = False, None
        self._write()

    def _clean(self, v):
        if isinstance(v, str):
            return v.replace(self._secret, "***") if self._secret else v
        if isinstance(v, dict):
            return {k: self._clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [self._clean(x) for x in v]
        return v

    def add(self, step: str, ok: bool | None, detail, ms: float, **extra) -> dict:
        entry = {"step": step, "ok": ok, "detail": self._clean(detail), "ms": int(round(ms)), **self._clean(extra)}
        with self._lock:
            self.steps.append(entry)
            self._write()
        log.info("selftest %s: %s", step, "ok" if ok else "not checked" if ok is None else "FAILED")
        return entry

    def finish(self, code: int) -> int:
        with self._lock:
            self._done, self._exit = True, code
            self._write()
        return code

    def _write(self) -> None:
        doc = {"version": 1, "done": self._done, "exit": self._exit,
               "ok": (self._exit == EXIT_OK) if self._done else None, "steps": self.steps}
        text = json.dumps(doc, indent=2, ensure_ascii=False)
        if self._secret:
            text = text.replace(self._secret, "***")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        for _ in range(50):                     # a reader holding the file open: Windows refuses the replace a moment
            try:
                os.replace(tmp, self.path)
                return
            except PermissionError:
                time.sleep(0.02)
        os.replace(tmp, self.path)


def exit_code(steps: list[dict]) -> int:
    """0 when there were steps and none failed (a step not checked, ok null, is not a failure); else 5."""
    return EXIT_OK if steps and not any(s.get("ok") is False for s in steps) else EXIT_FAILED


def fail(path: Path, step: str, detail: str, keep: bool = False) -> int:
    """One failed step in selftest.json, and 5.  `keep`: after the steps already in the file (a crash part-way)."""
    steps = []
    if keep:
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
            steps = [s for s in doc.get("steps", []) if isinstance(s, dict)]
        except (OSError, ValueError, AttributeError):
            steps = []
    r = Result(path, None, steps)
    r.add(step, False, detail, 0)
    return r.finish(EXIT_FAILED)


def _timed(fn: Callable[[], tuple]) -> tuple:
    """(ok, detail, ms[, ...]) of `fn`, an exception being a failure in words (its type and text; scrubbed later)."""
    t0 = time.monotonic()
    try:
        ok, detail, *more = fn()
    except Exception as e:  # noqa: BLE001 - a step that raises is a failed step, never a crash
        log.error("a selftest step raised %s", type(e).__name__, exc_info=True)
        ok, detail, more = False, f"{type(e).__name__}: {e}", []
    return (ok, detail, (time.monotonic() - t0) * 1000, *more)


# ------------------------------------------------------------------------------------------------ the ports
class QuietTray:
    """The Controller's TrayPort with no icon: it keeps the latest state (the selftest waits on it) and counts the
    notices.  pystray is never imported."""

    def __init__(self):
        self._cond = threading.Condition()
        self.state = TrayState("idle")
        self.notices = 0
        self.stopped = False

    def update(self, s: TrayState) -> None:
        with self._cond:
            self.state = s
            self._cond.notify_all()

    def notify(self, title: str, text: str) -> None:
        self.notices += 1

    def stop(self) -> None:
        self.stopped = True

    def wait(self, pred: Callable[[TrayState], bool], timeout: float) -> TrayState:
        end = time.monotonic() + timeout
        with self._cond:
            while not pred(self.state):
                left = end - time.monotonic()
                if left <= 0:
                    break
                self._cond.wait(left)
            return self.state


class NoWindow:
    """The WindowPort of a selftest with no window (the in-process test): nothing loads, so nothing is written."""

    def show_servers(self) -> None: ...
    def hide_servers(self) -> None: ...
    def render(self, view: dict) -> None: ...
    def navigate_and_inject(self, base: str, script: str, timeout: float = 15.0) -> bool:
        return False
    def open_app(self, base: str, server: str | None = None) -> None: ...
    def show_app(self) -> None: ...
    def hide_app(self) -> None: ...
    def destroy(self) -> None: ...


class Watched:
    """The WindowPort the Controller is given: every call goes to `inner`.  It records what the selftest checks: each
    inject's and clear's answer, that the app was opened, and - on `destroy`, Quit's step right after the clear - the
    page read once more (`read_back`) before the windows close."""

    def __init__(self, inner, read_back: Callable[[], object] | None = None):
        self.inner, self._read_back = inner, read_back
        self.injects: list[bool] = []
        self.clears: list[bool] = []
        self.opened = threading.Event()
        self.read_back = None

    def show_servers(self) -> None:
        self.inner.show_servers()

    def hide_servers(self) -> None:
        self.inner.hide_servers()

    def render(self, view: dict) -> None:
        self.inner.render(view)

    def navigate_and_inject(self, base: str, script: str, timeout: float = 15.0) -> bool:
        ok = self.inner.navigate_and_inject(base, script, timeout=timeout)
        (self.clears if script == inject.clear_script() else self.injects).append(ok is True)
        return ok

    def open_app(self, base: str, server: str | None = None) -> None:
        self.inner.open_app(base, server)
        self.opened.set()

    def show_app(self) -> None:
        self.inner.show_app()

    def hide_app(self) -> None:
        self.inner.hide_app()

    def destroy(self) -> None:
        if self._read_back is not None:
            try:
                self.read_back = self._read_back()
            except Exception as e:  # noqa: BLE001
                log.error("the read-back before closing raised %s", type(e).__name__)
        self.inner.destroy()


# ------------------------------------------------------------------------------------------------ helpers
def get_json(base: str, path: str, token: str, device: dict, label: str, timeout: float = 10.0):
    """(status, JSON body or None) of GET `<base><path>` with the token and this PC's device headers (http.client: no
    redirect, no proxy, as desktop/probe.py)."""
    u = urlsplit(base)
    if u.scheme == "https":
        conn = http.client.HTTPSConnection(u.hostname, u.port, timeout=timeout, context=ssl.create_default_context())
    else:
        conn = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
    headers = {**probe.device_headers(device, label), "Authorization": "Bearer " + token,
               "User-Agent": probe.USER_AGENT, "Accept": "application/json"}
    try:
        conn.request("GET", u.path.rstrip("/") + path, headers=headers)
        r = conn.getresponse()
        raw = r.read(4 << 20)
    finally:
        conn.close()
    try:
        return r.status, json.loads(raw) if raw else None
    except ValueError:
        return r.status, None


def child_processes(pid: int) -> list[dict]:
    """The processes whose parent is `pid`, as {"pid", "name"}; [] off Windows."""
    return [{"pid": p, "name": name} for p, parent, name in processes() if parent == pid and p != pid]


def processes() -> list[tuple[int, int, str]]:
    """Every process as (pid, parent pid, exe name), from a Toolhelp snapshot; [] off Windows."""
    if os.name != "nt":
        return []
    import ctypes
    from ctypes import wintypes

    class Entry(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD),
                    ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                    ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD),
                    ("szExeFile", ctypes.c_wchar * 260)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    k32.Process32FirstW.argtypes = k32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
    k32.Process32FirstW.restype = k32.Process32NextW.restype = wintypes.BOOL
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    snap = k32.CreateToolhelp32Snapshot(0x2, 0)              # TH32CS_SNAPPROCESS
    if not snap or snap == wintypes.HANDLE(-1).value:
        return []
    out = []
    try:
        e = Entry()
        e.dwSize = ctypes.sizeof(Entry)
        ok = k32.Process32FirstW(snap, ctypes.byref(e))
        while ok:
            out.append((int(e.th32ProcessID), int(e.th32ParentProcessID), e.szExeFile))
            ok = k32.Process32NextW(snap, ctypes.byref(e))
    finally:
        k32.CloseHandle(snap)
    return out


def _worker_threads() -> list[str]:
    return sorted(t.name for t in threading.enumerate() if t.is_alive() and t.name.startswith(WORKER_THREADS))


def _settle(check: Callable[[], list], timeout: float) -> list:
    """`check()` until it is empty or `timeout` passed; its last value."""
    end = time.monotonic() + timeout
    left = check()
    while left and time.monotonic() < end:
        time.sleep(0.1)
        left = check()
    return left


def _read_script(token: str) -> str:
    """The page's values: the token compared inside the page (only true/false comes back), the others as stored."""
    keys = inject.js_literal(list(STORAGE_KEYS[1:]))
    return ("(function(){try{var s=window.localStorage,r=document.getElementById(\"root\"),o={origin:location.origin,"
            "apikey:s.getItem(\"strata.apikey\")===" + inject.js_literal(json.dumps(token)) + ","
            "root:r?r.childElementCount:-1};" + keys + ".forEach(function(k){o[k]=s.getItem(k)});"
            "return JSON.stringify(o)}catch(e){return JSON.stringify({error:String(e&&e.name)})}})()")


GATE_SCRIPT = ("(function(){var r=document.getElementById(\"root\");return JSON.stringify({gate:document.querySelector("
               "\"[data-device-gate]\")!==null,root:r?r.childElementCount:-1})})()")
SERVERS_SCRIPT = ("(function(){try{var s=window.localStorage;return " + inject.js_literal(list(STORAGE_KEYS)) +
                  ".every(function(k){return s.getItem(k)===null})}catch(e){return \"no storage\"}})()")
READ_BACK_SCRIPT = ("(function(){try{return JSON.stringify({origin:location.origin,apikey_absent:window.localStorage"
                    ".getItem(\"strata.apikey\")===null,device_kept:window.localStorage.getItem(\"strata.device\")!==null})"
                    "}catch(e){return JSON.stringify({error:String(e&&e.name)})}})()")


def _json(v) -> dict:
    try:
        out = json.loads(v) if isinstance(v, str) else None
    except ValueError:
        out = None
    return out if isinstance(out, dict) else {"value": None if v is None else str(v)[:200]}


# ------------------------------------------------------------------------------------------------ the steps
def run(server: str, data_dir: Path, token: str, result_path: Path, window=None, *,
        runner_wait: float = RUNNER_WAIT_S, gate_wait: float = GATE_WAIT_S) -> int:
    """The selftest; 0 when every step passed, 5 when one failed.  `window`: the hidden `desktop.gui.Gui` (it must
    already run its loop: `run_hidden`), or None to check everything but the window."""
    res = Result(result_path, token)
    if not token:
        res.add("store", False, "no token was given", 0)
        return res.finish(EXIT_FAILED)
    applog.add_secret(token)
    ctl = watched = tray = None
    try:
        from serve import runner
        from desktop.servers import ServerStore

        data = Path(data_dir)
        data.mkdir(parents=True, exist_ok=True)
        runner_folder = paths.runner_dir(data, True)            # never the real device
        device = runner.load_device(runner_folder)
        label = inject.page_label(runner.device_label(device, paths.APP_NAME))

        # store -------------------------------------------------------------------------------------------------
        store_box: list = []

        def do_store():
            first = ServerStore(paths.servers_file(data))
            if first.get(NAME) is not None:
                first.remove(NAME)
            first.save(NAME, server, token, None)
            again = ServerStore(paths.servers_file(data))
            entry = again.get(NAME)
            text = paths.servers_file(data).read_text(encoding="utf-8")
            back = entry is not None and entry.token.reveal() == token
            store_box.append(again)
            return (back and token not in text,
                    "saved, read back by a new ServerStore: the token came back through DPAPI and is not in "
                    "servers.json" if back and token not in text else "the token did not come back from the store")
        ok, detail, ms = _timed(do_store)
        res.add("store", ok, detail, ms)
        if not ok:
            return res.finish(EXIT_FAILED)
        store = store_box[0]

        # probe -------------------------------------------------------------------------------------------------
        def do_probe():
            r = probe.test_server(server, token, device, label)
            if not r.reachable or r.error:
                return False, r.error or words.t("unreachable")
            return True, {"role": r.role, "model": r.model, "engine_running": r.engine_running,
                          "pending_code": r.pending_code}
        ok, detail, ms = _timed(do_probe)
        res.add("probe", ok, detail, ms)
        if not ok:
            return res.finish(EXIT_FAILED)

        # runner ------------------------------------------------------------------------------------------------
        from desktop.controller import Controller
        tray = QuietTray()
        inner = window if window is not None else NoWindow()
        watched = Watched(inner, (lambda: _json(window.evaluate("app", READ_BACK_SCRIPT))) if window is not None
                          else None)
        ctl = Controller(store, device, label, runner_folder, watched, tray, clipboard=lambda text: False,
                         exit_process=_watchdog_exit)
        t0 = time.monotonic()
        ctl.connect(NAME)
        s = ctl.state
        if s.kind == "idle":                                   # the connect stopped before a runner started
            res.add("runner", False, ctl.notice or "no runner was started", (time.monotonic() - t0) * 1000)
            return res.finish(EXIT_FAILED)
        end, pending_told = t0 + runner_wait, False
        while True:
            s = tray.wait(lambda x: x.kind in ("online", "refused", "error") or (x.kind == "pending" and not
                                                                                 pending_told),
                          max(0.0, end - time.monotonic()))
            if s.kind == "pending" and not pending_told:
                from desktop.controller import code_text
                res.add("pending", True, "this PC waits for the owner's approval", (time.monotonic() - t0) * 1000,
                        code=code_text(s.code))
                pending_told = True
                continue
            break
        online = s.kind == "online" and inject.RUNNER_ID.fullmatch(s.runner_id or "") is not None
        res.add("runner", online, {"state": s.kind, "who": s.who, "runner_id": s.runner_id, "device": device["id"],
                                   "label": label, "no_shell": s.no_shell, "message": s.message or None},
                (time.monotonic() - t0) * 1000)
        if not online:
            return res.finish(EXIT_FAILED)

        # window ------------------------------------------------------------------------------------------------
        page: dict = {}
        if window is None:
            res.add("window", None, "not checked: no window (the in-process selftest)", 0)
        else:
            ok, detail, ms = _timed(lambda: _window_step(window, watched, ctl, server, token, device, label, s,
                                                         gate_wait, page))
            res.add("window", ok, detail, ms)

        # runners -----------------------------------------------------------------------------------------------
        def do_runners():
            code, body = get_json(server, "/runners", token, device, label)
            rows = body.get("runners") if isinstance(body, dict) else None
            if code != 200 or not isinstance(rows, list):
                return False, {"status": code}
            own = [r for r in rows if isinstance(r, dict) and r.get("id") == s.runner_id]
            r = own[0] if len(own) == 1 else {}
            want_page = window is None or page.get("runner") == s.runner_id
            good = bool(r) and r.get("online") is True and r.get("device") == device["id"] and r.get("who") == s.who \
                and want_page
            return good, {"id": r.get("id"), "who": r.get("who"), "device": r.get("device"), "online": r.get("online"),
                          "machine": r.get("machine"), "listed": len(rows),
                          "page_runner": page.get("runner") if window is not None else "not checked"}
        ok, detail, ms = _timed(do_runners)
        res.add("runners", ok, detail, ms)
    except Exception as e:  # noqa: BLE001 - the selftest reports, it never crashes
        log.error("the selftest raised %s", type(e).__name__, exc_info=True)
        res.add("crash", False, f"{type(e).__name__}: {e}", 0)
    finally:
        if ctl is not None:
            ok, detail, ms = _timed(lambda: _quit_step(ctl, watched, tray, data_dir, window is not None))
            res.add("quit", ok, detail, ms)
    return res.finish(exit_code(res.steps))


def _watchdog_exit(code: int) -> None:
    """Quit's watchdog (a step hung for 10 s) ends the selftest as a failure, never as exit 0."""
    os._exit(EXIT_FAILED)


def _window_step(window, watched: Watched, ctl, server: str, token: str, device: dict, label: str, s: TrayState,
                 gate_wait: float, page: dict) -> tuple:
    if not watched.opened.wait(OPEN_WAIT_S):
        return False, {"inject": watched.injects, "opened": False, "notice": ctl.notice or None}
    loaded = window.wait_app_loaded(LOAD_WAIT_S)
    t_loaded = time.monotonic()
    read = _json(window.evaluate("app", _read_script(token)))
    end = time.monotonic() + LOAD_WAIT_S
    while read.get("root", -1) <= 0 and time.monotonic() < end:     # the app mounts after the load event
        time.sleep(0.2)
        read = _json(window.evaluate("app", _read_script(token)))
    time.sleep(max(0.0, gate_wait - (time.monotonic() - t_loaded)))
    gate = _json(window.evaluate("app", GATE_SCRIPT))
    servers_clean = window.evaluate("servers", SERVERS_SCRIPT)

    def decoded(key):
        try:
            return json.loads(read.get(key)) if isinstance(read.get(key), str) else None
        except ValueError:
            return None
    page.update({"runner": decoded("strata.runner"), "device": decoded("strata.device")})
    checks = {
        "injected": bool(watched.injects) and watched.injects[-1] is True,
        "loaded": loaded,
        "origin": read.get("origin") == inject.origin(server),
        "apikey": read.get("apikey") is True,
        "device": decoded("strata.device") == device["id"],
        "deviceName": decoded("strata.deviceName") == label,
        "runner": decoded("strata.runner") == s.runner_id,
        "root_mounted": isinstance(read.get("root"), int) and read["root"] > 0,
        "no_device_gate": gate.get("gate") is False,
        "servers_window_has_none": servers_clean in (True, "no storage"),
    }
    return all(checks.values()), {"checks": checks, "origin": read.get("origin"), "root": read.get("root"),
                                  "servers_window": servers_clean, "error": read.get("error")}


def _quit_step(ctl, watched: Watched, tray: QuietTray, data_dir, with_window: bool) -> tuple:
    from desktop.servers import ServerStore
    ctl.quit()
    done = ctl.wait_quit(QUIT_WAIT_S)
    threads = _settle(_worker_threads, SETTLE_S)
    me = os.getpid()
    children = _settle(lambda: [c for c in child_processes(me) if c["name"].lower() != WEBVIEW_PROCESS], SETTLE_S)
    injected = ServerStore(paths.servers_file(Path(data_dir))).injected
    checks = {"quit_finished": done, "no_worker_thread": threads == [], "no_child_process": children == [],
              "no_tray_icon": "pystray" not in sys.modules and tray.stopped}
    detail = {"threads": threads, "children": children, "injected": injected, "tray_notices": tray.notices}
    if with_window:
        back = watched.read_back if isinstance(watched.read_back, dict) else {}
        checks["token_cleared"] = bool(watched.clears) and watched.clears[-1] is True \
            and back.get("apikey_absent") is True
        checks["injected_empty"] = injected == []
        detail.update({"read_back": back, "clears": watched.clears})
    else:
        detail["token_cleared"] = "not checked: no window"
    detail["checks"] = checks
    return all(checks.values()), detail


# ------------------------------------------------------------------------------------------------ the hidden window
def run_hidden(server: str, data: Path, token: str, result: Path) -> int:
    """The selftest inside pywebview's loop with both windows hidden for good (`Gui(headless=True)`): the loop runs
    on this (the main) thread, the selftest on pywebview's own thread, and Quit's `destroy` ends the loop."""
    from desktop.gui import Gui

    g = Gui(Path(data), None, headless=True)
    out: dict = {}
    began, finished = threading.Event(), threading.Event()

    def body():
        began.set()
        try:
            out["code"] = run(server, data, token, result, g)
        except Exception as e:  # noqa: BLE001
            log.error("the selftest raised %s", type(e).__name__, exc_info=True)
            out["code"] = fail(result, "crash", type(e).__name__, keep=True)
        finally:
            finished.set()
            g.destroy()                                         # nothing to do when Quit already closed them
    rc = g.run(body)
    # Quit's `destroy` ends the loop while the selftest still checks what Quit left: wait for its end.
    if began.is_set() and not finished.wait(QUIT_WAIT_S + 2 * SETTLE_S + 10.0):
        return fail(result, "quit", "the selftest did not end after the windows closed", keep=True)
    if "code" not in out:
        return fail(result, "webview2", f"the hidden web view did not start (exit {rc})", keep=True)
    return out["code"] if rc == 0 else EXIT_FAILED
