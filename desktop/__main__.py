"""Strata Desktop (#310 part 3): `python -m desktop`, and the built `StrataDesktop.exe`.

    python -m desktop [--data-dir PATH] [--debug]
    python -m desktop --selftest --server URL --data-dir PATH        (the token in STRATA_DESKTOP_SELFTEST_TOKEN;
                                                                      the result in PATH\\selftest.json)

Exit codes: 0 normal (a second start that handed over to the first included), 1 Windows refused the single-instance
lock (after a message box), 4 no WebView2 Runtime or .NET 4.6.2 (after a message box), 5 the selftest failed.

Order: the log (`applog.setup`, the exception hooks, stdout/stderr into it, `serve.runner._say` into it) -> the
WebView2 check -> one instance per data folder (a second start asks the first to show its window and exits 0; when it
cannot reach the first - an elevated one - it says so in a box) -> the server list, this PC's device and its one label
-> every stored token registered with the log's scrubber -> the tray, the windows, the controller.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

from desktop import applog, inject, paths, single, webview2, words

log = logging.getLogger("strata.desktop")

EXIT_OK = 0
EXIT_START_FAILED = 1
EXIT_SELFTEST_FAILED = 5
SELFTEST_ENV = "STRATA_DESKTOP_SELFTEST_TOKEN"
APP_NAME = paths.APP_NAME


def parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="StrataDesktop", description="Strata Desktop: your Strata servers in a window, "
                                "with this PC's tools (My PC) running beside them.")
    p.add_argument("--data-dir", metavar="PATH", help="keep servers.json, the web view's profile and desktop.log here "
                   "(default %%LOCALAPPDATA%%\\StrataDesktop); the runner's device then lives in PATH\\runner")
    p.add_argument("--selftest", action="store_true", help="check the program against --server without a window, "
                   "in an empty --data-dir of its own; the token comes from " + SELFTEST_ENV + ". Each run in a new "
                   "folder is a new device on that key: the server's owner sees one more device row for it")
    p.add_argument("--server", metavar="URL", help="the server --selftest checks")
    p.add_argument("--debug", action="store_true", help="pywebview's debug mode (the web inspector)")
    a = p.parse_args(argv)
    if a.server and not a.selftest:
        p.error("--server is for --selftest only")
    if a.selftest:                                 # its own folder: never the real server list, profile or device
        why = selftest_folder_refused(a.data_dir)
        if why:
            p.error(why)
    return a


def selftest_folder_refused(folder: str | None) -> str | None:
    """Why `--selftest` may not use `folder`, or None.  Its folder is its own: not the app's (the selftest replaces the
    server named "selftest" there and writes `last` and `injected` beside a live app), and not one holding a server
    list that an earlier selftest did not make."""
    if not folder:
        return "--selftest needs --data-dir PATH (an empty folder of its own)"

    def real(p) -> str:
        return os.path.normcase(os.path.realpath(os.fspath(p)))
    if real(folder) == real(paths.data_dir(None)):
        return "--selftest may not use Strata Desktop's own folder: give it an empty --data-dir"
    d = paths.data_dir(folder)
    if paths.servers_file(d).exists() and not (d / "selftest.json").exists():
        return "--data-dir holds a server list the selftest did not make: give it an empty folder"
    return None


def main(argv: list[str] | None = None) -> int:
    selftest_token = os.environ.pop(SELFTEST_ENV, None)     # at once: nothing the app starts inherits it
    args = parse(argv)
    data = paths.data_dir(args.data_dir)
    try:
        applog.setup(paths.log_file(data))
        applog.install_hooks()
        applog.capture_stdio()
        if selftest_token:
            applog.add_secret(selftest_token)
        from serve import runner
        runner._say = applog.say
    except Exception:  # noqa: BLE001
        if args.selftest:                          # an unwritable folder: exit 5, never PyInstaller's error box
            return EXIT_SELFTEST_FAILED
        raise

    if args.selftest:
        return _selftest(args, data, selftest_token)

    code = webview2.preflight()
    if code is not None:
        return code

    name = single.instance_name(data, bool(args.data_dir))
    try:
        instance = single.acquire(name)
    except OSError as e:                               # words, never PyInstaller's error box
        log.error("the single-instance lock could not be made: %s %s", type(e).__name__, e.errno)
        webview2.message_box(words.t("cannot_start"))
        return EXIT_START_FAILED
    if instance is None:                               # no log line: a second start never opens desktop.log
        if not single.signal_show(name):
            webview2.message_box(words.t("already_running"))
        return EXIT_OK
    log.info("Strata Desktop starts (data folder %s)", data)
    try:
        return _run(args, data, instance)
    finally:
        instance.release()


def _run(args: argparse.Namespace, data, instance) -> int:
    from serve import runner
    from desktop.controller import QUIT_WATCHDOG_S, Controller
    from desktop.gui import Gui, ServersApi
    from desktop.servers import ServerStore
    from desktop.tray import Tray

    store = ServerStore(paths.servers_file(data))
    runner_folder = paths.runner_dir(data, bool(args.data_dir))
    device = runner.load_device(runner_folder)
    label = inject.page_label(runner.device_label(device, APP_NAME))   # one name: the runner, the test, the page
    for e in store.entries():
        applog.add_secret(e.token.reveal())

    ctl: Controller | None = None
    tray = Tray(on_open=lambda: ctl.open(), on_switch=lambda: ctl.switch(), on_copy_code=lambda: ctl.copy_code(),
                on_quit=lambda: ctl.quit())
    gui = Gui(data, None, quitting=lambda: ctl is not None and ctl.quitting, debug=args.debug,
              on_session_end=lambda: ctl.end_session())
    ctl = Controller(store, device, label, runner_folder, gui, tray, release=instance.release)
    gui.attach(ServersApi(store, device, label, ctl))
    instance.listen(ctl.open)
    code = EXIT_OK
    try:
        tray.start()                               # without its icon the app could not be quit: a failure ends it
        code = gui.run(ctl.start)
    finally:
        if not ctl.quitting:                       # the loop ended without Quit (a failure): the same clean-up
            ctl.quit()
        ctl.wait_quit(QUIT_WATCHDOG_S + 1.0)
        tray.stop()
    return code


def _selftest(args: argparse.Namespace, data, token: str | None) -> int:
    """`--selftest` (desktop/selftest.py): no window, no tray icon, no message box; the result in
    `<data>\\selftest.json`.  Whatever it raises is exit 5 (a windowed exe would otherwise end in PyInstaller's error
    box).  It takes the data folder's single-instance mutex, so it never runs beside an app (or another selftest) on
    that folder."""
    from desktop import selftest
    result = data / selftest.RESULT_FILE
    instance = None
    try:
        instance = single.acquire(single.instance_name(data, True))
        if instance is None:
            return selftest.fail(result, "start", "another Strata Desktop (or selftest) is using this data folder")
        if not args.server:
            return selftest.fail(result, "start", "--selftest needs --server URL")
        if not token:
            return selftest.fail(result, "start", f"--selftest takes its token from {SELFTEST_ENV}, which was not set")
        if webview2.installed() is None:
            return selftest.fail(result, "webview2", words.t("no_webview2"))
        if not webview2.dotnet_ok():
            return selftest.fail(result, "webview2", words.t("no_dotnet"))
        log.info("Strata Desktop selftest starts against %s (data folder %s)", args.server, data)
        return selftest.run_hidden(args.server, data, token, result)
    except Exception as e:  # noqa: BLE001
        log.error("the selftest raised %s", type(e).__name__, exc_info=True)
        return selftest.fail(result, "crash", type(e).__name__, keep=True)
    finally:
        if instance is not None:
            instance.release()


if __name__ == "__main__":
    sys.exit(main())
