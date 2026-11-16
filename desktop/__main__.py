"""Strata Desktop (#310 part 3): `python -m desktop`, and the built `StrataDesktop.exe`.

    python -m desktop [--data-dir PATH] [--debug]
    python -m desktop --selftest --server URL --data-dir PATH        (the token in STRATA_DESKTOP_SELFTEST_TOKEN;
                                                                      the result in PATH\\selftest.json)

Exit codes: 0 normal (a second start that handed over to the first included), 4 no WebView2 Runtime or .NET 4.6.2
(after a message box), 5 the selftest failed.

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
EXIT_SELFTEST_FAILED = 5
SELFTEST_ENV = "STRATA_DESKTOP_SELFTEST_TOKEN"
APP_NAME = paths.APP_NAME


def parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="StrataDesktop", description="Strata Desktop: your Strata servers in a window, "
                                "with this PC's tools (My PC) running beside them.")
    p.add_argument("--data-dir", metavar="PATH", help="keep servers.json, the web view's profile and desktop.log here "
                   "(default %%LOCALAPPDATA%%\\StrataDesktop); the runner's device then lives in PATH\\runner")
    p.add_argument("--selftest", action="store_true", help="check the program against --server without a window; "
                   "the token comes from " + SELFTEST_ENV)
    p.add_argument("--server", metavar="URL", help="the server --selftest checks")
    p.add_argument("--debug", action="store_true", help="pywebview's debug mode (the web inspector)")
    a = p.parse_args(argv)
    if a.server and not a.selftest:
        p.error("--server is for --selftest only")
    if a.selftest and not a.data_dir:              # its own folder: never the real server list, profile or device
        p.error("--selftest needs --data-dir PATH")
    return a


def main(argv: list[str] | None = None) -> int:
    selftest_token = os.environ.pop(SELFTEST_ENV, None)     # at once: nothing the app starts inherits it
    args = parse(argv)
    data = paths.data_dir(args.data_dir)
    applog.setup(paths.log_file(data))
    applog.install_hooks()
    applog.capture_stdio()
    if selftest_token:
        applog.add_secret(selftest_token)
    from serve import runner
    runner._say = applog.say

    if args.selftest:
        return _selftest(args, data, selftest_token)

    code = webview2.preflight()
    if code is not None:
        return code

    name = single.instance_name(data, bool(args.data_dir))
    instance = single.acquire(name)
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
    gui = Gui(data, None, quitting=lambda: ctl is not None and ctl.quitting, debug=args.debug)
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
    box).  No single-instance mutex: the data folder is the selftest's own."""
    from desktop import selftest
    result = data / selftest.RESULT_FILE
    try:
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


if __name__ == "__main__":
    sys.exit(main())
