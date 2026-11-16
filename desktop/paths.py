"""Where Strata Desktop keeps its files (#310 part 3).  Nothing here creates a folder.

`data_dir()`: %LOCALAPPDATA%\\StrataDesktop on Windows, ~/.strata-desktop elsewhere, or the `--data-dir` folder.  In it:
`servers.json` (the server list, `desktop/servers.py`), `webview\\` (the WebView2 profile: the web app's localStorage,
so chats survive a restart) and `desktop.log`.  The runner keeps `serve.runner.runner_dir()` (%LOCALAPPDATA%\\StrataRunner:
`device.json`, `runner.json`, `checkpoints\\`), so the app and `python -m serve.runner` on one PC are one device; with
`--data-dir D` the runner's folder is `D\\runner`, so a test run never touches the real device.
"""
from __future__ import annotations

import os
from pathlib import Path

APP = "StrataDesktop"
APP_NAME = "Strata Desktop"          # the device's name is "<APP_NAME> on <machine>" (serve.runner.device_label)


def data_dir(override: str | None = None) -> Path:
    if override:
        return Path(override)
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / APP
    return Path.home() / ".strata-desktop"


def runner_dir(data: Path, override_used: bool) -> Path:
    if override_used:
        return Path(data) / "runner"
    from serve import runner          # imported here: serve.runner pulls in the agent modules
    return runner.runner_dir()


def servers_file(data: Path) -> Path:
    return Path(data) / "servers.json"


def webview_dir(data: Path) -> Path:
    return Path(data) / "webview"


def log_file(data: Path) -> Path:
    return Path(data) / "desktop.log"
