"""The check before any window (#310 part 3, RF 5): Strata Desktop needs the Microsoft Edge WebView2 Runtime, and
pywebview's Edge Chromium backend needs .NET Framework 4.6.2 or later.  When either is missing, pywebview does not fail:
it silently falls back to the old MSHTML engine, which cannot run the web app.  So `preflight()` checks both first,
and a missing one gives a native message box and exit code 4.

WebView2, as Microsoft Learn documents it ("Detect if a WebView2 Runtime is already installed", verified on this PC in
Task 1): the REG_SZ `pv` under `HKEY_LOCAL_MACHINE\\SOFTWARE\\WOW6432Node\\Microsoft\\EdgeUpdate\\Clients\\{GUID}`
(32-bit Windows: without `WOW6432Node`) or `HKEY_CURRENT_USER\\Software\\Microsoft\\EdgeUpdate\\Clients\\{GUID}`;
missing, empty or 0.0.0.0 is not installed.  A version below 86.0.622.0 counts as missing too, because pywebview
(`webview/platforms/winforms.py` `_is_chromium`) would not use it.  .NET: `Release` under
`HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\NET Framework Setup\\NDP\\v4\\Full`, at least 394802 (4.6.2), pywebview's
own condition.
"""
from __future__ import annotations

import os
import platform

from desktop import words

GUID = "{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"           # the Evergreen WebView2 Runtime
CLIENTS = r"Microsoft\EdgeUpdate\Clients" + "\\" + GUID
MIN_VERSION = (86, 0, 622, 0)                           # pywebview's lowest
NET_KEY = r"SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full"
NET_462 = 394802
EXIT_MISSING = 4
MB_OK, MB_ICONERROR, MB_SETFOREGROUND = 0x0, 0x10, 0x10000


def _machine() -> str:
    return platform.machine()


def _read(root: str, path: str, value: str):
    """One registry value (`root` is a `winreg` HKEY name), or None when the key or the value is absent."""
    if os.name != "nt":
        return None
    import winreg
    try:
        with winreg.OpenKey(getattr(winreg, root), path) as key:
            data, _ = winreg.QueryValueEx(key, value)
            return data
    except OSError:
        return None


def _version(v) -> tuple[int, ...] | None:
    if not isinstance(v, str):
        return None
    try:
        parts = tuple(int(p) for p in v.strip().split("."))
    except ValueError:
        return None
    return parts if len(parts) == 4 else None


def installed() -> str | None:
    """The WebView2 Runtime's version, or None when it is not installed (or too old for pywebview)."""
    machine_wide = "SOFTWARE\\" + CLIENTS if _machine() == "x86" else "SOFTWARE\\WOW6432Node\\" + CLIENTS
    for root, path in (("HKEY_LOCAL_MACHINE", machine_wide), ("HKEY_CURRENT_USER", "Software\\" + CLIENTS)):
        pv = _read(root, path, "pv")
        v = _version(pv)
        if v is not None and any(v) and v >= MIN_VERSION:
            return pv.strip()
    return None


def dotnet_release() -> int | None:
    r = _read("HKEY_LOCAL_MACHINE", NET_KEY, "Release")
    return r if isinstance(r, int) else None


def dotnet_ok() -> bool:
    r = dotnet_release()
    return r is not None and r >= NET_462


def missing_message() -> str:
    return words.t("no_webview2")


def message_box(text: str) -> None:
    """A native error box (no window of ours exists yet)."""
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes
    box = ctypes.WinDLL("user32").MessageBoxW
    box.argtypes = [wintypes.HWND, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.UINT]
    box.restype = ctypes.c_int
    box(None, text, "Strata Desktop", MB_OK | MB_ICONERROR | MB_SETFOREGROUND)


def preflight() -> int | None:
    """None when the app can start; else shows the box and returns 4."""
    if installed() is None:
        message_box(missing_message())
        return EXIT_MISSING
    if not dotnet_ok():
        message_box(words.t("no_dotnet"))
        return EXIT_MISSING
    return None
