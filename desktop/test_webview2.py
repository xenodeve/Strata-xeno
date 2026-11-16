"""desktop/test_webview2.py - the check before any window (#310 part 3 task 4, RF 5): the WebView2 Runtime as Microsoft
documents finding it, and .NET Framework 4.6.2 (pywebview's own condition for Edge Chromium); either missing gives a
message box and exit code 4, never pywebview's fallback to the old MSHTML engine.  The registry and the box are
patched: no test opens a box or depends on this PC's installs (one test reads the real registry and only checks the
type).

    python -m pytest desktop/test_webview2.py -q
"""
from __future__ import annotations

import os

import pytest

from desktop import webview2, words

GUID = "{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
HKLM_WOW = ("HKEY_LOCAL_MACHINE", rf"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{GUID}", "pv")
HKLM_X86 = ("HKEY_LOCAL_MACHINE", rf"SOFTWARE\Microsoft\EdgeUpdate\Clients\{GUID}", "pv")
HKCU = ("HKEY_CURRENT_USER", rf"Software\Microsoft\EdgeUpdate\Clients\{GUID}", "pv")
NET = ("HKEY_LOCAL_MACHINE", r"SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full", "Release")


@pytest.fixture
def registry(monkeypatch):
    """A fake registry {(root, path, value): data}; anything not in it is absent."""
    reg: dict = {}
    monkeypatch.setattr(webview2, "_read", lambda root, path, value: reg.get((root, path, value)))
    monkeypatch.setattr(webview2, "_machine", lambda: "AMD64")
    return reg


@pytest.fixture
def box(monkeypatch):
    shown: list[str] = []
    monkeypatch.setattr(webview2, "message_box", shown.append)
    return shown


def test_installed_reads_version(registry):
    registry[HKLM_WOW] = "157.0.4341.46"
    assert webview2.installed() == "157.0.4341.46"


def test_installed_per_user(registry):
    registry[HKCU] = "140.0.1.2"
    assert webview2.installed() == "140.0.1.2"


def test_x86_windows_reads_without_wow6432node(registry, monkeypatch):
    monkeypatch.setattr(webview2, "_machine", lambda: "x86")
    registry[HKLM_WOW] = "157.0.4341.46"                            # not where 32-bit Windows keeps it
    assert webview2.installed() is None
    registry[HKLM_X86] = "157.0.4341.46"
    assert webview2.installed() == "157.0.4341.46"


@pytest.mark.parametrize("pv", [None, "", "0.0.0.0", "  ", "not a version", "85.0.1.1", "86.0.621.9", 157])
def test_not_installed(registry, pv):
    if pv is not None:
        registry[HKLM_WOW] = pv
    assert webview2.installed() is None


def test_oldest_pywebview_takes(registry):
    registry[HKLM_WOW] = "86.0.622.0"
    assert webview2.installed() == "86.0.622.0"


def test_missing_webview2_message(registry, box):
    registry[NET] = 533320
    assert webview2.preflight() == 4
    assert box == [words.t("no_webview2")]
    assert webview2.missing_message() == words.t("no_webview2")


def test_missing_dotnet_message(registry, box):
    registry[HKLM_WOW] = "157.0.4341.46"
    registry[NET] = 394801                                           # one below 4.6.2
    assert webview2.preflight() == 4
    assert box == [words.t("no_dotnet")]
    del registry[NET]
    box.clear()
    assert webview2.preflight() == 4
    assert box == [words.t("no_dotnet")]


def test_preflight_passes(registry, box):
    registry[HKLM_WOW] = "157.0.4341.46"
    registry[NET] = 394802
    assert webview2.preflight() is None
    assert box == []


@pytest.mark.skipif(os.name != "nt", reason="the registry is Windows'")
def test_real_registry_answers_a_version_or_none():
    v = webview2.installed()
    assert v is None or (isinstance(v, str) and v.count(".") == 3)
    r = webview2.dotnet_release()
    assert r is None or isinstance(r, int)
