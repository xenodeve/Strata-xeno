# PyInstaller spec of Strata Desktop (#310 part 3): one folder, windowed.
#
#     desktop\.venv\Scripts\python desktop\build.py        (runs PyInstaller on this file with no window)
#
# Out: desktop\dist\StrataDesktop\StrataDesktop.exe (git-ignored, as desktop\build\).  One folder, not one file: a
# one-file exe unpacks itself to %TEMP% on every start (a slower start, and antivirus noise).
#
# The entry is desktop/__main__.py.  Its own imports (and the lazy ones inside functions) bring the desktop package;
# serve's modules come in as hidden imports, every one but the tests, the fakes, the UI hash tool and serve.dev, and
# but those that do not import in the venv: collect_submodules leaves out a submodule whose import fails. jinja2
# is not in the venv, so serve.server, serve.frontend, serve.identity, serve.web, serve.subagent and serve.agent_run
# are not in the bundle (build\StrataDesktop\PYZ-00.toc, 2026-11-16).  The app never imports them: every module the
# runner imports is there.  PyInstaller 6.22.3 says nothing in its log about the ones it leaves out, even with
# on_error="warn": read PYZ-00.toc after a build to see what came in.  Not bundled either: pypdf and
# pypdfium2 (optional in serve.media), so a My PC Read of a PDF in the exe takes the "not installed" path.
# Data: desktop/ui/* (servers.html, icon.png), at desktop/ui, where gui.py and tray.py look for it beside
# themselves.  The runner's serve modules load no file of their own (only serve/server.py and serve/ui_hash.py use
# __file__).
import os

from PyInstaller.utils.hooks import collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, os.pardir))  # noqa: F821 - SPECPATH is PyInstaller's


def _wanted(name: str) -> bool:
    last = name.rsplit(".", 1)[-1]
    return not (last.startswith("test_") or "_fake" in last or name == "serve.ui_hash"
                or name == "serve.dev" or name.startswith("serve.dev."))


def _desktop(name: str) -> bool:
    last = name.rsplit(".", 1)[-1]
    return not (last.startswith("test_") or last in ("build", "check_selftest", "__main__"))


hidden = collect_submodules("serve", filter=_wanted, on_error="warn") + \
    collect_submodules("desktop", filter=_desktop, on_error="warn")

a = Analysis(
    [os.path.join(ROOT, "desktop", "__main__.py")],
    pathex=[ROOT],
    binaries=[],
    datas=[(os.path.join(ROOT, "desktop", "ui", "*"), os.path.join("desktop", "ui"))],
    hiddenimports=hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="StrataDesktop",
    icon=os.path.join(ROOT, "desktop", "ui", "icon.png"),
    console=False,
    debug=False,
    strip=False,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="StrataDesktop")
