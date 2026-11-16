"""desktop/test_imports.py - the GUI libraries stay out of every module's import (#310 part 3 task 4): with
`webview`, `pystray` and `PIL` blocked, every `desktop` module imports.  Run in a fresh python, so modules this test
run already imported prove nothing.

    python -m pytest desktop/test_imports.py -q
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NO_WINDOW = 0x08000000 if os.name == "nt" else 0

PROBE = r"""
import pkgutil, sys
for lib in ("webview", "pystray", "PIL"):
    sys.modules[lib] = None                 # `import webview` now raises ImportError
import desktop
names = sorted(m.name for m in pkgutil.iter_modules(desktop.__path__)
               if not m.name.startswith("test_") and m.name != "__main__")
for name in names:
    __import__("desktop." + name)
loaded = sorted(k for k in sys.modules if k.split(".")[0] in ("webview", "pystray", "PIL") and sys.modules[k] is not None)
print("OK", ",".join(names), "LOADED", ",".join(loaded))
"""


def test_imports_without_gui_libs():
    out = subprocess.run([sys.executable, "-c", PROBE], cwd=ROOT, capture_output=True, text=True, timeout=120,
                         creationflags=NO_WINDOW)
    assert out.returncode == 0, out.stderr
    line = out.stdout.strip().splitlines()[-1]
    assert line.startswith("OK "), out.stdout
    names = line.split()[1].split(",")
    for must in ("applog", "inject", "single", "webview2", "gui", "servers", "probe", "runner_host", "state", "words",
                 "controller", "tray"):
        assert must in names, names
    assert line.endswith("LOADED"), line                            # nothing of the GUI libraries came in
