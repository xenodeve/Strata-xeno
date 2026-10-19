"""pytest: setup.py's data-folder migration (#198) saves the data folder it settled on in the user's settings file
(%APPDATA%\Strata\settings.json, ~/.config/strata/settings.json) and can MOVE model files from a folder it finds there.
A test that reaches it must never touch the real file, so the whole session gets its own config home."""
import os
import tempfile

_home = tempfile.mkdtemp(prefix="strata-pytest-config-")
os.environ["APPDATA"] = _home
os.environ["XDG_CONFIG_HOME"] = _home

# tools/*_test.py are not unit tests: they are scripts that drive a running engine or server and read sys.argv at
# import (python tools/early_close_test.py http://127.0.0.1:8090, python tools/cancel_park_test.py CFG, ...; each
# file's docstring has its command).  Upstream runs the tests with `unittest discover -p "test_*.py"` (CI:
# .github/workflows/terminal-minimal-server.yml), which never matches them; pytest's default python_files does, and
# collecting them ran the scripts (merge #268).
collect_ignore_glob = ["tools/*_test.py"]
