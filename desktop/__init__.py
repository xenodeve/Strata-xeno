"""Strata Desktop (#310 part 3): the Windows program that opens a Strata server's web app in its own window and runs
the runner (`serve/runner.py`) in the background, with a tray icon.  Its GUI libraries live in `desktop\\.venv`
(`desktop/requirements.txt`); only `desktop/gui.py` imports them, inside functions, so the rest is plain Python."""
