"""The folders of this PC, for the page that makes a project (issue #96): a project is a folder, and the coding tools work in it.

`GET /agent/folders?path=` answers with where `path` is and the folders in it, so the page can offer a choice instead of a guess. Only the
folders' names are given, never a file's, and the same callers may ask as may use the coding tools (this PC, or the API key)."""
from __future__ import annotations

import os
import string

LIMIT = 500                          # more than this in one folder is cut (and says so)


def _drives() -> list[str]:
    return [f"{c}:\\" for c in string.ascii_uppercase if os.path.isdir(f"{c}:\\")]


def _hidden(name: str, full: str) -> bool:
    if name.startswith("."):
        return True
    try:
        import stat
        return bool(os.stat(full).st_file_attributes & stat.FILE_ATTRIBUTE_HIDDEN)      # Windows only; absent elsewhere
    except (AttributeError, OSError):
        return False


def look(raw: str | None) -> dict | None:
    """Where `raw` is and the folders in it; None when `raw` is not a folder. Blank: the user's home. "@drives" (Windows): the drives."""
    text = (raw or "").strip()
    if "\0" in text or len(text) > 1000:
        return None
    if text == "@drives" and os.name == "nt":
        return {"ok": True, "path": "", "parent": None, "dirs": [{"name": d, "path": d} for d in _drives()], "truncated": False}
    path = os.path.realpath(os.path.expanduser(text or "~"))
    if not os.path.isdir(path):
        return None
    parent = os.path.dirname(path)
    if parent == path:                                                  # a root: on Windows the drives are above it
        parent = "@drives" if os.name == "nt" else None
    dirs: list[dict] = []
    try:
        with os.scandir(path) as it:
            for e in it:
                try:
                    if e.is_dir() and not _hidden(e.name, e.path):
                        dirs.append({"name": e.name, "path": os.path.join(path, e.name)})
                except OSError:
                    continue
    except OSError:                                                     # a folder that may not be listed is still a folder
        pass
    dirs.sort(key=lambda d: d["name"].lower())
    return {"ok": True, "path": path, "parent": parent, "dirs": dirs[:LIMIT], "truncated": len(dirs) > LIMIT}
