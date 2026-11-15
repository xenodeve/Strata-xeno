"""The folders of this PC, for the page that makes a project (issue #96): a project is a folder, and the coding tools work in it.

`GET /agent/folders?path=` answers with where `path` is and the folders in it, so the page can offer a choice instead of a guess. Only the
folders' names are given, never a file's, and the same callers may ask as may use the coding tools (this PC, or the API key).

`resolve` and `in_git` are a coding run's folder rules (#310 part 2): the server applies them to a project on this PC, and the runner to
one on the engineer's own PC."""
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


def resolve(cwd_raw, dirs_raw, max_dirs: int) -> tuple[str | None, list[str]]:
    """A coding run's folders as the page named them (#96, #310): the main one when it is a folder (its real path), and the
    project's other folders - the real ones, once each, not the main one, at most `max_dirs`; none without a main one.
    Anything odd is ignored.  The server calls it for a project on this PC, the runner for one on the engineer's PC."""
    raw = cwd_raw if isinstance(cwd_raw, str) else None
    folder = os.path.realpath(raw) if raw and raw.strip() and "\0" not in raw and os.path.isdir(raw) else None
    dirs: list[str] = []
    for d in dirs_raw if folder and isinstance(dirs_raw, list) else []:
        if isinstance(d, str) and d.strip() and "\0" not in d and os.path.isdir(d):
            r = os.path.realpath(d)
            if os.path.normcase(r) != os.path.normcase(folder) and os.path.normcase(r) not in [os.path.normcase(x) for x in dirs]:
                dirs.append(r)
    return folder, dirs[:max_dirs]


def in_git(folder: str | None) -> bool:
    """The folder or one of the five folders above it holds `.git` (a folder, or a worktree's file)."""
    d = folder
    for _ in range(6):
        if not d:
            break
        if os.path.exists(os.path.join(d, ".git")):
            return True
        parent = os.path.dirname(d)
        d = parent if parent != d else None
    return False
