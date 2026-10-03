"""serve/files.py - the files of a project's folders, for `@file` in the prompt (issue #99): which files go with a few typed letters, and the text of one that was mentioned.

Read only, and only inside the project's folders (a path that leads out of them by `..` or a link is refused); files that look like secrets (keys, tokens, .env) are not listed and not
read; `.git`, `node_modules` and the like are not walked. Who may ask is decided by the caller (the same callers as the coding tools)."""
from __future__ import annotations

import os

from serve import permissions

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache", ".idea", ".vscode", "dist", "build", ".next", "target", ".gradle"}
MAX_SCAN = 25_000            # entries looked at in all, so a huge tree answers in time
MAX_DEPTH = 10
LIMIT = 20                   # files offered
MAX_TEXT = 60_000            # characters of one mentioned file that go to the model


def _folders(folders: list[str]) -> list[str]:
    out: list[str] = []
    for f in folders:
        if isinstance(f, str) and f.strip() and "\0" not in f and os.path.isdir(f):
            r = os.path.realpath(f)
            if r not in out:
                out.append(r)
    return out


def _rank(rel: str, q: str) -> tuple[int, int, int] | None:
    """How well a relative path goes with what was typed (any case): the name starts with it, the name has it, the path has it; shorter and shallower first. None: not at all."""
    low, name, ql = rel.lower(), os.path.basename(rel).lower(), q.lower()
    if not ql:
        return (3, rel.count("/"), len(rel))
    if name.startswith(ql):
        return (0, rel.count("/"), len(rel))
    if ql in name:
        return (1, rel.count("/"), len(rel))
    if ql in low:
        return (2, rel.count("/"), len(rel))
    return None


def find(folders: list[str], q: str, limit: int = LIMIT) -> list[dict]:
    """The files of the folders that go with `q`, best first: [{"path": relative with /, "folder": index}]. A tree is walked breadth first, so shallow files are found first."""
    roots = _folders(folders)
    q = (q or "").strip().replace("\\", "/")
    if len(q) > 200 or "\0" in q:
        return []
    found: list[tuple[tuple[int, int, int], int, str]] = []
    scanned = 0
    for i, root in enumerate(roots):
        level = [root]
        depth = 0
        while level and depth <= MAX_DEPTH and scanned < MAX_SCAN:
            nxt: list[str] = []
            for d in level:
                try:
                    entries = sorted(os.scandir(d), key=lambda e: e.name.lower())
                except OSError:
                    continue
                for e in entries:
                    scanned += 1
                    if scanned > MAX_SCAN:
                        break
                    try:
                        if e.is_dir(follow_symlinks=False):
                            if e.name not in SKIP_DIRS and not e.name.startswith("."):
                                nxt.append(e.path)
                        elif e.is_file() and not permissions.is_secret(e.path):
                            rel = os.path.relpath(e.path, root).replace("\\", "/")
                            r = _rank(rel, q)
                            if r is not None:
                                found.append((r, i, rel))
                    except OSError:
                        continue
            level = nxt
            depth += 1
    found.sort(key=lambda x: (x[0], x[1], x[2]))
    return [{"path": rel, "folder": i} for _, i, rel in found[:limit]]


def read(folders: list[str], rel: str) -> dict:
    """The text of the file `rel` names, in the first folder that has it. {"ok": True, "name", "folder", "text", "cut"}, or {"ok": False, "error"}."""
    if not isinstance(rel, str) or not rel.strip() or "\0" in rel or os.path.isabs(rel) or rel.startswith(("/", "\\")) or len(rel) > 500:
        return {"ok": False, "error": "that is not a path inside the project"}
    for i, root in enumerate(_folders(folders)):
        full = os.path.realpath(os.path.join(root, rel))
        if not permissions.inside(full, root):
            continue
        if not os.path.isfile(full):
            continue
        if permissions.is_secret(full):
            return {"ok": False, "error": "it looks like a secret (a key, a token, an .env file)"}
        try:
            with open(full, "rb") as f:
                raw = f.read(MAX_TEXT * 4 + 1)
        except OSError:
            return {"ok": False, "error": "it cannot be read"}
        if b"\0" in raw[:8192]:
            return {"ok": False, "error": "it is a binary file"}
        text = raw.decode("utf-8", errors="replace")
        return {"ok": True, "name": rel.replace("\\", "/"), "folder": i, "text": text[:MAX_TEXT], "cut": len(text) > MAX_TEXT}
    return {"ok": False, "error": "no such file in the project's folders"}
