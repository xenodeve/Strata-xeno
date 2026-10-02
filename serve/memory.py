"""serve/memory.py - the memory and instruction files the chat reads (issue #99), as Claude Code reads its own.

Three kinds, all read only:
- the project's own instruction files, in every folder of the project: `CLAUDE.md`, `CLAUDE.local.md`, `.claude/CLAUDE.md`, `AGENTS.md` (when there is no `CLAUDE.md`) and
  `.claude/rules/*.md`; a line `@path/to/file.md` brings that file in (inside the folder, a few levels deep). Always on: they are the project's. A file in a sub-folder is read when the
  model works in that sub-folder (`nested`).
- the instruction file each other app keeps for its user (Claude Code's `~/.claude/CLAUDE.md`, Codex's `AGENTS.md`, ...): another app's words, so **off until switched on**.
- the memory another app wrote for a project (Claude Code's `~/.claude/projects/<folder>/memory/`): off until switched on, per app.

What is read is the user's own notes about how to work with them, handed to the model after the rules and under them: it never changes what the model may do (the permission rules
decide that), and it is cut to a size so it does not eat the context. Nothing here writes. The home folder is a parameter (tests, mock servers).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

MAX_FILE = 20_000            # characters of one file that are used
MAX_TOTAL = 60_000           # characters of all of them
MAX_NESTED = 8_000
IMPORT_DEPTH = 3
RULES_MAX = 30               # rule files read from one folder

PROJECT_FILES = ("CLAUDE.md", "CLAUDE.local.md", os.path.join(".claude", "CLAUDE.md"))
NESTED_FILES = ("CLAUDE.md", "AGENTS.md")
USER_FILES = {"claude": ".claude/CLAUDE.md", "codex": ".codex/AGENTS.md", "agents": ".agents/AGENTS.md", "gemini": ".gemini/GEMINI.md"}
LABELS = {"claude": "Claude Code", "codex": "Codex", "agents": "Shared agents folder", "gemini": "Gemini CLI"}
IMPORT = re.compile(r"(?m)(?:^|[\s(])@((?:\.{1,2}/)?[A-Za-z0-9_][\w./-]*\.(?:md|markdown|txt|mdc))")


# ------------------------------------------------------------------------------------------------ the switches
def settings(cfg) -> dict:
    """The run config's `import.memory`: {"on": [source ids]}. Everything of another app is off until it is in the list; anything odd is ignored."""
    block = (cfg.get("import") or {}).get("memory") if isinstance(cfg, dict) and isinstance(cfg.get("import"), dict) else None
    on = block.get("on") if isinstance(block, dict) else None
    return {"on": [x for x in on if isinstance(x, str) and 0 < len(x) <= 100][:100] if isinstance(on, list) else []}


def check_settings(value) -> tuple[dict | None, list[dict]]:
    """What a page sends, strictly: (the settings, []) or (None, errors)."""
    err = lambda field, message: {"field": field, "message": message}      # noqa: E731
    if not isinstance(value, dict):
        return None, [err("memory", "memory is an object")]
    errors = [err(k, "not a setting that can be changed here") for k in value if k != "on"]
    on = value.get("on", [])
    if not isinstance(on, list) or len(on) > 100 or not all(isinstance(x, str) and re.fullmatch(r"[a-z-]+:(instructions|memory)", x) for x in on):
        errors.append(err("on", "on is a list of sources such as \"claude:memory\""))
    if errors:
        return None, errors
    return settings({"import": {"memory": value}}), []


# ------------------------------------------------------------------------------------------------ reading
def slug(folder: str) -> str:
    """The name Claude Code gives the folder of a project's memory: the path with everything that is not a letter or a digit as `-`."""
    return re.sub(r"[^A-Za-z0-9]", "-", os.path.realpath(folder))


def _read(path: str, cap: int = MAX_FILE) -> tuple[str, bool] | None:
    """(text, cut) of a text file, or None when it cannot be read or is binary."""
    try:
        with open(path, "rb") as f:
            raw = f.read(cap * 4 + 1)
    except OSError:
        return None
    if b"\0" in raw[:8192]:
        return None
    text = raw.decode("utf-8", errors="replace")
    return (text[:cap], True) if len(text) > cap else (text, False)


def _inside(path: str, root: str) -> bool:
    a, b = os.path.normcase(os.path.realpath(path)), os.path.normcase(os.path.realpath(root))
    try:
        return os.path.commonpath([a, b]) == b
    except ValueError:
        return False


def with_imports(path: str, root: str, depth: int = 0, seen: set | None = None) -> tuple[str, bool]:
    """A file's text followed by the files it names with `@path` lines (inside `root`, `IMPORT_DEPTH` levels, each once)."""
    seen = set() if seen is None else seen
    real = os.path.normcase(os.path.realpath(path))
    got = _read(path)
    if got is None or real in seen:
        return "", False
    seen.add(real)
    text, cut = got
    if depth >= IMPORT_DEPTH:
        return text, cut
    extra = []
    for rel in dict.fromkeys(IMPORT.findall(text)):
        target = os.path.join(os.path.dirname(path), rel)
        if os.path.isfile(target) and _inside(target, root):
            more, c = with_imports(target, root, depth + 1, seen)
            if more:
                extra.append(f"[imported from {rel}]\n{more.rstrip()}")
                cut = cut or c
    return (text.rstrip() + "\n\n" + "\n\n".join(extra) if extra else text), cut


def _shown(path: str, home: str) -> str:
    p = os.path.realpath(path)
    h = os.path.realpath(home)
    return "~" + p[len(h):].replace("\\", "/") if os.path.normcase(p).startswith(os.path.normcase(h) + os.sep) else p.replace("\\", "/")


# ------------------------------------------------------------------------------------------------ what there is
def _project_files(folder: str) -> list[str]:
    """The project's own instruction files in one folder, in the order they are read."""
    found = [os.path.join(folder, n) for n in PROJECT_FILES if os.path.isfile(os.path.join(folder, n))]
    if not os.path.isfile(os.path.join(folder, "CLAUDE.md")) and os.path.isfile(os.path.join(folder, "AGENTS.md")):
        found.insert(0, os.path.join(folder, "AGENTS.md"))
    rules = os.path.join(folder, ".claude", "rules")
    if os.path.isdir(rules):
        try:
            names = sorted(n for n in os.listdir(rules) if n.lower().endswith(".md") and os.path.isfile(os.path.join(rules, n)))
        except OSError:
            names = []
        found += [os.path.join(rules, n) for n in names[:RULES_MAX]]
    return found


def _memory_files(home: str, folder: str) -> list[str]:
    """Claude Code's memory of a project: its index first, then the topic files."""
    d = os.path.join(home, ".claude", "projects", slug(folder), "memory")
    if not os.path.isdir(d):
        return []
    try:
        names = sorted(n for n in os.listdir(d) if n.lower().endswith(".md") and os.path.isfile(os.path.join(d, n)))
    except OSError:
        return []
    names.sort(key=lambda n: (n.upper() != "MEMORY.md".upper(), n.lower()))
    return [os.path.join(d, n) for n in names]


def _size(files: list[str]) -> int:
    n = 0
    for f in files:
        try:
            n += os.path.getsize(f)
        except OSError:
            pass
    return n


def discover(home: str, folders: list[str], on: list[str]) -> list[dict]:
    """Every source there is for a project: the project's own files (always on), then each other app's user file and memory of the project (on when switched on).
    A source: {id, app, label, kind, files, shown, bytes, on}."""
    home = str(home)
    out: list[dict] = []
    for i, folder in enumerate(f for f in folders if f and os.path.isdir(f)):
        files = _project_files(folder)
        if files:
            out.append({"id": f"project:{i}", "app": "project", "label": os.path.basename(os.path.realpath(folder)) or folder, "kind": "project", "files": files,
                        "shown": [os.path.relpath(f, folder).replace("\\", "/") for f in files], "bytes": _size(files), "on": True, "folder": os.path.realpath(folder)})
    for app, rel in USER_FILES.items():
        path = os.path.join(home, *rel.split("/"))
        if os.path.isfile(path):
            out.append({"id": f"{app}:instructions", "app": app, "label": LABELS[app], "kind": "instructions", "files": [path], "shown": [_shown(path, home)], "bytes": _size([path]),
                        "on": f"{app}:instructions" in on})
    mem: list[str] = []
    seen = set()
    for folder in (f for f in folders if f and os.path.isdir(f)):
        for f in _memory_files(home, folder):
            if f not in seen:
                seen.add(f)
                mem.append(f)
    if mem:
        out.append({"id": "claude:memory", "app": "claude", "label": LABELS["claude"], "kind": "memory", "files": mem, "shown": [_shown(f, home) for f in mem], "bytes": _size(mem),
                    "on": "claude:memory" in on})
    return out


def collect(home: str, folders: list[str], on: list[str]) -> list[dict]:
    """The blocks the model is handed: for each source that is on, each file's text (with its imports), cut to a size. {id, kind, label, name, text, cut}."""
    blocks: list[dict] = []
    left = MAX_TOTAL
    for src in discover(home, folders, on):
        if not src["on"] or left <= 0:
            continue
        root = src.get("folder") or str(home)
        for path in src["files"]:
            if left <= 0:
                break
            text, cut = with_imports(path, root) if src["kind"] != "memory" else (_read(path) or ("", False))
            text = text.strip()
            if not text:
                continue
            if len(text) > left:
                text, cut = text[:left], True
            left -= len(text)
            blocks.append({"id": src["id"], "kind": src["kind"], "label": src["label"], "name": os.path.basename(path), "path": path, "text": text, "cut": cut})
    return blocks


def nested(path: str, roots: list[str], loaded: set) -> str | None:
    """The instruction files in the folders between a project folder and the file the model is working on (the folder's own were read at the start): once each, so a file in a
    sub-folder brings that sub-folder's rules with it. None when there are none."""
    p = os.path.realpath(path)
    root = next((r for r in roots if r and _inside(p, r)), None)
    if root is None:
        return None
    rr = os.path.realpath(root)
    chain: list[str] = []
    d = os.path.dirname(p)
    while os.path.normcase(d) != os.path.normcase(rr) and _inside(d, rr):
        chain.append(d)
        d = os.path.dirname(d)
    parts = []
    for folder in reversed(chain):                                   # the outer folder first
        for name in NESTED_FILES:
            f = os.path.join(folder, name)
            key = os.path.normcase(f)
            if key in loaded or not os.path.isfile(f):
                continue
            loaded.add(key)
            got = _read(f, MAX_NESTED)
            if got and got[0].strip():
                rel = os.path.relpath(f, rr).replace("\\", "/")
                parts.append(f"Instructions from {rel} (the folder you are working in):\n{got[0].strip()}" + ("\n[cut]" if got[1] else ""))
            break                                                    # CLAUDE.md, else AGENTS.md
    return "\n\n".join(parts) if parts else None
