"""The Git state of a folder, read only, for the Chat's right panel (issue #99): the branch, the other branches, the changed files, the last commits, the worktrees, and the diff of a file.

Nothing here writes. Git is run as a program (no shell) with a short list of read commands, no pager, no terminal prompt, no optional locks (so a status does not touch the index), no
external diff or text conversion (a repository's own configuration cannot run a program through it) and no file-system monitor hook; with a time limit and a limit on what is read. Who
may ask is decided by the caller (the same callers as the coding tools: this PC, or the API key)."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading

TIMEOUT = 8.0                         # seconds one git command may take
MAX_OUT = 1_500_000                   # bytes read from one command
MAX_FILES = 300                       # changed files listed (with a note when there are more)
MAX_DIFF = 120_000                    # bytes of a diff shown
MAX_BRANCHES = 100
COMMITS = 12

SAFE = ["--no-pager", "-c", "core.quotepath=off", "-c", "core.fsmonitor=false", "-c", "diff.external=", "-c", "core.pager=cat"]


def available() -> str | None:
    return shutil.which("git")


def _env() -> dict:
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("STRATA_")}
    env.update(GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0", LC_ALL="C", GIT_PAGER="cat")
    env.pop("GIT_EXTERNAL_DIFF", None)                             # a program the environment names to show diffs is not run
    return env


def _run(folder: str, *args: str, limit: int = MAX_OUT) -> tuple[int, bytes, bool]:
    """(exit code, what it wrote, whether that was cut at `limit`). Exit code -1: git is missing or the command timed out."""
    git = available()
    if not git:
        return -1, b"", False
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    try:
        p = subprocess.Popen([git, *SAFE, *args], cwd=folder, env=_env(), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=flags)
    except OSError:
        return -1, b"", False
    timer = threading.Timer(TIMEOUT, p.kill)
    timer.start()
    try:
        out = p.stdout.read(limit + 1) if p.stdout else b""
        cut = len(out) > limit
        if cut:
            p.kill()
        p.wait()
        return (-1 if (timer.finished.is_set() and not cut) else p.returncode), out[:limit], cut
    finally:
        timer.cancel()
        if p.stdout:
            p.stdout.close()


def _text(b: bytes) -> str:
    return b.decode("utf-8", "replace")


# ------------------------------------------------------------------------------------------------ status
def _parse_status(raw: str) -> dict:
    """`git status --porcelain=v2 --branch -z`: the branch lines and the changed files."""
    head = {"branch": None, "detached": False, "oid": None, "upstream": None, "ahead": 0, "behind": 0}
    staged: list[dict] = []
    unstaged: list[dict] = []
    untracked: list[dict] = []
    conflicted: list[dict] = []
    parts = raw.split("\0")
    i = 0
    while i < len(parts):
        line = parts[i]
        i += 1
        if not line:
            continue
        if line.startswith("# "):
            key, _, value = line[2:].partition(" ")
            if key == "branch.head":
                head["detached"] = value == "(detached)"
                head["branch"] = None if head["detached"] else value
            elif key == "branch.oid":
                head["oid"] = None if value == "(initial)" else value[:9]
            elif key == "branch.upstream":
                head["upstream"] = value
            elif key == "branch.ab":
                a, _, b = value.partition(" ")
                head["ahead"], head["behind"] = int(a.lstrip("+") or 0), int(b.lstrip("-") or 0)
        elif line[0] == "1" and len(line) > 3:                         # an ordinary change: "1 XY sub mH mI mW hH hI path"
            fields = line.split(" ", 8)
            if len(fields) == 9:
                x, y, path = fields[1][0], fields[1][1], fields[8]
                if x != ".":
                    staged.append({"path": path, "status": x})
                if y != ".":
                    unstaged.append({"path": path, "status": y})
        elif line[0] == "2" and len(line) > 3:                         # a rename or a copy: "2 XY sub mH mI mW hH hI Xscore path", then the old path
            fields = line.split(" ", 9)
            old = parts[i] if i < len(parts) else ""
            i += 1
            if len(fields) == 10:
                x, y, path = fields[1][0], fields[1][1], fields[9]
                if x != ".":
                    staged.append({"path": path, "status": x, "from": old})
                if y != ".":
                    unstaged.append({"path": path, "status": y, "from": old})
        elif line[0] == "u":                                           # not merged: "u XY sub m1 m2 m3 mW h1 h2 h3 path"
            fields = line.split(" ", 10)
            if len(fields) == 11:
                conflicted.append({"path": fields[10], "status": "U"})
        elif line[0] == "?":
            untracked.append({"path": line[2:], "status": "?"})
    return {"head": head, "staged": staged, "unstaged": unstaged, "untracked": untracked, "conflicted": conflicted}


def _branches(folder: str) -> list[dict]:
    code, out, _ = _run(folder, "for-each-ref", f"--count={MAX_BRANCHES + 1}", "--sort=-committerdate",
                        "--format=%(refname:short)%09%(HEAD)%09%(upstream:short)%09%(upstream:track)", "refs/heads")
    if code != 0:
        return []
    rows = []
    for ln in _text(out).splitlines():
        name, _, rest = ln.partition("\t")
        mark, _, rest = rest.partition("\t")
        upstream, _, track = rest.partition("\t")
        if name:
            rows.append({"name": name, "current": mark == "*", **({"upstream": upstream} if upstream else {}), **({"track": track} if track else {})})
    return rows


def _worktrees(folder: str, here: str) -> list[dict]:
    code, out, _ = _run(folder, "worktree", "list", "--porcelain")
    if code != 0:
        return []
    rows: list[dict] = []
    cur: dict = {}
    for ln in _text(out).splitlines() + [""]:
        if not ln.strip():
            if cur.get("path"):
                rows.append({**cur, "current": os.path.normcase(os.path.realpath(cur["path"])) == os.path.normcase(here)})
            cur = {}
        elif ln.startswith("worktree "):
            cur["path"] = ln[9:]
        elif ln.startswith("HEAD "):
            cur["head"] = ln[5:14]
        elif ln.startswith("branch "):
            cur["branch"] = ln[7:].removeprefix("refs/heads/")
        elif ln == "detached":
            cur["detached"] = True
        elif ln == "bare":
            cur["bare"] = True
    return rows


def _commits(folder: str) -> list[dict]:
    code, out, _ = _run(folder, "log", f"-n{COMMITS}", "--format=%h%x09%an%x09%at%x09%s")
    if code != 0:
        return []
    rows = []
    for ln in _text(out).splitlines():
        sha, _, rest = ln.partition("\t")
        who, _, rest = rest.partition("\t")
        at, _, subject = rest.partition("\t")
        if sha:
            rows.append({"sha": sha, "author": who, "time": int(at) if at.isdigit() else 0, "subject": subject})
    return rows


def info(folder: str) -> dict:
    """What the panel shows for one folder. `repo` false: it is not inside a repository. `git` false: Git is not installed."""
    if not available():
        return {"ok": True, "git": False, "repo": False}
    path = os.path.realpath(folder) if isinstance(folder, str) and folder.strip() and "\0" not in folder else ""
    if not path or not os.path.isdir(path):
        return {"ok": False, "error": "not a folder"}
    code, out, _ = _run(path, "rev-parse", "--show-toplevel")
    if code != 0 or not out.strip():
        return {"ok": True, "git": True, "repo": False}
    root = os.path.realpath(_text(out).strip())
    code, out, cut = _run(path, "status", "--porcelain=v2", "--branch", "-z", "--untracked-files=normal")
    if code != 0:
        return {"ok": True, "git": True, "repo": True, "root": root, "error": "git status failed"}
    st = _parse_status(_text(out))
    groups = {k: st[k] for k in ("staged", "unstaged", "untracked", "conflicted")}
    counts = {k: len(v) for k, v in groups.items()}
    truncated = cut or sum(counts.values()) > MAX_FILES
    budget = MAX_FILES
    for k in groups:                                              # at most MAX_FILES files in all, the conflicts and the staged ones first
        groups[k] = groups[k][:budget]
        budget -= len(groups[k])
    branches = _branches(path)
    return {"ok": True, "git": True, "repo": True, "root": root, "folder": path, **st["head"], "counts": counts, "truncated": truncated, **groups,
            "branches": branches[:MAX_BRANCHES], "moreBranches": len(branches) > MAX_BRANCHES, "worktrees": _worktrees(path, root), "commits": _commits(path)}


# ------------------------------------------------------------------------------------------------ one file's diff
def _inside(root: str, rel: str) -> str | None:
    if not isinstance(rel, str) or not rel or "\0" in rel or os.path.isabs(rel) or rel.startswith(("/", "\\")):
        return None
    full = os.path.realpath(os.path.join(root, rel))
    a, b = os.path.normcase(full), os.path.normcase(os.path.realpath(root))
    try:
        return full if os.path.commonpath([a, b]) == b else None
    except ValueError:
        return None


def diff(folder: str, rel: str, staged: bool = False, untracked: bool = False) -> dict:
    """The diff of one changed file of the repository `folder` is in; an untracked file is shown as all added. Binary files and long diffs are cut with a note."""
    if not available():
        return {"ok": False, "error": "git is not installed"}
    path = os.path.realpath(folder) if isinstance(folder, str) and folder.strip() and "\0" not in folder and os.path.isdir(folder) else ""
    if not path:
        return {"ok": False, "error": "not a folder"}
    code, out, _ = _run(path, "rev-parse", "--show-toplevel")
    if code != 0 or not out.strip():
        return {"ok": False, "error": "not a repository"}
    root = os.path.realpath(_text(out).strip())
    full = _inside(root, rel)
    if full is None:
        return {"ok": False, "error": "that file is outside the repository"}
    if untracked:
        try:
            if not os.path.isfile(full):
                return {"ok": False, "error": "not a file"}
            with open(full, "rb") as f:
                raw = f.read(MAX_DIFF + 1)
        except OSError:
            return {"ok": False, "error": "cannot be read"}
        if b"\0" in raw[:8000]:
            return {"ok": True, "binary": True, "diff": "", "truncated": False}
        cut = len(raw) > MAX_DIFF
        lines = _text(raw[:MAX_DIFF]).splitlines()
        body = "\n".join(f"+{ln}" for ln in lines)
        return {"ok": True, "binary": False, "truncated": cut, "diff": f"--- /dev/null\n+++ b/{rel}\n@@ -0,0 +1,{len(lines)} @@\n{body}\n"}
    args = ["diff", "--no-color", "--no-ext-diff", "--no-textconv", "-U3"] + (["--cached"] if staged else []) + ["--", rel]
    code, out, cut = _run(path, *args, limit=MAX_DIFF)
    if code not in (0, 1) and not cut:
        return {"ok": False, "error": "git diff failed"}
    text = _text(out)
    return {"ok": True, "binary": "Binary files" in text[:2000] and "@@" not in text, "truncated": cut, "diff": text}
