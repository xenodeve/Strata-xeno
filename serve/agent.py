"""serve/agent.py - the chat's coding tools, with Claude Code's names and parameters (Read, Write, Edit, Glob, Grep, TodoWrite, Bash, ...).

They are a built-in server of the MCP hub (like serve/skills.py): the model sees them as `agent__Read` and so on and calls them through the
chat's own tool loop.  Two things make them different from other tools:

  * every call goes through the permission gate in `AgentServer.call` - there is no way to run a tool without a context that says which
    project folder, mode and rules apply and how to ask the user (serve/permissions.py decides; the context's `ask` waits for the page's
    answer).  A call with no context is refused.
  * the tools keep a little state per chat (`session`): which files were read (Write and Edit refuse a file that was not read in this chat, or
    that changed since), and the todo list.

The descriptions the model reads are written for this app; the parameters and the behaviours (line-numbered reads, unique-match edits,
newest-first globs, ripgrep-like search) follow Claude Code's.  Nothing here reaches outside what the permission gate allowed.
"""
from __future__ import annotations

import collections
import fnmatch
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from serve import permissions

MAX_READ_LINES = 2000
MAX_LINE_CHARS = 2000
MAX_READ_BYTES = 256 * 1024
MAX_GLOB = 100
MAX_GREP_FILE = 2_000_000
MAX_GREP_LINES = 250
GREP_SECONDS = 10.0
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache", ".idea", ".vscode"}
TYPES = {"py": ("*.py", "*.pyi"), "js": ("*.js", "*.mjs", "*.cjs", "*.jsx"), "ts": ("*.ts", "*.tsx"), "json": ("*.json",), "md": ("*.md", "*.markdown"), "css": ("*.css",),
         "html": ("*.html", "*.htm"), "rust": ("*.rs",), "go": ("*.go",), "java": ("*.java",), "c": ("*.c", "*.h"), "cpp": ("*.cpp", "*.cc", "*.cxx", "*.hpp", "*.h"),
         "yaml": ("*.yml", "*.yaml"), "toml": ("*.toml",), "sh": ("*.sh", "*.bash"), "txt": ("*.txt",), "cs": ("*.cs",), "rb": ("*.rb",), "php": ("*.php",),
         "swift": ("*.swift",), "kt": ("*.kt", "*.kts"), "sql": ("*.sql",), "xml": ("*.xml",), "vue": ("*.vue",), "svelte": ("*.svelte",)}
TODO_STATES = ("pending", "in_progress", "completed")
NESTED = re.compile(r"\((?:[^()\\]|\\.)*[+*](?:[^()\\]|\\.)*\)\s*(?:[+*]|\{\d*,\d*\})")        # a repeated group that repeats inside: (a+)+  (a*)*  (.*)+


def _ok(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": False}


def _err(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": True}


@dataclass
class AgentContext:
    """What one request brings: the rules that apply, which chat it is, how to ask the user, and whether the request was cancelled."""
    policy: permissions.Policy
    session: str = ""
    ask: Callable[[dict], str] = lambda req: "deny"        # "allow" | "allow_chat" | "deny" | "cancelled"
    cancel: threading.Event = field(default_factory=threading.Event)
    emit: Callable[[dict], None] = lambda event: None      # tool activity for the page (a todo list, a shell's output)


class Session:
    def __init__(self):
        self.reads: dict[str, tuple[int, int]] = {}        # real path -> (mtime_ns, size) when it was last read or written here
        self.todos: list[dict] = []
        self.shells: dict = {}                              # background commands of this chat (serve/shell.py)
        self.shell_count = 0
        self.lock = threading.Lock()


# ------------------------------------------------------------------------------------------------ text helpers
def _read_text(path: str) -> tuple[str, str]:
    """The file's text and its line ending ("\\r\\n" or "\\n")."""
    with open(path, "rb") as f:
        raw = f.read()
    text = raw.decode("utf-8", errors="replace")
    return text, "\r\n" if "\r\n" in text else "\n"


def _is_binary(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return b"\0" in f.read(8192)
    except OSError:
        return True


def _write_atomic(path: str, text: str) -> None:
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".strata-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(text.encode("utf-8"))
        try:
            os.chmod(tmp, os.stat(path).st_mode & 0o7777)
        except OSError:
            pass
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _numbered(lines: list[str], start: int) -> str:
    out = []
    for i, ln in enumerate(lines, start):
        if len(ln) > MAX_LINE_CHARS:
            ln = ln[:MAX_LINE_CHARS] + f"... [line truncated: {len(ln) - MAX_LINE_CHARS:,} more characters]"
        out.append(f"{i:6}\t{ln}")
    return "\n".join(out)


def _int(v, name: str, low: int) -> tuple[int | None, str]:
    if v is None:
        return None, ""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or int(v) != v or int(v) < low:
        return None, f"{name} must be a whole number of at least {low}"
    return int(v), ""


# ------------------------------------------------------------------------------------------------ the server
SCHEMAS = {
    "Read": ("Read a text file from the project (or another place, if the user allows it). Returns the lines with their numbers. Give an absolute path, or one "
             "relative to the project folder. Long files are cut at 2000 lines: use offset and limit to read a window. Read a file before you Write or Edit it.",
             {"file_path": {"type": "string", "description": "the file to read"}, "offset": {"type": "integer", "description": "the line to start at (1 is the first)"},
              "limit": {"type": "integer", "description": "how many lines to read"}}, ["file_path"]),
    "Write": ("Write a whole file (create it, or replace it). To replace an existing file you must have read it first in this chat. To change a part of a file use Edit.",
              {"file_path": {"type": "string"}, "content": {"type": "string", "description": "the complete new content"}}, ["file_path", "content"]),
    "Edit": ("Replace an exact piece of text in a file. old_string must appear once in the file (add surrounding lines to make it unique) unless replace_all is true. "
             "You must have read the file first in this chat. An empty old_string creates a new file with new_string.",
             {"file_path": {"type": "string"}, "old_string": {"type": "string", "description": "the exact text to replace"}, "new_string": {"type": "string", "description": "the text to put instead"},
              "replace_all": {"type": "boolean", "description": "replace every occurrence (default false)"}}, ["file_path", "old_string", "new_string"]),
    "Glob": ("Find files by a name pattern such as **/*.py or src/**/*.ts. Returns paths, the most recently changed first (at most 100).",
             {"pattern": {"type": "string", "description": "the glob pattern"}, "path": {"type": "string", "description": "the folder to search (default: the project folder)"}}, ["pattern"]),
    "Grep": ("Search file contents with a regular expression. output_mode: files_with_matches (default) lists the files, content shows the matching lines, count counts them per file. "
             "Narrow it with glob or type. Skips .git, node_modules and binary files.",
             {"pattern": {"type": "string", "description": "the regular expression"}, "path": {"type": "string", "description": "a file or folder (default: the project folder)"},
              "glob": {"type": "string", "description": "only files matching this glob, e.g. *.tsx"}, "type": {"type": "string", "description": "only this kind of file: py, js, ts, md, ..."},
              "output_mode": {"type": "string", "enum": ["content", "files_with_matches", "count"]}, "-i": {"type": "boolean", "description": "ignore case"},
              "-n": {"type": "boolean", "description": "line numbers (content mode; default true)"}, "-A": {"type": "integer", "description": "lines after each match"},
              "-B": {"type": "integer", "description": "lines before each match"}, "-C": {"type": "integer", "description": "lines before and after each match"},
              "head_limit": {"type": "integer", "description": "show only the first N results (default 250)"}, "offset": {"type": "integer", "description": "skip the first N results"},
              "multiline": {"type": "boolean", "description": "let . and \\n match across lines"}}, ["pattern"]),
    "ExitPlanMode": ("Use this in plan mode when your plan is ready: send the plan, and the user decides whether you may start. Do not use it for anything else.",
                     {"plan": {"type": "string", "description": "the plan, in Markdown"}}, ["plan"]),
    "TodoWrite": ("Keep a list of the steps of a longer task and mark them as you go: exactly one in_progress at a time. Send the whole list every time.",
                  {"todos": {"type": "array", "items": {"type": "object", "properties": {
                      "content": {"type": "string", "description": "the step, as a command"}, "status": {"type": "string", "enum": list(TODO_STATES)},
                      "activeForm": {"type": "string", "description": "the step while it is being done, e.g. Running the tests"}},
                      "required": ["content", "status", "activeForm"]}}}, ["todos"]),
}


class AgentServer:
    """Looks like an McpServer to the hub (name, kind, status, tools, call, start, close) but has no process and no transport."""

    name = "agent"
    kind = "builtin"
    wants_context = True                                        # the hub passes the request's AgentContext to call()
    plain_names = True                                          # the model sees Read, Bash... as Claude Code names them, not agent__Read
    hidden = True                                               # not listed with the MCP servers (GET /mcp): the page has GET /agent for it

    def __init__(self, extra: dict | None = None):
        self.status = "ready"
        self.error = None
        self.info = {"name": "agent", "title": "Coding tools (Claude Code's: files, search, commands)"}
        self.transport = None
        self.lock = threading.Lock()
        self.last_start = 0.0
        self._sessions: collections.OrderedDict[str, Session] = collections.OrderedDict()
        self._tools = dict(SCHEMAS)
        self._run = {"Read": self._read, "Write": self._write, "Edit": self._edit, "Glob": self._glob, "Grep": self._grep, "TodoWrite": self._todo, "ExitPlanMode": self._exit_plan}
        for name, (desc, props, req, fn) in (extra or {}).items():            # more tools (Bash lives in serve/shell.py)
            self._tools[name] = (desc, props, req)
            self._run[name] = fn
        self.tools = [self._schema(n) for n in self._tools]

    def _schema(self, name: str) -> dict:
        desc, props, req = self._tools[name]
        return {"name": name, "description": desc, "inputSchema": {"type": "object", "properties": props, "required": req}}

    def start(self) -> bool:
        return True

    def close(self) -> None:
        """The background commands end with the server."""
        with self.lock:
            sessions = list(self._sessions.values())
        for s in sessions:
            for b in list(s.shells.values()):
                b.kill()

    def add_tool(self, name: str, description: str, properties: dict, required: list, fn) -> None:
        self._tools[name] = (description, properties, required)
        self._run[name] = fn
        self.tools = [self._schema(n) for n in self._tools]

    # -------------------------------------------------------------------------------------------- per chat
    def session(self, key: str) -> Session:
        with self.lock:
            s = self._sessions.get(key)
            if s is None:
                s = self._sessions[key] = Session()
                while len(self._sessions) > 100:
                    self._sessions.popitem(last=False)
            else:
                self._sessions.move_to_end(key)
            return s

    def todos(self, key: str) -> list[dict]:
        with self.lock:
            s = self._sessions.get(key)
        return list(s.todos) if s else []

    # -------------------------------------------------------------------------------------------- the gate
    def call(self, tool: str, arguments: dict, timeout: float = 0, cancel=None, ctx: AgentContext | None = None) -> dict:
        if ctx is None:
            return _err("the coding tools run only inside a chat that says which project folder and rules apply")
        if tool not in self._run:
            return _err(f"there is no tool named {tool!r} in the coding tools")
        args = arguments if isinstance(arguments, dict) else {}
        d = permissions.decide(tool, args, ctx.policy)
        if d.kind == "deny":
            return _err(f"{tool} is denied: {d.why}")
        if d.kind == "ask":
            rule = permissions.rule_for(tool, args)
            answer = ctx.ask({"tool": tool, "arguments": args, "why": d.why, "danger": d.danger, "judgeable": d.judgeable, "rule": rule})
            if answer == "allow_chat":
                if rule:
                    ctx.policy.allow.append(rule)                    # for the rest of this request; the page remembers it for the chat
            elif isinstance(answer, str) and answer.startswith("blocked:"):
                return _err(f"{tool} was blocked: {answer[8:].strip()}. Do not retry it; do something safer or ask the user.")
            elif answer == "cancelled":
                return _err("The request was cancelled before the user answered.")
            elif answer != "allow":
                return _err(f"The user did not allow this {tool} call ({d.why}). Do not try the same thing again; ask them what they want instead.")
        try:
            return self._run[tool](args, ctx)
        except PermissionError as e:
            return _err(f"{tool} could not do it: {e.strerror or e}")
        except OSError as e:
            return _err(f"{tool} could not do it: {e.strerror or e}")

    # -------------------------------------------------------------------------------------------- Read
    def _path(self, raw, ctx: AgentContext) -> str | None:
        return permissions.real(raw, ctx.policy.cwd) if isinstance(raw, str) else None

    def _read(self, a: dict, ctx: AgentContext) -> dict:
        path = self._path(a.get("file_path"), ctx)
        if path is None:
            return _err("file_path must be a path (absolute, or relative to the project folder)")
        offset, e1 = _int(a.get("offset"), "offset", 1)
        limit, e2 = _int(a.get("limit"), "limit", 1)
        if e1 or e2:
            return _err(e1 or e2)
        if not os.path.exists(path):
            return _err(f"File does not exist: {a['file_path']}")
        if os.path.isdir(path):
            return _err(f"{a['file_path']} is a directory, not a file (use Glob to list what is in it)")
        if _is_binary(path):
            return _err("This tool cannot read binary files.")
        size = os.path.getsize(path)
        if size > MAX_READ_BYTES and limit is None:
            return _err(f"The file is {size:,} bytes, more than the {MAX_READ_BYTES:,} that can be read at once. Use offset and limit to read a part of it, or Grep to find what you need.")
        text, _eol = _read_text(path)
        st = os.stat(path)
        s = self.session(ctx.session)
        s.reads[path] = (st.st_mtime_ns, st.st_size)
        lines = text.splitlines()
        if not lines:
            return _ok("<system-reminder>The file exists but its contents are empty.</system-reminder>")
        start = (offset or 1) - 1
        count = limit or MAX_READ_LINES
        chunk = lines[start:start + count]
        if not chunk:
            return _ok(f"<system-reminder>The file has {len(lines)} lines, fewer than the offset {offset}.</system-reminder>")
        out = _numbered(chunk, start + 1)
        if limit is None and start + len(chunk) < len(lines):
            out += f"\n\n[The file has {len(lines)} lines; showing {start + 1}-{start + len(chunk)}. Use offset to read more.]"
        return _ok(out)

    # -------------------------------------------------------------------------------------------- Write and Edit
    def _stale(self, path: str, s: Session) -> str | None:
        """Why a file that exists may not be changed now (None: it may)."""
        seen = s.reads.get(path)
        if seen is None:
            return "File has not been read yet. Read it first before changing it."
        st = os.stat(path)
        if st.st_mtime_ns != seen[0] or st.st_size != seen[1]:
            return "File has been modified since it was read, by the user or a tool. Read it again before changing it."
        return None

    def _remember(self, path: str, s: Session) -> None:
        st = os.stat(path)
        s.reads[path] = (st.st_mtime_ns, st.st_size)

    def _write(self, a: dict, ctx: AgentContext) -> dict:
        path = self._path(a.get("file_path"), ctx)
        content = a.get("content")
        if path is None:
            return _err("file_path must be a path (absolute, or relative to the project folder)")
        if not isinstance(content, str):
            return _err("content must be text (the whole new content of the file)")
        s = self.session(ctx.session)
        if os.path.isdir(path):
            return _err(f"{a['file_path']} is a directory, not a file")
        existed = os.path.exists(path)
        if existed:
            why = self._stale(path, s)
            if why:
                return _err(why)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _write_atomic(path, content)
        self._remember(path, s)
        return _ok(f"File {'updated' if existed else 'created'} successfully at: {a['file_path']}" if existed else f"File created successfully at: {a['file_path']}")

    def _edit(self, a: dict, ctx: AgentContext) -> dict:
        path = self._path(a.get("file_path"), ctx)
        old, new, every = a.get("old_string"), a.get("new_string"), a.get("replace_all", False)
        if path is None:
            return _err("file_path must be a path (absolute, or relative to the project folder)")
        if not isinstance(old, str) or not isinstance(new, str) or not isinstance(every, bool):
            return _err("old_string and new_string must be text, and replace_all true or false")
        if old == new:
            return _err("No changes to make: old_string and new_string are exactly the same.")
        s = self.session(ctx.session)
        if not os.path.exists(path):
            if old == "":
                os.makedirs(os.path.dirname(path), exist_ok=True)
                _write_atomic(path, new)
                self._remember(path, s)
                return _ok(f"Created {a['file_path']}")
            return _err(f"File does not exist: {a['file_path']}")
        if os.path.isdir(path):
            return _err(f"{a['file_path']} is a directory, not a file")
        why = self._stale(path, s)
        if why:
            return _err(why)
        if old == "":
            return _err("old_string is empty, but the file already exists: give the text to replace.")
        text, eol = _read_text(path)
        if eol == "\r\n":                                                    # the model writes \n; the file keeps its own line ends
            old, new = old.replace("\r\n", "\n").replace("\n", "\r\n"), new.replace("\r\n", "\n").replace("\n", "\r\n")
        n = text.count(old)
        if n == 0:
            return _err(f"String to replace not found in file.\nString: {a['old_string']}")
        if n > 1 and not every:
            return _err(f"Found {n} matches of the string to replace, but replace_all is false. To replace all occurrences, set replace_all to true. "
                        f"To replace only one occurrence, provide more context to identify the instance.\nString: {a['old_string']}")
        updated = text.replace(old, new) if every else text.replace(old, new, 1)
        _write_atomic(path, updated)
        self._remember(path, s)
        at = text.find(old)
        line = text[:at].count("\n") + 1
        lines = updated.splitlines()
        lo, hi = max(0, line - 1 - 4), min(len(lines), line - 1 + new.count("\n") + 1 + 4)
        return _ok(f"The file {a['file_path']} has been updated successfully" + (f" ({n} replacements)" if every and n > 1 else "") + f". A part of it now reads:\n" + _numbered(lines[lo:hi], lo + 1))

    # -------------------------------------------------------------------------------------------- Glob and Grep
    def _walk(self, root: str, cancel: threading.Event):
        for dirpath, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            if cancel.is_set():
                return
            for f in files:
                yield os.path.join(dirpath, f)

    def _glob(self, a: dict, ctx: AgentContext) -> dict:
        pat = a.get("pattern")
        if not isinstance(pat, str) or not pat.strip():
            return _err("pattern must be a glob such as **/*.py")
        raw = a.get("path")
        root = self._path(raw, ctx) if raw else ctx.policy.cwd
        if root is None or not os.path.isdir(root):
            return _err(f"path is not a folder: {raw or '(no project folder)'}")
        pat = pat.replace("\\", "/")
        if os.path.isabs(pat) or re.match(r"^[A-Za-z]:", pat):
            head = re.split(r"[*?\[{]", pat)[0]
            base = os.path.dirname(head) if not head.endswith("/") else head.rstrip("/")
            root, pat = base or root, pat[len(base):].lstrip("/") if base else pat
        rx = permissions._glob_re(pat)
        found = []
        for p in self._walk(root, ctx.cancel):
            rel = os.path.relpath(p, root).replace("\\", "/")
            if rx.match(rel):
                try:
                    found.append((os.stat(p).st_mtime, p))
                except OSError:
                    pass
        if not found:
            return _ok("No files found")
        found.sort(key=lambda x: -x[0])
        out = [p for _, p in found[:MAX_GLOB]]
        text = "\n".join(out)
        if len(found) > MAX_GLOB:
            text += f"\n\n[Results are truncated: {len(found):,} files match, the first {MAX_GLOB} are shown. Use a more specific pattern or path.]"
        return _ok(text)

    def _grep(self, a: dict, ctx: AgentContext) -> dict:
        pat = a.get("pattern")
        if not isinstance(pat, str) or pat == "":
            return _err("pattern must be a regular expression")
        mode = a.get("output_mode", "files_with_matches")
        if mode not in ("content", "files_with_matches", "count"):
            return _err("output_mode must be content, files_with_matches or count")
        flags = (re.I if a.get("-i") is True else 0) | (re.M | re.S if a.get("multiline") is True else 0)
        try:
            rx = re.compile(pat, flags | (0 if a.get("multiline") is True else re.M))
        except re.error as e:
            return _err(f"The pattern is not a valid regular expression: {e}")
        if NESTED.search(pat):                                              # Python's re cannot be interrupted: a nested repetition can run for hours
            return _err("The pattern repeats a repeated group (like (a+)+), which can take far too long to match. Write it without the nested repetition.")
        ctx_n, e1 = _int(a.get("-C"), "-C", 0)
        after, e2 = _int(a.get("-A"), "-A", 0)
        before, e3 = _int(a.get("-B"), "-B", 0)
        head, e4 = _int(a.get("head_limit"), "head_limit", 0)
        skip, e5 = _int(a.get("offset"), "offset", 0)
        if e1 or e2 or e3 or e4 or e5:
            return _err(e1 or e2 or e3 or e4 or e5)
        after, before = (after if after is not None else ctx_n or 0), (before if before is not None else ctx_n or 0)
        raw = a.get("path")
        root = self._path(raw, ctx) if raw else ctx.policy.cwd
        if root is None or not os.path.exists(root):
            return _err(f"path does not exist: {raw or '(no project folder)'}")
        globs = [g for g in (a.get("glob"),) if isinstance(g, str) and g]
        kind = a.get("type")
        if isinstance(kind, str) and kind:
            if kind not in TYPES:
                return _err(f"type {kind!r} is not known (try py, js, ts, md, json, ...)")
            globs += list(TYPES[kind])
        files = [root] if os.path.isfile(root) else self._walk(root, ctx.cancel)
        deadline = time.monotonic() + GREP_SECONDS
        hits: list[tuple[float, str, list]] = []
        for p in files:
            if globs and not any(fnmatch.fnmatch(os.path.basename(p), g) or fnmatch.fnmatch(os.path.relpath(p, root).replace("\\", "/"), g) for g in globs):
                continue
            try:
                st = os.stat(p)
                if st.st_size > MAX_GREP_FILE or _is_binary(p):
                    continue
                text, _ = _read_text(p)
            except OSError:
                continue
            if time.monotonic() > deadline or ctx.cancel.is_set():
                break
            found = []
            if a.get("multiline") is True:
                for m in rx.finditer(text):
                    found.append(text.count("\n", 0, m.start()) + 1)
                    if time.monotonic() > deadline:
                        break
            else:
                lines = text.split("\n")
                for i, ln in enumerate(lines, 1):
                    if time.monotonic() > deadline:
                        break
                    if rx.search(ln):
                        found.append(i)
            if found:
                hits.append((st.st_mtime, p, found))
        if not hits:
            return _ok("No files found" if mode == "files_with_matches" else "No matches found")
        hits.sort(key=lambda x: -x[0])
        lim = MAX_GREP_LINES if head is None else head
        start = skip or 0
        if mode == "files_with_matches":
            rows = [p for _, p, _ in hits]
        elif mode == "count":
            rows = [f"{p}:{len(f)}" for _, p, f in hits]
        else:
            rows = []
            numbered = a.get("-n", True) is not False
            for _, p, found in hits:
                text, _ = _read_text(p)
                lines = text.split("\n")
                shown: set[int] = set()
                for ln in found:
                    for j in range(max(1, ln - before), min(len(lines), ln + after) + 1):
                        shown.add(j)
                last = None
                for j in sorted(shown):
                    if last is not None and j != last + 1:
                        rows.append("--")
                    sep = ":" if j in found else "-"
                    rows.append(f"{p}{sep}{j}{sep}{lines[j - 1]}" if numbered else f"{p}{sep}{lines[j - 1]}")
                    last = j
        total = len(rows)
        rows = rows[start:start + lim] if lim else rows[start:]
        out = "\n".join(rows)
        if start + len(rows) < total:
            out += f"\n\n[Showing {start + 1}-{start + len(rows)} of {total}; use offset to see more.]"
        return _ok(out)

    # -------------------------------------------------------------------------------------------- ExitPlanMode
    def _exit_plan(self, a: dict, ctx: AgentContext) -> dict:
        if not isinstance(a.get("plan"), str) or not a["plan"].strip():
            return _err("plan must be the text of your plan")
        ctx.policy.mode = None                                              # the user approved it (the gate asked): the default rules apply from now on
        ctx.emit({"event": "mode", "mode": "ask"})
        return _ok("The user has approved your plan. You can start now: changes and commands are allowed again (the usual questions still apply).")

    # -------------------------------------------------------------------------------------------- TodoWrite
    def _todo(self, a: dict, ctx: AgentContext) -> dict:
        todos = a.get("todos")
        if not isinstance(todos, list) or len(todos) > 100:
            return _err("todos must be a list of {content, status, activeForm}")
        clean = []
        for t in todos:
            if (not isinstance(t, dict) or not isinstance(t.get("content"), str) or not t["content"].strip() or t.get("status") not in TODO_STATES
                    or not isinstance(t.get("activeForm"), str) or not t["activeForm"].strip()):
                return _err("each todo needs content, activeForm and a status of pending, in_progress or completed")
            clean.append({"content": t["content"], "status": t["status"], "activeForm": t["activeForm"]})
        s = self.session(ctx.session)
        with s.lock:
            s.todos = clean
        ctx.emit({"event": "todos", "todos": clean})
        return _ok("Todos have been modified successfully. Keep using the list to track progress, and carry on with the current task.")
