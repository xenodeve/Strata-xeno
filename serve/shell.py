"""serve/shell.py - the Bash tool of the chat's coding tools (with BashOutput and KillShell), with Claude Code's parameters.

`command` runs in a shell (bash: Git Bash on Windows; PowerShell when there is no bash) in the chat's project folder, its output (both streams)
comes back with the exit code, and it is stopped when `timeout` (milliseconds, default 120000, at most 600000) runs out or the request is
cancelled - the whole process tree, not only the shell.  `run_in_background` starts it and returns an id; BashOutput reads what is new,
KillShell stops it.  The permission gate in serve/agent.py has already decided by the time a command gets here.

A call starts where the last one ended, when that is inside the project's folders (a `cd` carries over, as in Claude Code); a directory outside them is forgotten and the next call
starts in the project folder again.  Environment variables and shell state do not carry over.  The server's own settings (environment variables that start with STRATA_) are not
given to a command.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass

from serve import agent, permissions

DEFAULT_TIMEOUT_MS = 120_000
MAX_TIMEOUT_MS = 600_000
MAX_OUTPUT = 30_000
HEAD, TAIL = 10_000, 20_000
MAX_KEPT = 4_000_000                       # bytes of one command's output kept in memory
MAX_BACKGROUND = 8
WIN = os.name == "nt"


@dataclass
class Shell:
    kind: str                              # "bash" | "powershell" | "cmd"
    argv: list[str]                        # the program and what makes it run one command string
    path: str


def _git_bash_candidates() -> list[str]:
    out = []
    for env in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
        base = os.environ.get(env)
        if base:
            root = os.path.join(base, "Programs", "Git") if env == "LOCALAPPDATA" else os.path.join(base, "Git")
            out += [os.path.join(root, "bin", "bash.exe"), os.path.join(root, "usr", "bin", "bash.exe")]
    return out


def find_shell(prefer: str | None = None) -> Shell | None:
    """The shell commands run in: STRATA_SHELL if set, else bash (Git Bash on Windows - not the WSL launcher), else PowerShell."""
    override = os.environ.get("STRATA_SHELL")
    if override and os.path.isfile(override):
        name = os.path.basename(override).lower()
        kind = "powershell" if "pwsh" in name or "powershell" in name else "cmd" if name.startswith("cmd") else "bash"
        return _make(kind, override)
    bash = None
    if WIN:
        bash = next((p for p in _git_bash_candidates() if os.path.isfile(p)), None)
    else:
        bash = shutil.which("bash") or shutil.which("sh")
    ps = shutil.which("pwsh") or shutil.which("powershell")
    if prefer == "powershell":
        return _make("powershell", ps) if ps else (_make("bash", bash) if bash else None)
    if bash:
        return _make("bash", bash)
    return _make("powershell", ps) if ps else None


def _make(kind: str, path: str) -> Shell:
    if kind == "bash":
        return Shell("bash", [path, "-lc"] if WIN else [path, "-c"], path)
    if kind == "powershell":
        return Shell("powershell", [path, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command"], path)
    return Shell("cmd", [path, "/d", "/s", "/c"], path)


def describe(sh: Shell) -> str:
    return {"bash": "bash", "powershell": "PowerShell", "cmd": "cmd.exe"}[sh.kind]


def _env() -> dict:
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("STRATA_")}
    env.update(CHERE_INVOKING="1", TERM="dumb", PAGER="cat", GIT_PAGER="cat", GIT_TERMINAL_PROMPT="0")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def _kill_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if WIN:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, timeout=10)
        else:
            import signal
            os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        try:
            proc.kill()
        except OSError:
            pass


class Running:
    """One command that is running (or has run): its process and what it has written."""

    def __init__(self, command: str, sh: Shell, cwd: str):
        self.command = command
        self.started = time.monotonic()
        self.buf = bytearray()
        self.lock = threading.Lock()
        self.pos = 0                                                  # how much BashOutput has given out
        self.killed = False
        flags = {}
        if WIN:
            flags["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        else:
            flags["start_new_session"] = True
        self.proc = subprocess.Popen([*sh.argv, command], cwd=cwd, env=_env(), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **flags)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self) -> None:
        out = self.proc.stdout
        while True:
            chunk = out.read1(8192) if hasattr(out, "read1") else out.read(8192)
            if not chunk:
                break
            with self.lock:
                if len(self.buf) < MAX_KEPT:
                    self.buf += chunk

    def kill(self) -> None:
        self.killed = True
        _kill_tree(self.proc)

    def text(self, start: int = 0) -> str:
        with self.lock:
            return bytes(self.buf[start:]).decode("utf-8", errors="replace")

    def size(self) -> int:
        with self.lock:
            return len(self.buf)


def _cap(text: str) -> str:
    if len(text) <= MAX_OUTPUT:
        return text
    cut = len(text) - HEAD - TAIL
    return f"{text[:HEAD]}\n\n[... output truncated: {cut:,} characters in the middle are not shown ...]\n\n{text[-TAIL:]}"


# ------------------------------------------------------------------------------------------------ the tools
def install(server: agent.AgentServer, sh: Shell | None) -> None:
    """Add Bash, BashOutput and KillShell to the coding tools (nothing when there is no shell on this PC)."""
    if sh is None:
        return
    server.shell = sh
    name = describe(sh)
    server.add_tool(
        "Bash",
        f"Run a command in {name}, in the project folder, and return its output (both streams) and exit code. A call starts in the directory the last one ended in "
        "(cd carries over inside the project's folders; environment variables and shell state do not). Prefer Read, Edit, Glob and Grep to cat, sed, find and grep. Default timeout 120 seconds, at most 600 (timeout is in milliseconds); "
        "output over 30,000 characters is cut in the middle. For a long-running command (a server, a watcher) set run_in_background and read it later with BashOutput. "
        "The user is asked before a command runs, unless it is a plain read-only one.",
        {"command": {"type": "string", "description": f"the command, written for {name}"}, "timeout": {"type": "integer", "description": "milliseconds before it is stopped (max 600000)"},
         "description": {"type": "string", "description": "what the command does, in a few words"}, "run_in_background": {"type": "boolean", "description": "start it and return at once"}},
        ["command"], lambda a, ctx: _bash(server, sh, a, ctx))
    server.add_tool(
        "BashOutput", "Read what a background command has written since the last time you asked (and whether it is still running). filter is a regular expression: only matching lines are returned.",
        {"bash_id": {"type": "string", "description": "the id Bash returned"}, "filter": {"type": "string", "description": "only lines matching this regular expression"}},
        ["bash_id"], lambda a, ctx: _output(server, a, ctx))
    server.add_tool(
        "KillShell", "Stop a background command.", {"shell_id": {"type": "string", "description": "the id Bash returned"}}, ["shell_id"], lambda a, ctx: _kill(server, a, ctx))


def _bash(server: agent.AgentServer, sh: Shell, a: dict, ctx: agent.AgentContext) -> dict:
    command, tm, bg = a.get("command"), a.get("timeout"), a.get("run_in_background", False)
    if not isinstance(command, str) or not command.strip():
        return agent._err("command must be the text of a command")
    if tm is None:
        tm = DEFAULT_TIMEOUT_MS
    if isinstance(tm, bool) or not isinstance(tm, int) or not 1 <= tm <= MAX_TIMEOUT_MS:
        return agent._err(f"timeout must be a whole number of milliseconds from 1 to {MAX_TIMEOUT_MS}")
    if not isinstance(bg, bool):
        return agent._err("run_in_background must be true or false")
    cwd = ctx.policy.cwd
    if not cwd or not os.path.isdir(cwd):
        return agent._err("This chat has no project folder to run commands in. Choose one for the chat's project first.")
    s = server.session(ctx.session)
    if bg and sum(1 for b in s.shells.values() if b.proc.poll() is None) >= MAX_BACKGROUND:
        return agent._err(f"There are already {MAX_BACKGROUND} background commands running; stop one with KillShell first.")
    start = _start_dir(s, ctx.policy)
    mark = None
    code = command
    if not bg and sh.kind == "bash":                                  # the shell says where it ended (an EXIT trap, so an `exit` in the command counts too)
        fd, mark = tempfile.mkstemp(prefix="strata-cwd-")
        os.close(fd)
        code = f"trap 'pwd -W > \"{mark.replace(chr(92), '/')}\" 2>/dev/null || pwd -P > \"{mark.replace(chr(92), '/')}\"' EXIT\n{command}"
    try:
        run = Running(code, sh, start)
    except OSError as e:
        return agent._err(f"the shell could not be started: {e.strerror or e}")
    if bg:
        with s.lock:
            s.shell_count += 1
            bid = f"bash_{s.shell_count}"
            s.shells[bid] = run
        return agent._ok(f"Command running in background with ID: {bid}")
    deadline = time.monotonic() + tm / 1000
    why = ""
    while run.proc.poll() is None:
        if ctx.cancel.is_set():
            run.kill()
            why = "cancelled"
            break
        if time.monotonic() > deadline:
            run.kill()
            why = "timeout"
            break
        time.sleep(0.05)
    try:
        run.proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    run.reader.join(timeout=2)
    out = _cap(run.text()).rstrip()
    moved = _ended(s, ctx.policy, mark, start)
    if moved:
        out = (out + "\n\n" if out else "") + moved
    if why == "timeout":
        return agent._err((out + "\n\n" if out else "") + f"Command timed out after {tm / 1000:g} seconds and was stopped.")
    if why == "cancelled":
        return agent._err((out + "\n\n" if out else "") + "Command cancelled and stopped.")
    rc = run.proc.returncode
    if rc:
        return agent._err((out + "\n" if out else "") + f"Exit code {rc}")
    return agent._ok(out or "(no output)")


def _start_dir(s, policy: permissions.Policy) -> str:
    """Where a command starts: where the last one ended, if that is still a folder inside the project's; else the project folder."""
    d = s.cwd
    if d and os.path.isdir(d) and permissions.root_of(os.path.realpath(d), policy) is not None:
        return d
    s.cwd = None
    return policy.cwd


def _ended(s, policy: permissions.Policy, mark: str | None, start: str) -> str:
    """Notes where the command ended (read from the file its shell wrote) for the next one; a note for the model when that is not where it started. Always removes the file."""
    if not mark:
        return ""
    try:
        with open(mark, "rb") as f:
            text = f.read(4096).decode("utf-8", errors="replace").strip()
    except OSError:
        text = ""
    try:
        os.remove(mark)
    except OSError:
        pass
    if not text:
        return ""
    where = os.path.realpath(text)
    if not os.path.isdir(where):
        return ""
    if permissions.root_of(where, policy) is None:
        s.cwd = None
        return f"[Shell cwd was reset to {policy.cwd}: {where} is outside the project's folders]"
    s.cwd = where
    return f"[Shell cwd is now {where}]" if os.path.normcase(where) != os.path.normcase(os.path.realpath(start)) else ""


def _find(server: agent.AgentServer, bid, ctx: agent.AgentContext, what: str):
    s = server.session(ctx.session)
    run = s.shells.get(bid) if isinstance(bid, str) else None
    return run


def _output(server: agent.AgentServer, a: dict, ctx: agent.AgentContext) -> dict:
    bid = a.get("bash_id")
    run = _find(server, bid, ctx, "bash_id")
    if run is None:
        return agent._err(f"There is no background command with the id {bid!r} in this chat.")
    flt = a.get("filter")
    rx = None
    if flt:
        try:
            rx = re.compile(flt)
        except (re.error, TypeError):
            return agent._err("filter must be a valid regular expression")
    size = run.size()
    new = run.text(run.pos)
    run.pos = size
    if rx:
        new = "\n".join(ln for ln in new.splitlines() if rx.search(ln))
    rc = run.proc.poll()
    status = "running" if rc is None else "killed" if run.killed else f"completed, exit code {rc}"
    body = _cap(new).rstrip()
    return agent._ok((body + "\n\n" if body else "") + f"[{status}]")


def _kill(server: agent.AgentServer, a: dict, ctx: agent.AgentContext) -> dict:
    bid = a.get("shell_id")
    run = _find(server, bid, ctx, "shell_id")
    if run is None:
        return agent._err(f"There is no background command with the id {bid!r} in this chat.")
    run.kill()
    return agent._ok(f"Stopped {bid} ({run.command[:80]}).")
