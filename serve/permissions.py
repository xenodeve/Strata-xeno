"""serve/permissions.py - what the chat's coding tools may do without asking (the rules are modelled on Claude Code's).

The tools are the ones Claude Code has, with its names and parameters (Read, Write, Edit, Glob, Grep, Bash, ...); the rule syntax is
Claude Code's too: `Tool` or `Tool(specifier)`, where the specifier of a file tool is a path glob (`Read(src/**)`, `Edit(/abs/dir/**)`)
and the specifier of Bash is a command or a prefix (`Bash(git commit:*)`).  Deny rules win over allow rules, which win over the defaults.

The defaults are the developer's choice for this app:
  * inside the chat's project folder, files are free to read and to change (not secrets, not `.git/` for writing);
  * outside it, anything asks (so does every file when the chat has no folder);
  * a command (Bash) asks every time, except a plain read-only one that names nothing outside the folder;
  * mode "plan" changes nothing (Write, Edit and any command that is not read-only are refused);
  * mode "auto" does not change this function's answer: it marks what a second judgement may settle (`judgeable`) and what never may
    (a dangerous command, a secret, the .git folder) - the caller asks the judge, and the user when the judge is unsure.
  * mode "yolo" (#329, the developer's choice for a model hosted on this PC) asks nothing: deny rules, a dangerous command and a secret
    leaving this PC (`leaks`, and the caller's check of a `git push`, `pushes` / `secret_paths`) are refused instead.  The leak checks
    are patterns: they cannot see every way out (a program the model writes and then runs can still send what it reads).
An unknown mode is the default one.

This module only decides; it reads no file and runs nothing (it looks at where paths lead, with links resolved, and nothing more).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit

FILE_READ = ("Read", "Glob", "Grep")
FILE_EDIT = ("Write", "Edit", "NotebookEdit")
WEB = ("WebFetch", "WebSearch")                                  # they send something off this PC: every call asks, unless the user wrote a rule for it
KNOWN = FILE_READ + FILE_EDIT + WEB + ("Bash", "TodoWrite", "BashOutput", "KillShell", "ExitPlanMode", "AskUserQuestion", "Task")
FREE = ("TodoWrite", "BashOutput", "KillShell", "AskUserQuestion", "Task")      # they only act on what the user already allowed in this chat, or only ask the user something


@dataclass
class Policy:
    cwd: str | None                          # the chat's project folder (None: it has none): where commands run and relative paths start
    mode: str | None = None                  # "plan" | "auto" | anything else is the default
    allow: list[str] = field(default_factory=list)
    deny: list[str] = field(default_factory=list)
    dirs: list[str] = field(default_factory=list)   # the project's other folders (Claude Code's additional directories, e.g. other worktrees): files in them are as free as in `cwd`
    protected: list[str] = field(default_factory=list)   # the server's own files (the run config, the --mcp-config file): they switch hooks, web access and helpers on and hold keys, so reading or changing them asks every time


@dataclass
class Decision:
    kind: str                                # "allow" | "ask" | "deny"
    why: str = ""
    danger: bool = False                     # a command that can do harm that is hard to undo: never settled by the judge
    judgeable: bool = False                  # in mode auto, a second judgement may settle this (only when it asks)
    rule: str | None = None                  # the rule that decided it, for the card


# ------------------------------------------------------------------------------------------------ paths
def _norm(p: str) -> str:
    return os.path.normcase(os.path.normpath(p))


def real(path: str, cwd: str | None) -> str | None:
    """Where a path leads: relative ones from the folder, `~` expanded, links and junctions resolved (also in the part that does not
    exist yet).  None when it cannot be told (no folder for a relative path, not text)."""
    if not isinstance(path, str) or not path.strip() or "\0" in path:
        return None
    p = os.path.expanduser(path)
    if not os.path.isabs(p):
        if not cwd:
            return None
        p = os.path.join(cwd, p)
    return os.path.realpath(p)


def inside(path: str, root: str) -> bool:
    """Whether `path` is `root` or below it (both already resolved)."""
    a, b = _norm(path), _norm(root)
    try:
        return os.path.commonpath([a, b]) == b
    except ValueError:                                                        # another drive
        return False


SECRET_FILES = re.compile(r"^(\.env(\..*)?|.*\.(pem|key|pfx|p12|jks|keystore)|id_(rsa|ed25519|ecdsa|dsa)(\..*)?|credentials(\..*)?|\.npmrc|\.pypirc|\.netrc|"
                          r"secrets?(\..*)?|\.git-credentials|service[-_]account.*\.json)$", re.I)
SECRET_DIRS = {".ssh", ".aws", ".gnupg", ".kube", ".docker"}
ENV_TEMPLATE = re.compile(r"^\.env\.(example|sample|template|dist)$", re.I)


def is_secret(path: str) -> bool:
    parts = [x for x in re.split(r"[\\/]+", path) if x]
    if not parts:
        return False
    if any(x.lower() in SECRET_DIRS for x in parts[:-1]) or parts[-1].lower() in SECRET_DIRS:
        return True
    last = parts[-1]
    return bool(SECRET_FILES.match(last)) and not ENV_TEMPLATE.match(last)


def root_of(path: str, pol: "Policy") -> str | None:
    """The folder of the project that holds `path` (both already resolved), or None when it is outside all of them. Other folders count only with a main one."""
    for root in ([pol.cwd, *pol.dirs] if pol.cwd else []):
        if inside(path, root):
            return root
    return None


def in_git(path: str, root: str) -> bool:
    rel = os.path.relpath(path, root) if inside(path, root) else ""
    return rel.split(os.sep)[0].lower() == ".git" if rel else False


# ------------------------------------------------------------------------------------------------ the rules (Claude Code's syntax)
def _rule_parts(rule: str) -> tuple[str, str | None]:
    m = re.fullmatch(r"\s*([A-Za-z_][\w]*)\s*(?:\((.*)\))?\s*", rule or "", re.S)
    return (m.group(1), m.group(2)) if m else ("", None)


def _glob_re(pattern: str) -> re.Pattern:
    out, i = [], 0
    while i < len(pattern):
        c = pattern[i]
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif c == "*":
            out.append("[^/]*")
            i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(c))
            i += 1
    return re.compile("".join(out) + r"\Z", re.I if os.name == "nt" else 0)


def _path_rule_matches(spec: str, path: str, cwd: str | None) -> bool:
    spec = spec.strip()
    if spec.startswith("~"):
        spec = os.path.expanduser(spec)
    spec = spec.replace("\\", "/")
    if not (spec.startswith("/") or re.match(r"^[A-Za-z]:/", spec)):
        if not cwd:
            return False
        spec = cwd.replace("\\", "/").rstrip("/") + "/" + spec
    target = path.replace("\\", "/")
    base = spec[:-3] if spec.endswith("/**") else None
    if base is not None and (target == base or (os.name == "nt" and target.lower() == base.lower())):
        return True
    return bool(_glob_re(spec).match(target))


def _web_rule_matches(spec: str | None, tool: str, args: dict) -> bool:
    """A rule for WebFetch or WebSearch: the tool alone, or WebFetch(domain:example.com) for that site and its sub-domains."""
    if spec is None:
        return True
    m = re.fullmatch(r"\s*domain:\s*([A-Za-z0-9.-]+)\s*", spec)
    if tool != "WebFetch" or not m:
        return False
    url = args.get("url")
    host = (urlsplit(url).hostname or "").lower().rstrip(".") if isinstance(url, str) else ""
    want = m.group(1).lower().rstrip(".")
    return bool(host) and (host == want or host.endswith("." + want))


def _family(tool: str) -> str:
    return "Read" if tool in FILE_READ else "Edit" if tool in FILE_EDIT else tool


def _tool_matches(rule_tool: str, tool: str) -> bool:
    return rule_tool == tool or (rule_tool in ("Read", "Edit") and _family(tool) == rule_tool)


# ------------------------------------------------------------------------------------------------ commands
OPERATORS = ("&&", "||", ";", "|", "&", "\n")


@dataclass
class Parsed:
    segments: list[list[str]]                # the commands, split at ; && || | & and newlines, each as words
    tainted: bool = False                    # a substitution, a redirection or something that hides what runs: not covered by a rule
    writes: bool = False                     # a redirection to a file


def parse(command: str) -> Parsed:
    """A small shell reader: words (quotes respected), the commands between operators, and whether anything in it hides what runs."""
    segs: list[list[str]] = [[]]
    word, has, i, q = "", False, 0, ""
    tainted = writes = False
    n = len(command)

    def end_word():
        nonlocal word, has
        if has:
            segs[-1].append(word)
        word, has = "", False

    while i < n:
        c = command[i]
        if q == "'":
            if c == "'":
                q = ""
            else:
                word += c
            i += 1
            continue
        if q == '"':
            if c == '"':
                q = ""
            elif c == "\\" and i + 1 < n and command[i + 1] in '"\\$`':
                word += command[i + 1]
                i += 1
            elif c == "`" or command.startswith("$(", i) or c == "$":
                tainted = True
                word += c
            else:
                word += c
            i += 1
            continue
        if c in "'\"":
            q, has = c, True
            i += 1
            continue
        if c == "\\" and i + 1 < n:
            word += command[i + 1]
            has = True
            i += 2
            continue
        if c == "`" or command.startswith("$(", i) or command.startswith("<(", i) or command.startswith(">(", i) or c == "$":
            tainted = True
            word += c
            has = True
            i += 1
            continue
        op = next((o for o in OPERATORS if command.startswith(o, i)), None)
        if op:
            end_word()
            if segs[-1]:
                segs.append([])
            i += len(op)
            continue
        if c == ">":
            tainted = writes = True
            word += c
            has = True
            i += 1
            continue
        if c in " \t\r":
            end_word()
            i += 1
            continue
        word += c
        has = True
        i += 1
    end_word()
    if q:
        tainted = True                                                         # an unclosed quote: not what it looks like
    return Parsed([s for s in segs if s], tainted, writes)


def _inner_commands(command: str) -> list[list[str]]:
    """What runs inside `$( )` and backticks, read as commands too (for deny rules, which err on the side of matching)."""
    out = []
    for m in re.finditer(r"\$\(([^()]*)\)|`([^`]*)`", command):
        out += parse(m.group(1) or m.group(2) or "").segments
    return out


def _segment_matches(spec: str, words: list[str]) -> bool:
    line = " ".join(words)
    spec = spec.strip()
    if spec.endswith(":*"):
        pre = spec[:-2].strip()
        return line == pre or line.startswith(pre + " ")
    return line == spec or (spec.endswith("*") and line.startswith(spec[:-1]))


READONLY = {"ls", "pwd", "cat", "head", "tail", "wc", "echo", "which", "where", "grep", "rg", "find", "diff", "cmp", "stat", "du", "df", "tree", "file", "sort", "uniq",
            "cut", "basename", "dirname", "realpath", "readlink", "date", "whoami", "hostname", "uname", "md5sum", "sha1sum", "sha256sum", "git", "nl", "tac", "rev",
            "column", "tr", "printf", "true", "id"}
NO_PATH_ARGS = {"pwd", "echo", "which", "where", "date", "whoami", "hostname", "uname", "printf", "true", "id"}
GIT_READONLY = {"status", "diff", "log", "show", "branch", "rev-parse", "blame", "ls-files", "remote", "describe", "shortlog", "grep", "tag", "stash list", "config --get"}
GIT_BAD_FLAGS = ("--output", "-o", "--ext-diff", "--textconv", "-c", "--exec", "--upload-pack", "--receive-pack", "-d", "-D", "-m", "-M", "-C", "--delete", "--set-upstream",
                 "-u", "--unset-upstream", "--edit-description", "-f", "--force", "add", "set-url", "rm", "rename", "prune")
FIND_BAD = ("-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fprintf", "-fls", "-fprint0")
SORT_BAD = ("-o", "--output")
VERSION_ONLY = {"--version", "-V", "-v", "version"}


def _readonly_segment(words: list[str], pol: Policy) -> tuple[bool, str]:
    """Whether one command is a plain read-only one, and if not why."""
    prog = os.path.basename(words[0]).lower()
    args = words[1:]
    if len(words) == 2 and words[1] in VERSION_ONLY and re.fullmatch(r"[\w.+-]+", prog):
        return True, ""
    if prog not in READONLY or words[0] != os.path.basename(words[0]):
        return False, "it is not a plain read-only command"
    if any("\\" in w or "$" in w or "`" in w for w in words):
        return False, "it uses something the shell would expand"
    if prog == "git":
        sub = next((a for a in args if not a.startswith("-")), "")
        flags = [a for a in args if a.startswith("-")]
        first = [a for a in args if not a.startswith("-")][:2]
        sub2 = " ".join(first[:2])
        if not (sub in GIT_READONLY or sub2 in GIT_READONLY) or any(f.split("=")[0] in GIT_BAD_FLAGS for f in flags):
            return False, "this git command can change something"
        if sub in ("branch", "tag", "remote") and len(first) > 1 and not set(flags) <= {"-a", "-v", "-vv", "-r", "-l", "--list", "--all", "--verbose"}:
            return False, "this git command can change something"
        if sub in ("branch", "tag") and len(first) > 1:
            return False, "this git command can change something"
        if sub == "remote" and len(first) > 1 and first[1] not in ("-v", "show", "get-url"):
            return False, "this git command can change something"
        args = [a for a in args if a != sub]
    if prog == "find" and any(a in FIND_BAD for a in args):
        return False, "find would delete or run something"
    if prog == "sort" and any(a.split("=")[0] in SORT_BAD for a in args):
        return False, "sort would write a file"
    if prog in NO_PATH_ARGS:
        return True, ""
    paths = [a for a in args if not a.startswith("-")]
    if not pol.cwd:
        return (False, "it names no project folder to read from") if not paths else (False, "this chat has no project folder")
    for a in paths or ["."]:
        r = real(a, pol.cwd)
        if r is None or root_of(r, pol) is None:
            return False, "it reads outside the project folder"
        if is_secret(r):
            return False, "it reads a file that looks like a secret"
    return True, ""


def command_is_readonly(command: str, pol: Policy) -> tuple[bool, str]:
    if "\\" in command:                                                       # the reader would take a Windows path's backslashes for escapes and lose where it leads
        return False, "it uses a backslash, so where its paths lead cannot be told"
    p = parse(command)
    if not p.segments:
        return False, "there is no command"
    if p.tainted:
        return False, "it writes to a file or hides what runs (a substitution, a redirection)"
    for words in p.segments:
        ok, why = _readonly_segment(words, pol)
        if not ok:
            return False, why
    return True, ""


DANGER = [re.compile(x, re.I) for x in (
    r"\brm\s+(-[a-z]*\s+)*-[a-z]*[rf][a-z]*\s+(-[a-z]*\s+)*(/|~|\*|\$home|\.\.?|[a-z]:\\?)(\s|$)",
    r"\brm\s+(-[a-z]*\s+)*-[a-z]*r[a-z]*\s+(-[a-z]*\s+)*(/|~|\*|\$home)(\s|$)",
    r"\bsudo\b", r"\bsu\s+-?\s*\w*\s*$", r"\bdoas\b",
    r"\b(curl|wget|iwr|invoke-webrequest|irm)\b[^|;&]*\|\s*(sudo\s+)?(sh|bash|zsh|python\d?|node|perl|ruby|iex|invoke-expression|pwsh|powershell)\b",
    r"\bgit\s+push\b[^;&|]*\s(-f\b|--force\b|--force-with-lease\b|--mirror\b)", r"\bgit\s+reset\s+--hard\b", r"\bgit\s+clean\s+-[a-z]*f", r"\bgit\s+checkout\s+(--\s+)?\.\s*$",
    r"\bdd\s+[^;&|]*\bof=", r"\bmkfs(\.\w+)?\b", r"\bchmod\s+-r\s+[0-7]*7\b[^;&|]*\s/", r"\bchown\s+-r\b[^;&|]*\s/(\s|$)",
    r":\(\)\s*\{", r"\bformat\s+[a-z]:", r"\bdel\s+(/[sfq]\s+)+", r"\brd\s+/s\b", r"\bremove-item\b[^;&|]*-recurse[^;&|]*-force|\bremove-item\b[^;&|]*-force[^;&|]*-recurse",
    r"\b(shutdown|reboot|halt|poweroff)\b", r">\s*/dev/(sd|nvme|hd)", r"\bdiskpart\b", r"\bnet\s+user\b", r"\breg\s+(delete|add)\b", r"\bcrontab\s+-r\b",
    r"\bkill\s+-9\s+(-1|1)\b", r"\bkillall\b", r"\bpkill\s+-9\b", r"\btruncate\s+-s\s*0\s+/",
)]


def is_dangerous(command: str) -> bool:
    return any(rx.search(command) for rx in DANGER)


# ------------------------------------------------------------------------------------------------ yolo: what leaves this PC (#329)
SENDERS = re.compile(r"(^|[\s;&|(`/\\\"'])(curl|wget|scp|sftp|rsync|ssh|nc|ncat|netcat|socat|ftp|tftp|telnet|iwr|irm|invoke-webrequest|invoke-restmethod|"
                     r"start-bitstransfer|bitsadmin|certutil|send-mailmessage|sendmail|mail|rclone|gsutil|http|https|xh|"
                     r"gh\s+(gist|api|issue|pr|release|secret)|aws\s+s3|az\s+storage|git\s+(push|send-email))(\.exe)?(?=[\s\"']|$)", re.I)
SECRET_WORDS = re.compile(r"(^|[\s/\\=@'\"<:`(])(\.env(\.(?!example\b|sample\b|template\b|dist\b)\w+)?|[\w.-]*\.(pem|key|pfx|p12|jks|keystore)|id_(rsa|ed25519|ecdsa|dsa)(\.pub)?|"
                          r"\.ssh|\.aws|\.gnupg|\.kube|\.docker|\.npmrc|\.pypirc|\.netrc|\.git-credentials|credentials(\.\w+)?|secrets?(\.\w+)?|service[-_]account[\w.-]*\.json)"
                          r"(?=$|[\s/\\'\";|&)>`])", re.I)
SECRET_VARS = re.compile(r"\$(\{)?(env:)?\w*(key|token|secret|passw(or)?d|credential)\w*", re.I)
# a key or certificate a sender USES to log in or to check the server is not sent: `ssh -i ~/.ssh/id_rsa`, `curl --cert c.pem --key c.key`
KEY_USE = re.compile(r"(?<=\s)(-i|-E|--cert|--key|--cacert|--capath|--proxy-cert|--proxy-key|--identity-file)(\s+|=)\S+|-o\s*IdentityFile=\S+", re.I)
ENV_DUMP = re.compile(r"(^|[\s;&|(])(env|printenv|set|export\s+-p|declare\s+-p|get-childitem\s+env:|gci\s+env:|dir\s+env:)(?=\s|$|\|)", re.I)
URLS = re.compile(r"\w+://\S+")                                 # an address's path is not a file of this PC (`.../api/secrets`)
OUTPUT_ARG = re.compile(r"(?<=\s)(-o|-O|--output|--output-document|-OutFile)(\s+|=)\S+")     # where a download is written: it comes in, not out
TOKENS = re.compile(r"(?<![A-Za-z0-9])(gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-(ant-|proj-)?[A-Za-z0-9_-]{20,}|(AKIA|ASIA)[0-9A-Z]{16}|"
                    r"xox[abposr]-[A-Za-z0-9-]{10,}|hf_[A-Za-z0-9]{30,}|AIza[0-9A-Za-z_-]{35}|glpat-[A-Za-z0-9_-]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)")


def leaks(tool: str, args: dict) -> str:
    """Why this call would send a secret off this PC ("" when it would not seem to): a command that sends data out and names a secret file,
    a variable named like a key or the whole environment, or a token-shaped string; a WebFetch URL or WebSearch query holding a token."""
    if tool == "Bash":
        c = args.get("command")
        if not isinstance(c, str) or not SENDERS.search(c):
            return ""
        c = OUTPUT_ARG.sub(" ", KEY_USE.sub(" ", c))
        c = re.sub(r"(\bgh\s+api\s+)\S+", r"\1", c, flags=re.I)     # `gh api <endpoint>`: a path of the GitHub API, not a file of this PC
        for part in re.split(r"&&|\|\||;|\n", c):         # one command at a time (`git push && echo $TOKEN_COUNT`); a pipe stays one: it carries data
            if not SENDERS.search(part):
                continue
            if TOKENS.search(part):
                return "the command sends something that looks like a token off this PC"
            if SECRET_WORDS.search(URLS.sub(" ", part)):
                return "the command sends data off this PC and names a secret file (.env, a key, credentials)"
            if SECRET_VARS.search(part) or ENV_DUMP.search(part):
                return "the command sends data off this PC and uses a secret variable (a key, a token, a password) or the whole environment"
        return ""
    if tool == "WebFetch" and isinstance(args.get("url"), str) and TOKENS.search(args["url"]):
        return "the address holds something that looks like a token: it would leave this PC"
    if tool == "WebSearch" and isinstance(args.get("query"), str) and TOKENS.search(args["query"]):
        return "the search words hold something that looks like a token: they would leave this PC"
    return ""


def _all_segments(command: str) -> list[list[str]]:
    """The commands of a line, with what runs in `$( )`, backticks and a shell's `-c` / `-Command` body, one level deep."""
    segs = parse(command).segments + _inner_commands(command)
    for w in list(segs):
        for i, x in enumerate(w[:-1]):
            if x.lower() in ("-c", "-command", "/c", "/k"):
                segs += parse(w[i + 1]).segments
    return segs


def _git_push_words(w: list[str]) -> list[str] | None:
    """The words after `push` when this command is `git [options] push ...` (git named by any path), else None."""
    if not w or re.split(r"[\\/]", w[0])[-1].lower() not in ("git", "git.exe") or "push" not in w[1:]:
        return None
    k = w.index("push")
    if not all(x.startswith("-") or w[i - 1] in ("-C", "-c") for i, x in enumerate(w[1:k], 1)):
        return None
    return w[:k] + ["push"] + w[k + 1:]


def pushes(command: str) -> bool:
    """Whether a command runs `git push` (git by any path, with or without its own options, also inside `$( )` or `bash -c`), so the caller
    checks what it would send."""
    return isinstance(command, str) and any(_git_push_words(w) is not None for w in _all_segments(command))


def push_remotes(command: str) -> list[str | None] | None:
    """The remotes the pushes of a command name (None: the default one), or None when what they send cannot be checked from the chat's
    folder: a push to an address or a path rather than a named remote, `git -C`, or a `cd` / `pushd` / `Set-Location` before it."""
    segs = _all_segments(command)
    if any(w and w[0].lower() in ("cd", "pushd", "set-location", "sl", "chdir") for w in segs):
        return None
    out: list[str | None] = []
    for w in segs:
        words = _git_push_words(w)
        if words is None:
            continue
        k = words.index("push")
        if "-C" in words[1:k] or any(x.startswith("--repo") for x in words[k + 1:]):
            return None
        rest, skip = [], False
        for x in words[k + 1:]:
            if skip:
                skip = False
            elif x in ("-o", "--push-option", "--receive-pack", "--exec"):
                skip = True
            elif not x.startswith("-"):
                rest.append(x)
        if rest and ("://" in rest[0] or "@" in rest[0] or "/" in rest[0] or "\\" in rest[0] or rest[0].endswith(".git") or rest[0] in (".", "..")):
            return None
        out.append(rest[0] if rest else None)
    return out


def secret_paths(paths: list[str]) -> list[str]:
    """The secret files (as `is_secret` tells them) in a list of paths, e.g. the files of the commits a push would send."""
    return [p for p in paths if isinstance(p, str) and p.strip() and is_secret(p)]


# ------------------------------------------------------------------------------------------------ the decision
MULTI = {"git", "npm", "pnpm", "yarn", "docker", "kubectl", "cargo", "go", "gh", "pip", "uv", "dotnet", "make"}


def rule_for(tool: str, args: dict) -> str | None:
    """The rule the page may remember when the user chooses "allow for this chat" (None when there is no sensible one)."""
    if tool == "WebFetch":
        url = args.get("url")
        host = (urlsplit(url).hostname or "").lower().rstrip(".") if isinstance(url, str) else ""
        return f"WebFetch(domain:{host})" if re.fullmatch(r"[a-z0-9.-]+", host) and "." in host else None      # a search has no sensible rule: it asks every time
    if tool == "Bash":
        c = args.get("command")
        p = parse(c) if isinstance(c, str) else None
        if not p or p.tainted or len(p.segments) != 1:
            return None
        w = p.segments[0]
        pre = w[0] if len(w) == 1 or w[0] not in MULTI or w[1].startswith("-") else f"{w[0]} {w[1]}"
        return f"Bash({pre}:*)"
    if tool in KNOWN and tool not in FREE and tool != "ExitPlanMode":
        raw = args.get("path") if tool in ("Glob", "Grep") else args.get("notebook_path") if tool == "NotebookEdit" else args.get("file_path")
        if isinstance(raw, str) and raw.strip():
            r = real(raw, None) or (os.path.realpath(os.path.expanduser(raw)) if os.path.isabs(os.path.expanduser(raw)) else None)
            if r:
                d = os.path.dirname(r) if tool not in ("Glob", "Grep") else r
                return f"{'Read' if tool in FILE_READ else tool}({d.replace(chr(92), '/').rstrip('/')}/**)"
    return None


def _file_target(tool: str, args: dict, pol: Policy) -> tuple[str | None, str]:
    """(where the call goes, why not when it cannot be told)."""
    if tool in ("Glob", "Grep"):
        raw = args.get("path")
        if raw is None or raw == "":
            if not pol.cwd:
                return None, "this chat has no project folder"
            raw = pol.cwd
        if tool == "Glob":
            pat = args.get("pattern")
            if isinstance(pat, str) and (os.path.isabs(pat) or re.match(r"^[A-Za-z]:", pat) or ".." in re.split(r"[\\/]", pat)):
                head = re.split(r"[*?\[{]", pat)[0]
                raw = os.path.join(raw if isinstance(raw, str) else "", head) if not os.path.isabs(pat) else head
    else:
        raw = args.get("notebook_path") if tool == "NotebookEdit" else args.get("file_path")
    if not isinstance(raw, str) or not raw.strip():
        return None, "the call does not say which path"
    r = real(raw, pol.cwd)
    if r is None:
        return None, "this chat has no project folder" if not pol.cwd else "the path cannot be read"
    return r, ""


def decide(tool: str, args: dict, pol: Policy) -> Decision:
    args = args if isinstance(args, dict) else {}
    # 1. deny rules
    for rule in pol.deny:
        rt, spec = _rule_parts(rule)
        if not _tool_matches(rt, tool):
            continue
        if spec is None:
            return Decision("deny", f"denied by the rule {rule}", rule=rule)
        if tool == "Bash":
            c = args.get("command")
            if isinstance(c, str) and any(_segment_matches(spec, w) for w in parse(c).segments + _inner_commands(c)):
                return Decision("deny", f"denied by the rule {rule}", rule=rule)
        elif tool in WEB:
            if _web_rule_matches(spec, tool, args):
                return Decision("deny", f"denied by the rule {rule}", rule=rule)
        else:
            t, _ = _file_target(tool, args, pol)
            if t and _path_rule_matches(spec, t, pol.cwd):
                return Decision("deny", f"denied by the rule {rule}", rule=rule)
    if pol.mode == "yolo":
        return _yolo(tool, args)
    plan = pol.mode == "plan"
    auto = pol.mode == "auto"

    def ask(why, *, danger=False, judge=True):
        return Decision("ask", why, danger=danger, judgeable=auto and judge and not danger)

    if tool == "ExitPlanMode":                                     # the plan is the user's to approve; there is nothing to leave outside plan mode
        return ask("approve the plan to leave plan mode", judge=False) if plan else Decision("deny", "not in plan mode: there is no plan to approve")
    if tool in FREE:
        return Decision("allow", "")
    if tool not in KNOWN:
        return ask(f"{tool} is not a tool this chat knows", judge=False)

    # 2. the web: it leaves this PC, so it asks every time (what the user wrote a rule for is theirs to have allowed); no judge settles it, and plan mode does not change that
    if tool in WEB:
        for rule in pol.allow:
            rt, spec = _rule_parts(rule)
            if rt == tool and _web_rule_matches(spec, tool, args):
                return Decision("allow", f"allowed by the rule {rule}", rule=rule)
        url = args.get("url")
        host = (urlsplit(url).hostname or "") if tool == "WebFetch" and isinstance(url, str) else ""
        return ask(f"it fetches a page from {host or 'the internet'}: the address leaves this PC" if tool == "WebFetch" else "it sends your search words to a search engine: they leave this PC", judge=False)

    # 3. a command
    if tool == "Bash":
        c = args.get("command")
        if not isinstance(c, str) or not c.strip():
            return ask("the call has no command", judge=False)
        danger = is_dangerous(c)
        p = parse(c)
        readonly, why_not = command_is_readonly(c, pol)
        if not danger and not p.tainted and p.segments:
            for rule in pol.allow:
                rt, spec = _rule_parts(rule)
                if rt == "Bash" and (spec is None or all(any(_rule_parts(r2)[0] == "Bash" and (_rule_parts(r2)[1] is None or _segment_matches(_rule_parts(r2)[1], w)) for r2 in pol.allow) for w in p.segments)):
                    return Decision("allow", f"allowed by the rule {rule}", rule=rule)
        if readonly and not danger:
            return Decision("allow", "a plain read-only command")
        if plan:
            return Decision("deny", "plan mode: nothing is changed, so no command that is not read-only runs")
        secret = bool(re.search(r"(^|[\s/\\=])(\.env(\.\w+)?|id_rsa|id_ed25519|\.ssh|\.aws|\.npmrc|\.netrc|credentials(\.json)?)(\s|$|/|\\)", c)) and "reads a file that looks like a secret" in why_not
        if danger:
            return ask("this command can do harm that is hard to undo", danger=True)
        return ask(why_not or "a command asks every time", judge=not secret)

    # 4. a file tool
    target, why = _file_target(tool, args, pol)
    if target is None:
        return ask(why, judge=False)
    if any(_norm(target) == _norm(real(p, None) or p) for p in pol.protected):   # before any rule: nothing settles these
        if plan and tool in FILE_EDIT:
            return Decision("deny", "plan mode: nothing is changed")
        return ask("it is the server's own config (hooks, web access, keys)", judge=False)
    secret = is_secret(target)
    root_here = root_of(target, pol)
    in_dot_git = tool in FILE_EDIT and root_here is not None and in_git(target, root_here)
    for rule in pol.allow:                                         # a rule, however it was written (even an "always" one), never settles a secret or a change in .git: those ask every time
        rt, spec = _rule_parts(rule)
        if _tool_matches(rt, tool) and not secret and not in_dot_git and (spec is None or _path_rule_matches(spec, target, pol.cwd)):
            if plan and tool in FILE_EDIT:
                break
            return Decision("allow", f"allowed by the rule {rule}", rule=rule)
    if plan and tool in FILE_EDIT:
        return Decision("deny", "plan mode: nothing is changed")
    if secret:
        return ask("it looks like a secret (a key, a token, an .env file)", judge=False)
    if not pol.cwd:
        return ask("this chat has no project folder")
    root = root_of(target, pol)
    if root is None:
        return ask("it is outside the project folder")
    if tool in FILE_EDIT and in_git(target, root):
        return ask("a change in .git can run programs (hooks) later", judge=False)
    return Decision("allow", "inside the project folder")


def mode_for(raw, *, guest: bool = False, on_runner: bool = False) -> str | None:
    """The mode a request asked for, as the run will use it: "plan", "auto", "yolo" or None (the default; anything unknown is that).
    #329: a team guest's YOLO runs only on the guest's own PC (the My PC runner); on this PC it is the default mode."""
    if raw == "yolo":
        return "yolo" if (not guest or on_runner) else None
    return raw if raw in ("auto", "plan") else None


def _yolo(tool: str, args: dict) -> Decision:
    """#329: after the deny rules, nothing asks; what is refused is refused with its reason, for the model to read."""
    if tool == "ExitPlanMode":
        return Decision("deny", "not in plan mode: there is no plan to approve")
    if tool == "Bash":
        c = args.get("command")
        if not isinstance(c, str) or not c.strip():
            return Decision("deny", "the call has no command")
        if is_dangerous(c):
            return Decision("deny", "YOLO mode refuses a command that can do harm that is hard to undo", danger=True)
    why = leaks(tool, args)
    if why:
        return Decision("deny", f"YOLO mode never lets a secret leave this PC: {why}")
    return Decision("allow", "YOLO mode")
