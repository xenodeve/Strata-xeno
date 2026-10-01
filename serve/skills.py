"""serve/skills.py - the skills imported from the other coding apps, as an in-process "server" of the MCP hub (issue #94).

The model gets three tools from it, so the catalog is not in the prompt (a PC here has about 600 skills):
  find_skills(query)         the best matches by a few words, each with a one-line description
  use_skill(name)            the skill's SKILL.md, and the list of files bundled with it
  read_skill_file(name, path)  one of those files
`use_skill`'s description lists the skills' names and descriptions only when there are few; with many it says to search.

A skill's files are read through a fence: the path is relative, has no `..`, no hidden part, is not a key or a secret by its name, and
its real location (links and junctions resolved) is inside the skill's real folder; the file is text and under MAX_FILE bytes.
Nothing here writes anything or runs anything.
"""
from __future__ import annotations

import os
import re
import threading

MAX_FILE = 100_000          # bytes a file of a skill may be
MAX_SKILL = 60_000          # characters of SKILL.md given in one answer
MAX_RESULTS = 20
INLINE_MAX = 15             # up to this many skills are listed in use_skill's description
MAX_LISTED = 80             # files named after a skill's SKILL.md
BLOCKED_SUFFIX = (".pem", ".key", ".pfx", ".p12", ".jks", ".keystore")
BLOCKED_PREFIX = ("id_rsa", "id_ed25519", "id_ecdsa", "id_dsa")
SKIP_DIRS = {"node_modules", "__pycache__", "venv", ".venv"}


def _ok(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": False}


def _err(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": True}


def _inside(root: str, path: str) -> bool:
    try:
        return os.path.normcase(os.path.commonpath([root, path])) == os.path.normcase(root)
    except ValueError:                                       # another drive
        return False


def _blocked_name(name: str) -> bool:
    low = name.lower()
    return low.endswith(BLOCKED_SUFFIX) or low.startswith(BLOCKED_PREFIX)


class SkillsServer:
    """Looks like an McpServer to the hub (name, kind, status, tools, call, start, close) but has no process and no transport."""

    name = "skills"
    kind = "builtin"

    def __init__(self, skills: list[dict]):
        self.status = "ready"
        self.error = None
        self.info = {"name": "skills", "title": "Skills from your other coding apps"}
        self.transport = None
        self.lock = threading.Lock()
        self.last_start = 0.0
        self.tools: list[dict] = []
        self._skills: dict[str, dict] = {}
        self.set_skills(skills)

    # -------------------------------------------------------------------------------------------- the hub's side
    def start(self) -> bool:
        return True

    def close(self) -> None:
        return None

    def set_skills(self, skills: list[dict]) -> None:
        """The skills in use now (the switches changed): the tools follow, with no restart of anything."""
        self._skills = {s["name"]: s for s in skills}
        self.tools = self._make_tools() if self._skills else []

    def _make_tools(self) -> list[dict]:
        n = len(self._skills)
        if n <= INLINE_MAX:
            listing = " Available: " + "; ".join(f"{s['name']}: {(s['description'] or '')[:110]}" for s in self._skills.values()) + "."
        else:
            listing = f" There are too many to list here: call find_skills first."
        obj = lambda props, req: {"type": "object", "properties": props, "required": req}      # noqa: E731
        return [
            {"name": "find_skills", "description": f"Search the {n} skills available (imported from the user's other coding apps) by a few words of what you need. "
                                                  "Returns the best matches with a one-line description each; then call use_skill with a name.",
             "inputSchema": obj({"query": {"type": "string", "description": "a few words about the task"}}, ["query"])},
            {"name": "use_skill", "description": f"Load a skill: returns its instructions (SKILL.md) and the files bundled with it. {n} skills are available.{listing}",
             "inputSchema": obj({"name": {"type": "string", "description": "the skill's name"}}, ["name"])},
            {"name": "read_skill_file", "description": "Read a text file that is bundled with a skill (named by use_skill), by its path relative to the skill.",
             "inputSchema": obj({"name": {"type": "string"}, "path": {"type": "string", "description": "relative to the skill, e.g. references/forms.md"}}, ["name", "path"])},
        ]

    def call(self, tool: str, arguments: dict, timeout: float = 0, cancel=None) -> dict:
        if not isinstance(arguments, dict):
            return _err("the arguments must be an object")
        try:
            if tool == "find_skills":
                return self._find(arguments.get("query"))
            if tool == "use_skill":
                return self._use(arguments.get("name"))
            if tool == "read_skill_file":
                return self._read(arguments.get("name"), arguments.get("path"))
        except OSError as e:                                  # a folder that went away
            return _err(f"could not read it: {e.strerror or e}")
        return _err(f"there is no tool named {tool!r} in the skills")

    # -------------------------------------------------------------------------------------------- the tools
    def _find(self, query) -> dict:
        if not isinstance(query, str) or not query.strip():
            return _err("query must be a few words about what you need")
        words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 1]
        if not words:
            return _err("query must be a few words about what you need")
        scored = []
        for s in self._skills.values():
            name, desc = s["name"].lower(), (s["description"] or "").lower()
            score = sum((3 if w in name else 0) + (1 if w in desc else 0) for w in words) + (5 if name == query.strip().lower() else 0)
            if score:
                scored.append((-score, s["name"], s))
        if not scored:
            return _ok("No skill matches. Try other words, or work without one.")
        scored.sort(key=lambda x: (x[0], x[1]))
        lines = [f"{s['name']} - {(s['description'] or '')[:160]}" for _, _, s in scored[:MAX_RESULTS]]
        return _ok("\n".join(lines) + "\n\nCall use_skill with a name to load one.")

    def _skill(self, name):
        return self._skills.get(name) if isinstance(name, str) else None

    def _missing(self, name) -> dict:
        return _err(f"there is no skill named {name!r}; call find_skills to see which there are" if isinstance(name, str) else "name must be the skill's name")

    def _use(self, name) -> dict:
        s = self._skill(name)
        if s is None:
            return self._missing(name)
        text, err = self._file(s, "SKILL.md", MAX_SKILL)
        if err:
            return _err(err)
        files = self._listing(s)
        if files:
            text += "\n\n---\nFiles in this skill (read one with read_skill_file):\n" + "\n".join("- " + f for f in files)
        return _ok(text)

    def _read(self, name, path) -> dict:
        s = self._skill(name)
        if s is None:
            return self._missing(name)
        text, err = self._file(s, path, MAX_FILE)
        return _err(err) if err else _ok(text)

    # -------------------------------------------------------------------------------------------- the fence
    def _file(self, skill: dict, path, cap: int) -> tuple[str, str | None]:
        """(text, None) for a file of the skill, or ("", why not). `cap`: bytes allowed (a longer SKILL.md is cut, not refused)."""
        if not isinstance(path, str) or not path or len(path) > 300 or "\x00" in path:
            return "", "path must be a file of the skill, relative to it"
        p = path.replace("\\", "/")
        if p.startswith("/") or re.match(r"^[A-Za-z]:", p):
            return "", "path must be relative to the skill, not absolute"
        parts = [x for x in p.split("/") if x not in ("", ".")]
        if not parts or ".." in parts:
            return "", "path must stay inside the skill (no ..)"
        if any(x.startswith(".") for x in parts):
            return "", "hidden files and folders are not available"
        if _blocked_name(parts[-1]):
            return "", "that kind of file (a key) is not available"
        root = os.path.realpath(skill["dir"])
        real = os.path.realpath(os.path.join(skill["dir"], *parts))
        if not _inside(root, real):
            return "", "that path leads outside the skill's folder"
        if not os.path.isfile(real):
            return "", "there is no such file in the skill"
        size = os.path.getsize(real)
        with open(real, "rb") as f:
            head = f.read(8192)
            if b"\x00" in head:
                return "", "that is a binary file; only text files can be read"
            data = head + f.read(min(size, cap) - len(head)) if size > len(head) else head
        text = data.decode("utf-8", "replace").replace("\r\n", "\n")          # (a file written on Windows has \r\n; the model reads \n)
        if size > cap and parts == ["SKILL.md"]:
            return text + f"\n\n[... cut: the file is {size:,} bytes; only the first {cap:,} are shown]", None
        if size > cap:
            return "", f"that file is too large ({size:,} bytes; the limit is {cap:,})"
        return text, None

    def _listing(self, skill: dict) -> list[str]:
        """The files that sit beside SKILL.md and may be read: not hidden, not keys, and not behind a link that leads out."""
        root = os.path.realpath(skill["dir"])
        out: list[str] = []
        for dirpath, dirnames, filenames in os.walk(skill["dir"]):
            here = os.path.realpath(dirpath)
            depth = os.path.relpath(dirpath, skill["dir"]).count(os.sep) + (0 if dirpath == skill["dir"] else 1)
            if not _inside(root, here) or depth > 4:
                dirnames[:] = []
                continue
            dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d not in SKIP_DIRS)
            for f in sorted(filenames):
                rel = os.path.relpath(os.path.join(dirpath, f), skill["dir"]).replace(os.sep, "/")
                if rel == "SKILL.md" or f.startswith(".") or _blocked_name(f):
                    continue
                full = os.path.realpath(os.path.join(dirpath, f))
                try:
                    with open(full, "rb") as fh:
                        if b"\x00" in fh.read(2048) or os.path.getsize(full) > MAX_FILE:
                            continue
                except OSError:
                    continue
                if _inside(root, full):
                    out.append(rel)
                if len(out) >= MAX_LISTED:
                    return out
        return out
