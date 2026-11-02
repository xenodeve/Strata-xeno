"""serve/sandbox/search.py - finding files and text in the sandbox's copy (the design's L1, 4.9), with the names Serena uses so that what a model knows of it carries over.

It looks at what `Workspace.listing` says the copy holds, so what the snapshot leaves out stays out, and a file the model changed a moment ago is seen as it is now. Text is kept in memory and read again when a
file's size or mtime moves. The answers are structures: the caller turns them into text and spends the budget of `output.py` on them.

`re` has no time limit and holds the GIL while it runs, so a pattern that can backtrack goes to `grep_worker.py`, a child with a kill timer. The check here catches only the common form of it (a repeat inside a repeat).
This belongs in the box, as 4.9 says, where the box's time limit ends it; in the server's own process it would stall every request."""
from __future__ import annotations

import difflib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from serve import permissions
from serve.sandbox.grep_worker import MAX_LINE, decode, hits_in
from serve.sandbox.workspace import MTIME_TICK_NS, NO_WINDOW, WorkspaceError

try:
    from re import _constants as _sre, _parser as _parse
except ImportError:                                                           # before Python 3.11
    import sre_constants as _sre
    import sre_parse as _parse

MAX_HITS = 100
READ_BYTES = 1024                      # what one `read` gives
SEARCH_SECONDS = 10.0                  # a pattern that has not finished by now is killed: a search of the whole copy takes under a second
WORKER = Path(__file__).with_name("grep_worker.py")
GENERATED = re.compile(r"(\.min\.(js|css)|_pb2(_grpc)?\.py|\.pb\.(go|cc|h)|\.generated\.\w+|\.g\.(cs|dart))\Z", re.I)        # kept out of search and symbols (design case 62), and counted so that "not found" stays honest


def _nested_repeat(items, under: bool = False) -> bool:
    for op, av in items:
        if op in (_sre.MAX_REPEAT, _sre.MIN_REPEAT):
            open_ended = av[1] is _sre.MAXREPEAT
            if (open_ended and under) or _nested_repeat(av[2], under or open_ended):
                return True
        elif op is _sre.SUBPATTERN:
            if _nested_repeat(av[3], under):
                return True
        elif op is _sre.BRANCH:
            if any(_nested_repeat(b, under) for b in av[1]):
                return True
    return False


def _may_backtrack(items) -> bool:
    """A repeat or a choice: anything else matches in time linear in the text."""
    for op, av in items:
        if op in (_sre.MAX_REPEAT, _sre.MIN_REPEAT, _sre.BRANCH):
            return True
        if op is _sre.SUBPATTERN and _may_backtrack(av[3]):
            return True
    return False


def _matcher(glob: str | None):
    """A glob with no / is about the file's name at any depth, as in ripgrep; one with a / is about the whole path."""
    if not glob:
        return lambda _rel: True
    pattern = glob.replace("\\", "/").removeprefix("./")
    rx = permissions._glob_re(pattern)
    if "/" in pattern:
        return lambda rel: bool(rx.match(rel))
    return lambda rel: bool(rx.match(rel.rpartition("/")[2]))


def clean_path(path: str) -> str:
    return path.replace("\\", "/").removeprefix("./").rstrip("/")


class Search:
    def __init__(self, workspace):
        self.ws = workspace
        self._texts: dict[str, tuple[tuple[int, int], str | None]] = {}

    def files(self) -> dict[str, tuple[int, int]]:
        now = self.ws.listing()
        for rel in self._texts.keys() - now.keys():
            del self._texts[rel]
        return now

    def text(self, rel: str, stamp: tuple[int, int]) -> str | None:
        """The text of a file that `files` listed. None: binary, too big, or it could not be read."""
        hit = self._texts.get(rel)
        if hit and hit[0] == stamp:
            return hit[1]
        text = None
        if stamp[0] <= self.ws.max_file:
            try:
                text = decode(self.ws.read_stamped(rel, stamp))
            except (OSError, WorkspaceError):
                pass
        if stamp[1] < time.time_ns() - MTIME_TICK_NS:                         # the rule of Workspace._hash_copy: a file written just now could be written again within the same tick
            self._texts[rel] = (stamp, text)
        return text

    def near(self, rel: str) -> list[str]:
        """Paths close to one that does not exist (design case 42): the same name in another folder first, then what looks like it."""
        paths = list(self.ws.listing())
        name = rel.rpartition("/")[2].lower()
        same = [p for p in paths if p.rpartition("/")[2].lower() == name]
        return (same + [p for p in difflib.get_close_matches(rel, paths, 5, 0.6) if p not in same])[:5]

    def tree(self, path: str = ".", depth: int = 2) -> dict:
        """The folders and files under `path`, `depth` levels deep; a folder cut off by the depth says how many files it holds."""
        base = clean_path(path)
        base = "" if base in ("", ".") else base + "/"
        found = [rel[len(base):].split("/") for rel in self.files() if rel.startswith(base)]
        if not found:
            return {"ok": False, "error": f"no such folder in the copy: {path}", "near": self.near(base.rstrip("/"))}
        folders: dict[tuple[str, ...], int] = {}
        entries: set[tuple[str, ...]] = set()
        for parts in found:
            for i in range(1, min(len(parts), depth + 1)):
                key = tuple(parts[:i])
                folders[key] = folders.get(key, 0) + 1
            if len(parts) <= depth:
                entries.add(tuple(parts))
        lines = []
        for key in sorted(entries | set(folders), key=lambda k: [x.lower() for x in k]):
            label = f"{key[-1]}/ ({folders[key]} files)" if key in folders else key[-1]
            lines.append("  " * (len(key) - 1) + label)
        return {"ok": True, "text": "\n".join(lines), "files": len(found)}

    def find_file(self, glob: str) -> dict:
        match = _matcher(glob)
        return {"ok": True, "paths": sorted(rel for rel in self.files() if match(rel))}

    hits_in = staticmethod(hits_in)

    def _in_child(self, pattern: str, rels: list[str], max_hits: int, generated: int) -> dict:
        job = {"root": str(self.ws.copy), "files": rels, "pattern": pattern, "max_hits": max_hits, "max_file": self.ws.max_file}
        try:
            r = subprocess.run([sys.executable, "-I", "-B", str(WORKER)], input=json.dumps(job).encode("utf-8"), capture_output=True, timeout=SEARCH_SECONDS, creationflags=NO_WINDOW)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"that pattern was still running after {SEARCH_SECONDS:.0f} s: write it without a choice or a repeat that can overlap"}
        if r.returncode:
            return {"ok": False, "error": "the search stopped: " + r.stderr.decode("utf-8", "replace").strip().splitlines()[-1][:200]}
        out = json.loads(r.stdout)
        return {"ok": True, "hits": out["hits"], "truncated": out["truncated"], "skipped": {"generated": generated, "binary": out["binary"]}}

    def search(self, pattern: str, glob: str | None = None, max_hits: int = MAX_HITS) -> dict:
        """Lines that match a regular expression. {"ok", "hits": [{"path", "line", "text"}], "truncated", "skipped": {"generated", "binary"}}; `skipped` is what was not looked at, so that no match does not read as none exists."""
        try:
            rx = re.compile(pattern, re.M)
        except (re.error, RecursionError, OverflowError) as e:
            return {"ok": False, "error": f"that is not a valid pattern: {e}"}
        parsed = _parse.parse(pattern)
        if _nested_repeat(parsed):
            return {"ok": False, "error": "a repeat inside a repeat can run for hours on one line: write it without the outer repeat"}
        match = _matcher(glob)
        if _may_backtrack(parsed):
            listed = [rel for rel in sorted(self.files()) if match(rel)]
            rels = [rel for rel in listed if not GENERATED.search(rel)]
            return self._in_child(pattern, rels, max_hits, len(listed) - len(rels))
        hits: list[dict] = []
        skipped = {"generated": 0, "binary": 0}
        truncated = False
        for rel, stamp in sorted(self.files().items()):
            if not match(rel):
                continue
            if GENERATED.search(rel):
                skipped["generated"] += 1
                continue
            text = self.text(rel, stamp)
            if text is None:
                skipped["binary"] += 1
                continue
            hits += self.hits_in(rel, text, rx, max_hits - len(hits))
            if len(hits) >= max_hits:
                truncated = True
                break
        return {"ok": True, "hits": hits, "truncated": truncated, "skipped": skipped}

    def read(self, path: str, offset: int = 0, limit: int = READ_BYTES) -> dict:
        """A piece of a file of the copy; a binary file or one past the size limit gives its size and kind, not its bytes (design case 9). `next`: where the following piece starts, None at the end.

        This is what the text of `output.py` points at, so it also reads the saved output, which `files` does not list."""
        rel = clean_path(path)
        saved = self.ws.is_output(rel)                                        # saved output can be any length and is read a piece at a time; a project file is read whole, to know its encoding
        try:
            size = self.ws.size_of(rel)
            raw = self.ws.read_bytes(rel, offset + limit + 8 if saved else self.ws.max_file + 1)
        except FileNotFoundError:
            return {"ok": False, "error": f"no such file in the copy: {path}", "near": self.near(rel)}
        except (OSError, WorkspaceError) as e:
            return {"ok": False, "error": str(e)}
        if size > self.ws.max_file and not saved:
            return {"ok": True, "path": rel, "size": size, "kind": "too big to show"}
        text = decode(raw)
        if text is None:
            return {"ok": True, "path": rel, "size": size, "kind": "binary"}
        data = text.encode("utf-8", "replace")                                # offsets count this form, whatever the file's own encoding
        total = size if saved else len(data)
        end = min(offset + limit, total)
        while end < min(total, len(data)) and (data[end] & 0xC0) == 0x80:     # not in the middle of a character
            end -= 1
        return {"ok": True, "path": rel, "size": total, "text": data[offset:end].decode("utf-8", "ignore"), "next": end if end < total else None}
