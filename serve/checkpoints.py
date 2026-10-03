"""serve/checkpoints.py - the way back for the files the chat's tools change (issue #99), as Claude Code's rewind.

Each prompt of a chat is a checkpoint. Before the Write, Edit or NotebookEdit tool first changes a file during a prompt, the file's text as it was (or the fact that it did not exist) is
kept here; after each change its fingerprint is noted. Rewinding to a prompt puts every file the tools changed in that prompt or later back to what it was before that prompt: a file
that was changed is restored, one that was made is deleted. A file that was changed by someone else since the tools last wrote it (you, a build, a command) is left alone unless asked for.
What the shell tool does to files is not tracked, so it is not undone. The size of what is kept is bounded, the oldest goes first, and a chat's checkpoints go when the chat is deleted."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
from pathlib import Path

MAX_FILE = 5 * 1024 * 1024         # a file bigger than this is not kept (it is listed as one that cannot be restored)
MAX_FILES = 200                    # files kept for one prompt
MAX_BYTES = 100 * 1024 * 1024      # bytes kept for one prompt
KEEP_PROMPTS = 40                  # checkpoints kept for one chat: the newest
KEEP_CHATS = 30                    # chats that have checkpoints: the ones used most lately


def default_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base) / "Strata" / "checkpoints"


def _safe(name: str) -> str | None:
    # no "." or ":": a name is joined onto the store, so "..", "." or a drive ("C:") would lead outside it (forget() deletes)
    return name if isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", name) else None


def _hash(path: str) -> str | None:
    try:
        h = hashlib.sha1()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


class Scope:
    """What the tools of one request use: this chat's checkpoint for this prompt."""

    def __init__(self, store: "Checkpoints", session: str, cp: str):
        self.store, self.session, self.cp = store, session, cp

    def before(self, path: str) -> None:
        self.store.before(self.session, self.cp, path)

    def after(self, path: str) -> None:
        self.store.after(self.session, self.cp, path)


class Checkpoints:
    def __init__(self, root: str | os.PathLike | None = None):
        self.root = Path(root) if root else default_dir()
        self.lock = threading.RLock()

    # ------------------------------------------------------------------------------------------------ storage
    def _dir(self, session: str) -> Path | None:
        s = _safe(session)
        return self.root / s if s else None

    def _load(self, d: Path) -> dict:
        try:
            data = json.loads((d / "index.json").read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("order"), list) and isinstance(data.get("cps"), dict):
                return data
        except (OSError, ValueError):
            pass
        return {"order": [], "cps": {}}

    def _save(self, d: Path, data: dict) -> None:
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "index.json.tmp"
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, d / "index.json")

    # ------------------------------------------------------------------------------------------------ recording
    def before(self, session: str, cp: str, path: str) -> None:
        """Keeps a file as it is, once per prompt, before the tools change it."""
        d, c = self._dir(session), _safe(cp)
        if d is None or c is None:
            return
        with self.lock:
            data = self._load(d)
            if c not in data["cps"]:
                data["cps"][c] = {"time": time.time(), "files": {}}
                data["order"].append(c)
            files = data["cps"][c]["files"]
            if path in files:
                return
            if len(files) >= MAX_FILES:
                files[path] = {"skipped": "too many files in one prompt"}
            elif not os.path.exists(path):
                files[path] = {"existed": False, "wrote": None}
            else:
                try:
                    size = os.path.getsize(path)
                except OSError:
                    size = -1
                used = sum(f.get("size", 0) for f in files.values())
                if size < 0 or not os.path.isfile(path):
                    files[path] = {"skipped": "it cannot be read"}
                elif size > MAX_FILE or used + size > MAX_BYTES:
                    files[path] = {"skipped": "it is too big to keep"}
                else:
                    name = hashlib.sha1(path.encode("utf-8", "replace")).hexdigest()[:16]
                    (d / c).mkdir(parents=True, exist_ok=True)
                    try:
                        shutil.copy2(path, d / c / name)
                        files[path] = {"existed": True, "backup": name, "size": size, "wrote": None}
                    except OSError:
                        files[path] = {"skipped": "it cannot be read"}
            self._save(d, data)

    def after(self, session: str, cp: str, path: str) -> None:
        """Notes what the file is after a change of the tools' own, to tell later whether someone else changed it."""
        d, c = self._dir(session), _safe(cp)
        if d is None or c is None:
            return
        with self.lock:
            data = self._load(d)
            entry = data["cps"].get(c, {}).get("files", {}).get(path)
            if entry is None or "skipped" in entry:
                return
            entry["wrote"] = _hash(path)
            self._save(d, data)

    def scope(self, session: str, cp: str) -> Scope | None:
        return Scope(self, session, cp) if _safe(session) and _safe(cp) else None

    # ------------------------------------------------------------------------------------------------ looking and rewinding
    def listing(self, session: str) -> list[dict]:
        d = self._dir(session)
        if d is None or not d.is_dir():
            return []
        with self.lock:
            data = self._load(d)
        return [{"id": c, "files": len(data["cps"].get(c, {}).get("files", {}))} for c in data["order"]]

    def _plan(self, data: dict, cp: str) -> list[dict]:
        """For each file the tools changed in `cp` or after it: what it was before that prompt (the first record), and what the tools last wrote (the last record)."""
        if cp not in data["order"]:
            return []
        first: dict[str, dict] = {}
        last: dict[str, dict] = {}
        for c in data["order"][data["order"].index(cp):]:
            for path, e in data["cps"].get(c, {}).get("files", {}).items():
                first.setdefault(path, {**e, "cp": c})
                if "skipped" not in e:
                    last[path] = e
        out = []
        for path, e in first.items():
            if "skipped" in e:
                out.append({"path": path, "action": "skip", "why": e["skipped"], "changed": False})
                continue
            wrote = (last.get(path) or {}).get("wrote")
            now = _hash(path) if os.path.exists(path) else None
            changed = wrote is not None and now != wrote                   # someone else changed it (or took it away) since the tools last wrote it
            out.append({"path": path, "action": "restore" if e["existed"] else "delete", "changed": bool(changed), "present": now is not None, "_e": e})
        return sorted(out, key=lambda x: x["path"].lower())

    def preview(self, session: str, cp: str) -> dict:
        d = self._dir(session)
        if d is None or _safe(cp) is None or not d.is_dir():
            return {"ok": True, "files": [], "known": False}
        with self.lock:
            data = self._load(d)
            plan = self._plan(data, cp)
        return {"ok": True, "known": cp in data["order"], "files": [{k: v for k, v in f.items() if k != "_e"} for f in plan]}

    def apply(self, session: str, cp: str, include_changed: bool = False) -> dict:
        """Puts the files back. A file someone else changed since is left (listed in `kept`) unless `include_changed`. The checkpoints from `cp` on are done with."""
        d = self._dir(session)
        if d is None or _safe(cp) is None or not d.is_dir():
            return {"ok": True, "restored": 0, "deleted": 0, "kept": [], "failed": [], "skipped": []}
        restored = deleted = 0
        kept: list[str] = []
        failed: list[str] = []
        skipped: list[str] = []
        with self.lock:
            data = self._load(d)
            for f in self._plan(data, cp):
                path, e = f["path"], f.get("_e")
                if f["action"] == "skip":
                    skipped.append(path)
                    continue
                if f["changed"] and not include_changed:
                    kept.append(path)
                    continue
                try:
                    if f["action"] == "restore":
                        src = d / e["cp"] / e["backup"]
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        tmp = path + ".strata-restore"
                        shutil.copy2(src, tmp)
                        os.replace(tmp, path)
                        restored += 1
                    elif os.path.exists(path):
                        os.remove(path)
                        deleted += 1
                except OSError:
                    failed.append(path)
            if cp in data["order"]:                                  # these prompts are gone: so are their checkpoints
                gone = data["order"][data["order"].index(cp):]
                data["order"] = data["order"][:data["order"].index(cp)]
                for c in gone:
                    data["cps"].pop(c, None)
                    shutil.rmtree(d / c, ignore_errors=True)
                self._save(d, data)
        return {"ok": True, "restored": restored, "deleted": deleted, "kept": kept, "failed": failed, "skipped": skipped}

    def forget(self, session: str) -> None:
        d = self._dir(session)
        if d is not None and d.is_dir():
            with self.lock:
                shutil.rmtree(d, ignore_errors=True)

    def trim(self) -> None:
        """The oldest checkpoints go, and the chats used longest ago: called now and then, not on every change."""
        with self.lock:
            if not self.root.is_dir():
                return
            chats = sorted((p for p in self.root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
            for p in chats[KEEP_CHATS:]:
                shutil.rmtree(p, ignore_errors=True)
            for p in chats[:KEEP_CHATS]:
                data = self._load(p)
                if len(data["order"]) > KEEP_PROMPTS:
                    for c in data["order"][:-KEEP_PROMPTS]:
                        data["cps"].pop(c, None)
                        shutil.rmtree(p / c, ignore_errors=True)
                    data["order"] = data["order"][-KEEP_PROMPTS:]
                    self._save(p, data)
