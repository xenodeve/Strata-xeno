"""serve/history.py - the request history on disk (xeno UI S3).

    <dir>/requests-YYYY-MM.jsonl     one summary line per request, kept forever (~1 KB each)
    <dir>/detail/<id>.json.gz        the rounds / chunks / stages of one request, capped in total, oldest deleted first

A deleted detail leaves its summary: the page says "detail deleted". Everything can be switched off (enabled=False).
"""
from __future__ import annotations

import gzip
import json
import os
import re
import threading
import time
from pathlib import Path

DEFAULT_DETAIL_CAP = 2 * 2**30              # 2 GB
_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def default_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base) / "Strata" / "history"


def request_meta(dialect: str, messages, tools, user_agent) -> dict:
    """What is kept of a request besides its numbers: an id, the endpoint's dialect, the client, the tool names and
    the first 200 characters of the last user message. Never the whole prompt (that is an opt-in, "keep for replay")."""
    preview = ""
    for m in reversed(messages or []):
        if isinstance(m, dict) and m.get("role") == "user":
            body = m.get("content")
            if isinstance(body, list):
                body = " ".join(p.get("text", "") for p in body if isinstance(p, dict))
            preview = str(body or "")[:200]
            break
    names = []
    for t in tools or []:
        n = (t.get("function") or {}).get("name") if isinstance(t.get("function"), dict) else None
        n = n or t.get("name")
        if n:
            names.append(str(n))
    return {"id": f"r{int(time.time() * 1000):x}{os.urandom(2).hex()}", "dialect": dialect,
            "client": str(user_agent or "")[:120], "tools": names, "preview": preview}


def summary_record(rec: dict) -> dict:
    """A request's summary line: the server's record plus the prefill speed over the tokens actually read."""
    read = (rec.get("prompt_tokens") or 0) - (rec.get("reused") or 0)
    ms = rec.get("prompt_ms")
    return {**rec, "prefill_tok_s": round(read / (ms / 1000), 1) if read > 0 and ms else None}


class HistoryStore:
    def __init__(self, directory, enabled=True, detail_cap_bytes=DEFAULT_DETAIL_CAP):
        self.dir = Path(directory)
        self.enabled = enabled
        self.cap = int(detail_cap_bytes)
        self.lock = threading.Lock()

    # ------------------------------------------------------------------------------------------ summaries
    def append(self, record: dict) -> None:
        if not self.enabled:
            return
        t = record.get("time") or time.time()
        name = f"requests-{time.strftime('%Y-%m', time.localtime(t))}.jsonl"
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self.lock:
            self.dir.mkdir(parents=True, exist_ok=True)
            with open(self.dir / name, "a", encoding="utf-8") as f:
                f.write(line)

    def _files_newest_first(self):
        return sorted(self.dir.glob("requests-*.jsonl"), reverse=True) if self.dir.is_dir() else []

    @staticmethod
    def _read(path: Path):
        out = []
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return out
        for line in text.splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:                      # a torn last line after a crash
                continue
        return out

    def page(self, page: int, size: int) -> dict:
        """Newest first. {"items": [...], "total": n, "page": p, "size": s}."""
        size = max(1, min(int(size), 500))
        page = max(0, int(page))
        with self.lock:
            files = self._files_newest_first()
            per_file = [self._read(f) for f in files]
        total = sum(len(r) for r in per_file)
        skip, items = page * size, []
        for recs in per_file:
            for r in reversed(recs):
                if skip:
                    skip -= 1
                elif len(items) < size:
                    items.append(r)
        return {"items": items, "total": total, "page": page, "size": size}

    def has_summary(self, rid: str) -> bool:
        if not isinstance(rid, str) or not _ID.match(rid):
            return False
        with self.lock:
            return any(r.get("id") == rid for f in self._files_newest_first() for r in self._read(f))

    def summary(self, rid: str):
        if not isinstance(rid, str) or not _ID.match(rid):
            return None
        with self.lock:
            for f in self._files_newest_first():
                for r in self._read(f):
                    if r.get("id") == rid:
                        return r
        return None

    # ------------------------------------------------------------------------------------------ details
    def _detail_path(self, rid: str) -> Path:
        if not isinstance(rid, str) or not _ID.match(rid):
            raise ValueError(f"bad request id: {rid!r}")
        return self.dir / "detail" / f"{rid}.json.gz"

    def write_detail(self, rid: str, data: dict) -> None:
        path = self._detail_path(rid)               # raises on a bad id, on or off
        if not self.enabled:
            return
        blob = gzip.compress(json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        with self.lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(blob)
            os.replace(tmp, path)
            self._prune()

    def _prune(self) -> None:
        files = []
        for p in (self.dir / "detail").glob("*.json.gz"):
            try:
                st = p.stat()
            except OSError:
                continue
            files.append((st.st_mtime, p.name, st.st_size, p))
        files.sort()                                # oldest first
        total = sum(f[2] for f in files)
        for _, _, size, p in files:
            if total <= self.cap:
                break
            try:
                p.unlink()
                total -= size
            except OSError:
                pass

    def detail(self, rid: str):
        """The request's detail, or None when it was never kept or has been deleted."""
        try:
            path = self._detail_path(rid)
        except ValueError:
            return None
        try:
            return json.loads(gzip.decompress(path.read_bytes()))
        except (OSError, ValueError):
            return None
