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


def prompt_for_keep(messages) -> list:
    """The prompt as kept for replay (Q8, opt-in): the messages in full, but a picture is a note, not its base64."""
    out = []
    for m in messages or []:
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, list):
            c = [{**p, "image_url": {"url": "[image omitted]"}} if isinstance(p, dict) and p.get("type") == "image_url"
                 else {**p, "source": "[image omitted]"} if isinstance(p, dict) and p.get("type") == "image"   # the Anthropic dialect, converted
                 else p for p in c]
            m = {**m, "content": c}
        out.append(m)
    return out


def summary_record(rec: dict) -> dict:
    """A request's summary line: the server's record plus the prefill speed over the tokens actually read."""
    read = (rec.get("prompt_tokens") or 0) - (rec.get("reused") or 0)
    ms = rec.get("prompt_ms")
    return {**rec, "prefill_tok_s": round(read / (ms / 1000), 1) if read > 0 and ms else None}


def chunk_stats(points, start: int):
    """Prefill speed per chunk from the engine's PP lines: `points` are (position reached, ms since that read began),
    `start` the position the read began at (the cached prefix). None when nothing was read."""
    items, pos, prev_ms = [], start, 0.0
    for p, ms in points:
        if ms < prev_ms:                            # a second engine call: its clock began again, the position did not
            prev_ms = 0.0
        tokens, dms = p - pos, ms - prev_ms
        if tokens > 0 and dms > 0:
            items.append([tokens, round(dms, 3)])
        pos, prev_ms = p, ms
    if not items:
        return None
    rates = [t / (ms / 1000) for t, ms in items]
    total_t, total_ms = sum(t for t, _ in items), sum(ms for _, ms in items)
    return {"chunks": len(items), "items": items, "tok_s_max": round(max(rates), 1), "tok_s_min": round(min(rates), 1),
            "tok_s_mean": round(total_t / (total_ms / 1000), 1)}


def window_rates(times, window: int = 16):
    """Decode speed from the time each token reached the server: tok/s over every sliding `window` of tokens (min /
    max / the series for the trend), and the mean over the whole run. Never a per-token speed. None below 2 tokens."""
    n = len(times)
    if n < 2:
        return None
    span = times[-1] - times[0]
    mean = round((n - 1) / span, 1) if span > 0 else None
    rates = [window / (times[i] - times[i - window]) for i in range(window, n) if times[i] > times[i - window]]
    if not rates:
        return {"windows": 0, "tok_s_min": None, "tok_s_max": None, "tok_s_mean": mean, "series": []}
    step = max(1, -(-len(rates) // 120))                       # at most 120 points
    return {"windows": len(rates), "tok_s_min": round(min(rates), 1), "tok_s_max": round(max(rates), 1),
            "tok_s_mean": mean, "series": [round(r, 1) for r in rates[::step]]}


class HistoryStore:
    def __init__(self, directory, enabled=True, detail_cap_bytes=DEFAULT_DETAIL_CAP):
        self.dir = Path(directory)
        self.enabled = enabled
        self.cap = int(detail_cap_bytes)
        self.lock = threading.Lock()              # writers only (append, write_detail, prune)
        self._counts: dict = {}

    # ------------------------------------------------------------------------------------------ summaries
    def append(self, record: dict) -> None:
        if not self.enabled:
            return
        t = record.get("time") or time.time()
        name = f"requests-{time.strftime('%Y-%m', time.localtime(t))}.jsonl"
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self.lock:
            self._private_dir(self.dir)
            fd = os.open(self.dir / name, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)       # xeno #71: owner only (a no-op on Windows)
            with os.fdopen(fd, "a", encoding="utf-8") as f:
                f.write(line)

    @staticmethod
    def _private_dir(d: Path) -> None:
        """The history holds the start of every prompt (and a whole one on request): the folder is the owner's alone where the
        OS has modes (on Windows it lives under the user's own profile)."""
        d.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name != "nt":
            os.chmod(d, 0o700)

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

    def _count(self, path: Path) -> int:
        """Rows in a month file, remembered per (size, mtime): an old month is parsed once, not on every page."""
        try:
            st = path.stat()
        except OSError:
            return 0
        key = (st.st_size, st.st_mtime_ns)
        hit = self._counts.get(path.name)
        if hit and hit[0] == key:
            return hit[1]
        n = len(self._read(path))
        self._counts[path.name] = (key, n)
        return n

    def page(self, page: int, size: int) -> dict:
        """Newest first. {"items": [...], "total": n, "page": p, "size": s}. Nothing here takes the write lock: the
        files are append-only and a torn last line is skipped, so a page read never holds up a request that is ending."""
        size = max(1, min(int(size), 500))
        page = max(0, int(page))
        files = self._files_newest_first()
        need, rows, total = (page + 1) * size, [], 0
        for f in files:
            if len(rows) < need:                    # newest files first, and only as many as this page reaches
                recs = self._read(f)
                self._counts[f.name] = (self._stat_key(f), len(recs))
                rows.extend(reversed(recs))
                total += len(recs)
            else:
                total += self._count(f)             # older months only counted (cached)
        return {"items": rows[page * size:need], "total": total, "page": page, "size": size}

    @staticmethod
    def _stat_key(path: Path):
        try:
            st = path.stat()
            return (st.st_size, st.st_mtime_ns)
        except OSError:
            return None

    def has_summary(self, rid: str) -> bool:
        return self.summary(rid) is not None

    def summary(self, rid: str):
        if not isinstance(rid, str) or not _ID.match(rid):
            return None
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
            self._private_dir(path.parent)
            tmp = path.with_suffix(".tmp")
            with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as f:
                f.write(blob)
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
