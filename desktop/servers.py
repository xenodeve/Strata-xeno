"""Strata Desktop's server list (#310 part 3): `servers.json` in the data folder, each token kept with DPAPI.

The file: `{"version": 1, "last": name | null, "injected": origin | null, "servers": [{"name", "url", "token_dpapi":
base64 of the DPAPI blob | null}]}`, written to `servers.json.tmp` and moved over the old one with `os.replace`, so a
failed write leaves the old file whole.  `last` is the server opened last; `injected` names the web view origin whose
localStorage holds a token right now (the next start removes it there first, so a crash leaves it only until then).

The token is never in the file, in an error message, in any `repr` / `str` (`Secret` shows `***`) or in `public()`.
A blob that no longer decrypts (the profile moved to another Windows user) gives its entry `Secret("")` and
`public()` says `"unreadable"`; the blob itself is kept as it was until the entry gets a new token or is removed.
"""
from __future__ import annotations

import base64
import binascii
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from desktop import dpapi

VERSION = 1
NAME_MAX = 64


class Secret:
    """A token in memory: `reveal()` gives it, every text form is `***`."""

    __slots__ = ("_v",)

    def __init__(self, value: str):
        self._v = value

    def reveal(self) -> str:
        return self._v

    def __repr__(self) -> str:
        return "***"

    __str__ = __repr__

    def __format__(self, spec: str) -> str:
        return "***"

    def __eq__(self, other) -> bool:
        return isinstance(other, Secret) and other._v == self._v

    def __hash__(self) -> int:
        return hash((Secret, self._v))

    def __copy__(self):
        return self                              # immutable: copy, deepcopy and dataclasses.asdict keep the object

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        raise TypeError("a Secret is not pickled")


@dataclass(frozen=True)
class ServerEntry:
    name: str
    url: str
    token: Secret


def _token_ok(token: str) -> bool:
    from serve.runner import KEY        # the runner sends the token as a header: the same rule
    return isinstance(token, str) and KEY.fullmatch(token) is not None


def _clean_url(url: str) -> str:
    u = (url or "").strip() if isinstance(url, str) else ""
    bad = ValueError("The URL must look like http://host:port or https://host, with no user name, query or #part.")
    try:
        p = urlsplit(u)
        p.port                                   # a port that is not a number in range raises here
    except ValueError:
        raise bad from None
    if p.scheme not in ("http", "https") or not p.hostname or "@" in p.netloc or p.query or p.fragment \
            or any(c.isspace() for c in u):
        raise bad
    return f"{p.scheme}://{p.netloc}{p.path.rstrip('/')}"


def _clean_name(name: str) -> str:
    n = name.strip() if isinstance(name, str) else ""
    if not 1 <= len(n) <= NAME_MAX:
        raise ValueError(f"The name must be 1 to {NAME_MAX} characters.")
    return n


class ServerStore:
    """The list in `path` (read once; every change is written at once).  Safe to call from several threads."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._doc = self._read()

    # ------------------------------------------------------------------ the file
    def _read(self) -> dict:
        empty = {"version": VERSION, "last": None, "injected": None, "servers": []}
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return empty
        try:
            doc = json.loads(raw)
            rows = [{"name": str(r["name"]), "url": str(r["url"]),
                     "token_dpapi": r.get("token_dpapi") if isinstance(r.get("token_dpapi"), str) else None}
                    for r in doc["servers"]]
            last = doc.get("last") if isinstance(doc.get("last"), str) else None
            injected = doc.get("injected") if isinstance(doc.get("injected"), str) else None
        except (ValueError, KeyError, TypeError, AttributeError):
            os.replace(self.path, self.path.with_name(self.path.name + ".broken"))      # kept, not overwritten
            return empty
        if last not in [r["name"] for r in rows]:
            last = None
        return {"version": VERSION, "last": last, "injected": injected, "servers": rows}

    def _write(self, doc: dict) -> None:
        """`doc` to disk through a .tmp file; the store takes it only once it is there."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            with open(tmp, "w", encoding="utf-8", newline="\n") as f:
                json.dump(doc, f, ensure_ascii=False, indent=1)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise
        self._doc = doc

    def _copy(self) -> dict:
        d = dict(self._doc)
        d["servers"] = [dict(r) for r in self._doc["servers"]]
        return d

    # ------------------------------------------------------------------ reading
    @staticmethod
    def _token(row: dict) -> tuple[Secret, str]:
        b64 = row["token_dpapi"]
        if not b64:
            return Secret(""), "missing"
        try:
            return Secret(dpapi.unprotect(base64.b64decode(b64, validate=True)).decode("ascii")), "set"
        except (dpapi.DpapiError, binascii.Error, ValueError):
            return Secret(""), "unreadable"

    def entries(self) -> list[ServerEntry]:
        with self._lock:
            rows = [dict(r) for r in self._doc["servers"]]
        return [ServerEntry(r["name"], r["url"], self._token(r)[0]) for r in rows]

    def get(self, name: str) -> ServerEntry | None:
        return next((e for e in self.entries() if e.name == name), None)

    def public(self) -> list[dict]:
        """What a page may see: name, URL and whether a token is "set", "missing" or "unreadable"."""
        with self._lock:
            rows = [dict(r) for r in self._doc["servers"]]
        return [{"name": r["name"], "url": r["url"], "token": self._token(r)[1]} for r in rows]

    @property
    def last(self) -> str | None:
        return self._doc["last"]

    @property
    def injected(self) -> str | None:
        return self._doc["injected"]

    # ------------------------------------------------------------------ changing
    def save(self, name: str, url: str, token: str | None, old_name: str | None = None) -> ServerEntry:
        """Add a server, or change `old_name` (renamed to `name`).  `token=None` keeps the stored token (a new entry
        then has none).  A bad field raises `ValueError` in words that name the field and never repeat the token."""
        name = _clean_name(name)
        url = _clean_url(url)
        if token is not None and not _token_ok(token):
            raise ValueError("The token must be 1 to 4096 printable ASCII characters with no spaces.")
        blob = None if token is None else base64.b64encode(dpapi.protect(token.encode("ascii"))).decode("ascii")
        with self._lock:
            doc = self._copy()
            rows = doc["servers"]
            names = [r["name"] for r in rows]
            if old_name is not None and old_name not in names:
                raise ValueError("There is no server with that name to change.")
            if name in names and name != old_name:
                raise ValueError("A server with this name is already in the list.")
            if old_name is None:
                row = {"name": name, "url": url, "token_dpapi": blob}
                rows.append(row)
            else:
                row = rows[names.index(old_name)]
                row.update(name=name, url=url)
                if blob is not None:
                    row["token_dpapi"] = blob
                if doc["last"] == old_name:
                    doc["last"] = name
            self._write(doc)
            row = dict(row)
        return ServerEntry(row["name"], row["url"], self._token(row)[0])

    def remove(self, name: str) -> bool:
        with self._lock:
            doc = self._copy()
            keep = [r for r in doc["servers"] if r["name"] != name]
            if len(keep) == len(doc["servers"]):
                return False
            doc["servers"] = keep
            if doc["last"] == name:
                doc["last"] = None
            self._write(doc)
            return True

    def set_last(self, name: str | None) -> None:
        with self._lock:
            if name is not None and name not in [r["name"] for r in self._doc["servers"]]:
                raise ValueError("There is no server with that name.")
            doc = self._copy()
            doc["last"] = name
            self._write(doc)

    def set_injected(self, origin: str | None) -> None:
        with self._lock:
            doc = self._copy()
            doc["injected"] = origin
            self._write(doc)
