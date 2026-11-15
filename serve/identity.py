"""serve/identity.py - who is asking, and what that person may do: named API keys with a role, and the approval of
devices (#310).

A team workstation needs more than one shared key.  The keys live in a file of their own beside the run config
(`"keys_file"`, default strata-keys.json), not in the run config, so a key change never rewrites the config.  The
file is read at start and again when it changes on disk, so the owner can add, rename, demote and revoke keys while
the server runs and the change counts at the next request.

  * `Identity(name, role)`: the answer to "who".  Roles are exactly "owner" and "guest".  The old single `"api_key"`
    keys are the identity `OWNER`.  With no key set at all, a request from this PC that names this PC is `OWNER` and
    any other is `LOCAL_GUEST` (as mcp_admin.may_edit decided before).
  * `KeyStore`: matches a given key to an identity, keeps the keys and each key's devices, and answers who may approve
    a device.  A new key is "sk-strata-" + 24 random bytes, shown once and kept only as its sha256.
  * Team mode is the keys file existing.  Without it nothing here changes how the server behaves today.
  * `device_of`, `device_addr`: which device a request came from (the X-Strata-Device header, else "addr:<ip>") and
    the address it is recorded with (forwarding headers count only from a proxy on this PC).
  * `visible_record`: what another person's request record may show a guest.

Standard library only.  Every read and write of the file state is under one lock.  A file that fails to parse keeps
the last good keys and prints one line; it never opens the server.
"""
from __future__ import annotations

import copy
import hashlib
import hmac
import ipaddress
import json
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass

ROLES = ("owner", "guest")
RESERVED = ("owner", "anonymous")          # "owner" is the legacy key's identity, "anonymous" the unauthenticated guest
KEY_PREFIX = "sk-strata-"
LOG_CAP = 500
MAX_DEVICES = 100                          # per key: a client chooses its device id, so the file must not grow without end
MAX_NAME = 64
MAX_TEXT = 80                              # a device's label and address
RELOAD_EVERY_S = 1.0                       # match() looks at the file's mtime at most this often
LAST_SEEN_EVERY_S = 60.0                   # a device's last_seen reaches the disk at most this often
CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"      # no 0/O/1/I: the code is read aloud and compared
_clock = time.monotonic


class KeysFileBad(ValueError):
    """The keys file on disk does not parse.  The store answers from the last good keys, but writes nothing until the
    file is fixed: a write would replace what the owner is in the middle of editing."""


@dataclass(frozen=True)
class Identity:
    name: str
    role: str                              # "owner" | "guest"

    @property
    def is_owner(self) -> bool:
        return self.role == "owner"


OWNER = Identity("owner", "owner")
LOCAL_GUEST = Identity("anonymous", "guest")


def _legacy_list(value) -> list[str]:
    """The old api_key forms (server.key_list): "k1,k2", or a list that is taken as it is."""
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        return [str(k) for k in value if str(k)]
    return [k.strip() for k in str(value).split(",") if k.strip()]


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class KeyStore:
    def __init__(self, path: str | None, legacy: str | list[str] = ""):
        self.path = os.fspath(path) if path else None
        self._legacy = _legacy_list(legacy)
        self._lock = threading.Lock()
        self._data: dict = {"keys": [], "log": []}
        self._index: list[tuple[str, Identity]] = []      # (sha256 hex, identity) of every key in the file
        self._sig = None                                  # (mtime_ns, size) of the file as last read or written
        self._checked = -RELOAD_EVERY_S                   # when match() last looked at the file
        self._seen_saved: dict = {}                       # (key, device) -> last_seen as last written
        self._team = False
        self._bad = False                                 # the file on disk failed to parse when it was last read
        with self._lock:
            self._reload_locked()

    # ------------------------------------------------------------------------------------------ the file
    def _stat_sig(self):
        try:
            st = os.stat(self.path)
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size)

    def _reload_locked(self) -> None:
        if not self.path:
            return
        sig = self._stat_sig()
        if sig is None:                                   # no file (yet), or gone: keep what is held
            self._bad = False                             # nothing on disk to destroy; a write makes a new file
            return
        self._team = True
        if sig == self._sig:
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = self._clean(json.load(f))
        except OSError as e:                              # not read at all (locked for a moment): look again next time
            if not self._bad:
                print(f"strata serve: the keys file {self.path} is not read ({e}); keeping the keys held", flush=True)
            self._bad = True
            return
        except ValueError as e:
            self._sig = sig                               # a bad file is reported once, not once a second
            print(f"strata serve: the keys file {self.path} is not read ({e}); keeping the keys held", flush=True)
            self._bad = True
            return
        self._sig = sig
        self._bad = False
        self._data = data
        self._rebuild()

    @staticmethod
    def _clean(raw) -> dict:
        """The file as the store holds it.  A wrong shape raises ValueError (a broken file); one entry that cannot be
        a key (an unknown role, no hash) is dropped, so it can never match."""
        if not isinstance(raw, dict) or not isinstance(raw.get("keys"), list):
            raise ValueError("expected an object with a list of keys")
        keys, names = [], set()
        for e in raw["keys"]:
            if not isinstance(e, dict) or not isinstance(e.get("name"), str) or e.get("role") not in ROLES:
                continue
            legacy = e.get("legacy") is True
            sha = e.get("sha256")
            if legacy:
                if e["name"] != "owner" or e["role"] != "owner":
                    continue
            elif not (isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha)):
                continue
            if not legacy and e["name"].casefold() in RESERVED:
                print(f"strata serve: the keys file has a key named {e['name']!r}, a reserved name; it is ignored",
                      flush=True)
                continue
            if e["name"].casefold() in names:
                continue
            names.add(e["name"].casefold())
            devices = e.get("devices")
            clean = {"name": e["name"], "role": e["role"], "created": e.get("created"),
                     "devices": {i: d for i, d in devices.items() if isinstance(d, dict)}
                     if isinstance(devices, dict) else {}}
            if legacy:
                clean["legacy"] = True
            else:
                clean["sha256"] = sha
            keys.append(clean)
        log = raw.get("log")
        return {"keys": keys, "log": [x for x in log if isinstance(x, dict)][-LOG_CAP:] if isinstance(log, list) else []}

    def _rebuild(self) -> None:
        self._index = [(e["sha256"], Identity(e["name"], e["role"])) for e in self._data["keys"] if "sha256" in e]
        self._seen_saved = {(e["name"], i): d.get("last_seen")
                            for e in self._data["keys"] for i, d in e["devices"].items()}

    def _save(self, data: dict) -> None:
        """Atomic: the new text is written beside the file and renamed over it.  The file is the owner's alone where
        the OS has modes (on Windows it lives in the user's own profile)."""
        tmp = f"{self.path}.{os.getpid()}.tmp"
        try:
            with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=1)
            for attempt in range(6):
                try:
                    os.replace(tmp, self.path)
                    break
                except PermissionError:                   # Windows: a scanner or indexer holds the target for a moment
                    if attempt == 5:
                        raise
                    time.sleep(0.02 * (attempt + 1))
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        self._sig = self._stat_sig()
        self._team = True

    def _change(self, fn):
        """Run fn(copy of the data) and write the result.  The file is read first (an outside edit is kept), and a
        failed write leaves the store as it was."""
        with self._lock:
            if not self.path:
                raise ValueError("no keys file: the keys are not kept")
            self._reload_locked()
            if self._bad:
                raise KeysFileBad(f"the keys file {self.path} does not parse: fix it (or delete it) before changing keys")
            new = copy.deepcopy(self._data)
            result = fn(new)
            new["log"] = new["log"][-LOG_CAP:]
            self._save(new)
            self._data = new
            self._rebuild()
            return result

    def reload_if_changed(self) -> None:
        with self._lock:
            self._reload_locked()

    def _maybe_reload(self) -> None:
        now = _clock()
        if now - self._checked >= RELOAD_EVERY_S:
            self._checked = now
            self.reload_if_changed()

    def team_mode(self) -> bool:
        """True when the keys file exists (or a key has been created): only then do device checks and the owner-only
        routes apply.  A store with no file path is never in team mode."""
        if not self.path:
            return False
        with self._lock:
            return os.path.exists(self.path) or self._team

    # ------------------------------------------------------------------------------------------ who
    def any_key(self) -> bool:
        """Does this server need a key?  A keys file that exists counts even when it is empty or unreadable: a
        broken file must not turn into an open server."""
        self._maybe_reload()
        with self._lock:
            return bool(self._legacy or self._data["keys"] or self._team)

    def match(self, given: str) -> Identity | None:
        """The identity of the key a request sent, or None.  Constant time over every key, whatever the outcome (#213).
        `given` is the header's value as http.server read it, each byte one character; a key with characters outside
        ASCII arrives as UTF-8 from most clients and as Latin-1 from some, so both forms are tried (as
        server.key_matches)."""
        if not isinstance(given, str):
            return None
        self._maybe_reload()
        with self._lock:
            index, legacy = self._index, self._legacy
        raw = given.encode("latin-1", "replace")
        utf = given.encode("utf-8", "replace")
        hashes = (_sha(raw), _sha(utf))
        found = None
        for k in legacy:
            ok = hmac.compare_digest(raw, k.encode())
            try:
                ok |= hmac.compare_digest(raw, k.encode("latin-1"))
            except UnicodeEncodeError:
                pass
            if ok and found is None:
                found = OWNER
        for digest, ident in index:
            ok = False
            for h in hashes:
                ok |= hmac.compare_digest(h, digest)
            if ok and found is None:
                found = ident
        return found

    # ------------------------------------------------------------------------------------------ keys
    @staticmethod
    def _name(name) -> str:
        if not isinstance(name, str):
            raise ValueError("a key name is text")
        name = name.strip()
        if not name or len(name) > MAX_NAME or any(ord(c) < 32 or ord(c) == 127 for c in name):
            raise ValueError(f"a key name is 1-{MAX_NAME} characters with no control characters")
        if name.casefold() in RESERVED:
            raise ValueError(f"the name {name!r} is reserved")
        return name

    @staticmethod
    def _role(role) -> str:
        if role not in ROLES:
            raise ValueError(f"a role is one of {', '.join(ROLES)}")
        return role

    @staticmethod
    def _entry(data: dict, name: str):
        return next((e for e in data["keys"] if e["name"] == name), None)

    @staticmethod
    def _log(data: dict, what: str, key: str, device: str = "", by: str = "") -> None:
        data["log"].append({"t": time.time(), "what": what, "key": key, "device": device, "by": by})

    @staticmethod
    def _unique(data: dict, name: str) -> None:
        if any(e["name"].casefold() == name.casefold() for e in data["keys"]):
            raise ValueError(f"a key named {name!r} exists")

    def create(self, name: str, role: str) -> str:
        """A new key.  The plain key is returned here and nowhere else: only its sha256 is kept."""
        name, role = self._name(name), self._role(role)
        key = KEY_PREFIX + secrets.token_urlsafe(24)

        def go(data):
            self._unique(data, name)
            data["keys"].append({"name": name, "role": role, "sha256": _sha(key.encode()),
                                 "created": time.time(), "devices": {}})
            self._log(data, "created", name)
        self._change(go)
        return key

    def rename(self, name: str, new: str) -> bool:
        if name == "owner":
            raise ValueError("the owner key is built in")
        new = self._name(new)

        def go(data):
            e = self._entry(data, name)
            if e is None:
                return False
            if new.casefold() != name.casefold():
                self._unique(data, new)
            e["name"] = new
            self._log(data, f"renamed from {name}", new)
            return True
        return self._change(go)

    def set_role(self, name: str, role: str) -> bool:
        role = self._role(role)
        if name == "owner":
            raise ValueError("the owner key is built in")

        def go(data):
            e = self._entry(data, name)
            if e is None:
                return False
            e["role"] = role
            self._log(data, "role:" + role, name)
            return True
        return self._change(go)

    def revoke(self, name: str) -> bool:
        if name == "owner":
            raise ValueError("the owner key is built in")

        def go(data):
            e = self._entry(data, name)
            if e is None:
                return False
            data["keys"].remove(e)
            self._log(data, "revoked", name)
            return True
        return self._change(go)

    @staticmethod
    def _device_row(i: str, d: dict) -> dict:
        return {"device": i, "label": d.get("label", ""), "addr": d.get("addr", ""), "status": d.get("status"),
                "first_seen": d.get("first_seen"), "last_seen": d.get("last_seen"), "decided_by": d.get("decided_by")}

    def view(self) -> list[dict]:
        """For the owner's Settings page: never a key or a hash."""
        with self._lock:
            self._reload_locked()
            rows = []
            for e in self._data["keys"]:
                row = {"name": e["name"], "role": e["role"], "created": e.get("created"),
                       "devices": sorted((self._device_row(i, d) for i, d in e["devices"].items()),
                                         key=lambda r: r["first_seen"] or 0)}
                if e.get("legacy"):
                    row["legacy"] = True
                rows.append(row)
            return rows

    # ------------------------------------------------------------------------------------------ devices
    def device(self, ident: Identity, device_id: str, label: str, addr: str, now: float) -> str:
        """The state of the device this request came from, "approved", "pending" or "refused".  A device seen for the
        first time is recorded as pending with a short code the owner compares.  Devices are kept per key: an
        identity with no key entry (the unauthenticated LOCAL_GUEST) is approved and nothing is recorded."""
        if not self.path:
            return "approved"
        legacy_owner = ident == OWNER
        with self._lock:
            self._reload_locked()
            e = self._entry(self._data, ident.name)
            if e is None and not legacy_owner:            # no key, no devices: nothing to check, whatever the id
                return "approved"
            if not isinstance(device_id, str) or not 0 < len(device_id) <= 128:
                return "refused"
            label, addr = str(label or "")[:MAX_TEXT], str(addr or "")[:MAX_TEXT]
            d = e["devices"].get(device_id) if e else None
            if d is not None:
                d["last_seen"] = now
                if label:
                    d["label"] = label
                stale = now - (self._seen_saved.get((ident.name, device_id)) or 0) >= LAST_SEEN_EVERY_S
                moved = d.get("addr") != addr
                d["addr"] = addr
                status = d["status"]
                if stale or moved:
                    self._persist_locked()
                    self._seen_saved[(ident.name, device_id)] = now
                return status
        return self._new_device(ident, legacy_owner, device_id, label, addr, now)

    def _persist_locked(self) -> None:
        if self._bad:                                     # the in-memory state still answers; the file is not touched
            return
        try:
            self._save(self._data)
        except OSError as e:                              # a request is not failed for a note about it
            print(f"strata serve: the keys file {self.path} is not written ({e})", flush=True)

    def _new_device(self, ident, legacy_owner, device_id, label, addr, now) -> str:
        def go(data):
            e = self._entry(data, ident.name)
            if e is None:                                 # the legacy owner's reserved entry, made on first use
                if not legacy_owner:
                    return "approved"
                e = {"name": "owner", "legacy": True, "role": "owner", "created": now, "devices": {}}
                data["keys"].append(e)
            d = e["devices"].get(device_id)
            if d is not None:                             # another request got here first
                return d["status"]
            if len(e["devices"]) >= MAX_DEVICES:
                return "refused"
            e["devices"][device_id] = {"label": label, "addr": addr, "status": "pending",
                                       "code": "".join(secrets.choice(CODE_ALPHABET) for _ in range(4)),
                                       "first_seen": now, "last_seen": now, "decided_by": None}
            self._log(data, "pending", ident.name, device_id)
            return "pending"
        try:
            return self._change(go)
        except KeysFileBad:                               # cannot be recorded: held, not let in
            return "pending"
        except OSError as e:                              # not written (a full disk, a locked file): held, not let in
            print(f"strata serve: the keys file {self.path} is not written ({e}); a new device is held", flush=True)
            return "pending"

    def device_code(self, key_name: str, device_id: str) -> str | None:
        """The short code of a device the store holds (what the 403 `device_pending` answer shows and the owner
        compares), or None: not recorded, as while the keys file cannot be read or written."""
        with self._lock:
            e = self._entry(self._data, key_name)
            d = e["devices"].get(device_id) if e else None
            return d.get("code") if d else None

    def decide_device(self, key_name: str, device_id: str, approve: bool, by: Identity) -> bool:
        """Approve or refuse a device.  Who may do so is `may_approve`, asked by the route; one rule is kept here: a
        refusal is the owner's, so only an owner can approve a device that was refused."""
        def go(data):
            e = self._entry(data, key_name)
            d = e["devices"].get(device_id) if e else None
            if d is None:
                return False
            if approve and d["status"] == "refused" and not by.is_owner:
                return False
            d["status"] = "approved" if approve else "refused"
            d["decided_by"] = by.name
            self._log(data, d["status"], key_name, device_id, by.name)
            return True
        return self._change(go)

    def clear_device(self, key_name: str, device_id: str) -> bool:
        """Forget a device (a refused one is let in again, as pending, the next time it is seen)."""
        def go(data):
            e = self._entry(data, key_name)
            if e is None or device_id not in e["devices"]:
                return False
            del e["devices"][device_id]
            self._log(data, "cleared", key_name, device_id)
            return True
        return self._change(go)

    def pending(self) -> list[dict]:
        with self._lock:
            self._reload_locked()
            rows = [{"key": e["name"], "device": i, "label": d.get("label", ""), "addr": d.get("addr", ""),
                     "code": d.get("code", ""), "first_seen": d.get("first_seen")}
                    for e in self._data["keys"] for i, d in e["devices"].items() if d.get("status") == "pending"]
        return sorted(rows, key=lambda r: r["first_seen"] or 0)

    def may_approve(self, by: Identity, key_name: str, by_device: str | None) -> bool:
        """The owner always.  A guest only for their own key, and only from a device of that key that is already
        approved, so the first device of a key needs the owner and a key that leaks before its holder uses it is still
        caught."""
        if by.is_owner:
            return True
        if by.name != key_name or not by_device:
            return False
        with self._lock:
            self._reload_locked()
            e = self._entry(self._data, key_name)
            d = e["devices"].get(by_device) if e else None
            return bool(d and d.get("status") == "approved")


# ---------------------------------------------------------------------------------------------- what a request says
DEVICE_ID = re.compile(r"[A-Za-z0-9_-]{8,64}")              # X-Strata-Device: Strata Desktop's id, or the browser's
FORWARDING = ("Cf-Connecting-Ip", "X-Forwarded-For", "Forwarded")   # a proxy on this PC relayed the request


def _header_text(value) -> str:
    """A header's text: http.server reads each byte as one character, so a UTF-8 device name arrives as Latin-1."""
    if not isinstance(value, str):
        return ""
    try:
        value = value.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return "".join(c for c in value if ord(c) >= 32 and ord(c) != 127).strip()


def device_of(header_id, header_name, user_agent, addr: str) -> tuple[str, str]:
    """(device id, label) of a request.  The id is X-Strata-Device when it is 8-64 of [A-Za-z0-9_-]; anything else
    (no header, or a malformed one such as "addr:<an approved address>") is the device of its address, "addr:<ip>", so
    a client cannot pick the id of another PC's address.  The label is X-Strata-Device-Name, else the User-Agent
    (Claude Code sends no device header: "claude-cli/..." is what the owner sees)."""
    did = header_id if isinstance(header_id, str) and DEVICE_ID.fullmatch(header_id) else f"addr:{addr}"
    return did, (_header_text(header_name) or _header_text(user_agent))[:MAX_TEXT]


def _ip(text) -> str | None:
    t = (text or "").strip().strip('"')
    if t.startswith("["):                                 # "[2001:db8::17]:4711"
        t = t[1:t.find("]")] if "]" in t else ""
    elif t.count(":") == 1:                               # "192.0.2.60:8080"
        t = t.rsplit(":", 1)[0]
    try:
        return str(ipaddress.ip_address(t))
    except ValueError:
        return None


def device_addr(peer: str, peer_is_loopback: bool, cf_connecting_ip, x_forwarded_for, forwarded) -> str:
    """The address a device is recorded with.  The socket peer, unless the peer is this PC itself and a proxy on it
    (cloudflared, tailscale serve, a reverse proxy) named the client: Cf-Connecting-Ip, else the last X-Forwarded-For
    entry (the one the nearest proxy wrote; the first ones are the client's own words), else the last Forwarded
    element's for=.  A remote peer's forwarding headers are its own words and are ignored."""
    if peer_is_loopback:
        if (ip := _ip(cf_connecting_ip)) is not None:
            return ip
        if isinstance(x_forwarded_for, str) and (ip := _ip(x_forwarded_for.split(",")[-1])) is not None:
            return ip
        if isinstance(forwarded, str):
            for part in forwarded.split(",")[-1].split(";"):
                k, _, v = part.partition("=")
                if k.strip().lower() == "for" and (ip := _ip(v)) is not None:
                    return ip
    return peer


HIDDEN = frozenset(("preview", "input", "output", "reasoning", "response", "prompt", "messages"))


def visible_record(record: dict, viewer: Identity) -> dict:
    """A request record as `viewer` may see it.  The owner sees everything, and so does the person whose request it
    is; anyone else sees who, state, queue place and timings, but not what was asked or answered (the preview, the
    prompt and answer fields, and every key that holds a detail record)."""
    if viewer.is_owner or record.get("who") == viewer.name:
        return record
    return {k: v for k, v in record.items() if k not in HIDDEN and "detail" not in k.lower()}
