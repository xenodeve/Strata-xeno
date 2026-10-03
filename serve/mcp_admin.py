"""serve/mcp_admin.py - setting up MCP servers from the web app (issue #79).

A program server is a program Strata starts, so changing the list is a write that runs commands on this PC. Who may do it
(`may_edit`), what is accepted (`validate_entry`, `validate_settings`), what the browser is shown (`public_entry`: secrets as a
mask) and how the run config is written back (`write_config`: other keys kept, a one-time backup, a temporary file) are in this
module as plain functions; serve/server.py only wires `GET /mcp/config` and `POST /mcp/config` to them.
"""
from __future__ import annotations

import atexit
import copy
import ipaddress
import json
import os
import re
import shutil
import tempfile
import threading
from pathlib import Path

from serve.mcp import DEFAULTS, hub_from_config

MASK = "********"                      # what a secret (an `env` or `headers` value) is shown as; sent back, it keeps the stored one
NAME = re.compile(r"^[A-Za-z0-9_.-]{1,40}$")
MAX_SERVERS, MAX_ARGS, MAX_TEXT = 50, 64, 4096
LIMITS = {"timeout_s": (1.0, 600.0), "max_result_chars": (1000, 500000), "max_rounds": (1, 30)}
_lock = threading.Lock()               # one change at a time: a write and the restart of the hub that follows it


# ------------------------------------------------------------------------------------------------ who may change them
def is_loopback(addr: str) -> bool:
    """Whether `addr` (an IP address as a socket gives it) is this PC itself."""
    try:
        ip = ipaddress.ip_address((addr or "").split("%")[0])
    except ValueError:
        return False
    return (ip.ipv4_mapped or ip).is_loopback if ip.version == 6 else ip.is_loopback


def host_is_loopback(host: str) -> bool:
    """Whether the Host a request names is this PC's loopback (`localhost`, 127.x, [::1]), with or without a port."""
    h = (host or "").strip().lower()
    if h.startswith("["):
        name = h[1:h.find("]")] if "]" in h else h
    else:
        name = h.rsplit(":", 1)[0] if h.count(":") == 1 else h
    name = name.rstrip(".")
    return name == "localhost" or is_loopback(name)


def may_edit(api_key_set: bool, client_ip: str, host: str) -> tuple[bool, str]:
    """(allowed, why not). With an API key set the key is the gate (the caller has sent it by the time this is asked);
    without one, only a request from this PC that also names this PC (a page on another site whose name was re-pointed at
    this PC names that site as Host) may change what programs Strata starts."""
    if api_key_set:
        return True, ""
    if is_loopback(client_ip) and host_is_loopback(host):
        return True, ""
    return False, ("MCP servers start programs on this PC, so they can be changed only from this PC itself (open the page "
                   "as localhost) or when the server has an API key and the request carries it; set \"api_key\" in the run config")


# ------------------------------------------------------------------------------------------------ what is accepted
def _err(field: str, message: str) -> dict:
    return {"field": field, "message": message}


def _text_map(value, field: str, errors: list) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, dict) or len(value) > MAX_ARGS or not all(
            isinstance(k, str) and k and isinstance(v, str) and len(k) <= 256 and len(v) <= MAX_TEXT for k, v in value.items()):
        errors.append(_err(field, f"{field} must be an object of text values (at most {MAX_ARGS})"))
        return None
    return dict(value)


def validate_entry(name: str, entry) -> tuple[dict, list[dict]]:
    """(the entry as it will be stored, [{"field", "message"}]).  A program ("command", "args", "env", "cwd") or an
    address ("url", "headers"), never both; `"disabled": true` keeps one in the file without starting it."""
    errors: list[dict] = []
    if not isinstance(name, str) or not NAME.match(name):
        errors.append(_err("name", "a name is 1-40 letters, digits, '.', '_' or '-' (no spaces)"))
    if not isinstance(entry, dict):
        return {}, errors + [_err("command", "a server is an object with \"command\" or \"url\"")]
    command, url = entry.get("command"), entry.get("url")
    out: dict = {}
    if command not in (None, "") and url not in (None, ""):
        errors += [_err("command", "a program or an address, not both"), _err("url", "a program or an address, not both")]
    elif url not in (None, ""):
        if not isinstance(url, str) or not re.match(r"^https?://[^\s]+$", url) or len(url) > 2048:
            errors.append(_err("url", "the address must start with http:// or https://"))
        elif entry.get("type") == "sse":
            errors.append(_err("url", "the old SSE transport is not supported: Strata speaks Streamable HTTP (often at /mcp)"))
        else:
            out["url"] = url
        headers = _text_map(entry.get("headers"), "headers", errors)
        if headers:
            out["headers"] = headers
    else:
        if not isinstance(command, str) or not command.strip() or "\n" in command or "\r" in command or len(command) > MAX_TEXT:
            errors.append(_err("command", "the program to start is required (one line)"))
        else:
            out["command"] = command.strip()
        args = entry.get("args")
        if args is not None:
            if not isinstance(args, list) or len(args) > MAX_ARGS or not all(isinstance(a, str) and len(a) <= MAX_TEXT for a in args):
                errors.append(_err("args", f"args is a list of at most {MAX_ARGS} text items"))
            elif args:
                out["args"] = list(args)
        env = _text_map(entry.get("env"), "env", errors)
        if env:
            out["env"] = env
        cwd = entry.get("cwd")
        if cwd not in (None, ""):
            if not isinstance(cwd, str) or len(cwd) > MAX_TEXT or "\n" in cwd:
                errors.append(_err("cwd", "the folder is one line of text"))
            else:
                out["cwd"] = cwd
    if entry.get("disabled") is True:
        out["disabled"] = True
    return out, errors


def validate_settings(settings) -> tuple[dict, list[dict]]:
    """The MCP limits that may be set here: timeout_s, max_result_chars, max_rounds, each inside its range."""
    out, errors = {}, []
    if not isinstance(settings, dict):
        return {}, [_err("settings", "settings is an object")]
    for key, value in settings.items():
        number = isinstance(value, (int, float)) and not isinstance(value, bool)
        if key not in LIMITS:
            errors.append(_err(key, "not a setting that can be changed here"))
            continue
        lo, hi = LIMITS[key]
        if not number or not lo <= value <= hi or (key != "timeout_s" and value != int(value)):
            errors.append(_err(key, f"{key} is a {'number of seconds' if key == 'timeout_s' else 'whole number'} from {lo:g} to {hi:g}"))
        else:
            out[key] = float(value) if key == "timeout_s" else int(value)
    return out, errors


# ------------------------------------------------------------------------------------------------ secrets
def public_entry(stored: dict) -> dict:
    """An entry as the browser sees it: the values of `env` and `headers` are a mask, never the secret."""
    out = copy.deepcopy(stored)
    for key in ("env", "headers"):
        if isinstance(out.get(key), dict):
            out[key] = {k: MASK for k in out[key]}
        else:
            out.pop(key, None)
    return out


def merge_masked(sent: dict, stored: dict | None) -> dict:
    """The entry sent by the browser with every masked value replaced by the stored one (a mask with nothing stored behind it
    is dropped); a value that is not the mask replaces the stored one."""
    out = copy.deepcopy(sent)
    for key in ("env", "headers"):
        if isinstance(out.get(key), dict):
            old = (stored or {}).get(key)
            old = old if isinstance(old, dict) else {}
            out[key] = {k: (old[k] if v == MASK and k in old else v) for k, v in out[key].items() if not (v == MASK and k not in old)}
    return out


# ------------------------------------------------------------------------------------------------ the run config
def _read_json(path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def _servers_in(cfg: dict) -> dict:
    """The servers a run config lists, as the hub reads them (mcp_servers, then mcpServers over it)."""
    out = {}
    for key in ("mcp_servers", "mcpServers"):
        if isinstance(cfg.get(key), dict):
            out.update(cfg[key])
    return out


def write_config(path, servers: dict, settings: dict) -> None:
    """Write `servers` (and the limits in `settings`) into the run config at `path`, keeping every other key.  The key is the
    one the file already uses (`mcpServers` if that is the only one, else `mcp_servers`).  The original is copied once to
    `<path>.bak-mcp`; the new file is written beside it and moved over, so a failed write leaves the file as it was."""
    path = Path(path)
    cfg = _read_json(path)
    key = "mcpServers" if "mcpServers" in cfg and "mcp_servers" not in cfg else "mcp_servers"
    cfg.pop("mcpServers" if key == "mcp_servers" else "mcp_servers", None)
    cfg[key] = servers
    if settings:
        cfg["mcp"] = {**(cfg["mcp"] if isinstance(cfg.get("mcp"), dict) else {}), **settings}
    _save(path, cfg)


def update_config(path, mutate) -> None:
    """Read the run config, let `mutate(cfg)` change it, and write it back the same careful way (other keys kept, one-time backup, atomic)."""
    path = Path(path)
    cfg = _read_json(path)
    mutate(cfg)
    _save(path, cfg)


def _save(path: Path, cfg: dict) -> None:
    text = json.dumps(cfg, indent=2, ensure_ascii=False) + "\n"       # a value that is not JSON fails here, before any file moves
    bak = Path(str(path) + ".bak-mcp")
    if not bak.exists():
        shutil.copy2(path, bak)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        try:
            shutil.copymode(path, tmp)
        except OSError:
            pass
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ------------------------------------------------------------------------------------------------ what the page is shown
def _kind(entry) -> str:
    if not isinstance(entry, dict):
        return "invalid"
    return "address" if entry.get("url") else "program"


def read_view(config_path: str | None, mcp_config_path: str | None) -> dict:
    """{"servers": [...]} - every configured server without its state: from the run config (editable) and from the
    --mcp-config file (shown, not edited; it wins a name, as the hub has it)."""
    config, other = {}, {}
    if config_path:
        try:
            config = _servers_in(_read_json(config_path))
        except (OSError, ValueError):
            pass
    if mcp_config_path:
        try:
            data = _read_json(mcp_config_path)
            block = data.get("mcpServers", data.get("mcp_servers"))
            other = block if isinstance(block, dict) else {}
        except (OSError, ValueError):
            pass

    def row(name, e, **extra):
        return {"name": str(name), "kind": _kind(e), **(public_entry(e) if isinstance(e, dict) else {}),
                "disabled": isinstance(e, dict) and e.get("disabled") is True, **extra}
    # a name in both: the file's entry is the one that runs (as the hub has it); the run config's stays listed so a save keeps it
    servers = [row(n, e, source="config", editable=True, overridden=str(n) in other) for n, e in config.items()]
    servers += [row(n, e, source="file", editable=False, shadowed=str(n) in config, file=os.path.basename(mcp_config_path or ""))
                for n, e in other.items()]
    return {"servers": servers}


def view(svc, client_ip: str, host: str) -> dict:
    """The answer of GET /mcp/config: the servers with their state and tools, the limits, and whether (and, if not, why)
    this caller may change them."""
    v = read_view(svc.config_path, svc.mcp_config_path)
    status = svc.mcp.status() if svc.mcp else {"servers": [], "tools": 0}
    live = {s["name"]: s for s in status["servers"]}
    for s in v["servers"]:
        r = live.get(s["name"])
        s["status"] = "disabled" if s["disabled"] else (r["status"] if r else "idle")
        s["error"] = r["error"] if r else None
        s["tools"] = r["tools"] if r else []
        s["info"] = r["info"] if r else {}
    ok, why = may_edit(bool(svc.api_key), client_ip, host)
    if not ok:                                               # how a server is started (a token can sit in an argument or a URL) is for those who may change it
        for s in v["servers"]:
            for k in ("command", "args", "cwd", "url", "env", "headers"):
                s.pop(k, None)
    if ok and not svc.config_path:
        ok, why = False, "this server was started without a run config file (--config), so there is nowhere to save the servers"
    v.update(editable=ok, reason=why, settings={k: (svc.mcp.settings if svc.mcp else DEFAULTS)[k] for k in LIMITS},
             config_file=os.path.basename(svc.config_path) if svc.config_path else None, tools=status["tools"])
    return v


def reload_hub(svc) -> None:
    """Close the running hub and start one from the run config (and --mcp-config) as they are now.  ValueError if they
    do not make a hub (the servers were checked before they were written, so this is a file edited by hand)."""
    cfg = _read_json(svc.config_path) if svc.config_path else {}
    try:
        importer = getattr(svc, "importer", None)                  # the skills of the other apps (serve/harness.py) are a built-in server
        builtins = dict(importer.builtins()) if importer else {}
        if getattr(svc, "agent", None) is not None:
            builtins["agent"] = svc.agent                        # the chat's coding tools (serve/agent.py) are a built-in server too
        hub = hub_from_config(cfg, svc.mcp_config_path, builtins=builtins or None)
    except SystemExit as e:                                  # hub_from_config stops the process at start-up; here it must not
        raise ValueError(str(e)) from None
    old, svc.mcp = svc.mcp, hub
    if old is not None:
        old.close()
    if hub is not None:
        hub.start()
    if not getattr(svc, "_mcp_atexit", False):               # the servers Strata started end with it, whichever hub is current
        svc._mcp_atexit = True
        atexit.register(lambda: svc.mcp.close() if svc.mcp else None)


def apply(svc, body) -> tuple[int, dict]:
    """POST /mcp/config (who may ask has been decided): validate, write the run config, restart the hub.
    Returns (status, what to send).  `body`: {"servers": {name: entry}, "settings": {...}}; a server not listed is deleted."""
    if not isinstance(body, dict) or not isinstance(body.get("servers"), dict):
        return 400, {"error": {"message": "send {\"servers\": {name: {...}}}", "fields": []}}
    if len(body["servers"]) > MAX_SERVERS:
        return 400, {"error": {"message": f"at most {MAX_SERVERS} servers", "fields": []}}
    errors, servers = [], {}
    with _lock:
        try:
            stored = _servers_in(_read_json(svc.config_path))
        except (OSError, ValueError) as e:
            return 500, {"error": {"message": f"the run config could not be read: {e}"}}
        for name, sent in body["servers"].items():
            if isinstance(sent, dict) and isinstance(name, str):
                sent = merge_masked(sent, stored.get(name) if isinstance(stored.get(name), dict) else None)
            entry, errs = validate_entry(name, sent)
            errors += [{"server": name, **e} for e in errs]
            servers[name] = entry
        settings, errs = validate_settings(body.get("settings", {}))
        errors += [{"server": None, **e} for e in errs]
        if errors:
            return 400, {"error": {"message": f"{errors[0]['message']} ({errors[0]['server'] or 'settings'}: {errors[0]['field']})",
                                   "fields": errors}}
        try:
            write_config(svc.config_path, servers, settings)
            reload_hub(svc)
        except (OSError, ValueError, TypeError) as e:
            return 500, {"error": {"message": f"the servers could not be saved: {e}"}}
    return 200, {"saved": True}
