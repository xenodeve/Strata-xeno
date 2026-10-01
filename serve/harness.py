"""serve/harness.py - the skills and the MCP servers of the other coding apps ("harnesses") on this PC (issue #94).

Read-only: it reads where Claude Code, Codex, Antigravity, Claude Desktop, Gemini CLI, Cursor and the shared ~/.agents folder keep them
and never writes there. The folder it reads (`home`, `appdata`) is a parameter, so tests and mock servers point at a fake one;
`home_dir()` is the one place that decides, from STRATA_HOME or the user's own.

Skills are imported by default and switched off by three off-lists in the run config (`import.skills`): the whole import, an app, a
skill; an item found later is on. MCP servers are only listed here; importing one is a click (see `import_server`).
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path

from serve import mcp_admin
from serve.skills import SkillsServer

try:
    import tomllib                                          # Python 3.11+; without it the Codex file is reported, not read
except ModuleNotFoundError:                                  # pragma: no cover
    tomllib = None

# the apps, in the order that decides whose copy of a skill that is in several apps is the one used
HARNESSES: list[dict] = [
    {"id": "claude", "label": "Claude Code"},
    {"id": "claude-desktop", "label": "Claude Desktop"},
    {"id": "codex", "label": "Codex"},
    {"id": "agents", "label": "Shared agents folder"},
    {"id": "antigravity", "label": "Antigravity"},
    {"id": "gemini", "label": "Gemini CLI"},
    {"id": "cursor", "label": "Cursor"},
]
IDS = [h["id"] for h in HARNESSES]
SKILL_ROOTS = {"claude": ".claude/skills", "codex": ".codex/skills", "agents": ".agents/skills", "antigravity": ".gemini/antigravity/skills",
               "gemini": ".gemini/skills", "cursor": ".cursor/skills"}
SKILL_READ = 8192                                           # the front matter is read from the first bytes: never 500 whole files


def home_dir(env=None) -> Path:
    env = os.environ if env is None else env
    return Path(env["STRATA_HOME"]) if env.get("STRATA_HOME") else Path.home()


def appdata_dir(home: Path, env=None) -> Path:
    env = os.environ if env is None else env
    if env.get("APPDATA"):
        return Path(env["APPDATA"])
    return home / "AppData" / "Roaming"


# ------------------------------------------------------------------------------------------------ SKILL.md front matter
def _unquote(v: str) -> str:
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        return v[1:-1]
    return v


def parse_front(text: str) -> dict:
    """The `name` and `description` (and the other top-level scalar keys) of a SKILL.md's front matter: the YAML subset skills use -
    plain, quoted, folded (>) and literal (|) values, and a value that goes on over indented lines. {} when there is none."""
    text = text.lstrip("﻿")
    if not text.startswith("---"):
        return {}
    lines = text.split("\n")
    end = next((i for i in range(1, len(lines)) if lines[i].rstrip("\r") == "---"), None)
    if end is None:
        return {}
    out: dict[str, str] = {}
    i = 1
    while i < end:
        line = lines[i].rstrip("\r")
        m = re.match(r"^([A-Za-z0-9_-]+):\s*(.*)$", line)
        if not m:
            i += 1
            continue
        key, val = m.group(1), m.group(2)
        i += 1
        cont: list[str] = []
        while i < end and (lines[i].startswith((" ", "\t")) or not lines[i].strip()):
            cont.append(lines[i].rstrip("\r"))
            i += 1
        while cont and not cont[-1].strip():
            cont.pop()
        if val and val[0] in ">|":
            body = [c.strip() for c in cont]
            out[key] = (" " if val[0] == ">" else "\n").join(b for b in body if b or val[0] == "|").strip()
        elif cont:
            out[key] = " ".join([_unquote(val)] + [c.strip() for c in cont if c.strip()]) if val else ""
            if not val:
                out.pop(key)                                  # a key that opens a nested block (metadata:) has no scalar value
        else:
            out[key] = _unquote(val)
    return out


# ------------------------------------------------------------------------------------------------ discovery
def _show(p: Path, home: Path, appdata: Path) -> str:
    for base, label in ((home, "~"), (appdata, "%APPDATA%")):
        try:
            return label + "/" + p.relative_to(base).as_posix()
        except ValueError:
            continue
    return p.as_posix()


def _skills_in(root: Path, harness: str, origin: str, errors: list, seen_real: set) -> list[dict]:
    try:
        entries = sorted(root.iterdir(), key=lambda p: p.name.lower())
    except OSError as e:
        errors.append(f"{root}: {e.strerror or e}")
        return []
    out = []
    for e in entries:
        try:
            if not e.is_dir() or not (e / "SKILL.md").is_file():
                continue
            real = os.path.realpath(e)
            if real in seen_real:                             # a link to a folder that is already listed
                continue
            seen_real.add(real)
            with open(e / "SKILL.md", "rb") as f:
                head = f.read(SKILL_READ).decode("utf-8", "replace")
        except OSError as err:
            errors.append(f"{e}: {err.strerror or err}")
            continue
        fm = parse_front(head)
        name = (fm.get("name") or e.name).strip() or e.name
        out.append({"id": f"{harness}:{name}", "harness": harness, "name": name, "description": (fm.get("description") or "").strip(),
                    "origin": origin, "dir": str(e), "real": real})
    return out


def _read_json(path: Path, errors: list):
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as e:
        errors.append(f"{path.name}: {e}")
        return None
    return data if isinstance(data, dict) else None


def _claude_plugin_skills(home: Path, errors: list, seen_real: set) -> list[dict]:
    """The skills of the plugins Claude Code has enabled (settings.json `enabledPlugins`), from where each was installed."""
    plugins = home / ".claude" / "plugins" / "installed_plugins.json"
    if not plugins.is_file():
        return []
    installed = _read_json(plugins, errors)
    settings = _read_json(home / ".claude" / "settings.json", errors) if (home / ".claude" / "settings.json").is_file() else None
    enabled = (settings or {}).get("enabledPlugins")
    if not isinstance(installed, dict) or not isinstance(installed.get("plugins"), dict) or not isinstance(enabled, dict):
        return []
    out = []
    for key, installs in installed["plugins"].items():
        if enabled.get(key) is not True or not isinstance(installs, list):
            continue
        for inst in installs:
            path = inst.get("installPath") if isinstance(inst, dict) else None
            if isinstance(path, str) and (Path(path) / "skills").is_dir():      # most plugins are about something else
                out += _skills_in(Path(path) / "skills", "claude", "plugin:" + key.split("@")[0], errors, seen_real)
    return out


def _mcp_file(h: str, home: Path, appdata: Path) -> tuple[Path, str] | None:
    return {
        "claude": (home / ".claude.json", "json"), "claude-desktop": (appdata / "Claude" / "claude_desktop_config.json", "json"),
        "codex": (home / ".codex" / "config.toml", "toml"), "gemini": (home / ".gemini" / "settings.json", "json"),
        "antigravity": (home / ".gemini" / "antigravity" / "mcp_config.json", "json"), "cursor": (home / ".cursor" / "mcp.json", "json"),
    }.get(h)


def _servers_of(path: Path, kind: str, errors: list) -> dict:
    try:
        if kind == "toml":
            if tomllib is None:
                errors.append(f"{path.name}: this Python cannot read TOML (3.11 or later is needed)")
                return {}
            data = tomllib.loads(path.read_text(encoding="utf-8-sig")).get("mcp_servers")
        else:
            data = (_read_json(path, errors) or {}).get("mcpServers")
    except (OSError, ValueError) as e:
        errors.append(f"{path.name}: {e}")
        return {}
    return {str(n): c for n, c in data.items() if isinstance(c, dict)} if isinstance(data, dict) else {}


def scan(home: Path, appdata: Path | None = None) -> dict:
    """{app id: {"label", "found", "files" (what was read, as ~/...), "skills", "servers" (name -> the entry as that app has it), "errors"}}.
    Nothing here raises for a missing or broken file: it is an entry in `errors`."""
    home = Path(home)
    appdata = Path(appdata) if appdata else appdata_dir(home)
    out = {}
    for h in HARNESSES:
        hid = h["id"]
        files: list[str] = []
        errors: list[str] = []
        skills: list[dict] = []
        servers: dict = {}
        seen: set = set()
        if hid in SKILL_ROOTS and (home / SKILL_ROOTS[hid]).is_dir():
            files.append(_show(home / SKILL_ROOTS[hid], home, appdata))
            skills += _skills_in(home / SKILL_ROOTS[hid], hid, "user", errors, seen)
        if hid == "claude":
            plug = _claude_plugin_skills(home, errors, seen)
            if plug:
                files.append(_show(home / ".claude" / "plugins", home, appdata))
            skills += plug
        mf = _mcp_file(hid, home, appdata)
        if mf and mf[0].is_file():
            files.append(_show(mf[0], home, appdata))
            servers = _servers_of(mf[0], mf[1], errors)
        seen_names: set = set()
        unique = []
        for s in skills:                                      # one skill of a name per app (the first folder wins)
            if s["name"] not in seen_names:
                seen_names.add(s["name"])
                unique.append(s)
        out[hid] = {"label": h["label"], "found": bool(files), "files": files, "skills": unique, "servers": servers, "errors": errors}
    return out


# ------------------------------------------------------------------------------------------------ the skill switches
def skill_settings(cfg) -> dict:
    """The run config's `import.skills`, read defensively: {"enabled": bool, "harness_off": [app ids], "off": {app id: [skill names]}}.
    Everything is on until it is in an off-list; anything that is not what it should be is ignored."""
    sk = (cfg.get("import") or {}).get("skills") if isinstance(cfg, dict) and isinstance(cfg.get("import"), dict) else None
    sk = sk if isinstance(sk, dict) else {}
    hoff = sk.get("harness_off")
    off = sk.get("off")
    return {
        "enabled": sk["enabled"] if isinstance(sk.get("enabled"), bool) else True,
        "harness_off": [h for h in hoff if h in IDS] if isinstance(hoff, list) else [],
        "off": {h: [n for n in v if isinstance(n, str)] for h, v in off.items() if h in IDS and isinstance(v, list)} if isinstance(off, dict) else {},
    }


def _eligible(s: dict, st: dict) -> bool:
    return st["enabled"] and s["harness"] not in st["harness_off"] and s["name"] not in st["off"].get(s["harness"], [])


def select_skills(found: dict, st: dict) -> list[dict]:
    """The skills in use: those not switched off, one of each name (the first app's copy that is on)."""
    chosen: dict[str, dict] = {}
    for hid in IDS:
        for s in found.get(hid, {}).get("skills", []):
            if _eligible(s, st) and s["name"] not in chosen:
                chosen[s["name"]] = s
    return list(chosen.values())


def skill_items(found: dict, st: dict) -> list[dict]:
    """Every skill found, for the page: whether its own switch is off, whether it is the copy in use, and which one is when it is a copy."""
    used = {s["id"]: s for s in select_skills(found, st)}
    by_name = {s["name"]: s["id"] for s in used.values()}
    items = []
    for hid in IDS:
        for s in found.get(hid, {}).get("skills", []):
            off = s["name"] in st["off"].get(hid, [])
            is_used = s["id"] in used
            same = by_name.get(s["name"]) if not is_used and not off and hid not in st["harness_off"] and st["enabled"] else None
            items.append({"id": s["id"], "harness": hid, "name": s["name"], "description": s["description"], "origin": s["origin"],
                          "off": off, "used": is_used, "same_as": same})
    return items


# ------------------------------------------------------------------------------------------------ MCP servers: read, listed, imported by a click
class ImportRefused(Exception):
    """A server cannot be imported: `code` is "missing", "unsupported" or "already"."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand(value, environ):
    """`${NAME}` and `${NAME:-default}` filled from the environment (as Claude Code does); one that is unset and has no default stays as written."""
    if not isinstance(value, str):
        return value
    return _VAR.sub(lambda m: environ[m.group(1)] if m.group(1) in environ else (m.group(2) if m.group(2) is not None else m.group(0)), value)


def _strmap(value, environ) -> dict:
    return {str(k): _expand(v, environ) for k, v in value.items() if isinstance(v, str)} if isinstance(value, dict) else {}


def translate(harness_id: str, name: str, raw, environ=None) -> dict:
    """One server as another app keeps it -> {"entry": Strata's entry or None, "reason": why not, "disabled": off in its own app, "note"}.
    The entry is checked with the same rules as one typed in Settings (mcp_admin.validate_entry)."""
    environ = os.environ if environ is None else environ
    none = lambda reason: {"entry": None, "reason": reason, "disabled": False, "note": None}      # noqa: E731
    if not isinstance(raw, dict):
        return none("not a server entry")
    disabled = raw.get("disabled") is True or raw.get("enabled") is False
    typ = str(raw.get("type") or "").lower()
    url = raw.get("url") or (raw.get("serverUrl") if harness_id == "antigravity" else None)
    if harness_id == "gemini" and raw.get("httpUrl"):
        url = raw["httpUrl"]                                  # Gemini CLI: `url` is the old SSE transport, `httpUrl` Streamable HTTP
    elif harness_id == "gemini" and url and typ not in ("http", "streamable-http"):
        return none("it speaks SSE only; Strata speaks Streamable HTTP")
    if typ == "sse":
        return none("it speaks SSE only; Strata speaks Streamable HTTP")
    note = None
    entry: dict = {}
    if isinstance(url, str) and url:
        entry["url"] = _expand(url, environ)
        headers = _strmap(raw.get("headers") or raw.get("http_headers"), environ)
        var = raw.get("bearer_token_env_var")
        if isinstance(var, str) and var:
            if environ.get(var):
                headers["Authorization"] = "Bearer " + environ[var]
            else:
                note = f"it takes its token from the environment variable {var}, which is not set here"
        if headers:
            entry["headers"] = headers
    elif isinstance(raw.get("command"), str) and raw["command"].strip():
        entry["command"] = _expand(raw["command"], environ)
        if isinstance(raw.get("args"), list):
            entry["args"] = [_expand(a, environ) for a in raw["args"] if isinstance(a, (str, int, float)) and not isinstance(a, bool)]
            entry["args"] = [a if isinstance(a, str) else str(a) for a in entry["args"]]
        env = _strmap(raw.get("env"), environ)
        if env:
            entry["env"] = env
        if isinstance(raw.get("cwd"), str) and raw["cwd"].strip():
            entry["cwd"] = _expand(raw["cwd"], environ)
    else:
        return {**none("no command or address to run"), "disabled": disabled}
    clean, errors = mcp_admin.validate_entry("x", entry)
    if errors:
        return {**none(f"{errors[0]['field']}: {errors[0]['message']}"), "disabled": disabled}
    return {"entry": clean, "reason": None, "disabled": disabled, "note": note}


def fingerprint(entry: dict) -> tuple:
    """What makes two servers the same one, whatever the app calls it: the address, or the program (its file name) and its arguments."""
    if entry.get("url"):
        return ("a", str(entry["url"]).rstrip("/").lower())
    cmd = os.path.basename(str(entry.get("command", "")).replace("\\", "/")).lower()
    for ext in (".cmd", ".exe", ".bat", ".ps1"):
        if cmd.endswith(ext):
            cmd = cmd[: -len(ext)]
    return ("p", cmd, tuple(entry.get("args") or []))


def mcp_candidates(found: dict, own: dict, environ=None) -> dict:
    """{app id: [one item per server the app has]}: its state ("available", "already" in Strata's own servers, "unsupported"), why, whether
    the app has it switched off, and (kept apart, never sent whole to a page) the `entry` that would be imported."""
    own_fp = {}
    for n, e in own.items():
        if isinstance(e, dict) and (e.get("url") or e.get("command")):
            own_fp.setdefault(fingerprint(e), n)
    out: dict[str, list] = {}
    for hid in IDS:
        items = []
        for name, raw in sorted(found.get(hid, {}).get("servers", {}).items()):
            t = translate(hid, name, raw, environ)
            kind = ("address" if (t["entry"] or {}).get("url") else "program") if t["entry"] else ("address" if isinstance(raw, dict) and (raw.get("url") or raw.get("httpUrl") or raw.get("serverUrl")) else "program" if isinstance(raw, dict) and raw.get("command") else None)
            fp = fingerprint(t["entry"]) if t["entry"] else None
            state = "unsupported" if t["entry"] is None else "already" if fp in own_fp else "available"
            items.append({"id": f"{hid}:{name}", "harness": hid, "name": name, "kind": kind, "state": state, "reason": t["reason"], "note": t["note"],
                          "disabled_in_source": t["disabled"], "already_as": own_fp.get(fp) if state == "already" else None, "entry": t["entry"]})
        out[hid] = items
    return out


def public_candidate(item: dict, details: bool) -> dict:
    """An item as a page may see it: always the name, kind and state; for a caller who may change the settings also the command line or
    the address (a token can sit in either), with the values of `env` and `headers` masked. Never the entry itself."""
    shown = {k: item[k] for k in ("id", "harness", "name", "kind", "state", "reason", "note", "disabled_in_source", "already_as")}
    if details and item.get("entry"):
        masked = mcp_admin.public_entry(item["entry"])
        shown.update({k: masked[k] for k in ("command", "args", "cwd", "url", "env", "headers") if k in masked})
    return shown


def _clean_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "-", text).strip("-.")[:40] or "server"


def import_server(own: dict, harness_id: str, name: str, found: dict, environ=None) -> tuple[dict, str]:
    """Strata's servers (`own`) with the server `name` of app `harness_id` added, and the name it was given (the app's name is added to
    one that is taken). Raises ImportRefused. `own` is not changed; nothing is written here."""
    cand = next((c for c in mcp_candidates(found, own, environ).get(harness_id, []) if c["name"] == name), None)
    if cand is None:
        raise ImportRefused("missing", f"there is no server {name!r} in {harness_id}")
    if cand["state"] == "unsupported":
        raise ImportRefused("unsupported", cand["reason"] or "it cannot be imported")
    if cand["state"] == "already":
        raise ImportRefused("already", f"Strata already has it, as {cand['already_as']!r}")
    final = _clean_name(name)
    if final in own:
        base = _clean_name(f"{name}-{harness_id}")
        final, n = base, 2
        while final in own:
            suffix = f"-{n}"
            final, n = base[: 40 - len(suffix)] + suffix, n + 1
    entry = dict(cand["entry"])
    if cand["disabled_in_source"]:
        entry["disabled"] = True
    return {**own, final: entry}, final


# ------------------------------------------------------------------------------------------------ what a running Strata holds
class Importer:
    """The last scan, and the built-in skills server that follows the skill switches. One per Strata; the folder it reads is its `home`."""

    def __init__(self, home=None, appdata=None, environ=None):
        self.environ = os.environ if environ is None else environ
        self.home = Path(home) if home else home_dir(self.environ)
        self.appdata = Path(appdata) if appdata else appdata_dir(self.home, self.environ)
        self.found: dict = {}
        self.skills = SkillsServer([])
        self.lock = threading.Lock()

    def rescan(self, cfg: dict) -> None:
        with self.lock:
            self.found = scan(self.home, self.appdata)
        self.apply(cfg)

    def apply(self, cfg: dict) -> None:
        """The skill switches changed (or the scan did): the skills in use follow, with no restart of anything."""
        self.skills.set_skills(select_skills(self.found, skill_settings(cfg)))

    def builtins(self) -> dict:
        return {"skills": self.skills}


# ------------------------------------------------------------------------------------------------ the page's side: GET /import and POST /import
NO_FILE = "this server was started without a run config file (--config), so there is nowhere to save the import settings"


def _cfg(svc) -> dict:
    try:
        return mcp_admin._read_json(svc.config_path) if svc.config_path else {}
    except (OSError, ValueError):
        return {}


def view(svc, client_ip: str, host: str, rescan: bool = False) -> dict:
    """The answer of GET /import: the apps found with what was read, every skill with its switches, every MCP server by app with its state,
    and whether this caller may change anything (and, if not, why). A caller who may not sees names only: not a command line or an address."""
    imp = getattr(svc, "importer", None)
    if imp is None:
        return {"available": False}
    cfg = _cfg(svc)
    if rescan:
        imp.rescan(cfg)
    ok, why = mcp_admin.may_edit(bool(svc.api_key), client_ip, host)
    if ok and not svc.config_path:
        ok, why = False, NO_FILE
    st = skill_settings(cfg)
    items = skill_items(imp.found, st)
    cands = mcp_candidates(imp.found, mcp_admin._servers_in(cfg), imp.environ)
    harnesses = []
    for h in HARNESSES:
        f = imp.found.get(h["id"], {})
        mine = [i for i in items if i["harness"] == h["id"]]
        harnesses.append({"id": h["id"], "label": h["label"], "found": bool(f.get("found")), "files": f.get("files", []),
                          "errors": [str(e)[:200] for e in f.get("errors", [])], "skills": len(mine), "skills_used": sum(1 for i in mine if i["used"]),
                          "skills_off": h["id"] in st["harness_off"], "servers": len(f.get("servers", {}))})
    return {
        "available": True, "editable": ok, "reason": why, "config_file": os.path.basename(svc.config_path) if svc.config_path else None,
        "harnesses": harnesses,
        "skills": {"settings": st, "used": sum(1 for i in items if i["used"]), "total": len(items), "items": items},
        "mcp": {"harnesses": [{"id": h["id"], "label": h["label"], "servers": [public_candidate(c, ok) for c in cands[h["id"]]]} for h in HARNESSES if cands[h["id"]]]},
    }


def _check_skills(value) -> tuple[dict | None, list[dict]]:
    """The skill switches a page sends, checked strictly (the saved file is read leniently, see skill_settings): (the settings, []) or (None, errors)."""
    err = lambda field, message: {"field": field, "message": message}      # noqa: E731
    if not isinstance(value, dict):
        return None, [err("skills", "skills is an object")]
    errors = [err(k, "not a setting that can be changed here") for k in value if k not in ("enabled", "harness_off", "off")]
    if "enabled" in value and not isinstance(value["enabled"], bool):
        errors.append(err("enabled", "enabled is true or false"))
    off = value.get("harness_off", [])
    if not isinstance(off, list) or len(off) > len(IDS) or not all(isinstance(x, str) and x in IDS for x in off):
        errors.append(err("harness_off", "harness_off is a list of the apps listed here"))
    items = value.get("off", {})
    if not isinstance(items, dict) or not all(h in IDS and isinstance(v, list) and len(v) <= 5000 and all(isinstance(n, str) and 0 < len(n) <= 200 for n in v)
                                              for h, v in items.items()):
        errors.append(err("off", "off is {app: [skill names]} for the apps listed here"))
    if errors:
        return None, errors
    return skill_settings({"import": {"skills": value}}), []


def apply(svc, body) -> tuple[int, dict | None]:
    """POST /import (who may ask has been decided): the skill switches, one MCP server to import, a rescan - in that order, any of them.
    (200, None) when done (the caller answers with `view`), else (status, what to send)."""
    imp = getattr(svc, "importer", None)
    if imp is None:
        return 404, {"error": {"message": "importing from other apps is not set up on this server"}}
    if not isinstance(body, dict) or not any(k in body for k in ("skills", "import_mcp", "rescan")):
        return 400, {"error": {"message": 'send {"skills": {...}}, {"import_mcp": {"harness", "name"}} or {"rescan": true}', "fields": []}}
    skills_new = None
    if "skills" in body:
        skills_new, errors = _check_skills(body["skills"])
        if errors:
            return 400, {"error": {"message": f"{errors[0]['message']} ({errors[0]['field']})", "fields": errors}}
    want = body.get("import_mcp")
    if "import_mcp" in body and not (isinstance(want, dict) and isinstance(want.get("harness"), str) and isinstance(want.get("name"), str) and want["name"]):
        return 400, {"error": {"message": "import_mcp is {\"harness\": ..., \"name\": ...}", "fields": [{"field": "import_mcp", "message": "harness and name are text"}]}}
    imported = False
    with mcp_admin._lock:
        try:
            if skills_new is not None:
                def put(cfg):
                    block = cfg.get("import") if isinstance(cfg.get("import"), dict) else {}
                    block["skills"] = skills_new
                    cfg["import"] = block
                mcp_admin.update_config(svc.config_path, put)
            if want is not None:
                try:
                    new_own, _name = import_server(mcp_admin._servers_in(mcp_admin._read_json(svc.config_path)), want["harness"], want["name"], imp.found, imp.environ)
                except ImportRefused as e:
                    return {"missing": 404, "unsupported": 422, "already": 409}[e.code], {"error": {"message": str(e), "code": e.code}}
                mcp_admin.write_config(svc.config_path, new_own, {})
                imported = True
            cfg = mcp_admin._read_json(svc.config_path)
            imp.rescan(cfg) if body.get("rescan") is True else imp.apply(cfg)
            if imported:
                mcp_admin.reload_hub(svc)
        except (OSError, ValueError, TypeError) as e:
            return 500, {"error": {"message": f"it could not be saved: {e}"}}
    return 200, None
