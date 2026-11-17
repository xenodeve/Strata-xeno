"""serve/app_settings.py - the writers of the app's settings that take effect without a restart (issue #339).

Each one is the body of a Settings page route, moved here so that the route and the chat's StrataSettingsChange tool (serve/strata_tools.py)
run the same code: the same checks, the same write, the same answer.  Who may ask has been decided by the caller (the route's owner gate, or
the tool's permission card).  Each returns (HTTP status, what the route sends).

  save_web(svc, body)       POST /agent/web      the run config's `web` (one setting or several; the others are kept)
  save_helpers(svc, body)   POST /agent/helpers  the run config's `agents`
  save_shared(svc, value)   POST /settings       the Chat settings every client gets (`defaults`; None clears them)

The MCP servers and the import switches already have theirs: serve/mcp_admin.py `apply` and serve/harness.py `apply`.
"""
from __future__ import annotations

import os

from serve import harness, mcp_admin, subagent as subagent_mod, web as web_mod


def _error(errors: list[dict]) -> tuple[int, dict]:
    return 400, {"error": {"message": f"{errors[0]['message']} ({errors[0]['field']})", "fields": errors}}


def _write(svc, key: str, value) -> tuple[int, dict] | None:
    with mcp_admin._lock:
        try:
            mcp_admin.update_config(svc.config_path, lambda cfg: cfg.__setitem__(key, value))
        except (OSError, ValueError, TypeError) as e:
            return 500, {"error": {"message": f"it could not be saved: {e}"}}
    return None


def web_view(svc) -> dict:
    """GET /agent/web's answer once there is a run config file: never the key, only whether there is one."""
    return {**web_mod.public_view(harness._cfg(svc)), "available": svc.agent is not None and bool(getattr(svc.agent, "web_tools", ())),
            "editable": bool(svc.config_path), "config_file": os.path.basename(svc.config_path) if svc.config_path else None}


def helpers_view(svc) -> dict:
    return {**subagent_mod.settings(harness._cfg(svc)), "available": svc.agent is not None and bool(getattr(svc.agent, "helper_tools", ())),
            "editable": bool(svc.config_path), "config_file": os.path.basename(svc.config_path) if svc.config_path else None}


def save_web(svc, body) -> tuple[int, dict]:
    """Web access: what is sent over what is stored (one setting changes, the others stay), checked strictly, written to the run config."""
    if not svc.config_path:
        return 409, {"error": {"message": harness.NO_FILE}}
    new, errors = web_mod.check_settings({**web_mod.settings(harness._cfg(svc)), **body} if isinstance(body, dict) else body)
    if errors:
        return _error(errors)
    return _write(svc, "web", new) or (200, web_view(svc))


def save_helpers(svc, body) -> tuple[int, dict]:
    """Helpers (sub-agents) on or off."""
    if not svc.config_path:
        return 409, {"error": {"message": harness.NO_FILE}}
    new, errors = subagent_mod.check_settings(body)
    if errors:
        return _error(errors)
    return _write(svc, "agents", new) or (200, helpers_view(svc))


def save_shared(svc, defaults) -> tuple[int, dict]:
    """The Chat settings shared with every client: the whole set (None or {} clears them).  A bad value is a 400 naming it."""
    try:
        shared = svc.set_shared(defaults)
    except ValueError as e:
        return 400, {"error": {"type": "invalid_request_error", "message": str(e)}}
    print("[strata] other apps now use the Chat settings: " + ", ".join(f"{k}={v}" for k, v in shared.items())
          if shared else "[strata] other apps use their own settings again", flush=True)
    return 200, {"shared": bool(shared), "defaults": shared}


SHARED_KEYS = ("reasoning_effort", "reasoning_budget_tokens", "temperature", "top_p", "top_k", "seed", "max_tokens",
               "experimental_speed_projection")


def clean_shared_defaults(d) -> dict:
    """The Chat settings other apps get (POST /settings): only known keys, each checked; ValueError names a bad one."""
    if d is None:
        return {}
    if not isinstance(d, dict):
        raise ValueError("defaults must be an object")
    out = {}
    for key, value in d.items():
        if value is None or value == "":
            continue
        number = isinstance(value, (int, float)) and not isinstance(value, bool)
        if key == "reasoning_effort":
            if value not in ("none", "low", "medium", "high", "xhigh"):
                raise ValueError("reasoning_effort: none, low, medium, high or xhigh")
        elif key == "temperature":
            if not number or not 0 <= value <= 2:
                raise ValueError("temperature: 0..2")
        elif key == "top_p":
            if not number or not 0 < value <= 1:
                raise ValueError("top_p: 0 < top_p <= 1")
        elif key == "top_k":
            if not number or value != int(value) or not 1 <= value <= 64:
                raise ValueError("top_k: an integer 1..64")
            value = int(value)
        elif key in ("seed", "max_tokens"):
            if not number or value != int(value) or value <= 0:
                raise ValueError(f"{key}: a positive integer")
            value = int(value)
        elif key == "reasoning_budget_tokens":                 # #123: 0 means no budget, so 0 is allowed here
            if not number or value != int(value) or value < 0:
                raise ValueError("reasoning_budget_tokens: a whole number of tokens (0: no budget)")
            value = int(value)
        elif key == "experimental_speed_projection":
            if not isinstance(value, bool):
                raise ValueError("experimental_speed_projection: true or false")
        else:
            raise ValueError(f"unknown setting {key!r}")
        out[key] = float(value) if key in ("temperature", "top_p") else value
    return out
