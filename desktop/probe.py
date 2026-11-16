"""The server "Test" of the servers window (#310 part 3): is a Strata server there, does it take this token from this
PC, and as whom.

    GET /health   no token (the route is open): 200, or 503 "engine_exited" / "loading" -> reachable; the model
    GET /props    the token and this PC's device headers: 401 -> the token is refused; 403 device_pending -> the
                  code the owner sees; 403 device_refused -> this PC is refused; 503 -> the engine is not running,
                  go on; 200 -> the model's alias
    GET /keys     the same headers: 200 -> owner, 403 -> guest, anything else -> role unknown

Every request carries the same X-Strata-Device and X-Strata-Device-Name as the runner and the page (RF 2: one device,
so the owner approves it once).  http.client, not urllib: no redirect is followed and no proxy variable is honoured,
so the token goes only to the address given.  `error` is always one of desktop/words.py's texts, never the server's
message or an exception's text (http.client puts a refused header's value, the token, in its message).
"""
from __future__ import annotations

import http.client
import json
import ssl
import time
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit

from desktop import words

USER_AGENT = "strata-desktop/1"
BODY_MOST = 4 << 20             # a larger answer is cut and read as no JSON (/props carries the chat template)
MODEL_MOST = 200
CODE_MOST = 16


@dataclass
class ProbeResult:
    reachable: bool
    model: str | None = None
    role: Literal["owner", "guest"] | None = None
    pending_code: str | None = None
    engine_running: bool = False
    error: str | None = None


class _Failed(Exception):
    def __init__(self, key: str):
        super().__init__(key)
        self.key = key


def device_headers(device: dict, label: str) -> dict[str, str]:
    """X-Strata-Device (the runner's 22-character id) and X-Strata-Device-Name (`label`, as
    `serve.runner.device_label(device, "Strata Desktop")` makes it), through the runner's printable-ASCII rule."""
    from serve.runner import _header
    return {"X-Strata-Device": device["id"], "X-Strata-Device-Name": _header(label)}


def _text(v, most: int) -> str | None:
    return v[:most] if isinstance(v, str) and v.strip() else None


def _code(v) -> str:
    s = "".join(c for c in str(v) if c.isascii() and (c.isalnum() or c == "-"))[:CODE_MOST]
    return s or "----"


def test_server(url: str, token: str, device: dict, label: str, timeout: float = 10.0) -> ProbeResult:
    """The test of one server.  `timeout` bounds the whole test (all three requests), in seconds."""
    try:
        u = urlsplit(url.strip())
        host, port = u.hostname, u.port
    except (ValueError, AttributeError):
        return ProbeResult(False, error=words.t("bad_url"))
    if u.scheme.lower() not in ("http", "https") or not host:
        return ProbeResult(False, error=words.t("bad_url"))
    https, prefix = u.scheme.lower() == "https", u.path.rstrip("/")
    deadline = time.monotonic() + timeout
    base_headers = {**device_headers(device, label), "User-Agent": USER_AGENT, "Accept": "application/json"}

    def get(path: str, with_token: bool) -> tuple[int, object]:
        left = deadline - time.monotonic()
        if left <= 0:
            raise _Failed("timed_out")
        headers = dict(base_headers)
        if with_token:
            headers["Authorization"] = "Bearer " + token
        try:
            if https:
                conn = http.client.HTTPSConnection(host, port, timeout=left, context=ssl.create_default_context())
            else:
                conn = http.client.HTTPConnection(host, port, timeout=left)
        except http.client.InvalidURL:           # "http://a b:1": a host http.client will not connect to
            raise _Failed("bad_url") from None
        try:
            conn.request("GET", prefix + path, headers=headers)
            r = conn.getresponse()
            raw = r.read(BODY_MOST)
        except http.client.InvalidURL:           # a path with a space or a control character
            raise _Failed("bad_url") from None
        except TimeoutError:                     # socket.timeout too
            raise _Failed("timed_out") from None
        except ssl.SSLError:
            raise _Failed("tls_failed") from None
        except (OSError, http.client.HTTPException):
            raise _Failed("unreachable") from None
        except ValueError:                       # a header value http.client refuses: its text holds the value
            raise _Failed("cannot_send") from None
        finally:
            conn.close()
        try:
            body = json.loads(raw) if raw else None
        except ValueError:
            body = None
        return r.status, body

    result = ProbeResult(False)
    try:
        code, body = get("/health", with_token=False)
        if code == 200 and isinstance(body, dict):
            result.engine_running = True
        elif not (code == 503 and isinstance(body, dict) and body.get("status") in ("engine_exited", "loading")):
            result.error = words.t("not_strata")
            return result
        result.reachable = True
        result.model = _text(body.get("model"), MODEL_MOST)

        code, body = get("/props", with_token=True)
        err = body.get("error") if isinstance(body, dict) and isinstance(body.get("error"), dict) else {}
        if code == 401:
            result.error = words.t("token_refused")
            return result
        if code == 403 and err.get("type") == "device_pending":
            result.pending_code = _code(err.get("code"))
            return result
        if code == 403 and err.get("type") == "device_refused":
            result.error = words.t("device_refused")
            return result
        if code == 503:
            result.engine_running = False
        elif code == 200:
            result.model = _text(body.get("model_alias"), MODEL_MOST) or result.model
        else:
            result.error = words.t("server_answered", code=code)
            return result

        code, _ = get("/keys", with_token=True)
        result.role = "owner" if code == 200 else "guest" if code == 403 else None
    except _Failed as f:
        result.error = words.t(f.key)
    return result
