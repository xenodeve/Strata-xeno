"""What Strata Desktop writes into the web view's localStorage before the web app loads (#310 part 3).

pywebview has no init-script call, so the app window first loads `<base>/health` (same origin as the app, no app
script, open without a key or a device approval), runs `inject_script(...)` there with `evaluate_js`, and only when
that returns `true` loads `<base>/`.  The web app then starts with the values already in place, as if a person had
typed them: `strata.apikey` (the token), `strata.device` and `strata.deviceName` (the runner's device, so the page and
the runner are one device to the server and approving one approves the other), and `strata.runner` (this PC's runner
id) - each JSON-encoded as `serve/ui/src/lib/store.ts` writes them.  Never on the app page itself: by then it has sent
its own device id.

The token sits in the WebView2 profile while its server is open (ruling P3-R1): `clear_script()` removes it on Switch
server, Quit and Remove, and on the next start after a crash.
"""
from __future__ import annotations

import json
import re
from urllib.parse import urlsplit

DEVICE_ID = re.compile(r"[A-Za-z0-9_-]{22}")      # serve/ui/src/lib/device.ts ID: any other value the page replaces
RUNNER_ID = re.compile(r"r[0-9a-f]{16}")          # serve/ui/src/lib/runner.ts RUNNER_RE: any other the page ignores
LABEL_MOST = 80                                   # device.ts headerSafe and the server (identity.MAX_TEXT) cut here
DEFAULT_PORTS = {"http": 80, "https": 443}


def page_label(label: str) -> str:
    """`label` as both the runner and the page send it: the runner's printable-ASCII rule (`serve.runner._header`),
    then the page's `headerSafe` (spaces folded, trimmed, at most 80 characters).  The result is unchanged by either
    rule, so the runner (`RunnerHost(label=...)`) and the page (`strata.deviceName`) given it send one name."""
    from serve.runner import _header
    text = re.sub(r"\s+", " ", _header(str(label))).strip()
    return text[:LABEL_MOST].strip()


def storage_values(token: str, runner_id: str | None, device: dict, label: str) -> tuple[dict[str, str], list[str]]:
    """The localStorage values to set and the keys to remove.  Raises ValueError (never repeating the token) for a
    value the page would drop and replace with its own: a device id that is not 22 of [A-Za-z0-9_-], a runner id that
    is not "r" + 16 hex digits, a label the page would send differently (see `page_label`), or no token (the token's
    own rule, `serve.runner.KEY`, is the server list's: whatever text comes here goes in JSON-encoded)."""
    if not isinstance(token, str) or not token:
        raise ValueError("There is no token to give the web app.")
    dev_id = device.get("id") if isinstance(device, dict) else None
    if not isinstance(dev_id, str) or not DEVICE_ID.fullmatch(dev_id):
        raise ValueError("The device id must be 22 of A-Z, a-z, 0-9, _ and -.")
    if runner_id is not None and (not isinstance(runner_id, str) or not RUNNER_ID.fullmatch(runner_id)):
        raise ValueError("The runner id must be r and 16 hex digits.")
    if not isinstance(label, str) or not label or page_label(label) != label:
        raise ValueError("The device's name must be what the page sends: use inject.page_label().")
    values = {"strata.apikey": json.dumps(token), "strata.device": json.dumps(dev_id),
              "strata.deviceName": json.dumps(label)}
    if runner_id is None:
        return values, ["strata.runner"]
    values["strata.runner"] = json.dumps(runner_id)
    return values, []


def js_literal(value) -> str:
    """A JSON literal safe anywhere in a script: ASCII only (U+2028 / U+2029 escaped), and `<`, `>`, `&` escaped so
    no `</script>` or `<!--` can appear."""
    text = json.dumps(value, ensure_ascii=True)
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def inject_script(values: dict[str, str], remove: list[str]) -> str:
    """One JS expression: sets `values`, removes `remove`, and is `true` only when every value reads back equal and
    every removed key is absent (`false` on any error, a full or refused storage included)."""
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in values.items()):
        raise TypeError("values must be text")
    if not all(isinstance(k, str) for k in remove):
        raise TypeError("the keys to remove must be text")
    data = js_literal({"set": values, "remove": list(remove)})
    return ("(function(){try{var d=" + data + ",s=window.localStorage,k=Object.keys(d.set),i;"
            "for(i=0;i<k.length;i++)s.setItem(k[i],d.set[k[i]]);"
            "for(i=0;i<d.remove.length;i++)s.removeItem(d.remove[i]);"
            "for(i=0;i<k.length;i++)if(s.getItem(k[i])!==d.set[k[i]])return false;"
            "for(i=0;i<d.remove.length;i++)if(s.getItem(d.remove[i])!==null)return false;"
            "return true}catch(e){return false}})()")


def clear_script() -> str:
    """Removes `strata.apikey` (and nothing else: the chats, the device and the runner id stay); `true` when it is
    gone."""
    return ('(function(){try{var s=window.localStorage;s.removeItem("strata.apikey");'
            'return s.getItem("strata.apikey")===null}catch(e){return false}})()')


def guarded(script: str, want_origin: str, want_path: str | None = None) -> str:
    """`script`, run only when the page's `location.origin` is `want_origin` and, when given, its `location.pathname`
    is `want_path` (else `false`): a page that navigated away, the browser's error page (origin "null"), and the
    document still current when the load was cancelled (the app page itself: by then it has sent its own device id)
    never get the values."""
    cond = "window.location.origin!==" + js_literal(want_origin)
    if want_path is not None:
        cond += "||window.location.pathname!==" + js_literal(want_path)
    return "(function(){try{if(" + cond + ")return false}catch(e){return false}return(" + script + ")})()"


def origin(url: str) -> str:
    """The web origin of an http(s) URL as a browser writes `location.origin`: scheme and host lower-cased, an IPv6
    host in brackets, the port only when it is not the scheme's default.  ValueError for anything else."""
    if not isinstance(url, str):
        raise ValueError("not a URL")
    try:
        u = urlsplit(url.strip())
        host, port = u.hostname, u.port
    except ValueError:
        raise ValueError("not a URL") from None
    scheme = u.scheme.lower()
    if scheme not in DEFAULT_PORTS or not host:
        raise ValueError("not an http(s) URL")
    if ":" in host:
        host = f"[{host}]"
    return f"{scheme}://{host}" + ("" if port is None or port == DEFAULT_PORTS[scheme] else f":{port}")


def health_url(base: str) -> str:
    return base.rstrip("/") + "/health"


def app_url(base: str) -> str:
    return base.rstrip("/") + "/"
