"""desktop/test_inject.py - the values Strata Desktop writes into the web view's localStorage before the app loads
(#310 part 3 task 4), the script that writes them, and the URLs and origin it is checked against.

    python -m pytest desktop/test_inject.py -q
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from desktop import inject
from serve import identity
from serve.runner import Client, device_label

TOKEN = "sk-strata-inject-SECRET-0123456789"
DEVICE = {"id": "AbCdEfGhIjKlMnOpQrStUv", "name": "ann-pc"}         # 22 characters, as load_device makes
RUNNER = "r0123456789abcdef"
LABEL = inject.page_label(device_label(DEVICE, "Strata Desktop"))
NODE = shutil.which("node")
NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def _header_safe(s: str) -> str:
    """serve/ui/src/lib/device.ts headerSafe, ported: what the page sends as X-Strata-Device-Name."""
    return re.sub(r"\s+", " ", re.sub(r"[^\x20-\x7E]+", " ", s)).strip()[:80].strip()


# ------------------------------------------------------------------------------------------------ the values
def test_values_are_json_encoded():
    token = 'ab"c\u2028</script>'
    values, remove = inject.storage_values(token, RUNNER, DEVICE, LABEL)
    assert values["strata.device"] == '"' + DEVICE["id"] + '"'
    assert json.loads(values["strata.apikey"]) == token
    assert json.loads(values["strata.runner"]) == RUNNER
    assert json.loads(values["strata.deviceName"]) == LABEL
    assert set(values) == {"strata.apikey", "strata.device", "strata.deviceName", "strata.runner"}
    assert remove == []


def test_without_runner_removes_runner_key():
    values, remove = inject.storage_values(TOKEN, None, DEVICE, LABEL)
    assert "strata.runner" not in values
    assert remove == ["strata.runner"]


@pytest.mark.parametrize("bad", ["A" * 21, "A" * 23, "AbCdEfGhIjKlMnOpQrSt+v", "AbCdEfGhIjKlMnOpQrSt/v", "", None, 7])
def test_rejects_device_id_not_22(bad):
    with pytest.raises(ValueError):
        inject.storage_values(TOKEN, RUNNER, {"id": bad, "name": "x"}, LABEL)


@pytest.mark.parametrize("bad", ["", "r0123", "r0123456789ABCDEF", "x0123456789abcdef", "r0123456789abcdef0", 5])
def test_rejects_bad_runner_id(bad):
    with pytest.raises(ValueError):
        inject.storage_values(TOKEN, bad, DEVICE, LABEL)


@pytest.mark.parametrize("bad", ["", None, 5])
def test_rejects_no_token(bad):
    with pytest.raises(ValueError):
        inject.storage_values(bad, RUNNER, DEVICE, LABEL)


def test_errors_never_repeat_the_token():
    for device, runner, label in [({"id": "x"}, RUNNER, LABEL), (DEVICE, "bad", LABEL), (DEVICE, RUNNER, "a  b")]:
        with pytest.raises(ValueError) as e:
            inject.storage_values(TOKEN, runner, device, label)
        assert TOKEN not in str(e.value)


def test_rejects_label_the_page_would_change():
    long_label = device_label({"id": DEVICE["id"], "name": "m" * 80}, "Strata Desktop")
    assert len(long_label) > 80
    with pytest.raises(ValueError):
        inject.storage_values(TOKEN, RUNNER, DEVICE, long_label)
    with pytest.raises(ValueError):
        inject.storage_values(TOKEN, RUNNER, DEVICE, "two  spaces")
    with pytest.raises(ValueError):
        inject.storage_values(TOKEN, RUNNER, DEVICE, "")


@pytest.mark.parametrize("machine", ["ann-pc", "เครื่องของแอน", "m" * 80, "a  b\tc", " pad "])
def test_page_and_runner_send_one_device(machine):
    """RF 2 / ruling P3-R2: the page (strata.device / strata.deviceName through device.ts) and the runner (Client's
    headers) give the server the same id and the same label, so approving one approves the other."""
    device = {"id": DEVICE["id"], "name": machine}
    label = inject.page_label(device_label(device, "Strata Desktop"))
    values, _ = inject.storage_values(TOKEN, RUNNER, device, label)
    page_id = json.loads(values["strata.device"])
    page_name = _header_safe(json.loads(values["strata.deviceName"]))
    runner = Client("http://127.0.0.1:1", TOKEN, device, label=label)._headers
    assert page_id == runner["X-Strata-Device"] == device["id"]
    assert page_name == runner["X-Strata-Device-Name"] == label
    seen_page = identity.device_of(page_id, page_name, "Mozilla/5.0", "127.0.0.1")
    seen_runner = identity.device_of(runner["X-Strata-Device"], runner["X-Strata-Device-Name"], "strata-runner/1",
                                     "127.0.0.1")
    assert seen_page == seen_runner == (device["id"], label)
    assert label.startswith("Strata Desktop on ")


def test_page_label_is_a_fixed_point():
    for raw in ["Strata Desktop on ann-pc", "Strata Desktop on " + "x" * 100, "a\u00e9b  c ", "\u0e01\u0e02"]:
        once = inject.page_label(raw)
        assert inject.page_label(once) == once
        assert _header_safe(once) == once
        assert len(once) <= 80


# ------------------------------------------------------------------------------------------------ the script
def test_script_is_one_safe_literal():
    values, remove = inject.storage_values('ab"c\u2028</script>', RUNNER, DEVICE, LABEL)
    script = inject.inject_script(values, remove)
    assert "</" not in script and "\u2028" not in script and "\u2029" not in script
    assert script.isascii()


def _node(script: str, store: dict[str, str], *, drop: str | None = None, origin: str = "http://h:1",
          path: str = "/health") -> dict:
    """Runs `script` in node against a Map-backed localStorage stub (a `setItem` that drops `drop`) on a page at
    `origin` + `path`, returns {"result", "store"}."""
    harness = """
const fs = require("fs");
const [scriptPath, storePath, drop, origin, pathname] = process.argv.slice(2);
const m = new Map(Object.entries(JSON.parse(fs.readFileSync(storePath, "utf8"))));
globalThis.localStorage = {
  getItem: (k) => (m.has(k) ? m.get(k) : null),
  setItem: (k, v) => { if (k !== drop) m.set(k, String(v)) },
  removeItem: (k) => { m.delete(k) },
};
globalThis.window = globalThis;
globalThis.location = { origin, pathname };
const result = (0, eval)(fs.readFileSync(scriptPath, "utf8"));
process.stdout.write(JSON.stringify({ result, store: Object.fromEntries(m) }));
"""
    with tempfile.TemporaryDirectory() as d:
        h, s, st = Path(d) / "h.js", Path(d) / "s.js", Path(d) / "store.json"
        h.write_text(harness, encoding="utf-8")
        s.write_text(script, encoding="utf-8")
        st.write_text(json.dumps(store), encoding="utf-8")
        out = subprocess.run([NODE, str(h), str(s), str(st), drop or "", origin, path], capture_output=True, text=True,
                             encoding="utf-8", timeout=30, creationflags=NO_WINDOW)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_script_runs_in_node():
    token = 'ab"c\u2028</script>'
    values, remove = inject.storage_values(token, RUNNER, DEVICE, LABEL)
    before = {"strata.chats": "[1]", "strata.apikey": '"old"'}
    got = _node(inject.inject_script(values, remove), before)
    assert got["result"] is True
    assert got["store"] == {**before, **values}
    assert json.loads(got["store"]["strata.apikey"]) == token

    # without a runner: strata.runner is removed, everything else set
    values, remove = inject.storage_values(TOKEN, None, DEVICE, LABEL)
    got = _node(inject.inject_script(values, remove), {"strata.runner": '"r0000000000000000"', "strata.chats": "[]"})
    assert got["result"] is True
    assert got["store"] == {"strata.chats": "[]", **values}

    # a store that drops the token: the script says false
    values, remove = inject.storage_values(TOKEN, RUNNER, DEVICE, LABEL)
    got = _node(inject.inject_script(values, remove), {}, drop="strata.apikey")
    assert got["result"] is False


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_clear_script_removes_only_apikey():
    before = {"strata.apikey": '"' + TOKEN + '"', "strata.chats": "[1]", "strata.device": '"' + DEVICE["id"] + '"',
              "strata.runner": '"' + RUNNER + '"'}
    got = _node(inject.clear_script(), before)
    assert got["result"] is True
    assert got["store"] == {k: v for k, v in before.items() if k != "strata.apikey"}
    assert _node(inject.clear_script(), {})["result"] is True


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_guarded_script_runs_only_on_its_origin():
    values, remove = inject.storage_values(TOKEN, RUNNER, DEVICE, LABEL)
    script = inject.guarded(inject.inject_script(values, remove), "http://h:1")
    assert _node(script, {}, origin="http://h:1")["result"] is True
    other = _node(script, {}, origin="http://evil:1")
    assert other == {"result": False, "store": {}}
    assert _node(script, {}, origin="null")["store"] == {}          # a browser error page


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_guarded_script_runs_only_on_its_path():
    """A /health load that was cancelled (a redirect off the origin) leaves the app page current: same origin, other
    path; the values must not land there (seen in the hidden-window check)."""
    values, remove = inject.storage_values(TOKEN, RUNNER, DEVICE, LABEL)
    script = inject.guarded(inject.inject_script(values, remove), "http://h:1", "/p/health")
    assert _node(script, {}, path="/p/health")["result"] is True
    assert _node(script, {}, path="/p/") == {"result": False, "store": {}}
    assert _node(script, {}, path="/") == {"result": False, "store": {}}


def test_script_refuses_what_is_not_text():
    with pytest.raises(TypeError):
        inject.inject_script({"strata.apikey": 1}, [])
    with pytest.raises(TypeError):
        inject.inject_script({}, ["strata.runner", 2])


# ------------------------------------------------------------------------------------------------ URLs and origins
def test_urls():
    base = "http://h:1/p"
    assert inject.health_url(base) == "http://h:1/p/health"
    assert inject.app_url(base) == "http://h:1/p/"
    assert inject.origin(base) == "http://h:1"
    assert inject.health_url("http://h:1/") == "http://h:1/health"
    assert inject.app_url("http://h:1") == "http://h:1/"


@pytest.mark.parametrize("url, want", [
    ("HTTP://Lab.Example.COM:8080/x", "http://lab.example.com:8080"),
    ("https://h:443/", "https://h"),
    ("http://h:80", "http://h"),
    ("https://h:80", "https://h:80"),
    ("http://[::1]:8080/p", "http://[::1]:8080"),
    ("http://[FE80::1]", "http://[fe80::1]"),
    ("http://127.0.0.1:8091/#/chat", "http://127.0.0.1:8091"),
])
def test_origin_is_lower_cased_as_a_browser_writes_it(url, want):
    assert inject.origin(url) == want


@pytest.mark.parametrize("bad", ["", "about:blank", "file:///c:/x", "http://", "chrome-error://chromewebdata/", None])
def test_origin_refuses_what_is_not_a_web_origin(bad):
    with pytest.raises(ValueError):
        inject.origin(bad)
