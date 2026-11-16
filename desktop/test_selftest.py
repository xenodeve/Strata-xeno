"""desktop/test_selftest.py - `--selftest` (#310 part 3 task 6): the result file (each step, atomic, never the token),
and the whole selftest against the real server in-process (team mode, the mock engine, the guest key `ann` on a
device that waits for the owner), with no window: `selftest.run(..., window=None)`.

The in-process server needs jinja2 (serve.frontend): the global python has it, desktop\\.venv does not, so there those
tests are skipped.  No test opens a window or a tray icon; the runner's home and AppData are temp folders.

    python -m pytest desktop/test_selftest.py -q
"""
from __future__ import annotations

import http.client
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from desktop import applog, inject, selftest, words

TOKEN = "sk-strata-selftest-SECRET-0123456789"


@pytest.fixture(autouse=True)
def english(monkeypatch):
    monkeypatch.setattr(words, "LANG", "en")
    yield
    applog.forget_secrets()


# ------------------------------------------------------------------------------------------------ the result file
def test_result_records_each_step_and_never_the_token(tmp_path):
    path = tmp_path / "selftest.json"
    r = selftest.Result(path, TOKEN)
    r.add("store", True, "saved", 12.4)
    r.add("probe", False, "refused " + TOKEN, 3.0)
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["done"] is False and doc["exit"] is None
    assert [(s["step"], s["ok"], s["ms"]) for s in doc["steps"]] == [("store", True, 12), ("probe", False, 3)]
    assert TOKEN not in path.read_text(encoding="utf-8")
    assert doc["steps"][1]["detail"] == "refused ***"
    r.finish(5)
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert (doc["done"], doc["exit"], doc["ok"]) == (True, 5, False)
    assert not list(tmp_path.glob("*.tmp"))                       # written to a .tmp, then replaced


def test_result_extra_fields_and_details_as_objects(tmp_path):
    r = selftest.Result(tmp_path / "s.json", TOKEN)
    r.add("pending", True, "waits for the owner", 0, code="K7QZ")
    r.add("runner", True, {"who": "ann", "note": TOKEN}, 1)
    doc = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    assert doc["steps"][0]["code"] == "K7QZ"
    assert doc["steps"][1]["detail"] == {"who": "ann", "note": "***"}


@pytest.mark.parametrize("oks, code", [([True, True], 0), ([True, None], 0), ([True, False], 5), ([], 5)])
def test_exit_code_from_the_steps(oks, code):
    steps = [{"step": str(i), "ok": ok} for i, ok in enumerate(oks)]
    assert selftest.exit_code(steps) == code


def test_fail_writes_one_step(tmp_path):
    path = tmp_path / "d" / "selftest.json"
    assert selftest.fail(path, "start", "no --server") == selftest.EXIT_FAILED
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["done"] and doc["exit"] == 5 and doc["steps"][0]["step"] == "start"


@pytest.mark.skipif(os.name != "nt", reason="a Toolhelp snapshot")
def test_child_processes_sees_a_child():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], creationflags=0x08000000)
    try:
        assert child.pid in [c["pid"] for c in selftest.child_processes(os.getpid())]
        assert (os.getpid(), os.getppid()) in [(p, parent) for p, parent, _ in selftest.processes()]
    finally:
        child.kill()
        child.wait(10)


def test_run_hidden_waits_for_the_selftest_after_the_loop_ends(tmp_path, monkeypatch):
    """Quit's destroy ends pywebview's loop while the selftest still checks what Quit left: run_hidden must return
    the selftest's own code, not a failure (the first exe run, 2026-11-16, exited 5 with every step ok)."""
    from desktop import gui

    class LoopGui:
        def __init__(self, data, api, quitting=lambda: False, *, headless=False, **kw):
            assert headless is True
            self.closed = threading.Event()

        def run(self, start):
            threading.Thread(target=start, daemon=True).start()
            self.closed.wait(10)                    # the loop ends at the first destroy
            return 0

        def destroy(self):
            self.closed.set()

    def selftest_run(server, data, token, result, window):
        window.destroy()                            # Quit closes the windows ...
        time.sleep(0.5)                             # ... and the quit step goes on checking
        return 0
    monkeypatch.setattr(gui, "Gui", LoopGui)
    monkeypatch.setattr(selftest, "run", selftest_run)
    assert selftest.run_hidden("http://h:1", tmp_path, TOKEN, tmp_path / "selftest.json") == 0


def test_no_token_fails_before_anything(tmp_path):
    path = tmp_path / "selftest.json"
    assert selftest.run("http://127.0.0.1:9", tmp_path, "", path, None) == 5
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["steps"][0]["step"] == "store" and doc["steps"][0]["ok"] is False
    assert not (tmp_path / "servers.json").exists()


# ------------------------------------------------------------------------------------------------ the real server
class Team:
    """serve.server.serve(svc, port=0), the mock engine, team mode; every request seen as coming from another PC
    (`peer_override`), so the guest's device waits for the owner as it would over the VPN.  The owner (the run
    config's api_key) uses the device OWNER_DEV, approved through the store before anything else."""

    OWNER_KEY = "owner-key-0123"
    OWNER_DEV = "ownerdevice00000000001"

    def __init__(self, tmp: Path):
        pytest.importorskip("jinja2")
        from serve.frontend import ChatTemplate
        from serve.identity import OWNER, KeyStore
        from serve.server import ByteTokenizer, MockEngine, Service, serve

        root = Path(__file__).resolve().parents[1]
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "</think>\n\nok", max_context=4096), tok,
                           ChatTemplate(root / "serve/chat_template.jinja"))
        keys_path, cfg = tmp / "strata-keys.json", tmp / "strata-test.json"
        cfg.write_text("{}", encoding="utf-8")
        keys_path.write_text(json.dumps({"keys": [], "log": []}), encoding="utf-8")      # team mode
        self.svc.api_key = self.OWNER_KEY
        self.svc.config_path = str(cfg)
        self.svc.keys = KeyStore(str(keys_path), self.OWNER_KEY)
        self.svc.peer_override = "26.1.2.3"
        self.httpd = serve(self.svc, port=0)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        code, body = self.req("GET", "/props", self.OWNER_KEY)
        assert (code, body["error"]["type"]) == (403, "device_pending")
        assert self.svc.keys.decide_device("owner", self.OWNER_DEV, True, OWNER)
        code, body = self.req("POST", "/keys", self.OWNER_KEY, {"action": "create", "name": "ann", "role": "guest"})
        assert code == 200, body
        self.ann = body["key"]

    def req(self, method: str, path: str, key: str, body: dict | None = None):
        c = http.client.HTTPConnection("127.0.0.1", self.httpd.server_address[1], timeout=30)
        h = {"Authorization": "Bearer " + key, "X-Strata-Device": self.OWNER_DEV}
        if body is not None:
            h["Content-Type"] = "application/json"
        try:
            c.request(method, path, body=json.dumps(body).encode() if body is not None else None, headers=h)
            r = c.getresponse()
            raw = r.read()
            return r.status, json.loads(raw) if raw[:1] in (b"{", b"[") else raw
        finally:
            c.close()

    def close(self):
        self.svc.runner_hub.close()
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def team(tmp_path, monkeypatch):
    from serve import runner, runner_hub
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("STRATA_HOME", str(home))                   # the runner reads no skill or MCP server of yours
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setattr(runner, "PENDING_RETRY_S", 0.3)            # the pending runner says hello again soon
    monkeypatch.setattr(runner_hub, "POLL_MAX_S", 2.0)
    monkeypatch.setattr(runner, "_say", lambda *a, **k: None)
    t = Team(tmp_path)
    yield t
    t.close()


def _approve_when_pending(team: Team, result: Path, seen: dict, stop: threading.Event) -> None:
    """What the harness does: once selftest.json names a pending code, the owner approves the device with that code
    through POST /devices."""
    while not stop.wait(0.05):
        try:
            steps = json.loads(result.read_text(encoding="utf-8"))["steps"]
        except (OSError, ValueError):
            continue
        codes = [s.get("code") for s in steps if s.get("step") == "pending"]
        if not codes:
            continue
        code, rows = team.req("GET", "/devices/pending", team.OWNER_KEY)
        row = next((r for r in rows["devices"] if r["key"] == "ann" and r["code"] == codes[0]), None)
        if row is None:
            continue
        seen["code"], seen["row"] = codes[0], row
        seen["answer"] = team.req("POST", "/devices", team.OWNER_KEY,
                                  {"key": "ann", "device": row["device"], "approve": True})
        return


def test_selftest_passes_in_process(team, tmp_path):
    from serve import runner
    data = tmp_path / "d"
    result = data / "selftest.json"
    seen: dict = {}
    stop = threading.Event()
    approver = threading.Thread(target=_approve_when_pending, args=(team, result, seen, stop), daemon=True)
    approver.start()
    try:
        t0 = time.monotonic()
        code = selftest.run(team.url, data, team.ann, result, None)
        took = time.monotonic() - t0
    finally:
        stop.set()
        approver.join(5)
    doc = json.loads(result.read_text(encoding="utf-8"))
    steps = {s["step"]: s for s in doc["steps"]}
    assert code == 0, doc
    assert (doc["done"], doc["exit"], doc["ok"]) == (True, 0, True)
    assert [s["step"] for s in doc["steps"]] == ["store", "probe", "pending", "runner", "window", "runners", "quit"]
    for name in ("store", "probe", "pending", "runner", "runners", "quit"):
        assert steps[name]["ok"] is True, steps[name]
    assert steps["window"]["ok"] is None and steps["window"]["detail"].startswith("not checked")
    assert seen["answer"][0] == 200 and steps["pending"]["code"] == seen["code"]
    assert team.ann not in result.read_text(encoding="utf-8")
    assert took < 60

    # the server's word: one device for ann, the runner's, under the one label (RF 2)
    device = json.loads((data / "runner" / "device.json").read_text(encoding="utf-8"))
    label = inject.page_label(runner.device_label(device, "Strata Desktop"))
    code, keys = team.req("GET", "/keys", team.OWNER_KEY)
    ann = next(k for k in keys["keys"] if k["name"] == "ann")
    assert [(d["device"], d["label"], d["status"]) for d in ann["devices"]] == [(device["id"], label, "approved")]
    assert label.startswith("Strata Desktop on ")
    assert steps["runner"]["detail"]["who"] == "ann" and steps["runner"]["detail"]["device"] == device["id"]
    assert steps["runners"]["detail"]["device"] == device["id"]
    # after Quit no poll of ann's runner stays held on the server (it still lists the runner as online for OFFLINE_S
    # after its last poll: runner_hub.Runner.online).  The server learns of a poll the runner cut only when that poll
    # ends (POLL_MAX_S, 2 s here), so it is given that long.
    end = time.monotonic() + 4.0
    while True:
        held = [r._held for r in list(team.svc.runner_hub._runners.values()) if r.who == "ann"]
        if (held and not any(held)) or time.monotonic() > end:
            break
        time.sleep(0.05)
    assert held and not any(held)


def test_selftest_fails_on_wrong_token(team, tmp_path):
    wrong = "sk-strata-" + "0" * 48
    result = tmp_path / "d" / "selftest.json"
    assert selftest.run(team.url, tmp_path / "d", wrong, result, None) == 5
    text = result.read_text(encoding="utf-8")
    doc = json.loads(text)
    probe = next(s for s in doc["steps"] if s["step"] == "probe")
    assert probe["ok"] is False and probe["detail"] == words.t("token_refused")
    assert "runner" not in [s["step"] for s in doc["steps"] if s["ok"] is True]
    assert wrong not in text and (doc["done"], doc["exit"]) == (True, 5)
    assert not [t for t in threading.enumerate() if t.name.startswith(("desktop-runner", "runner-job"))]


def test_selftest_never_needs_jinja2():
    """The exe's selftest runs where jinja2 is not bundled: selftest.py names neither it nor serve.frontend."""
    src = Path(selftest.__file__).read_text(encoding="utf-8")
    assert "serve.frontend" not in src and "jinja2" not in src