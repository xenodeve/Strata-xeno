"""desktop/test_probe.py - the server "Test" of the servers window (#310 part 3 task 2): /health, then /props with the
token and the device headers, then /keys for the role.  A scripted http.server on port 0 for the answers the brief
names, and (where jinja2 is installed: not in desktop\\.venv) the real server in-process with the mock engine.

    python -m pytest desktop/test_probe.py -q
"""
from __future__ import annotations

import dataclasses
import json
import socket
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from desktop import probe, words
from serve.runner import Client, device_label

TOKEN = "sk-strata-probe-SECRET-0123456789"
DEVICE = {"id": "AbCdEfGhIjKlMnOpQrStUv", "name": "ann-pc"}       # 22 characters, as load_device makes
LABEL = device_label(DEVICE, "Strata Desktop")


class Scripted:
    """An http.server on 127.0.0.1:0 that answers each path from `answers` {path: (status, body)} and records every
    request's path and headers.  A path missing from `answers` is 404; `hang` paths never answer (until shutdown)."""

    def __init__(self, answers: dict, hang: tuple = ()):
        self.answers, self.hang, self.seen = answers, set(hang), []
        self.release = threading.Event()
        owner = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                owner.seen.append((self.path, dict(self.headers.items())))
                if self.path in owner.hang:
                    owner.release.wait(30)
                    return
                status, body = owner.answers.get(self.path, (404, {"error": {"message": "not found"}}))
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def paths(self):
        return [p for p, _ in self.seen]

    def close(self):
        self.release.set()
        self.httpd.shutdown()
        self.httpd.server_close()


HEALTH_OK = (200, {"status": "ok", "model": "health-model", "service": "strata"})
PROPS_OK = (200, {"model_alias": "Qwen3.8-Flash-Next", "default_generation_settings": {"n_ctx": 4096}})
KEYS_OK = (200, {"keys": [], "log": [], "retired": [], "team": True, "keys_file": "k.json"})
FORBIDDEN = (403, {"error": {"type": "forbidden", "message": "keys can be managed only with an owner key"}})


def run(answers: dict, *, url_suffix: str = "", hang: tuple = (), timeout: float = 10.0, token: str = TOKEN):
    s = Scripted(answers, hang)
    try:
        r = probe.test_server(s.url + url_suffix, token, DEVICE, LABEL, timeout=timeout)
    finally:
        s.close()
    assert_no_token(r, token)                     # RF 1, on every scripted path
    return r, s


def assert_no_token(result, token: str = TOKEN):
    for secret in {token, TOKEN}:
        assert secret not in repr(result)
        assert secret not in str(result)
        assert secret not in json.dumps(dataclasses.asdict(result), ensure_ascii=False)


def test_device_headers():
    assert probe.device_headers(DEVICE, LABEL) == {"X-Strata-Device": DEVICE["id"],
                                                   "X-Strata-Device-Name": "Strata Desktop on ann-pc"}
    # the label goes through the runner's ASCII rule, so a Thai machine name cannot break a request
    assert probe.device_headers(DEVICE, "Strata Desktop on เครื่อง")["X-Strata-Device-Name"].isascii()


def test_probe_and_runner_send_one_device():
    # RF 2: the page, the test and the runner are one device with one name
    c = Client("http://127.0.0.1:9", TOKEN, DEVICE, label=LABEL)
    h = probe.device_headers(DEVICE, LABEL)
    assert h == {k: c._headers[k] for k in h}


def test_owner():
    r, s = run({"/health": HEALTH_OK, "/props": PROPS_OK, "/keys": KEYS_OK})
    assert (r.reachable, r.role, r.model, r.pending_code, r.engine_running, r.error) == \
        (True, "owner", "Qwen3.8-Flash-Next", None, True, None)
    assert s.paths() == ["/health", "/props", "/keys"]
    assert_no_token(r)


def test_guest():
    r, _ = run({"/health": HEALTH_OK, "/props": PROPS_OK, "/keys": FORBIDDEN})
    assert (r.reachable, r.role, r.error) == (True, "guest", None)


def test_token_refused():
    r, s = run({"/health": HEALTH_OK,
                "/props": (401, {"error": {"type": "authentication_error", "message": "missing or wrong API key"}})})
    assert (r.reachable, r.role, r.error) == (True, None, words.t("token_refused"))
    assert s.paths() == ["/health", "/props"]
    assert_no_token(r)


def test_device_refused():
    r, s = run({"/health": HEALTH_OK, "/props": (403, {"error": {"type": "device_refused", "message": "no"}})})
    assert (r.reachable, r.role, r.error, r.pending_code) == (True, None, words.t("device_refused"), None)
    assert s.paths() == ["/health", "/props"]


def test_pending_code():
    r, s = run({"/health": HEALTH_OK, "/props": (403, {"error": {"type": "device_pending", "code": "K7QZ"}}),
                "/keys": KEYS_OK})
    assert (r.reachable, r.pending_code, r.role, r.error) == (True, "K7QZ", None, None)
    assert "/keys" not in s.paths()
    assert r.model == "health-model"


def test_pending_code_is_bounded():
    r, _ = run({"/health": HEALTH_OK, "/props": (403, {"error": {"type": "device_pending", "code": 1234 * 10 ** 30}})})
    assert isinstance(r.pending_code, str) and len(r.pending_code) <= 16


def test_engine_down_still_reports_role():
    r, s = run({"/health": (503, {"status": "engine_exited", "model": "m"}),
                "/props": (503, {"error": {"message": "the engine is not running"}}), "/keys": KEYS_OK})
    assert (r.reachable, r.engine_running, r.role, r.model, r.error) == (True, False, "owner", "m", None)
    assert s.paths() == ["/health", "/props", "/keys"]


def test_loading_server_is_reachable():
    # serve.server.loading_server answers every GET 503 {"status": "loading"} while the model loads
    loading = (503, {"status": "loading"})
    r, _ = run({"/health": loading, "/props": loading, "/keys": loading})
    assert (r.reachable, r.engine_running, r.role, r.error) == (True, False, None, None)


def test_not_strata():
    r, s = run({"/health": (404, b"<html>nope</html>")})
    assert (r.reachable, r.role, r.error) == (False, None, words.t("not_strata"))
    assert s.paths() == ["/health"]
    r, _ = run({"/health": (503, {"status": "down for maintenance"})})
    assert (r.reachable, r.error) == (False, words.t("not_strata"))
    r, _ = run({"/health": (200, b"ok")})
    assert (r.reachable, r.error) == (False, words.t("not_strata"))


def test_props_other_answer_stops():
    r, s = run({"/health": HEALTH_OK, "/props": (502, b"bad gateway"), "/keys": KEYS_OK})
    assert (r.reachable, r.role, r.error) == (True, None, words.t("server_answered", code=502))
    assert "/keys" not in s.paths()


def test_unreachable():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()                                  # a port nothing listens on
    t0 = time.monotonic()
    r = probe.test_server(f"http://127.0.0.1:{port}", TOKEN, DEVICE, LABEL)
    assert time.monotonic() - t0 < 11
    assert (r.reachable, r.role, r.engine_running) == (False, None, False)
    assert r.error == words.t("unreachable")
    assert_no_token(r)


def test_timeout_is_for_the_whole_test():
    t0 = time.monotonic()
    r, _ = run({"/health": HEALTH_OK, "/props": PROPS_OK}, hang=("/keys",), timeout=1.5)
    assert time.monotonic() - t0 < 4
    assert (r.reachable, r.role, r.error) == (True, None, words.t("timed_out"))
    r, _ = run({}, hang=("/health",), timeout=1.0)
    assert (r.reachable, r.error) == (False, words.t("timed_out"))


def test_url_path_prefix_kept():
    r, s = run({"/strata/health": HEALTH_OK, "/strata/props": PROPS_OK, "/strata/keys": KEYS_OK}, url_suffix="/strata/")
    assert r.role == "owner"
    assert s.paths() == ["/strata/health", "/strata/props", "/strata/keys"]


def test_probe_sends_runner_device_headers():
    r, s = run({"/health": HEALTH_OK, "/props": PROPS_OK, "/keys": KEYS_OK})
    assert r.role == "owner"
    assert len(s.seen) == 3
    for path, h in s.seen[1:]:                    # RF 2: every request after /health carried the runner's device
        assert h.get("X-Strata-Device") == DEVICE["id"], path
        assert h.get("X-Strata-Device-Name") == "Strata Desktop on ann-pc", path
        assert h.get("Authorization") == "Bearer " + TOKEN, path
    health_headers = s.seen[0][1]
    assert "Authorization" not in health_headers     # /health is open: the token is not sent there


def test_bad_token_text_does_not_leak():
    # http.client refuses a header value with a newline; its message holds the value, so it must not be shown
    bad = TOKEN + "\nX-Evil: 1"
    r, _ = run({"/health": HEALTH_OK, "/props": PROPS_OK, "/keys": KEYS_OK}, token=bad)
    assert r.error and TOKEN not in r.error and "X-Evil" not in r.error
    assert TOKEN not in repr(r)


def test_api_never_returns_token():
    # RF 1, the probe's share: what ServersApi.test (Task 4/5) returns is asdict(ProbeResult); the probe never puts
    # the token in it, nor a server's error text that echoes it (a model name the server chooses is the server's own)
    echo = (401, {"error": {"type": "authentication_error", "message": "wrong key " + TOKEN}})
    for answers in ({"/health": HEALTH_OK, "/props": PROPS_OK, "/keys": KEYS_OK},
                    {"/health": HEALTH_OK, "/props": echo},
                    {"/health": HEALTH_OK, "/props": (500, {"error": {"message": TOKEN}})},
                    {"/health": HEALTH_OK, "/props": (403, {"error": {"type": "device_refused", "message": TOKEN}})}):
        r, _ = run(answers)
        assert r.reachable                       # (run() asserts the token is absent)


def test_bad_url():
    for url in ("", "ftp://x", "http://", "nonsense", "http://a b:1", "http://127.0.0.1:9/a b"):
        r = probe.test_server(url, TOKEN, DEVICE, LABEL)
        assert (r.reachable, r.error) == (False, words.t("bad_url"))


# ---------------------------------------------------------------------------------- the real server, in-process
class RealServer(unittest.TestCase):
    """serve.server.serve(svc, port=0) with the mock engine in team mode, as serve/test_identity_http.py builds it.
    Needs jinja2 (serve.frontend): the global python has it, desktop\\.venv does not, so there it is skipped."""

    def setUp(self):
        pytest.importorskip("jinja2")
        from serve.frontend import ChatTemplate
        from serve.identity import KeyStore
        from serve.server import ByteTokenizer, MockEngine, Service, serve

        root = Path(__file__).resolve().parents[1]
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "</think>\n\nok", max_context=4096), tok,
                           ChatTemplate(root / "serve/chat_template.jinja"))
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = Path(tmp.name)
        keys_path, cfg = d / "strata-keys.json", d / "strata-test.json"
        cfg.write_text("{}", encoding="utf-8")
        keys_path.write_text(json.dumps({"keys": [], "log": []}), encoding="utf-8")      # team mode
        self.svc.api_key = "owner-key-0123"
        self.svc.config_path = str(cfg)
        self.svc.keys = KeyStore(str(keys_path), self.svc.api_key)
        self.httpd = serve(self.svc, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def test_owner_from_this_pc(self):
        r = probe.test_server(self.url, "owner-key-0123", DEVICE, LABEL)
        self.assertEqual((r.reachable, r.role, r.engine_running, r.error), (True, "owner", True, None))
        self.assertEqual(r.model, self.svc.model)

    def test_pending_then_guest(self):
        from serve.identity import OWNER
        self.svc.peer_override = "26.1.2.3"                     # another PC, over Radmin
        ann = self.svc.keys.create("ann", "guest")
        r = probe.test_server(self.url, ann, DEVICE, LABEL)
        self.assertEqual((r.reachable, r.role, r.error), (True, None, None))
        self.assertEqual(r.pending_code, self.svc.keys.device_code("ann", DEVICE["id"]))
        rows = [row for row in self.svc.keys.pending() if row["device"] == DEVICE["id"]]
        self.assertEqual([row["label"] for row in rows], ["Strata Desktop on ann-pc"])   # the label the owner sees
        self.svc.keys.decide_device("ann", DEVICE["id"], True, OWNER)
        r = probe.test_server(self.url, ann, DEVICE, LABEL)
        self.assertEqual((r.reachable, r.role, r.pending_code, r.error), (True, "guest", None, None))

    def test_wrong_token(self):
        r = probe.test_server(self.url, "not-a-key", DEVICE, LABEL)
        self.assertEqual((r.reachable, r.role, r.error), (True, None, words.t("token_refused")))
