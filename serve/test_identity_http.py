"""serve/test_identity_http.py - one identity per request, the owner-only routes and device approval over HTTP
(#310), against the mock engine (no GPU, no pack).

    python -m pytest serve/test_identity_http.py -q
"""
from __future__ import annotations

import contextlib
import http.client
import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import agent  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.history import HistoryStore  # noqa: E402
from serve.identity import OWNER, KeyStore  # noqa: E402
from serve.mcp import hub_from_config  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, keys_file_of, serve  # noqa: E402
from serve.test_harness import tool_call  # noqa: E402
from serve.test_mcp import ScriptedEngine  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEV1, DEV2 = "device-0001", "device-0002"         # valid X-Strata-Device ids (8-64 of [A-Za-z0-9_-])
REMOTE = "26.1.2.3"                               # a Radmin address: what a team member's PC is seen as


class Base(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "</think>\n\nok", max_context=4096), tok,
                           ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.keys_path = self.dir / "strata-keys.json"
        self.cfg_path = self.dir / "strata-test.json"
        self.cfg_path.write_text("{}", encoding="utf-8")
        self.httpd = None

    def start(self, api_key="", team=True, **attrs):
        """`team`: write an empty keys file first, so the server is in team mode (ruling R2)."""
        self.svc.api_key = api_key
        self.svc.config_path = str(self.cfg_path)
        self.svc.keys = KeyStore(str(self.keys_path), api_key)
        if team:
            self.keys_path.write_text(json.dumps({"keys": [], "log": []}), encoding="utf-8")
        for k, v in attrs.items():
            setattr(self.svc, k, v)
        self.httpd = serve(self.svc, port=0)
        self.port = self.httpd.server_address[1]

    def tearDown(self):
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()

    def req(self, method, path, body=None, key=None, device=None, headers=None, host=None):
        """(status, JSON or raw bytes)."""
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        h = {"Content-Type": "application/json"} if body is not None else {}
        if key is not None:
            h["Authorization"] = "Bearer " + key
        if device is not None:
            h["X-Strata-Device"] = device
        h.update(headers or {})
        if host is not None:
            h["Host"] = host
        data = body if isinstance(body, (bytes, type(None))) else json.dumps(body).encode()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                c.request(method, path, body=data, headers=h)
                r = c.getresponse()
                raw = r.read()
            return r.status, json.loads(raw) if raw[:1] in (b"{", b"[") else raw
        finally:
            c.close()

    def chat(self, **kw):
        return self.req("POST", "/v1/chat/completions",
                        {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16}, **kw)

    def approved(self, name, role="guest", device=DEV1):
        """A new key whose device is approved: one request records it as pending, then the owner approves it."""
        key = self.svc.keys.create(name, role)
        code, body = self.req("GET", "/props", key=key, device=device)
        self.assertEqual((code, body["error"]["type"]), (403, "device_pending"))
        self.assertTrue(self.svc.keys.decide_device(name, device, True, OWNER))
        return key


class Identity(Base):
    def test_single_legacy_key_is_owner_everywhere(self):
        self.start(api_key="k", trust_loopback=False)
        code, body = self.req("GET", "/mcp/config", key="k", device=DEV1)
        self.assertEqual((code, body["error"]["type"]), (403, "device_pending"))
        self.assertTrue(self.svc.keys.decide_device("owner", DEV1, True, OWNER))
        code, body = self.req("GET", "/mcp/config", key="k", device=DEV1)
        self.assertEqual(code, 200)
        self.assertIs(body["editable"], True)
        code, body = self.req("POST", "/settings", {"defaults": None}, key="k", device=DEV1)
        self.assertEqual(code, 200)
        self.assertEqual(self.req("GET", "/props", key="wrong", device=DEV1)[0], 401)

    def test_outside_team_mode_nothing_changes(self):
        self.start(api_key="k", team=False, peer_override=REMOTE)
        self.assertEqual(self.chat(key="k", device=DEV1)[0], 200)
        self.assertEqual(self.req("POST", "/settings", {"defaults": None}, key="k", device=DEV1)[0], 200)
        code, body = self.req("GET", "/mcp/config", key="k", device=DEV1)
        self.assertIs(body["editable"], True)                     # a key set: every key holder may edit, as today
        self.assertEqual(self.req("GET", "/props", key="wrong")[0], 401)
        self.assertFalse(self.keys_path.exists())                 # nothing recorded, no file made

    def test_no_key_configured_is_unchanged(self):
        self.start(team=False)
        code, body = self.req("GET", "/mcp/config")
        self.assertIs(body["editable"], True)                     # this PC, naming this PC
        self.assertEqual(self.req("POST", "/agent/hooks", {"off": []})[0], 200)
        code, body = self.req("POST", "/mcp/config", {"servers": {}})          # past the gate: the servers are saved
        self.assertEqual(code, 200)
        self.assertIs(body["editable"], True)
        self.assertIs(self.req("GET", "/agent")[1]["allowed"], True)
        self.svc.peer_override = REMOTE                           # another PC: today's may_edit refusal, word for word
        code, body = self.req("GET", "/mcp/config")
        self.assertIs(body["editable"], False)
        self.assertIn("MCP servers start programs on this PC", body["reason"])
        code, body = self.req("POST", "/agent/hooks", {"off": []})
        self.assertEqual(code, 403)
        self.assertNotIn("type", body["error"])
        self.assertIn("MCP servers start programs on this PC", body["error"]["message"])
        self.assertIs(self.req("GET", "/agent")[1]["allowed"], False)
        self.assertEqual(self.req("POST", "/settings", {"defaults": None})[0], 200)   # not owner-only outside team mode
        self.assertEqual(self.chat()[0], 200)
        self.assertFalse(self.keys_path.exists())

    def test_guest_cannot_edit(self):
        self.start(api_key="k", peer_override=REMOTE)
        ann = self.approved("ann")
        code, body = self.req("POST", "/mcp/config", {"servers": {}}, key=ann, device=DEV1)
        self.assertEqual((code, body["error"]["type"]), (403, "forbidden"))
        code, body = self.req("GET", "/mcp/config", key=ann, device=DEV1)
        self.assertEqual(code, 200)
        self.assertIs(body["editable"], False)
        code, body = self.req("GET", "/agent", key=ann, device=DEV1)
        self.assertIs(body["allowed"], False)
        self.assertEqual(self.chat(key=ann, device=DEV1)[0], 200)
        self.svc.peer_override = None                             # the owner, from this PC: implicitly approved
        code, body = self.chat(key="k")
        self.assertEqual(code, 200)
        self.assertIs(self.req("GET", "/agent", key="k")[1]["allowed"], True)

    def test_owner_only_routes(self):
        self.start(api_key="k", peer_override=REMOTE)
        ann = self.approved("ann")
        for path in ("/settings", "/config", "/import", "/unload", "/load", "/v1/load", "/v1/unload", "/v1/vram",
                     "/slots/0?action=save", "/metrics/keep", "/agent/hooks", "/agent/web", "/agent/helpers",
                     "/agent/rewind", "/agent/checkpoints/forget"):
            code, body = self.req("POST", path, {}, key=ann, device=DEV1)
            self.assertEqual((code, body["error"]["type"]), (403, "forbidden"), path)
            self.assertIn("only with an owner key", body["error"]["message"], path)
        for path in ("/agent/folders", "/agent/git", "/agent/git/diff", "/agent/files", "/agent/mention",
                     "/agent/memory", "/agent/memory/file", "/agent/hooks", "/agent/web", "/agent/helpers"):
            code, body = self.req("GET", path, key=ann, device=DEV1)
            self.assertEqual((code, body["error"]["type"]), (403, "forbidden"), path)
        # strata_agent: the coding tools are the owner's
        code, body = self.req("POST", "/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}],
                                                               "strata_agent": {}}, key=ann, device=DEV1)
        self.assertEqual((code, body["error"]["type"]), (403, "forbidden"))
        # the owner passes the gate (and gets the route's own answer); from this PC, implicitly approved
        self.svc.peer_override = None
        self.assertEqual(self.req("POST", "/settings", {"defaults": None}, key="k")[0], 200)
        self.assertEqual(self.req("POST", "/metrics/keep", {"next": 0}, key="k")[0], 200)
        self.assertEqual(self.req("GET", "/agent/folders", key="k")[0], 200)


class Devices(Base):
    def test_pending_device_gets_code_and_health_stays_open(self):
        self.start(api_key="k", peer_override=REMOTE)
        ann = self.svc.keys.create("ann", "guest")
        code, body = self.chat(key=ann, device=DEV1)
        self.assertEqual((code, body["error"]["type"]), (403, "device_pending"))
        self.assertRegex(body["error"]["code"], r"^[A-Z0-9]{4}$")
        self.assertEqual(body["error"]["code"], self.svc.keys.device_code("ann", DEV1))
        self.assertEqual(self.req("GET", "/health")[0], 200)
        self.assertIs(self.req("GET", "/health")[1]["api_key"], True)
        self.assertEqual(self.req("GET", "/")[0], 200)
        self.assertEqual(self.req("GET", "/props", key=ann, device=DEV1)[1]["error"]["code"], body["error"]["code"])
        self.svc.keys.decide_device("ann", DEV1, False, OWNER)
        code, body = self.chat(key=ann, device=DEV1)
        self.assertEqual((code, body["error"]["type"]), (403, "device_refused"))
        self.assertNotIn("code", body["error"])

    def test_a_named_key_needs_a_key_even_without_a_legacy_one(self):
        self.start(peer_override=REMOTE)                          # team mode, no api_key at all
        self.assertEqual(self.req("GET", "/props")[0], 401)
        self.assertEqual(self.chat()[0], 401)
        self.assertIs(self.req("GET", "/health")[1]["api_key"], True)     # the web app asks for a key
        ann = self.approved("ann")
        self.assertEqual(self.req("GET", "/props", key=ann, device=DEV1)[0], 200)
        self.assertEqual(self.req("GET", "/props", key=ann, device=DEV2)[1]["error"]["type"], "device_pending")

    def test_tunnel_request_is_not_loopback(self):
        self.start(api_key="k")
        code, body = self.req("GET", "/props", key="k", device=DEV1, host="strata.example.com",
                              headers={"Cf-Connecting-Ip": "203.0.113.5"})
        self.assertEqual((code, body["error"]["type"]), (403, "device_pending"))
        self.assertEqual(self.svc.keys.pending()[0]["addr"], "203.0.113.5")
        for headers in ({"X-Forwarded-For": "203.0.113.6"}, {"Forwarded": "for=203.0.113.7"}):
            code, body = self.req("GET", "/props", key="k", device=DEV2, headers=headers)   # loopback Host, but relayed
            self.assertEqual((code, body["error"]["type"]), (403, "device_pending"), headers)
        self.assertEqual(self.req("GET", "/props", key="k", device=DEV2)[0], 200)            # this PC itself

    def test_forwarded_for_from_remote_peer_is_ignored(self):
        self.start(api_key="k", trust_loopback=False)
        ann = self.svc.keys.create("ann", "guest")
        self.req("GET", "/props", key=ann, device=DEV1, headers={"X-Forwarded-For": "26.9.9.9"})
        self.assertEqual(self.svc.keys.pending()[0]["addr"], "26.9.9.9")    # the peer is loopback: a proxy names it
        self.svc.peer_override = "198.51.100.7"
        self.req("GET", "/props", key=ann, device=DEV2,
                 headers={"X-Forwarded-For": "26.9.9.9", "Cf-Connecting-Ip": "26.9.9.8", "Forwarded": "for=26.9.9.7"})
        rows = {r["device"]: r for r in self.svc.keys.pending()}
        self.assertEqual(rows[DEV2]["addr"], "198.51.100.7")

    def test_headerless_client_is_pending_by_address(self):
        self.start(api_key="k", peer_override=REMOTE)
        ann = self.svc.keys.create("ann", "guest")
        code, body = self.chat(key=ann, headers={"User-Agent": "claude-cli/2.1.281 (external, cli)"})
        self.assertEqual((code, body["error"]["type"]), (403, "device_pending"))
        row = self.svc.keys.pending()[0]
        self.assertEqual((row["key"], row["device"], row["addr"]), ("ann", "addr:" + REMOTE, REMOTE))
        self.assertIn("claude-cli", row["label"])
        self.svc.keys.decide_device("ann", "addr:" + REMOTE, True, OWNER)
        self.assertEqual(self.chat(key=ann, headers={"User-Agent": "claude-cli/2"})[0], 200)

    def test_a_malformed_device_id_is_the_address(self):
        self.start(api_key="k", peer_override=REMOTE)
        ann = self.svc.keys.create("ann", "guest")
        self.req("GET", "/props", key=ann)                        # addr:26.1.2.3 recorded, then approved
        self.svc.keys.decide_device("ann", "addr:" + REMOTE, True, OWNER)
        self.svc.peer_override = "26.7.7.7"                       # another PC claims the approved device's id
        for bad in ("addr:" + REMOTE, "short", "x" * 65, "has space1"):
            code, body = self.req("GET", "/props", key=ann, device=bad)
            self.assertEqual((code, body["error"]["type"]), (403, "device_pending"), bad)
        self.assertEqual([r["device"] for r in self.svc.keys.pending()], ["addr:26.7.7.7"])

    def test_device_name_header_is_the_label(self):
        self.start(api_key="k", peer_override=REMOTE)
        ann = self.svc.keys.create("ann", "guest")
        self.req("GET", "/props", key=ann, device=DEV1,
                 headers={"X-Strata-Device-Name": "Ann's laptop " + "x" * 100, "User-Agent": "Mozilla/5.0"})
        row = self.svc.keys.pending()[0]
        self.assertTrue(row["label"].startswith("Ann's laptop"))
        self.assertLessEqual(len(row["label"]), 80)

    def test_a_broken_keys_file_holds_a_new_device_without_a_code(self):
        self.start(api_key="k", peer_override=REMOTE)
        ann = self.approved("ann")
        self.keys_path.write_text("{ not json", encoding="utf-8")
        code, body = self.req("GET", "/props", key=ann, device=DEV2)
        self.assertEqual((code, body["error"]["type"], body["error"]["code"]), (403, "device_pending", "----"))
        self.assertIn("keys file", body["error"]["message"])
        self.assertEqual(self.req("GET", "/props", key=ann, device=DEV1)[0], 200)   # the last good keys still answer

    def test_implicit_loopback_records_nothing(self):
        self.start(api_key="k")
        before = self.keys_path.read_text(encoding="utf-8")
        self.assertEqual(self.req("GET", "/props", key="k", device=DEV1)[0], 200)
        self.assertEqual(self.keys_path.read_text(encoding="utf-8"), before)


class FakeVision:
    """Stands in for the image encoder: records which sources reached it (Vision.load is where a file is read)."""

    def __init__(self, d):
        self.dir, self.seen = Path(d), []

    def encode(self, source):
        self.seen.append(source)
        rows = self.dir / f"img{len(self.seen)}.sve"
        rows.write_bytes(b"rows")
        return rows, 3


class LocalImages(Base):
    """A file on the workstation named as an image: in team mode only an owner key has it read (review of #310)."""

    def bodies(self, src):
        text = {"type": "text", "text": "what is it?"}
        anth = {"model": "m", "max_tokens": 16, "messages": [{"role": "user", "content": [
            text, {"type": "image", "source": {"type": "url", "url": src}}]}]}
        return {"/v1/chat/completions": {"model": "m", "max_tokens": 16, "messages": [{"role": "user", "content": [
                    text, {"type": "image_url", "image_url": {"url": src}}]}]},
                "/v1/messages": anth,
                "/v1/messages/count_tokens": anth,
                "/v1/responses": {"model": "m", "max_output_tokens": 16, "input": [{"role": "user", "content": [
                    {"type": "input_text", "text": "what is it?"}, {"type": "input_image", "image_url": src}]}]}}

    def setUp(self):
        super().setUp()
        self.vision = FakeVision(self.dir)
        self.img = self.dir / "secret.png"
        self.img.write_bytes(b"PNG not decoded by the fake encoder")

    def test_a_guest_cannot_name_a_file_and_learns_nothing_of_it(self):
        self.start(api_key="k", peer_override=REMOTE, vision=self.vision)
        ann = self.approved("ann")
        for src in (str(self.img), "file:///" + str(self.img).replace("\\", "/"), str(self.dir / "missing.png")):
            for path, body in self.bodies(src).items():
                with self.subTest(src=src, path=path):
                    code, r = self.req("POST", path, body, key=ann, device=DEV1)
                    self.assertEqual(code, 400, r)
                    self.assertIn("only with an owner key", json.dumps(r))
                    self.assertNotIn("missing", json.dumps(r))      # the same answer whether the file exists
        self.assertEqual(self.vision.seen, [])
        for path, body in self.bodies("data:image/png;base64,iVBORw0KGgo=").items():     # the guest's own bytes
            self.assertEqual(self.req("POST", path, body, key=ann, device=DEV1)[0], 200, path)

    def test_the_owner_may(self):
        self.start(api_key="k", vision=self.vision)                     # the owner, from this PC
        for path, body in self.bodies(str(self.img)).items():
            self.assertEqual(self.req("POST", path, body, key="k")[0], 200, path)
        self.assertEqual(self.vision.seen.count(str(self.img)), 4)

    def test_outside_team_mode_unchanged(self):
        self.start(api_key="k", team=False, peer_override=REMOTE, vision=self.vision)
        for path, body in self.bodies(str(self.img)).items():
            self.assertEqual(self.req("POST", path, body, key="k")[0], 200, path)
        self.assertEqual(self.vision.seen.count(str(self.img)), 4)


class RunsBelongToTheirIdentity(Base):
    """A run, and the permission and question prompts it opens, are its holder's (#310 task 3, ruling R3): an owner key
    "ann" launches a run that waits for her say-so; a guest "bob" cannot read, steer, stop or answer it, and is told
    404 - the same as for an id that does not exist, so ids cannot be probed."""
    RID = "run-team-0001"

    def setUp(self):
        super().setUp()
        self.proj = self.dir / "proj"
        self.proj.mkdir()
        (self.proj / "a.txt").write_text("hello\n", encoding="utf-8")
        self.outside = self.dir / "elsewhere"
        self.outside.mkdir()
        (self.outside / "b.txt").write_text("not in the project\n", encoding="utf-8")

    def start_agent(self, *scripts, **attrs):
        """The coding tools behind a scripted model (as serve/test_agent_chat.Fixture), in team mode."""
        tools = agent.AgentServer()
        hub = hub_from_config({}, None, builtins={"agent": tools})
        hub.start(wait=True)
        self.addCleanup(hub.close)
        tok = ByteTokenizer()
        self.svc = Service(ScriptedEngine(tok, list(scripts), max_context=200_000), tok,
                           ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.mcp, self.svc.agent = hub, tools
        self.start(**attrs)

    def launch(self, key, device):
        """ann's request, read on a thread of its own: (thread, set once a permission or question prompt came, the events read)."""
        asked, events = threading.Event(), []
        body = {"model": "m", "stream": True, "messages": [{"role": "user", "content": "look at b.txt"}],
                "strata_agent": {"cwd": str(self.proj), "session": "chat-1", "run": self.RID}}

        def read():
            c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
            try:
                c.request("POST", "/v1/chat/completions", body=json.dumps(body).encode(), headers={
                    "Content-Type": "application/json", "Authorization": "Bearer " + key, "X-Strata-Device": device})
                r = c.getresponse()
                events.append(r.status)
                for raw in r:
                    ln = raw.decode().strip()
                    if ln.startswith("data: {"):
                        ev = json.loads(ln[6:]).get("strata_mcp")
                        if ev:
                            events.append(ev)
                            if ev.get("event") in ("permission", "question"):
                                asked.set()
            finally:
                c.close()
        t = threading.Thread(target=read, daemon=True)
        t.start()
        return t, asked, events

    def test_guest_cannot_read_or_answer_another_guest_run(self):
        self.start_agent(tool_call("Read", file_path=str(self.outside / "b.txt")), "</think>\n\nDone.",
                         api_key="k", peer_override=REMOTE)
        ann = self.approved("ann", "owner")                       # R3: only an owner key launches coding runs in part 1
        bob = self.approved("bob", "guest", device=DEV2)
        t, asked, events = self.launch(ann, DEV1)
        self.assertTrue(asked.wait(20), events)
        pid = next(e["id"] for e in events[1:] if e["event"] == "permission")
        rid = self.RID
        nothing = {}
        for method, path, body in (("GET", f"/agent/run?id={rid}&from=0", None),
                                   ("POST", "/agent/steer", {"id": rid, "text": "from bob"}),
                                   ("POST", "/agent/cancel", {"id": rid}),
                                   ("POST", "/agent/permission", {"id": pid, "decision": "allow"}),
                                   ("POST", "/agent/question", {"id": pid, "answers": None})):
            with self.subTest(path=path):
                code, got = self.req(method, path, body, key=bob, device=DEV2)
                self.assertEqual(code, 404, got)
                gone = path.replace(rid, "run-none-0001") if method == "GET" else path
                other = {**body, "id": "run-none-0001" if body.get("id") == rid else "f" * 16} if body else None
                nothing[path] = self.req(method, gone, other, key=bob, device=DEV2)
                self.assertEqual(nothing[path], (code, got))             # word for word what an id that is not there gets
        self.assertTrue(t.is_alive())                                     # bob's tries neither stopped the run nor used up the prompt
        self.assertEqual(self.req("POST", "/agent/permission", {"id": pid, "decision": "deny"}, key=ann, device=DEV1),
                         (200, {"ok": True}))
        t.join(20)
        self.assertFalse(t.is_alive())
        self.assertEqual(events[0], 200)
        self.assertEqual([(e["id"], e["answer"]) for e in events[1:] if e["event"] == "answered"], [(pid, "deny")])
        # after the run: still not bob's; ann reads it again; the owner role may stop anyone's run
        self.assertEqual(self.req("GET", f"/agent/run?id={rid}&from=0", key=bob, device=DEV2)[0], 404)
        code, raw = self.req("GET", f"/agent/run?id={rid}&from=0", key=ann, device=DEV1)
        self.assertEqual(code, 200)
        self.assertIn(b"[DONE]", raw)
        self.assertEqual(self.req("GET", "/props", key="k", device=DEV1)[0], 403)        # the owner's own device waits for approval too
        self.assertTrue(self.svc.keys.decide_device("owner", DEV1, True, OWNER))
        self.assertEqual(self.req("POST", "/agent/cancel", {"id": rid}, key="k", device=DEV1), (200, {"ok": True, "found": True}))
        self.assertEqual(self.svc.runs.get(rid).who, "ann")

    def test_only_the_holder_answers_a_question_the_model_asked(self):
        q = [{"question": "Which library should we use?", "header": "Library", "multiSelect": False,
              "options": [{"label": "requests", "description": "the usual one"}, {"label": "httpx", "description": "async too"}]}]
        self.start_agent(tool_call("AskUserQuestion", questions=json.dumps(q)), "</think>\n\nDone.",
                         api_key="k", peer_override=REMOTE)
        ann = self.approved("ann", "owner")
        bob = self.approved("bob", "guest", device=DEV2)
        t, asked, events = self.launch(ann, DEV1)
        self.assertTrue(asked.wait(20), events)
        qid = next(e["id"] for e in events[1:] if e["event"] == "question")
        pick = {"Which library should we use?": ["httpx"]}
        code, got = self.req("POST", "/agent/question", {"id": qid, "answers": pick}, key=bob, device=DEV2)
        self.assertEqual(code, 404, got)
        self.assertEqual(self.req("POST", "/agent/question", {"id": "f" * 16, "answers": pick}, key=bob, device=DEV2), (code, got))
        self.assertTrue(t.is_alive())                                     # bob's try did not use the question up
        self.assertEqual(self.req("POST", "/agent/question", {"id": qid, "answers": pick}, key=ann, device=DEV1), (200, {"ok": True}))
        t.join(20)
        self.assertFalse(t.is_alive())
        self.assertEqual([(e["id"], e["answers"]) for e in events[1:] if e["event"] == "question_answered"], [(qid, pick)])


class GatedEngine(MockEngine):
    """The mock engine, slowed (delay_s) and held before its last two tokens while `go` is clear: a request that holds
    the engine, part of its answer written, for as long as a test needs."""

    def __init__(self, tok, script):
        super().__init__(tok, script, max_context=4096, delay_s=0.001)
        self.go = threading.Event()
        self.go.set()

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        hold = len(self.script) - 2
        for i, t in enumerate(super().generate(ids, max_new, sampling, cancel, embeddings)):
            if i == hold:
                while not self.go.wait(0.05):
                    if cancel.is_set():
                        return
            yield t


class MonitorByIdentity(Base):
    """#310 task 4: every request record says who sent it.  Another person's records show who, state, queue place and
    timings, never what was asked or answered (spec decision 2); the owner sees everything, as everyone does outside
    team mode.  Live: who holds the engine and who waits, in arrival order."""
    SCRIPT = "</think>\n\nthe answer is forty-two"

    def setUp(self):
        super().setUp()
        tok = ByteTokenizer()
        self.engine = GatedEngine(tok, self.SCRIPT)
        self.svc = Service(self.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.hstore = HistoryStore(self.dir / "history")
        self.svc.api_monitor = True
        self.start(api_key="k", peer_override=REMOTE)
        self.ann = self.approved("ann")
        self.bob = self.approved("bob", device=DEV2)
        self.boss = self.approved("boss", "owner")

    def tearDown(self):
        self.engine.go.set()                                      # a held request ends before the server closes
        super().tearDown()

    def ask(self, key, device, text):
        return self.req("POST", "/v1/chat/completions",
                        {"model": "m", "messages": [{"role": "user", "content": text}], "max_tokens": 64},
                        key=key, device=device)

    def ask_on_a_thread(self, key, device, text):
        """The same request from a thread of its own (no stdout redirect: that is not safe across threads)."""
        out = {}

        def go():
            c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
            try:
                c.request("POST", "/v1/chat/completions", body=json.dumps(
                    {"model": "m", "messages": [{"role": "user", "content": text}], "max_tokens": 64}).encode(),
                    headers={"Content-Type": "application/json", "Authorization": "Bearer " + key,
                             "X-Strata-Device": device})
                r = c.getresponse()
                out["status"], out["body"] = r.status, r.read()
            finally:
                c.close()
        t = threading.Thread(target=go, daemon=True)
        t.start()
        return t, out

    def wait_for(self, fn, timeout=20.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            v = fn()
            if v:
                return v
            time.sleep(0.02)
        self.fail("timed out")

    def live(self, key, device):
        code, body = self.req("GET", "/metrics", key=key, device=device)
        self.assertEqual(code, 200, body)
        return body["live"]

    def test_guest_sees_others_without_content(self):
        self.assertEqual(self.ask(self.ann, DEV1, "ann asks about apples")[0], 200)
        self.assertEqual(self.ask(self.bob, DEV2, "bob asks about pears")[0], 200)
        # the history on disk, the Monitor's last requests and the API monitor, as bob sees them
        for path, items in (("/metrics/requests", "items"), ("/metrics", "requests"), ("/api/requests", "requests")):
            with self.subTest(path=path):
                code, body = self.req("GET", path, key=self.bob, device=DEV2)
                self.assertEqual(code, 200, body)
                rows = {r["who"]: r for r in body[items]}
                self.assertEqual(set(rows), {"ann", "bob"})
                self.assertNotIn("apples", json.dumps(body))
                self.assertNotIn("preview", rows["ann"])
                if path == "/api/requests":
                    self.assertEqual(rows["ann"]["state"], "completed")
                    for k in ("started_at", "queue_s", "load_s", "first_token_s", "wallclock_s"):
                        self.assertIn(k, rows["ann"])
                else:
                    self.assertEqual(rows["bob"]["preview"], "bob asks about pears")
                    for k in ("finish", "time", "duration_s", "output_tokens", "prompt_tokens"):
                        self.assertIn(k, rows["ann"])
        # the owner sees both
        for path, items in (("/metrics/requests", "items"), ("/metrics", "requests")):
            code, body = self.req("GET", path, key=self.boss, device=DEV1)
            self.assertEqual({r["who"]: r["preview"] for r in body[items]},
                             {"ann": "ann asks about apples", "bob": "bob asks about pears"})

    def test_detail_of_another_guest_is_404(self):
        self.assertEqual(self.ask(self.ann, DEV1, "ann asks about apples")[0], 200)
        rid = self.svc.hstore.page(0, 1)["items"][0]["id"]
        mid = self.svc.api_requests[-1]["id"]
        nothing = self.req("GET", "/metrics/requests/r00000000ffff", key=self.bob, device=DEV2)
        self.assertEqual(nothing[0], 404)
        self.assertEqual(self.req("GET", f"/metrics/requests/{rid}", key=self.bob, device=DEV2), nothing)
        gone = self.req("GET", "/api/requests?id=000000000000", key=self.bob, device=DEV2)
        self.assertEqual(gone[0], 404)
        self.assertEqual(self.req("GET", f"/api/requests?id={mid}", key=self.bob, device=DEV2), gone)
        for key, device in ((self.ann, DEV1), (self.boss, DEV1)):
            code, body = self.req("GET", f"/metrics/requests/{rid}", key=key, device=device)
            self.assertEqual(code, 200)
            self.assertEqual((body["summary"]["who"], body["summary"]["preview"]), ("ann", "ann asks about apples"))
            self.assertEqual(body["detail"]["who"], "ann")
            code, body = self.req("GET", f"/api/requests?id={mid}", key=key, device=device)
            self.assertEqual(code, 200)
            self.assertIn("ann asks about apples", body["input"])

    def test_live_shows_holder_and_waiting(self):
        self.engine.go.clear()
        a, ra = self.ask_on_a_thread(self.ann, DEV1, "ann first")
        self.wait_for(lambda: self.live(self.bob, DEV2)["holder"] == "ann")
        b, rb = self.ask_on_a_thread(self.bob, DEV2, "bob second")
        live = self.wait_for(lambda: (x := self.live(self.bob, DEV2))["waiting"] and x)
        self.assertEqual(live["holder"], "ann")
        self.assertEqual([w["who"] for w in live["waiting"]], ["bob"])
        self.assertIsInstance(live["waiting"][0]["since"], float)
        self.assertEqual(live["queued"], 1)
        self.engine.go.set()
        a.join(20)
        b.join(20)
        self.assertEqual((ra["status"], rb["status"]), (200, 200))
        live = self.live(self.boss, DEV1)
        self.assertIsNone(live["holder"])
        self.assertEqual(live["waiting"], [])
        self.assertEqual(live["queued"], 0)

    def test_status_shows_the_answer_only_to_its_holder_and_the_owner(self):
        """GET /status carries the end of the answer being written (#212): in team mode only to the identity whose
        request it is and to the owner; anyone else gets the same shape with no text."""
        self.engine.go.clear()
        a, ra = self.ask_on_a_thread(self.ann, DEV1, "ann asks about apples")
        mine = self.wait_for(lambda: "the answer is" in ((x := self.req("GET", "/status", key=self.ann,
                                                                          device=DEV1)[1]).get("tail") or "") and x)
        self.assertEqual(mine["who"], "ann")
        code, other = self.req("GET", "/status", key=self.bob, device=DEV2)
        self.assertEqual(code, 200)
        self.assertEqual((other["who"], other["busy"], other["tail"]), ("ann", True, ""))
        self.assertEqual(set(other), set(mine))
        self.assertIn("the answer is", self.req("GET", "/status", key=self.boss, device=DEV1)[1]["tail"])
        # /v1/status and the Prometheus pages carry counts and timings only, while it runs and after
        for when in ("running", "done"):
            for path, headers in (("/v1/status", None), ("/metrics/prometheus", None),
                                  ("/metrics", {"Accept": "text/plain"})):
                with self.subTest(when=when, path=path):
                    code, body = self.req("GET", path, key=self.bob, device=DEV2, headers=headers)
                    self.assertEqual(code, 200)
                    text = body if isinstance(body, bytes) else json.dumps(body).encode()
                    self.assertNotIn(b"apples", text)
                    self.assertNotIn(b"answer", text)
            self.engine.go.set()
            a.join(20)
        self.assertEqual(ra["status"], 200)


class _Reached(Exception):
    """raised by the patched serve(): main() got as far as serving."""


class KeysFile(unittest.TestCase):
    def test_keys_file_of(self):
        self.assertIsNone(keys_file_of({}, None))                                  # no run config, no keys file
        self.assertIsNone(keys_file_of({"keys_file": "x.json"}, None))
        base = os.path.dirname(os.path.abspath("cfg/strata-x.json"))
        self.assertEqual(keys_file_of({}, "cfg/strata-x.json"), os.path.join(base, "strata-keys.json"))
        self.assertEqual(keys_file_of({"keys_file": "team/k.json"}, "cfg/strata-x.json"),
                         os.path.join(base, "team/k.json"))
        absolute = os.path.abspath("elsewhere/k.json")
        self.assertEqual(keys_file_of({"keys_file": absolute}, "cfg/strata-x.json"), absolute)
        for bad in ("", "  ", 5, None, ["k.json"]):
            with self.assertRaises(ValueError):
                keys_file_of({"keys_file": bad}, "cfg/strata-x.json")

    def run_main(self, cfg, keys=None):
        """-> (the service main() built, stdout).  `keys`: the keys file to write beside the config first."""
        import serve.server as S
        seen = {}

        def fake_serve(svc, host=None, port=None):
            seen["svc"] = svc
            raise _Reached

        out = io.StringIO()
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ),                 mock.patch.object(S, "serve", fake_serve), contextlib.redirect_stdout(out),                 contextlib.redirect_stderr(io.StringIO()):
            os.environ.pop("STRATA_API_KEY", None)
            p = Path(d) / "cfg.json"
            p.write_text(json.dumps(cfg), encoding="utf-8")
            if keys is not None:
                (Path(d) / "strata-keys.json").write_text(json.dumps(keys), encoding="utf-8")
            with mock.patch.object(sys, "argv", ["server.py", "--engine", "mock", "--port", "0", "--config", str(p)]):
                with self.assertRaises(_Reached):
                    S.main()
            svc = seen["svc"]
            return svc, out.getvalue(), os.path.join(d, "strata-keys.json")

    def test_main_builds_the_store_beside_the_config(self):
        svc, out, path = self.run_main({"api_key": "k"})
        self.assertEqual(os.path.normcase(svc.keys.path), os.path.normcase(path))
        self.assertIs(svc.keys.team_mode(), False)                                 # no file: as before
        self.assertEqual(svc.keys.match("k"), OWNER)                               # the legacy key is the owner
        self.assertNotIn("team keys", out)
        svc, out, path = self.run_main({"api_key": "k"}, keys={"keys": [], "log": []})
        self.assertIs(svc.keys.team_mode(), True)
        self.assertIn("team keys", out)

    def test_a_bad_keys_file_value_stops_the_start(self):
        with self.assertRaises(SystemExit):
            self.run_main({"keys_file": 5})


if __name__ == "__main__":
    unittest.main()
