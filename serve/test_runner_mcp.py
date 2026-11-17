"""serve/test_runner_mcp.py - My PC's MCP servers set up from the web app (#338).

The runner's jobs `mcp.config`, `mcp.config.set`, `mcp.import.list` and `mcp.import.add` (serve/runner.py Executor), and
the server's routes that relay them for the caller's own runner on the caller's own device (`GET /mcp/config?runner=`,
`POST /mcp/config {"runner"}`, `GET /import?runner=`, `POST /import {"runner"}`; serve/server.py).

    python -m pytest serve/test_runner_mcp.py -q
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import mcp_admin, page_routes, runner as runner_mod  # noqa: E402
from serve.identity import OWNER  # noqa: E402
from serve.test_identity_http import DEV1, DEV2  # noqa: E402
from serve.test_harness import tool_call  # noqa: E402
from serve.test_runner import FAKE_MCP, Base, wait_for  # noqa: E402
from serve.test_runner_routing import DONE  # noqa: E402

FAKE = {"command": sys.executable, "args": [FAKE_MCP]}


def job(ex, kind, **payload):
    return ex.handle({"kind": kind, **payload}, None)


class Jobs(unittest.TestCase):
    """The runner's own jobs, on an Executor with no server."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.d = Path(self._tmp.name)
        self.home = self.d / "home"
        self.home.mkdir()
        self.cfg = self.d / "runner.json"
        p = mock.patch.object(runner_mod, "_say", lambda text: None)
        p.start()
        self.addCleanup(p.stop)
        p = mock.patch.dict("os.environ", {"APPDATA": str(self.home / "AppData")})    # not this PC's own Claude Desktop
        p.start()
        self.addCleanup(p.stop)

    def executor(self, cfg=None):
        if cfg is not None:
            self.cfg.write_text(json.dumps(cfg), encoding="utf-8")
        ex = runner_mod.Executor(self.home, self.cfg, self.d)
        self.addCleanup(ex.close)
        return ex

    def test_view_without_runner_json(self):
        ex = self.executor()
        got = job(ex, "mcp.config")
        self.assertEqual(got["status"], 200, got)
        body = got["body"]
        self.assertEqual((body["servers"], body["editable"], body["config_file"]), ([], True, "runner.json"))
        self.assertEqual(set(body["settings"]), {"timeout_s", "max_result_chars", "max_rounds"})

    def test_first_add_makes_runner_json(self):
        ex = self.executor()
        self.assertFalse(self.cfg.exists())
        got = job(ex, "mcp.config.set", body={"servers": {"fake": FAKE}})
        self.assertEqual(got["status"], 200, got)
        self.assertEqual(json.loads(self.cfg.read_text(encoding="utf-8")), {"mcp_servers": {"fake": FAKE}})
        self.assertEqual([s["name"] for s in got["body"]["servers"]], ["fake"])
        self.assertIn("fake", ex.hub.servers)                          # the hub was started again from the file
        self.assertTrue(wait_for(lambda: any(g["name"] == "fake" and g["tools"] for g in ex.groups()), 20))
        self.assertEqual(ex.cfg["mcp_servers"], {"fake": FAKE})        # the executor reads its config again

    def test_other_keys_and_the_key_in_use_are_kept(self):
        ex = self.executor({"mcpServers": {"old": {"command": "x"}}, "import": {"skills": {"enabled": False}}})
        got = job(ex, "mcp.config.set", body={"servers": {"new": {"url": "http://127.0.0.1:9/mcp"}}})
        self.assertEqual(got["status"], 200, got)
        cfg = json.loads(self.cfg.read_text(encoding="utf-8"))
        self.assertEqual(cfg, {"mcpServers": {"new": {"url": "http://127.0.0.1:9/mcp"}}, "import": {"skills": {"enabled": False}}})

    def test_secrets_are_masked_and_kept(self):
        ex = self.executor({"mcp_servers": {"s": {"command": "x", "env": {"TOKEN": "secret"}}}})
        body = job(ex, "mcp.config")["body"]
        self.assertEqual(body["servers"][0]["env"], {"TOKEN": mcp_admin.MASK})
        self.assertNotIn("secret", json.dumps(body))
        got = job(ex, "mcp.config.set", body={"servers": {"s": {"command": "y", "env": {"TOKEN": mcp_admin.MASK}}}})
        self.assertEqual(got["status"], 200, got)
        self.assertEqual(json.loads(self.cfg.read_text(encoding="utf-8"))["mcp_servers"]["s"],
                         {"command": "y", "env": {"TOKEN": "secret"}})

    def test_a_bad_change_gives_the_field_errors_and_writes_nothing(self):
        ex = self.executor({"mcp_servers": {}})
        before = self.cfg.read_bytes()
        got = job(ex, "mcp.config.set", body={"servers": {"bad name": {"command": ""}}})
        self.assertEqual(got["status"], 400, got)
        fields = {(e["server"], e["field"]) for e in got["body"]["error"]["fields"]}
        self.assertEqual(fields, {("bad name", "name"), ("bad name", "command")})
        self.assertEqual(self.cfg.read_bytes(), before)

    def test_the_coding_tools_name_and_guest_marks_are_refused(self):
        ex = self.executor({"mcp_servers": {}})
        got = job(ex, "mcp.config.set", body={"servers": {"agent": {"command": "x"}}})
        self.assertEqual(got["status"], 400, got)
        self.assertEqual(got["body"]["error"]["fields"][0]["field"], "name")
        got = job(ex, "mcp.config.set", body={"servers": {}, "guest_mcp": []})
        self.assertEqual(got["status"], 400, got)
        self.assertEqual(got["body"]["error"]["fields"][0]["field"], "guest_mcp")
        self.assertEqual(json.loads(self.cfg.read_text(encoding="utf-8")), {"mcp_servers": {}})

    def test_not_an_object(self):
        ex = self.executor()
        self.assertEqual(job(ex, "mcp.config.set", body=[1])["status"], 400)
        self.assertEqual(job(ex, "mcp.config.set")["status"], 400)

    def test_delete_stops_the_server(self):
        ex = self.executor({"mcp_servers": {"fake": FAKE}})
        self.assertIn("fake", ex.hub.servers)
        got = job(ex, "mcp.config.set", body={"servers": {}})
        self.assertEqual(got["status"], 200, got)
        self.assertNotIn("fake", ex.hub.servers if ex.hub is not None else {})
        self.assertFalse(any(g["name"] == "fake" for g in ex.groups()))

    def _claude_json(self, servers):
        (self.home / ".claude.json").write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")

    def test_import_list_and_add(self):
        self._claude_json({"files": {"command": "npx", "args": ["-y", "fs"], "env": {"K": "v"}}})
        ex = self.executor()
        got = job(ex, "mcp.import.list")
        self.assertEqual(got["status"], 200, got)
        body = got["body"]
        self.assertNotIn("skills", body)                               # only the MCP servers: skills stay read-only here
        self.assertNotIn("memory", body)
        [claude] = body["mcp"]["harnesses"]
        [cand] = claude["servers"]
        self.assertEqual((cand["name"], cand["state"], cand["env"]), ("files", "available", {"K": mcp_admin.MASK}))
        got = job(ex, "mcp.import.add", body={"import_mcp": {"harness": "claude", "name": "files"}})
        self.assertEqual(got["status"], 200, got)
        self.assertEqual(got["body"]["mcp"]["harnesses"][0]["servers"][0]["state"], "already")
        self.assertEqual(json.loads(self.cfg.read_text(encoding="utf-8"))["mcp_servers"]["files"],
                         {"command": "npx", "args": ["-y", "fs"], "env": {"K": "v"}})
        self.assertIn("files", ex.hub.servers)
        got = job(ex, "mcp.import.add", body={"import_mcp": {"harness": "claude", "name": "files"}})
        self.assertEqual(got["status"], 409, got)

    def test_import_rescan_sees_a_new_file(self):
        ex = self.executor()
        self.assertEqual(job(ex, "mcp.import.list")["body"]["mcp"]["harnesses"], [])
        self._claude_json({"files": {"command": "npx"}})
        self.assertEqual(job(ex, "mcp.import.list")["body"]["mcp"]["harnesses"], [])
        self.assertEqual(len(job(ex, "mcp.import.list", rescan=True)["body"]["mcp"]["harnesses"]), 1)

    def test_import_add_takes_only_an_import(self):
        ex = self.executor()
        for body in ({"skills": {"enabled": False}}, {"memory": {"on": []}}, {}, None):
            got = job(ex, "mcp.import.add", body=body)
            self.assertEqual(got["status"], 400, (body, got))
        self.assertFalse(self.cfg.exists())

    def test_the_change_is_announced(self):
        ex = self.executor({"mcp_servers": {"a": {"command": "x"}, "b": {"command": "y"}}})
        seen = []
        ex.on_change = seen.append
        job(ex, "mcp.config.set", body={"servers": {"a": {"command": "x2"}, "c": {"command": "z"}}})
        self.assertEqual(seen, [{"added": ["c"], "removed": ["b"], "changed": ["a"]}])
        job(ex, "mcp.config.set", body={"servers": {"bad name": {}}})
        self.assertEqual(len(seen), 1)                                   # a refused change is no change


class Relay(unittest.TestCase):
    def test_statuses_and_an_old_runner(self):
        self.assertEqual(page_routes.relay({"status": 409, "body": {}}, "pc", page_routes.ADMIN_STATUSES), (409, {}))
        self.assertEqual(page_routes.relay({"status": 409, "body": {}}, "pc")[0], 502)        # a page job: as before
        code, body = page_routes.relay({"error": "unknown", "message": "this runner does not know the job"}, "pc")
        self.assertEqual((code, body["error"]["type"]), (409, "runner_outdated"))


class PlaceTags(unittest.TestCase):
    """#338: where each MCP tool runs, in its description, when the user's PC's MCP servers are offered."""

    @staticmethod
    def server(name, kind, **more):
        return SimpleNamespace(name=name, kind=kind, status="ready", tools=[{"name": "go", "description": "Does it."}], **more)

    def hub(self, *servers):
        from serve.mcp import McpHub
        return McpHub({}).with_servers({s.name: s for s in servers})

    def test_both_pcs_are_tagged(self):
        mine = self.server("my_play", "runner", group_kind="mcp", info={"machine": "ann-pc"})
        got = {t["name"]: t["description"] for t in self.hub(self.server("files", "stdio"), mine,
                                                              self.server("my_agent_x", "runner", group_kind="coding"),
                                                              self.server("skills", "builtin")).template_tools()}
        self.assertEqual(got["my_play__go"], "[Runs on the user's PC (ann-pc) — localhost there is the user's machine.] Does it.")
        self.assertEqual(got["files__go"], "[Runs on the central PC (the server).] Does it.")
        self.assertEqual(got["my_agent_x__go"], "Does it.")
        self.assertEqual(got["skills__go"], "Does it.")

    def test_without_a_runner_group_the_tools_are_byte_identical(self):
        hub = self.hub(self.server("files", "stdio"), self.server("web", "http"), self.server("my_agent", "runner", group_kind="coding"))
        self.assertEqual([t["description"] for t in hub.template_tools()], ["Does it."] * 3)
        mine = self.server("my_play", "runner", group_kind="mcp", info={"machine": "ann-pc"})
        self.assertEqual([t["description"] for t in self.hub(self.server("files", "stdio"), mine).template_tools(skip_servers={"my_play"})],
                         ["Does it."])                            # switched off: not offered, so no tags either


class Routes(Base):
    """The server's routes for My PC's MCP servers, with a real runner (team mode: owner "k" on DEV0, ann on DEV1)."""

    def setUp(self):
        super().setUp()
        p = mock.patch.dict("os.environ", {"APPDATA": str(self.dir / "AppData")})    # not this PC's own Claude Desktop
        p.start()
        self.addCleanup(p.stop)
        self.start_routing()

    def my_tools(self, loop):
        code, body = self.req("GET", f"/mcp?runner={loop.id}", key=self.ann, device=DEV1)
        return {t["name"] for s in body.get("servers", []) for t in s["tools"]} if code == 200 else set()

    def test_ann_sets_up_her_own_pc(self):
        loop = self.runner(self.ann)
        code, body = self.req("GET", f"/mcp/config?runner={loop.id}", key=self.ann, device=DEV1)
        self.assertEqual((code, body["servers"], body["editable"]), (200, [], True), body)
        code, body = self.req("POST", "/mcp/config", {"runner": loop.id, "servers": {"fake": FAKE}}, key=self.ann, device=DEV1)
        self.assertEqual(code, 200, body)
        self.assertEqual([s["name"] for s in body["servers"]], ["fake"])
        self.assertEqual(json.loads((loop.dir / "runner.json").read_text(encoding="utf-8")), {"mcp_servers": {"fake": FAKE}})
        self.assertEqual(json.loads(self.cfg_path.read_text(encoding="utf-8")), {})       # the server's own config: untouched
        t0 = time.monotonic()
        self.assertTrue(wait_for(lambda: "my_fake__echo" in self.my_tools(loop), 20), self.said)
        self.assertLess(time.monotonic() - t0, runner_mod.runner_hub.POLL_MAX_S + 15)
        code, body = self.req("POST", "/mcp/config", {"runner": loop.id, "servers": {}}, key=self.ann, device=DEV1)
        self.assertEqual(code, 200, body)
        self.assertTrue(wait_for(lambda: "my_fake__echo" not in self.my_tools(loop), 10), self.said)

    def test_the_hello_follows_at_once(self):
        loop = self.runner(self.ann)
        with mock.patch.object(runner_mod.runner_hub, "POLL_MAX_S", 25.0):
            loop.poll_s = 25.0
            wait_for(lambda: False, 2.5)                                   # the runner's poll now waits the long hold
            code, body = self.req("POST", "/mcp/config", {"runner": loop.id, "servers": {"fake": FAKE}}, key=self.ann, device=DEV1)
            self.assertEqual(code, 200, body)
            self.assertTrue(wait_for(lambda: "my_fake__echo" in self.my_tools(loop), 10), self.said)

    def test_a_bad_change_gives_the_field_errors(self):
        loop = self.runner(self.ann)
        code, body = self.req("POST", "/mcp/config", {"runner": loop.id, "servers": {"x": {"url": "ftp://a"}}}, key=self.ann, device=DEV1)
        self.assertEqual(code, 400, body)
        self.assertEqual(body["error"]["fields"][0]["field"], "url")

    def test_the_owner_cannot_touch_a_guests_pc(self):
        loop = self.runner(self.ann)
        for method, path, body in (("GET", f"/mcp/config?runner={loop.id}", None),
                                   ("POST", "/mcp/config", {"runner": loop.id, "servers": {"fake": FAKE}}),
                                   ("GET", f"/import?runner={loop.id}", None),
                                   ("POST", "/import", {"runner": loop.id, "import_mcp": {"harness": "claude", "name": "x"}})):
            code, got = self.req(method, path, body, key="k", device="device-0000")
            self.assertEqual((code, got["error"]["type"]), (404, "runner_not_found"), (method, path))
        self.assertEqual(json.loads((loop.dir / "runner.json").read_text(encoding="utf-8")), {})

    def test_another_guest_cannot_either(self):
        loop = self.runner(self.ann)
        code, got = self.req("POST", "/mcp/config", {"runner": loop.id, "servers": {"fake": FAKE}}, key=self.bob, device=DEV1)
        self.assertEqual((code, got["error"]["type"]), (404, "runner_not_found"))

    def test_only_from_the_runners_own_device(self):
        loop = self.runner(self.ann)
        self.assertEqual(self.req("GET", "/props", key=self.ann, device=DEV2)[0], 403)      # it asks to be let in
        self.assertTrue(self.svc.keys.decide_device("ann", DEV2, True, OWNER))
        for method, path, body in (("GET", f"/mcp/config?runner={loop.id}", None),
                                   ("POST", "/mcp/config", {"runner": loop.id, "servers": {"fake": FAKE}}),
                                   ("GET", f"/import?runner={loop.id}", None),
                                   ("POST", "/import", {"runner": loop.id, "import_mcp": {"harness": "claude", "name": "x"}})):
            code, got = self.req(method, path, body, key=self.ann, device=DEV2)
            self.assertEqual((code, got["error"]["type"]), (403, "runner_device"), (method, path))
        self.assertEqual(json.loads((loop.dir / "runner.json").read_text(encoding="utf-8")), {})

    def test_a_guest_without_a_runner_still_cannot_change_the_servers_pc(self):
        self.runner(self.ann)
        code, got = self.req("POST", "/mcp/config", {"servers": {"fake": FAKE}}, key=self.ann, device=DEV1)
        self.assertEqual(code, 403, got)
        code, got = self.req("POST", "/mcp/config", {"runner": None, "servers": {"fake": FAKE}}, key=self.ann, device=DEV1)
        self.assertEqual(code, 403, got)
        self.assertEqual(json.loads(self.cfg_path.read_text(encoding="utf-8")), {})

    def test_csrf_guards(self):
        loop = self.runner(self.ann)
        code, _ = self.req("POST", "/mcp/config", {"runner": loop.id, "servers": {}}, key=self.ann, device=DEV1,
                           headers={"Origin": "https://evil.example"})
        self.assertEqual(code, 403)
        code, _ = self.req("POST", "/import", json.dumps({"runner": loop.id, "rescan": True}).encode(), key=self.ann, device=DEV1,
                           headers={"Content-Type": "text/plain"})
        self.assertEqual(code, 415)

    def test_import_from_the_pcs_apps(self):
        loop = self.runner(self.ann)
        (loop.dir / "home" / ".claude.json").write_text(json.dumps({"mcpServers": {"fake": FAKE}}), encoding="utf-8")
        code, body = self.req("GET", f"/import?runner={loop.id}&rescan=1", key=self.ann, device=DEV1)
        self.assertEqual(code, 200, body)
        self.assertEqual(body["mcp"]["harnesses"][0]["servers"][0]["state"], "available")
        code, body = self.req("POST", "/import", {"runner": loop.id, "skills": {"enabled": False}}, key=self.ann, device=DEV1)
        self.assertEqual(code, 400, body)
        code, body = self.req("POST", "/import", {"runner": loop.id, "import_mcp": {"harness": "claude", "name": "fake"}},
                              key=self.ann, device=DEV1)
        self.assertEqual(code, 200, body)
        self.assertEqual(body["mcp"]["harnesses"][0]["servers"][0]["state"], "already")
        self.assertEqual(json.loads((loop.dir / "runner.json").read_text(encoding="utf-8")), {"mcp_servers": {"fake": FAKE}})
        self.assertTrue(wait_for(lambda: "my_fake__echo" in self.my_tools(loop), 20), self.said)

    def test_offline_runner(self):
        loop = self.runner(self.ann)
        r = self.svc.runner_hub.get_for(loop.id, SimpleNamespace(name="ann"))
        loop.stop.set()
        loop.thread.join(10)
        with mock.patch.object(runner_mod.runner_hub, "OFFLINE_S", 0.0):
            wait_for(lambda: not r.online(), 5)
            code, body = self.req("GET", f"/mcp/config?runner={loop.id}", key=self.ann, device=DEV1)
        self.assertEqual((code, body["error"]["type"]), (409, "runner_offline"))

    def test_an_old_runner_says_so(self):
        loop = self.runner(self.ann)
        real = loop.executor.handle

        def old(msg, link):
            if str(msg.get("kind")).startswith("mcp."):
                return {"error": "unknown", "message": "this runner does not know the job 'mcp.config': update Strata Desktop"}
            return real(msg, link)
        with mock.patch.object(loop.executor, "handle", old):
            code, body = self.req("GET", f"/mcp/config?runner={loop.id}", key=self.ann, device=DEV1)
        self.assertEqual((code, body["error"]["type"]), (409, "runner_outdated"), body)
        self.assertIn("update Strata Desktop", body["error"]["message"])

    def test_the_state_callback_hears_of_the_change(self):
        events = []
        loop = self.runner(self.ann)
        loop.on_state = lambda e, d: events.append((e, d))
        code, body = self.req("POST", "/mcp/config", {"runner": loop.id, "servers": {"s": {"command": "x"}}}, key=self.ann, device=DEV1)
        self.assertEqual(code, 200, body)
        self.assertTrue(wait_for(lambda: ("mcp_changed", {"added": ["s"], "removed": [], "changed": []}) in events, 5), events)



class ServerChat(Base):
    """#338: My PC's MCP servers in a chat whose project is on the Server PC (`strata_mcp_runner`): offered beside the server's own
    servers while the runner is online, run through the runner; its coding tools and skills are not."""

    def setUp(self):
        super().setUp()
        p = mock.patch.dict("os.environ", {"APPDATA": str(self.dir / "AppData")})
        p.start()
        self.addCleanup(p.stop)

    def chat(self, key, extra, device=DEV1):
        """A streamed chat with the MCP tools on and no coding tools -> (status, the strata_mcp events)."""
        body = {"model": "m", "stream": True, "max_tokens": 1000, "strata_mcp": True,
                "messages": [{"role": "user", "content": "do it"}], **extra}
        code, raw = self.req("POST", "/v1/chat/completions", body, key=key, device=device)
        if code != 200:
            return code, raw
        events = []
        for ln in raw.decode().splitlines():
            if ln.startswith("data: {"):
                ev = json.loads(ln[6:]).get("strata_mcp")
                if ev:
                    events.append(ev)
        return code, events

    def ready(self, loop):
        def offered():
            code, body = self.req("GET", f"/mcp?runner={loop.id}", key=self.ann, device=DEV1)
            return code == 200 and any(t["name"] == "my_fake__echo" for s in body["servers"] for t in s["tools"])
        self.assertTrue(wait_for(offered, 20), self.said)

    def test_offered_and_run_through_the_runner(self):
        self.start_routing(tool_call("my_fake__echo", text="hi from my PC"), DONE)
        loop = self.runner(self.ann, cfg={"mcp_servers": {"fake": FAKE}})
        self.ready(loop)
        code, events = self.chat(self.ann, {"strata_mcp_runner": loop.id})
        self.assertEqual(code, 200, events)
        names = self.offered[0][0]
        self.assertIn("my_fake__echo", names)
        self.assertFalse([n for n in names if n in ("Read", "Bash", "use_skill") or n.startswith("my_skills")], names)
        desc = next(t["description"] for t in self.offered[0][1].template_tools() if t["name"] == "my_fake__echo")
        self.assertTrue(desc.startswith(f"[Runs on the user's PC ({loop.executor.machine}) — localhost there is the user's machine.] "), desc)
        call = next(e for e in events if e["event"] == "call")
        self.assertEqual((call["name"], call["server"], call.get("machine")), ("my_fake__echo", "my_fake", "my"))
        self.assertIn("hi from my PC", next(e for e in events if e["event"] == "result")["text"])

    def test_switched_off_by_name(self):
        self.start_routing()
        loop = self.runner(self.ann, cfg={"mcp_servers": {"fake": FAKE}})
        self.ready(loop)
        code, _ = self.chat(self.ann, {"strata_mcp_runner": loop.id, "strata_mcp_off": ["my_fake"]})
        self.assertEqual(code, 200)
        self.assertNotIn("my_fake__echo", self.offered[0][0] if self.offered else [])     # (the guest has no other server)

    def test_offline_is_not_offered(self):
        self.start_routing()
        loop = self.runner(self.ann, cfg={"mcp_servers": {"fake": FAKE}})
        self.ready(loop)
        r = self.svc.runner_hub.get_for(loop.id, SimpleNamespace(name="ann"))
        loop.stop.set()
        loop.thread.join(10)
        with mock.patch.object(runner_mod.runner_hub, "OFFLINE_S", 0.0):
            wait_for(lambda: not r.online(), 5)
            code, _ = self.chat(self.ann, {"strata_mcp_runner": loop.id})
        self.assertEqual(code, 200)
        self.assertFalse([n for n in (self.offered[0][0] if self.offered else []) if n.startswith("my_")])

    def test_another_identitys_runner_is_refused(self):
        self.start_routing()
        loop = self.runner(self.ann, cfg={"mcp_servers": {"fake": FAKE}})
        for key, device in ((self.bob, DEV1), ("k", "device-0000")):
            code, body = self.chat(key, {"strata_mcp_runner": loop.id}, device=device)
            self.assertEqual((code, body["error"]["type"]), (404, "runner_not_found"), key)
        code, body = self.chat(self.ann, {"strata_mcp_runner": 7})
        self.assertEqual((code, body["error"]["type"]), (404, "runner_not_found"))

    def test_without_a_runner_the_request_is_as_before(self):
        self.start_routing()
        loop = self.runner(self.ann, cfg={"mcp_servers": {"fake": FAKE}})
        self.ready(loop)
        code, _ = self.chat(self.ann, {})
        self.assertEqual(code, 200)
        self.assertFalse([n for n in (self.offered[0][0] if self.offered else []) if n.startswith("my_")])


if __name__ == "__main__":
    unittest.main()
