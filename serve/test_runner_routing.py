"""serve/test_runner_routing.py - each request routes to its own runner's groups (#310 part 2, task 4): a chat that names
`strata_agent.runner` gets its own view of the tool groups (the server's, the server's `agent` cut to WebFetch /
WebSearch / Task, then the runner's `my_agent`, `my_skills`, `my_<name>`), its coding calls go to that runner, its
`call` events say `machine: "my"` and are followed by `remote` events; another identity's runner is a 404, an offline
one a 409; a request without a runner is unchanged.  Over real HTTP with a scripted model and `FakeRunner`.

    python -m pytest serve/test_runner_routing.py -q
"""
from __future__ import annotations

import contextlib
import http.client
import io
import json
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import agent, runner_hub, subagent, web  # noqa: E402
from serve import server as server_mod  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.mcp import McpHub, hub_from_config  # noqa: E402
from serve.runner_fake import FakeRunner  # noqa: E402
from serve.server import ByteTokenizer, Service  # noqa: E402
from serve.test_harness import tool_call  # noqa: E402
from serve.test_identity_http import DEV1, DEV2, REMOTE, ROOT, FakeMcp  # noqa: E402
from serve.test_mcp import ScriptedEngine  # noqa: E402
from serve.test_runner_http import DEV0, RunnerBase  # noqa: E402

CWD = "C:\\Users\\ann\\proj"                    # a folder on the engineer's PC, absent on this disk
CODING_NAMES = ["Read", "Write", "Edit", "Glob", "Grep", "Bash", "TodoWrite", "ExitPlanMode"]


def coding(names=CODING_NAMES):
    return {"name": "agent", "kind": "coding", "status": "ready", "error": None,
            "tools": [{"name": n, "description": f"{n} on the PC", "inputSchema": {"type": "object"}} for n in names]}


FS = {"name": "fs", "kind": "mcp", "status": "ready", "error": None,
      "tools": [{"name": "x", "description": "fs x on the PC", "inputSchema": {"type": "object"}}]}
DONE = "</think>\n\nDone."
CALL_KEYS = {"event", "id", "name", "server", "tool", "arguments", "round"}


class Fixture(RunnerBase):
    def setUp(self):
        super().setUp()
        self.jobs: dict[str, list] = {}                 # runner name -> the messages its fake got
        self.server_calls: list[str] = []              # the server's own coding tools that ran

    def start_routing(self, *scripts, cfg=None, servers=None, with_agent=True, with_hub=True, team=True):
        """The server's coding tools (call recorded) and `servers` (FakeMcp) behind a scripted model."""
        tools = None
        builtins = dict(servers or {})
        if with_agent:
            tools = agent.AgentServer()
            web.install(tools)
            subagent.install(tools)
            server_call = tools.call

            def call(tool, *a, **k):
                self.server_calls.append(tool)
                return server_call(tool, *a, **k)
            tools.call = call
            builtins["agent"] = tools
        tok = ByteTokenizer()
        self.engine = ScriptedEngine(tok, list(scripts) or [DONE], max_context=200_000)
        self.svc = Service(self.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        if with_hub:
            hub = hub_from_config({}, None, builtins=builtins)
            hub.start(wait=True)
            self.addCleanup(hub.close)
            self.svc.mcp = hub
        self.svc.agent = tools
        if team:
            self.team()
        else:
            self.start(team=False)
        self.cfg_path.write_text(json.dumps(cfg or {}), encoding="utf-8")
        self.offered: list[tuple[list[str], object]] = []
        orig = server_mod.run_with_mcp

        def wrap(svc, hub, messages, tools_, *a, **k):
            self.offered.append(([t["name"] for t in tools_ or []], hub))
            hook, self.on_offer = getattr(self, "on_offer", None), None     # once, between the list and the first call
            if hook is not None:
                hook()
            return orig(svc, hub, messages, tools_, *a, **k)
        p = mock.patch.object(server_mod, "run_with_mcp", wrap)
        p.start()
        self.addCleanup(p.stop)

    def fake(self, key, device=DEV1, groups=None, machine="ann-pc", facts=True, delay=0.0, name=None, vanish=False):
        """A runner for `key` that answers run.start with facts (unless `facts` is False) and every tool job (unless
        `vanish`: it takes the tool job and stops polling without an answer, as a PC whose lid was closed)."""
        name = name or machine
        got = self.jobs.setdefault(name, [])
        groups = [coding(), FS] if groups is None else groups
        me = []

        def on_job(msg, reply):
            got.append(msg)
            if msg.get("kind") == "run.start" and facts:
                names = [t["name"] for g in groups if g["kind"] == "coding" for t in g["tools"]]
                reply({"result": {"run": "run-" + name, "cwd": msg.get("cwd"), "dirs": [], "shell": "Git Bash",
                                  "platform": "win32", "git": False, "memory": [], "tools": names or []}, "exec_ms": 1})
            elif msg.get("kind") == "tool" and vanish:
                me[0].stop()
            elif msg.get("kind") == "tool":
                time.sleep(delay)
                reply({"result": {"content": [{"type": "text", "text": f"ran {msg['tool']} on {machine}"}]},
                       "exec_ms": 2})
        f = FakeRunner(self.port, key, device, groups, on_job, machine=machine)
        me.append(f)
        self.fakes.append(f)
        return f

    def chat(self, key, sa=None, device=DEV1, mcp=False):
        """POST a streamed chat -> (status, the strata_mcp events, the answer's text) or (status, error body, None)."""
        body = {"model": "m", "stream": True, "max_tokens": 1000, "messages": [{"role": "user", "content": "look at a.txt"}]}
        if sa is not None:
            body["strata_agent"] = sa
        if mcp:
            body["strata_mcp"] = True
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        try:
            headers = {"Content-Type": "application/json", "X-Strata-Device": device}
            if key:
                headers["Authorization"] = "Bearer " + key
            c.request("POST", "/v1/chat/completions", body=json.dumps(body).encode(), headers=headers)
            r = c.getresponse()
            if r.status != 200:
                return r.status, json.loads(r.read()), None
            events, text = [], []
            for raw in r:
                ln = raw.decode().strip()
                if not ln.startswith("data: {"):
                    continue
                d = json.loads(ln[6:])
                if d.get("strata_mcp"):
                    events.append(d["strata_mcp"])
                for ch in d.get("choices") or []:
                    text.append((ch.get("delta") or {}).get("content") or "")
            return 200, events, "".join(text)
        finally:
            c.close()

    def tool_jobs(self, name="ann-pc"):
        return [m for m in self.jobs.get(name, []) if m.get("kind") == "tool"]


class Routing(Fixture):
    def test_guest_runs_tools_on_own_runner(self):
        self.start_routing(tool_call("Read", file_path=CWD + "\\a.txt"), DONE)
        f = self.fake(self.ann)
        code, events, text = self.chat(self.ann, {"cwd": CWD, "session": "chat-1", "runner": f.id})
        self.assertEqual(code, 200, events)
        self.assertTrue(text.endswith("Done."), text)
        jobs = self.tool_jobs()
        self.assertEqual(len(jobs), 1, self.jobs)
        self.assertEqual((jobs[0]["group"], jobs[0]["tool"], jobs[0]["run"], jobs[0]["session"]),
                         ("agent", "Read", "run-ann-pc", "chat-1"))
        self.assertEqual(self.server_calls, [])             # the server's own coding tools were never reached
        result = next(e for e in events if e["event"] == "result")
        self.assertIn("ran Read on ann-pc", result["text"])

    def test_guest_cannot_route_to_another_runner(self):
        self.start_routing(tool_call("Read", file_path=CWD + "\\a.txt"), DONE)
        bobs = self.fake(self.bob, machine="bob-pc")
        code, body, _ = self.chat(self.ann, {"cwd": CWD, "runner": bobs.id})
        self.assertEqual(code, 404)
        self.assertEqual(body, {"error": {"type": "runner_not_found",
                                          "message": "there is no runner with that id for this key"}})
        self.assertEqual(self.chat(self.ann, {"cwd": CWD, "runner": "r0000000000000000"})[:2], (code, body))
        self.assertEqual(self.chat("k", {"cwd": CWD, "runner": bobs.id}, device=DEV0)[:2], (code, body))   # the owner neither
        time.sleep(0.3)
        self.assertEqual(self.jobs["bob-pc"], [])           # nothing reached bob's PC
        self.assertEqual(self.server_calls, [])

    def test_guest_without_runner_still_403(self):
        self.start_routing()
        for sa in ({"cwd": CWD}, {"cwd": CWD, "runner": 5}, {"cwd": CWD, "runner": None}):
            code, body, _ = self.chat(self.ann, sa)
            self.assertEqual((code, body["error"]["type"]), (403, "forbidden"), sa)

    def test_offline_runner_is_409(self):
        self.start_routing()
        f = self.fake(self.ann)
        f.stop()
        with mock.patch.object(runner_hub, "OFFLINE_S", 0.2):
            time.sleep(1.5)                                  # the held poll notices the hang-up, then OFFLINE_S
            code, body, _ = self.chat(self.ann, {"cwd": CWD, "runner": f.id})
        self.assertEqual(code, 409)
        self.assertEqual(body, {"error": {"type": "runner_offline",
                                          "message": "your PC (ann-pc) is offline: start the runner on it and send again"}})
        self.assertEqual(self.tool_jobs(), [])

    def test_facts_job_unanswered_is_409(self):
        self.start_routing()
        f = self.fake(self.ann, facts=False)                 # online, but never answers run.start
        with mock.patch.object(runner_hub, "REQUEST_TIMEOUT_S", 0.5):
            t0 = time.monotonic()
            code, body, _ = self.chat(self.ann, {"cwd": CWD, "runner": f.id})
        self.assertLess(time.monotonic() - t0, 5)
        self.assertEqual((code, body["error"]["type"]), (409, "runner_offline"))
        self.assertIn("did not answer within", body["error"]["message"])
        self.assertEqual(self.svc.broker.pending, {})

    def test_runner_gone_while_waiting_is_refused(self):
        self.start_routing(tool_call("Read", file_path=CWD + "\\a.txt"), DONE)
        f = self.fake(self.ann)
        orig = self.svc.load

        def load(*a, **k):                                   # the runner ends while the request waits for the model
            self.svc.runner_hub.end_all("ann")
            return orig(*a, **k)
        with mock.patch.object(self.svc, "load", load):
            code, body, _ = self.chat(self.ann, {"cwd": CWD, "runner": f.id})
        self.assertEqual((code, body["error"]["type"]), (409, "runner_offline"), body)
        self.assertEqual(self.tool_jobs(), [])

    def names_and_order(self, web_on):
        self.start_routing(DONE, servers={"fs": FakeMcp("fs", ("x",))}, cfg={"web": {"on": web_on}})
        f = self.fake("k", device=DEV0, groups=[coding(CODING_NAMES + ["WebFetch"]), FS], machine="owner-pc")
        code, events, _ = self.chat("k", {"cwd": CWD, "runner": f.id}, device=DEV0, mcp=True)
        self.assertEqual(code, 200, events)
        names, hub = self.offered[0]
        web_names = ["WebFetch", "WebSearch"] if web_on else []
        self.assertEqual(names, ["fs__x", *web_names, *CODING_NAMES, "my_fs__x"])     # no Read_2, no WebFetch_2
        self.assertIsNot(hub, self.svc.mcp)                  # the request's own view
        self.assertEqual(list(self.svc.mcp.servers), ["fs", "agent"])     # the server's hub is untouched
        self.assertIs(self.svc.mcp.servers["agent"], self.svc.agent)

    def test_names_and_order(self):
        self.names_and_order(False)

    def test_names_and_order_with_web_access(self):
        self.names_and_order(True)

    def test_guest_gets_the_server_subset_and_her_groups(self):
        """R11 keeps the server's unmarked MCP servers from a guest, but not the server's WebFetch / WebSearch / Task."""
        self.start_routing(DONE, servers={"fs": FakeMcp("fs", ("x",))}, cfg={"web": {"on": True}, "guest_mcp": []})
        f = self.fake(self.ann)
        code, events, _ = self.chat(self.ann, {"cwd": CWD, "runner": f.id}, mcp=True)
        self.assertEqual(code, 200, events)
        self.assertEqual(self.offered[0][0], ["WebFetch", "WebSearch", *CODING_NAMES, "my_fs__x"])
        code, events, _ = self.chat(self.ann, {"cwd": CWD, "runner": f.id})        # the coding tools alone
        self.assertEqual(self.offered[1][0], ["WebFetch", "WebSearch", *CODING_NAMES])

    def test_call_event_names_machine(self):
        self.start_routing(tool_call("Read", file_path=CWD + "\\a.txt"), DONE)
        f = self.fake(self.ann, delay=0.6)
        code, events, _ = self.chat(self.ann, {"cwd": CWD, "runner": f.id})
        self.assertEqual(code, 200, events)
        call = next(e for e in events if e["event"] == "call")
        self.assertEqual((call["server"], call["machine"], call["tool"], call["name"]), ("my_agent", "my", "Read", "Read"))
        mine = [e for e in events if e.get("id") == call["id"] and e["event"] in ("call", "remote", "result")]
        self.assertEqual([(e["event"], e.get("state")) for e in mine],
                         [("call", None), ("remote", "sending"), ("remote", "running"), ("result", None)])

    def test_helper_uses_runner_group(self):
        self.start_routing(tool_call("Task", description="look around", prompt="read a.txt and report", subagent_type="explore"),
                           tool_call("Read", file_path=CWD + "\\a.txt"), "</think>\n\nReport: it says hello.", DONE,
                           cfg={"agents": {"on": True}})
        f = self.fake(self.ann)
        code, events, text = self.chat(self.ann, {"cwd": CWD, "session": "chat-1", "runner": f.id})
        self.assertEqual(code, 200, events)
        jobs = self.tool_jobs()
        self.assertEqual(len(jobs), 1, self.jobs)
        self.assertEqual((jobs[0]["tool"], jobs[0]["helper"], jobs[0]["mode"], jobs[0]["run"], jobs[0]["session"]),
                         ("Read", True, "plan", "run-ann-pc", "chat-1"))
        self.assertEqual(self.server_calls, ["Task"])       # Task runs on the server; the helper's Read never does
        helper_names, helper_hub = self.offered[1]
        self.assertEqual(helper_names, ["Read", "Glob", "Grep"])
        self.assertIs(helper_hub, self.offered[0][1])        # the run's own view

    def test_runner_goes_offline_during_the_call(self):
        """Review Focus 1 in the stream: sending, running, offline, then a result the model reads; no fallback."""
        self.start_routing(tool_call("Read", file_path=CWD + "\\a.txt"), DONE)
        f = self.fake(self.ann, vanish=True)
        with mock.patch.object(runner_hub, "OFFLINE_S", 0.5):
            t0 = time.monotonic()
            code, events, text = self.chat(self.ann, {"cwd": CWD, "runner": f.id})
        self.assertEqual(code, 200, events)
        self.assertLess(time.monotonic() - t0, 10)
        call = next(e for e in events if e["event"] == "call")
        mine = [e for e in events if e.get("id") == call["id"] and e["event"] in ("remote", "result")]
        self.assertEqual([e.get("state") for e in mine[:-1]], ["sending", "running", "offline"])
        self.assertEqual(mine[-1]["event"], "result")
        self.assertIn("went offline while Read was running", mine[-1]["text"])
        self.assertEqual(self.server_calls, [])

    def test_a_server_called_agent_is_an_mcp_server_for_guests(self):
        """Without server coding tools no subset is put in: a config MCP server named "agent" keeps R11."""
        self.start_routing(DONE, DONE, with_agent=False, servers={"agent": FakeMcp("agent", ("x",))})
        f = self.fake(self.ann)
        self.assertEqual(self.chat(self.ann, {"cwd": CWD, "runner": f.id}, mcp=True)[0], 200)
        self.assertEqual(self.offered[0][0], [*CODING_NAMES, "my_fs__x"])
        self.assertEqual(self.chat(self.ann, {"cwd": CWD, "runner": f.id})[0], 200)
        self.assertEqual(self.offered[1][0], CODING_NAMES)
        self.cfg_path.write_text(json.dumps({"guest_mcp": ["agent"]}), encoding="utf-8")     # "agent" cannot be marked (mcp_admin)
        self.assertEqual(self.chat(self.ann, {"cwd": CWD, "runner": f.id}, mcp=True)[0], 200)
        self.assertEqual(self.offered[2][0], [*CODING_NAMES, "my_fs__x"])
        self.svc.peer_override = None                                       # the owner gets it, as an MCP server
        k = self.fake("k", device=DEV0, machine="owner-pc")
        self.assertEqual(self.chat("k", {"cwd": CWD, "runner": k.id}, device=DEV0, mcp=True)[0], 200)
        self.assertEqual(self.offered[3][0], ["agent__x", *CODING_NAMES, "my_fs__x"])

    def starting_fs(self):
        """A server MCP `fs` still starting (no tools yet), and the hook that makes it ready."""
        fs = FakeMcp("fs", ("x",))
        tools, fs.status, fs.tools = fs.tools, "starting", []

        def ready():
            fs.status, fs.tools = "ready", tools
        return fs, ready

    def test_a_server_ready_later_never_takes_an_offered_name(self):
        """Review I-1: the request's names are pinned when its list is made.  `fs` (not marked for guests) becomes
        ready between the list and the call; the guest's `fs__x` still goes to her runner, never to the server's `fs`."""
        fs, ready = self.starting_fs()
        self.start_routing(tool_call("fs__x", path="p"), DONE, servers={"fs": fs})
        with mock.patch.object(runner_hub, "plain_name_ok", lambda n: True, create=True):     # a runner that names a tool fs__x
            f = self.fake(self.ann, groups=[coding(CODING_NAMES + ["fs__x"]), FS])
            self.on_offer = ready
            code, events, _ = self.chat(self.ann, {"cwd": CWD, "runner": f.id}, mcp=True)
        self.assertEqual(code, 200, events)
        self.assertIn("fs__x", self.offered[0][0])
        self.assertEqual(fs.status, "ready")
        self.assertEqual([(j["group"], j["tool"]) for j in self.tool_jobs()], [("agent", "fs__x")])
        self.assertEqual(fs.calls, [])
        hub = self.offered[0][1]
        self.assertEqual(hub.routes()["fs__x"][0].name, "my_agent")       # still pinned after fs is ready
        self.assertEqual(hub.call("fs__x_2", {})["text"], "error: there is no tool named 'fs__x_2'")
        self.assertEqual(fs.calls, [])

    def test_a_helper_uses_the_pinned_names(self):
        fs, ready = self.starting_fs()
        self.start_routing(tool_call("Task", description="look around", prompt="read a.txt", subagent_type="explore"),
                           tool_call("Read", file_path=CWD + "\\a.txt"), tool_call("fs__x", path="p"), DONE,
                           servers={"fs": fs}, cfg={"agents": {"on": True}})
        with mock.patch.object(runner_hub, "plain_name_ok", lambda n: True, create=True):
            f = self.fake(self.ann, groups=[coding(CODING_NAMES + ["fs__x"]), FS])
            self.on_offer = ready
            code, events, _ = self.chat(self.ann, {"cwd": CWD, "runner": f.id}, mcp=True)
        self.assertEqual(code, 200, events)
        self.assertEqual([j["tool"] for j in self.tool_jobs()], ["Read"])      # the helper's Read, on the PC
        self.assertEqual(fs.calls, [])
        helper_hub = self.offered[1][1]
        self.assertIs(helper_hub, self.offered[0][1])
        self.assertEqual(helper_hub.routes()["fs__x"][0].name, "my_agent")
        self.assertNotIn("fs", {s.name for s, _ in helper_hub.routes().values()})

    def test_a_coding_tool_named_like_a_group_is_left_out(self):
        self.start_routing(DONE, servers={"fs": FakeMcp("fs", ("x",))}, cfg={"guest_mcp": ["fs"]})
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            f = self.fake(self.ann, groups=[coding(CODING_NAMES + ["fs__x", "my_fs", "a__b"]), FS])
        self.assertIn("coding tool 'fs__x' left out", out.getvalue())
        self.assertIn("coding tool 'my_fs' left out", out.getvalue())
        code, body = self.req("GET", f"/agent?runner={f.id}", key=self.ann, device=DEV1)
        self.assertEqual(body["tools"], CODING_NAMES)
        code, events, _ = self.chat(self.ann, {"cwd": CWD, "runner": f.id}, mcp=True)
        self.assertEqual(code, 200, events)
        self.assertEqual(self.offered[0][0], ["fs__x", *CODING_NAMES, "my_fs__x"])     # fs__x is the server's (marked)
        self.assertIs(self.offered[0][1].routes()["fs__x"][0], self.svc.mcp.servers["fs"])

    def test_runner_without_server_hub(self):
        self.start_routing(tool_call("Read", file_path=CWD + "\\a.txt"), DONE, with_agent=False, with_hub=False)
        f = self.fake(self.ann)
        code, events, text = self.chat(self.ann, {"cwd": CWD, "runner": f.id})
        self.assertEqual(code, 200, events)
        self.assertEqual(self.offered[0][0], CODING_NAMES)
        self.assertEqual([j["tool"] for j in self.tool_jobs()], ["Read"])

    def test_outside_team_mode_runner_is_the_owners(self):
        self.start_routing(tool_call("Read", file_path=CWD + "\\a.txt"), DONE, team=False)
        f = self.fake(None, device=DEV1, machine="my-pc")
        code, events, _ = self.chat(None, {"cwd": CWD, "runner": f.id})           # this PC itself: the owner
        self.assertEqual(code, 200, events)
        self.assertEqual([j["tool"] for j in self.tool_jobs("my-pc")], ["Read"])
        self.svc.peer_override = REMOTE                      # another PC on the network, no key: not the owner
        code, body, _ = self.chat(None, {"cwd": CWD, "runner": f.id}, device=DEV2)
        self.assertEqual(code, 403, body)
        code, body = self.req("GET", f"/agent?runner={f.id}", device=DEV2)
        self.assertEqual(code, 403, body)
        self.assertEqual(len(self.tool_jobs("my-pc")), 1)

    def test_default_request_unchanged(self):
        proj = self.dir / "proj"
        proj.mkdir()
        self.start_routing(tool_call("fs__x", path="p"), DONE, servers={"fs": FakeMcp("fs", ("x",))})
        code, events, _ = self.chat("k", {"cwd": str(proj), "session": "chat-1"}, device=DEV0, mcp=True)
        self.assertEqual(code, 200, events)
        names, hub = self.offered[0]
        self.assertIs(hub, self.svc.mcp)
        hidden = {"WebFetch", "WebSearch", "Task"}          # web access and helpers are off
        self.assertEqual(names, [t["name"] for t in self.svc.mcp.template_tools(exclude=hidden, skip_servers=set())])
        self.assertEqual(names[0], "fs__x")
        call = next(e for e in events if e["event"] == "call")
        self.assertEqual(set(call), CALL_KEYS)
        self.assertFalse([e for e in events if e["event"] == "remote"])
        code, body = self.req("GET", "/agent", key="k", device=DEV0)
        self.assertEqual(code, 200)
        self.assertEqual(set(body), {"available", "allowed", "reason", "shell", "tools"})

    def test_get_agent_with_runner(self):
        self.start_routing()
        f = self.fake(self.ann, groups=[coding(CODING_NAMES + ["WebFetch"]), FS])
        bobs = self.fake(self.bob, machine="bob-pc")
        code, body = self.req("GET", f"/agent?runner={f.id}", key=self.ann, device=DEV1)
        self.assertEqual(code, 200, body)
        self.assertEqual(body, {"available": True, "allowed": True, "reason": "", "shell": "Git Bash",
                                "tools": CODING_NAMES, "machine": "ann-pc", "online": True})
        nobody = self.req("GET", "/agent?runner=r0000000000000000", key=self.ann, device=DEV1)
        self.assertEqual(nobody[0], 404)
        self.assertEqual(self.req("GET", f"/agent?runner={bobs.id}", key=self.ann, device=DEV1), nobody)
        code, body = self.req("GET", "/agent", key=self.ann, device=DEV1)              # without runner=: as before
        self.assertEqual(set(body), {"available", "allowed", "reason", "shell", "tools"})
        self.assertFalse(body["allowed"])
        f.stop()
        with mock.patch.object(runner_hub, "OFFLINE_S", 0.2):
            time.sleep(1.5)                                  # the held poll notices the hang-up, then OFFLINE_S
            code, body = self.req("GET", f"/agent?runner={f.id}", key=self.ann, device=DEV1)
        self.assertEqual((body["allowed"], body["reason"], body["online"]), (False, "your PC is offline", False))


class View(unittest.TestCase):
    """McpHub.with_servers and runner_hub.agent_subset, without HTTP."""

    def test_with_servers_keeps_the_hub_and_renames_a_clash(self):
        hub = McpHub({}, {"timeout_s": 7}, builtins={"my_fs": FakeMcp("my_fs", ("x",)), "agent": FakeMcp("agent", ("WebFetch", "Read"))})
        group = FakeMcp("my_fs", ("x",))
        sub = FakeMcp("agent", ("WebFetch",))
        view = hub.with_servers({"my_fs": group}, replace={"agent": sub, "absent": FakeMcp("absent")})
        self.assertEqual(list(view.servers), ["my_fs", "agent", "my_fs_2"])
        self.assertIs(view.servers["agent"], sub)
        self.assertEqual(view.servers["my_fs_2"].name, "my_fs_2")
        self.assertEqual(group.name, "my_fs")                       # the runner's own object is not renamed
        self.assertIs(view.settings, hub.settings)
        self.assertIs(view.threads, hub.threads)
        self.assertEqual(list(view.routes()), ["my_fs__x", "agent__WebFetch", "my_fs_2__x"])
        self.assertEqual(list(hub.servers), ["my_fs", "agent"])
        self.assertEqual(list(hub.routes()), ["my_fs__x", "agent__WebFetch", "agent__Read"])

    def test_agent_subset(self):
        a = agent.AgentServer()
        web.install(a)
        subagent.install(a)
        seen = []
        a.call = lambda tool, *rest, **k: seen.append(tool) or {"content": [{"type": "text", "text": tool}]}
        s = runner_hub.agent_subset(a, runner_hub.SERVER_TOOLS)
        self.assertEqual(s.name, "agent")
        self.assertEqual([t["name"] for t in s.tools], ["WebFetch", "WebSearch", "Task"])
        for attr in ("kind", "wants_context", "plain_names", "hidden", "status", "error", "info"):
            self.assertEqual(getattr(s, attr), getattr(a, attr), attr)
        self.assertEqual(s.call("WebFetch", {}, 0, None, None)["content"][0]["text"], "WebFetch")
        self.assertTrue(s.call("Read", {}, 0, None, None)["isError"])
        self.assertEqual(seen, ["WebFetch"])
        s.close()                                                    # a view closes nothing


if __name__ == "__main__":
    unittest.main()
