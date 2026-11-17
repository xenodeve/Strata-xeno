"""serve/test_runner_run.py - a coding run whose project is on My PC (#310 part 2, task 3): `Service.start_agent_run`
with a runner takes the folder, shell, git and notes from the runner's `run.start` answer (the facts job) and never
looks at the server's own disk, hooks or checkpoints for it; without a runner nothing changes.

    python -m pytest serve/test_runner_run.py -q
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from serve import agent, agent_prompt, hooks, runner_hub, shell, subagent, web  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, Service  # noqa: E402
from serve.test_agent_chat import DONE  # noqa: E402
from serve.test_mcp import ScriptedEngine  # noqa: E402

SH = shell.find_shell(prefer="bash")
INFO = {"machine": "ann-pc", "platform": "linux", "shell": "bash 5", "home": "/home/ann"}
CODING_NAMES = ["Read", "Write", "Edit", "Glob", "Grep", "Bash", "TodoWrite", "ExitPlanMode"]
CODING = {"name": "agent", "kind": "coding", "status": "ready", "error": None,
          "tools": [{"name": n, "description": n, "inputSchema": {"type": "object"}} for n in CODING_NAMES]}
CWD = "C:\\Users\\ann\\proj"                    # a folder on the engineer's PC, absent on this disk
BLOCK = {"id": "project:CLAUDE.md", "kind": "project", "label": "Project", "name": "CLAUDE.md",
         "text": "Always answer in haiku.", "cut": False}


def facts(**over):
    out = {"run": "run-" + uuid.uuid4().hex[:12], "cwd": CWD, "dirs": [], "shell": "bash 5", "platform": "linux",
           "git": True, "memory": [dict(BLOCK)], "tools": list(CODING_NAMES)}
    out.update(over)
    return out


class Fixture(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name).resolve()
        self.agent = agent.AgentServer()
        shell.install(self.agent, SH)
        web.install(self.agent)
        subagent.install(self.agent)
        tok = ByteTokenizer()
        self.svc = Service(ScriptedEngine(tok, DONE), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.agent = self.agent
        self.hub = runner_hub.RunnerHub()
        self.addCleanup(self.hub.close)
        self.runner = self.hub.hello("ann", "device-0001", "inst-aaaa", dict(INFO), [CODING], [])
        self.sent: list[tuple[str, dict]] = []

    def config(self, cfg: dict) -> None:
        path = self.base / "strata.json"
        path.write_text(json.dumps(cfg), encoding="utf-8")
        self.svc.config_path = str(path)

    def answer(self, result):
        """Make the runner's `run.start` answer `result` (a dict, or a callable of the payload)."""
        def request(kind, payload, timeout=None):
            self.sent.append((kind, payload))
            return result(payload) if callable(result) else result
        self.runner.request = request

    def sa(self, **over):
        return {"cwd": CWD, "session": "chat-1", **over}

    def where(self) -> dict:
        """#339: the prompt of a run on My PC says which PC each tool runs on."""
        return {"machine": "ann-pc", "my_tools": any(getattr(g, "group_kind", "") == "mcp" for g in self.runner.servers().values())}

    def messages(self, text="fix the bug"):
        return [{"role": "user", "content": text}]


class OnMyPc(Fixture):
    def test_runner_facts_build_the_prompt(self):
        f = facts()
        self.answer(f)
        run = self.svc.start_agent_run(self.sa(), self.messages(), "ann", runner=self.runner)
        self.assertIn(CWD, run.prompt)
        self.assertIn("- Platform: linux", run.prompt)
        self.assertIn("- Shell: bash 5", run.prompt)
        self.assertIn("Is a git repository: yes", run.prompt)
        self.assertIn("Always answer in haiku.", run.prompt)
        self.assertEqual(run.coding, "my_agent")
        self.assertIs(run.runner, self.runner)
        self.assertIsNone(run.hub)                      # the request's route view is task 4's
        self.assertEqual(run.ctx.run_key, f["run"])
        self.assertEqual(run.shell, "bash 5")
        self.assertEqual((run.policy.cwd, run.policy.dirs, run.policy.protected), (CWD, [], []))
        self.assertIsNone(run.ctx.checkpoint)
        self.assertNotIn("WebFetch", run.prompt)        # web access and helpers are off in an empty config: hidden as today
        self.assertEqual(run.hidden, {"WebFetch", "WebSearch", "Task"})

    def test_run_start_payload_is_the_servers_sanitised_values(self):
        self.config({"import": {"memory": {"on": ["claude:memory"]}}})
        self.answer(facts())
        sa = self.sa(dirs=["D:\\other", "", "a\0b", 7], mode="plan", allow=["Bash(ls)", 3, "x" * 600],
                     deny=["Read(.env)"], checkpoint="cp-1")
        self.svc.start_agent_run(sa, self.messages(), "ann", runner=self.runner)
        self.assertEqual(len(self.sent), 1)
        kind, payload = self.sent[0]
        self.assertEqual(kind, "run.start")
        self.assertEqual(payload, {"cwd": CWD, "dirs": ["D:\\other"], "mode": "plan", "allow": ["Bash(ls)"],
                                   "deny": ["Read(.env)"], "session": "chat-1", "checkpoint": "cp-1",
                                   "memory_on": ["claude:memory"]})
        json.dumps(payload)
        self.answer(facts())
        self.svc.config_path = None                     # no run config: no other app's notes
        self.svc.start_agent_run({"cwd": "a\0b", "session": "default", "checkpoint": "cp-1", "mode": "nope"},
                                 self.messages(), "ann", runner=self.runner)
        self.assertEqual(self.sent[-1][1], {"cwd": None, "dirs": [], "mode": None, "allow": [], "deny": [],
                                            "session": "default", "checkpoint": None, "memory_on": []})

    def test_tools_for_the_prompt(self):
        self.config({"web": {"on": True}, "agents": {"on": True}})
        self.answer(facts())
        run = self.svc.start_agent_run(self.sa(), self.messages(), "ann", runner=self.runner)
        tools = [n for n in CODING_NAMES if n != "ExitPlanMode"] + ["WebFetch", "WebSearch", "Task"]
        self.assertEqual(run.prompt, agent_prompt.build(CWD, "bash 5", None, time.strftime("%Y-%m-%d"), "linux", True,
                                                        None, tools, [], [BLOCK], **self.where()))
        self.assertIsNotNone(run.ctx.web)
        self.assertIsNotNone(run.ctx.spawn)
        self.answer(facts())
        run = self.svc.start_agent_run(self.sa(mode="plan"), self.messages(), "ann", runner=self.runner)
        self.assertEqual(run.prompt, agent_prompt.build(CWD, "bash 5", "plan", time.strftime("%Y-%m-%d"), "linux",
                                                        True, None, CODING_NAMES + ["WebFetch", "WebSearch", "Task"],
                                                        [], [BLOCK], **self.where()))

    def test_the_rules_say_where_each_tool_runs(self):
        """#339: the coding tools on ann-pc, and an MCP server of that PC (my_browser__...) runs there too; the others on the central PC."""
        self.answer(facts())
        run = self.svc.start_agent_run(self.sa(), self.messages(), "ann", runner=self.runner)
        self.assertIn("run on the user's PC (ann-pc)", run.prompt)
        self.assertNotIn("starts with my_", run.prompt)                                  # only the coding group: no my_ MCP tool
        skills = {"name": "skills", "kind": "skills", "status": "ready", "error": None, "tools": [{"name": "find_skills", "description": "x", "inputSchema": {"type": "object"}}]}
        self.runner = self.hub.hello("ann", "device-0001", "inst-aaaa", dict(INFO), [CODING, skills], [])
        self.answer(facts())
        run = self.svc.start_agent_run(self.sa(), self.messages(), "ann", runner=self.runner)
        self.assertNotIn("starts with my_", run.prompt)                                  # my_skills is not an MCP group
        mcp = {"name": "browser", "kind": "mcp", "status": "ready", "error": None, "tools": [{"name": "open", "description": "x", "inputSchema": {"type": "object"}}]}
        self.runner = self.hub.hello("ann", "device-0001", "inst-aaaa", dict(INFO), [CODING, mcp], [])
        self.answer(facts())
        run = self.svc.start_agent_run(self.sa(), self.messages(), "ann", runner=self.runner)
        self.assertIn("starts with my_ (my_<server>__<tool>) runs on the user's PC (ann-pc)", run.prompt)
        self.assertIn("Playwright", run.prompt)

    def test_server_without_coding_tools(self):
        self.svc.agent = None                           # a server started without its own coding tools: the runner's alone
        self.config({"web": {"on": True}, "agents": {"on": True}})
        self.answer(facts())
        run = self.svc.start_agent_run(self.sa(), self.messages(), "ann", runner=self.runner)
        tools = [n for n in CODING_NAMES if n != "ExitPlanMode"]
        self.assertEqual(run.prompt, agent_prompt.build(CWD, "bash 5", None, time.strftime("%Y-%m-%d"), "linux", True,
                                                        None, tools, [], [BLOCK], **self.where()))

    @unittest.skipIf(SH is None, "no shell on this PC: the server would make no hook runner either")
    def test_runner_run_has_no_hooks(self):
        self.config({"hooks": [{"event": "prompt", "command": "echo from-the-servers-hook"}]})
        own = self.base / "own"
        own.mkdir()
        server_run = self.svc.start_agent_run({"cwd": str(own), "session": "chat-1"}, self.messages(), "owner")
        self.assertIsNotNone(server_run.hooks)          # the same config does make hooks for a server project
        self.assertIn("from-the-servers-hook", server_run.prompt)
        self.answer(facts())
        with mock.patch.object(hooks, "Runner", side_effect=AssertionError("hooks on a runner project")), \
                mock.patch.object(hooks, "load", side_effect=AssertionError("hooks read for a runner project")):
            run = self.svc.start_agent_run(self.sa(), self.messages(), "ann", runner=self.runner)
        self.assertIsNone(run.hooks)
        self.assertIsNone(run.ctx.hooks)
        self.assertNotIn("from-the-servers-hook", run.prompt)

    def test_runner_run_touches_no_server_disk(self):
        self.answer(facts(dirs=["D:\\x"]))
        boom = AssertionError("the server's disk was read for a runner project")
        with mock.patch("os.path.isdir", side_effect=boom), mock.patch("os.path.realpath", side_effect=boom), \
                mock.patch("os.path.exists", side_effect=boom), \
                mock.patch.object(self.svc, "memory_blocks", side_effect=boom), \
                mock.patch.object(self.svc.checkpoints, "scope", side_effect=boom):
            run = self.svc.start_agent_run(self.sa(dirs=["D:\\x"], checkpoint="cp-1"), self.messages(), "ann",
                                           runner=self.runner)
        self.assertEqual(run.policy.dirs, ["D:\\x"])
        self.assertIn("D:\\x", run.prompt)

    def test_offline_runner_raises(self):
        with mock.patch.object(runner_hub, "OFFLINE_S", 0.0):
            with self.assertRaises(runner_hub.RunnerUnavailable) as caught:
                self.svc.start_agent_run(self.sa(), self.messages(), "ann", runner=self.runner)
        self.assertIn("ann-pc", caught.exception.message)
        self.assertIn("offline", caught.exception.message)
        self.assertEqual(len(self.svc.broker.runs), 0)     # no run was made, so none is left tracked
        self.assertEqual(self.runner.jobs, {})

    def test_unreadable_facts_raise(self):
        for bad in ({"error": "timeout", "message": "your PC (ann-pc) did not answer within 30 s"}, {}, [],
                    facts(run=""), facts(run=7), facts(run="x" * 200), facts(cwd=3), facts(dirs="D:\\x"),
                    facts(dirs=[3]), facts(git="yes"), facts(memory=[{"name": "x"}]), facts(memory="notes"),
                    facts(tools=[1]), facts(shell=5), facts(platform=None)):
            self.answer(bad)
            with self.assertRaises(runner_hub.RunnerUnavailable, msg=repr(bad)) as caught:
                self.svc.start_agent_run(self.sa(), self.messages(), "ann", runner=self.runner)
            self.assertIsInstance(caught.exception.message, str)
            self.assertTrue(caught.exception.message)
        self.assertEqual(len(self.svc.broker.runs), 0)

    def test_facts_are_copied_and_cut(self):
        big = dict(BLOCK, name="a.md", text="a" * 50, extra="dropped")
        got = runner_hub.run_facts(facts(memory=[big, dict(BLOCK, name="b.md", text="b" * 50), dict(BLOCK, name="c.md")],
                                         tools=["Read", "Read", "Bash"], dirs=[f"D:\\{i}" for i in range(30)]),
                                   "ann-pc", 20, 80)
        self.assertEqual([(b["name"], len(b["text"]), b["cut"]) for b in got["memory"]], [("a.md", 50, False), ("b.md", 30, True)])
        self.assertNotIn("extra", got["memory"][0])
        self.assertEqual(got["tools"], ["Read", "Bash"])
        self.assertEqual(len(got["dirs"]), 20)

    def test_facts_without_a_folder(self):
        self.answer(facts(cwd=None, dirs=["D:\\x"], git=True))      # the runner found no folder: no others, no git line
        run = self.svc.start_agent_run(self.sa(), self.messages(), "ann", runner=self.runner)
        self.assertEqual((run.policy.cwd, run.policy.dirs), (None, []))
        self.assertIn("There is no project folder for this chat", run.prompt)
        self.assertNotIn("Is a git repository", run.prompt)

    def test_two_chats_on_one_runner_get_their_own_run_keys(self):
        self.answer(lambda payload: facts(run=f"run-{payload['session']}-{uuid.uuid4().hex[:8]}"))
        one = self.svc.start_agent_run(self.sa(session="s1"), self.messages(), "ann", runner=self.runner)
        two = self.svc.start_agent_run(self.sa(session="s2"), self.messages(), "ann", runner=self.runner)
        self.assertNotEqual(one.ctx.run_key, two.ctx.run_key)
        del self.runner.request                         # the real channel from here
        group = self.runner.servers()["my_agent"]
        for run in (one, two):
            out = []
            t = threading.Thread(target=lambda r=run: out.append(group.call("Read", {"file_path": "a.txt"}, 0,
                                                                            threading.Event(), r.ctx)))
            t.start()
            msg = self.runner.take(2.0)[-1]
            self.assertEqual((msg["kind"], msg["tool"]), ("tool", "Read"))
            self.assertEqual((msg["run"], msg["session"]), (run.ctx.run_key, run.ctx.session))
            self.runner.deliver({"job": msg["job"], "rid": 1, "result": {"content": [], "isError": False}})
            t.join(5)
            self.assertEqual(out, [{"content": [], "isError": False}])


class OnTheServer(Fixture):
    def test_no_runner_run_unchanged(self):
        proj = self.base / "proj"
        proj.mkdir()
        (proj / ".git").mkdir()
        sa = {"cwd": str(proj), "session": "chat-1", "mode": "auto"}
        run = self.svc.start_agent_run(sa, self.messages(), "owner")
        folder = os.path.realpath(str(proj))
        tools = [t["name"] for t in self.agent.tools if t["name"] != "ExitPlanMode" and t["name"] not in run.hidden]
        self.assertEqual(run.prompt, agent_prompt.build(folder, shell.describe(SH) if SH else None, "auto",
                                                        time.strftime("%Y-%m-%d"), sys.platform, True, None, tools,
                                                        [], self.svc.memory_blocks([folder])))
        self.assertEqual(run.coding, "agent")
        self.assertEqual(run.ctx.run_key, "")
        self.assertIsNone(run.runner)
        self.assertIsNone(run.hub)
        self.assertEqual(run.policy.cwd, folder)


if __name__ == "__main__":
    unittest.main()
