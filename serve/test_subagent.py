"""Sub-agents: the Task tool hands a side task to a helper with its own context (serve/subagent.py, issue #99).  Off by default; when on, one helper at a time, a short list of tools (read-only for
`explore`), no helper of its own and no questions to the user, a limit of rounds, its permission questions come to the user, its steps are streamed on the Task call, its last message is the result."""
from __future__ import annotations

import json
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from serve import agent, permissions, subagent  # noqa: E402
from serve.mcp import hub_from_config  # noqa: E402
from serve.test_agent_chat import DONE, Fixture  # noqa: E402
from serve.test_harness import tool_call  # noqa: E402


def report(text):
    return f"</think>\n\n{text}"


def two_calls(a, b):
    """One answer with two Task calls."""
    first, second = tool_call("Task", **a), tool_call("Task", **b)
    return first.rsplit("</tool_call>", 1)[0] + "</tool_call>\n" + second[second.index("<tool_call>"):]


class Settings(unittest.TestCase):
    def test_off_by_default_and_anything_odd_is_off(self):
        self.assertEqual(subagent.settings({}), {"on": False})
        for odd in ({"agents": "yes"}, {"agents": {"on": 1}}, {"agents": {"on": "true"}}, {"agents": None}, None, []):
            self.assertFalse(subagent.settings(odd)["on"], odd)
        self.assertTrue(subagent.settings({"agents": {"on": True}})["on"])

    def test_what_a_page_sends_is_checked(self):
        self.assertEqual(subagent.check_settings({"on": True})[0], {"on": True})
        for bad in ("x", [], {"on": "yes"}, {"on": 1}, {"other": 1}, {"on": True, "x": 1}):
            self.assertIsNone(subagent.check_settings(bad)[0], bad)


class TheTool(unittest.TestCase):
    def setUp(self):
        self.srv = agent.AgentServer()
        subagent.install(self.srv)

    def call(self, args, **ctx_kw):
        ctx = agent.AgentContext(policy=permissions.Policy(cwd=None), session="s")
        for k, v in ctx_kw.items():
            setattr(ctx, k, v)
        return self.srv.call("Task", args, ctx=ctx)

    def text(self, r):
        return r["content"][0]["text"]

    def test_it_is_a_tool_with_a_description_a_prompt_and_a_kind(self):
        t = {x["name"]: x for x in self.srv.tools}["Task"]
        self.assertEqual(set(t["inputSchema"]["properties"]), {"description", "prompt", "subagent_type"})
        self.assertEqual(t["inputSchema"]["required"], ["description", "prompt"])
        self.assertEqual(t["inputSchema"]["properties"]["subagent_type"]["enum"], ["explore", "general"])
        self.assertIn("does not see this conversation", t["description"])
        self.assertEqual(self.srv.helper_tools, ("Task",))

    def test_without_helpers_switched_on_it_refuses_and_says_how(self):
        r = self.call({"description": "look", "prompt": "find x"})
        self.assertTrue(r["isError"])
        self.assertIn("Settings > Sub-agents", self.text(r))

    def test_it_needs_no_permission_of_its_own_in_any_mode(self):
        for mode in (None, "plan", "auto"):
            self.assertEqual(permissions.decide("Task", {"prompt": "x"}, permissions.Policy(cwd=None, mode=mode)).kind, "allow", mode)
        self.assertEqual(permissions.decide("Task", {"prompt": "x"}, permissions.Policy(cwd=None, deny=["Task"])).kind, "deny")        # a "never" rule still wins

    def test_what_the_model_sends_is_checked_and_a_helper_cannot_start_a_helper(self):
        calls = []
        spawn = lambda kind, prompt, description: (calls.append((kind, prompt, description)), (True, "ok"))[1]          # noqa: E731
        for bad in ({}, {"description": "d"}, {"description": "d", "prompt": ""}, {"description": "d", "prompt": "p" * 20_001}, {"description": "d", "prompt": "p", "subagent_type": "root"},
                    {"prompt": "p"}, {"description": "", "prompt": "p"}, {"description": 5, "prompt": "p"}):
            self.assertTrue(self.call(bad, spawn=spawn)["isError"], bad)
        self.assertEqual(calls, [])
        self.assertIn("cannot start a helper", self.text(self.call({"description": "d", "prompt": "p"}, spawn=spawn, in_helper=True)))
        self.assertEqual(calls, [])
        self.assertFalse(self.call({"description": "d", "prompt": "p"}, spawn=spawn).get("isError"))
        self.assertEqual(calls, [("explore", "p", "d")])                                      # explore is the default kind

    def test_a_helper_that_failed_is_an_error_result(self):
        r = self.call({"description": "d", "prompt": "p"}, spawn=lambda *a: (False, "it broke"))
        self.assertTrue(r["isError"])
        self.assertEqual(self.text(r), "it broke")


class InTheChat(Fixture):
    def setUp(self):
        super().setUp()
        self.config = self.base / "run.json"
        self.config.write_text("{}", encoding="utf-8")

    def start(self, *scripts, agents=True, **kw):
        super().start(*scripts, with_shell=True, **kw)
        subagent.install(self.agent)
        self.hub.close()
        self.hub = hub_from_config({}, None, builtins={"agent": self.agent})
        self.hub.start(wait=True)
        self.svc.mcp = self.hub
        self.svc.config_path = str(self.config)
        self.config.write_text(json.dumps({"agents": {"on": True}} if agents else {}), encoding="utf-8")

    def get(self, path):
        with urllib.request.urlopen(self.base_url + path, timeout=10) as r:
            return json.loads(r.read())

    def answer(self, rid, decision):
        with urllib.request.urlopen(self.post("/agent/permission", {"id": rid, "decision": decision}), timeout=10) as r:
            return r.status

    def helper_prompt(self, i):
        return self.engine.prompt_text(i)

    def test_in_a_fresh_install_the_model_is_not_offered_the_tool(self):
        self.start(DONE, agents=False)
        self.chat()
        self.assertNotIn("hand a side task", self.engine.prompt_text(0))

    def test_when_on_the_model_is_offered_it(self):
        self.start(DONE)
        self.chat()
        self.assertIn("hand a side task", self.engine.prompt_text(0))

    def test_a_helper_does_a_task_with_its_own_context_and_its_report_is_the_result(self):
        self.start(tool_call("Task", description="find hello", prompt="Find where hello is said in the project and report the file and line."),
                   tool_call("Read", file_path="a.txt"), report("a.txt line 1 says hello."), DONE)
        status, said, events = self.chat()
        self.assertEqual(status, 200)
        self.assertTrue(said.strip().endswith("Done."))
        task = [e for e in events if e["event"] == "call" and e["name"] == "Task"][0]
        steps = [e for e in events if e["event"] == "step"]
        self.assertEqual([(s["name"], s["call_id"]) for s in steps], [("Read", task["id"])])         # its steps are on the Task call
        self.assertEqual(steps[0]["arguments"], {"file_path": "a.txt"})
        done = [e for e in events if e["event"] == "step_result"]
        self.assertTrue(done[0]["ok"] and done[0]["call_id"] == task["id"])
        (res,) = [e for e in events if e["event"] == "result" and e["id"] == task["id"]]
        self.assertTrue(res["ok"])
        self.assertIn("a.txt line 1 says hello.", res["text"])
        self.assertIn("Report of the helper (explore, 1 step)", res["text"])
        helper = self.helper_prompt(1)                                                           # the helper's own conversation
        self.assertIn("You are a helper", helper)
        self.assertIn("Find where hello is said", helper)
        self.assertNotIn("look at a.txt", helper)                                                # none of the main chat
        self.assertIn("a.txt line 1 says hello.", self.helper_prompt(3))                          # and the main model reads the report
        self.assertEqual(len(self.engine.prompts), 4)

    def test_the_helper_says_when_it_starts_and_ends(self):
        self.start(tool_call("Task", description="find it", prompt="p"), report("nothing found"), DONE)
        _, _, events = self.chat()
        marks = [(e["state"], e.get("kind"), e.get("description")) for e in events if e["event"] == "helper"]
        self.assertEqual(marks[0], ("start", "explore", "find it"))
        self.assertEqual(marks[-1][0], "end")

    def test_explore_has_only_the_tools_that_read(self):
        self.start(tool_call("Task", description="d", prompt="look"), report("ok"), DONE)
        self.chat()
        p = self.helper_prompt(1)
        for name in ("Read", "Glob", "Grep"):
            self.assertIn(f'"name": "{name}"', p, name)
        for name in ("Write", "Edit", "NotebookEdit", "Bash", "Task", "AskUserQuestion", "TodoWrite", "ExitPlanMode", "WebFetch"):
            self.assertNotIn(f'"name": "{name}"', p, name)

    def test_general_can_also_change_things_but_still_has_no_helper_and_no_questions(self):
        self.start(tool_call("Task", description="d", prompt="make it", subagent_type="general"), report("made"), DONE)
        self.chat()
        p = self.helper_prompt(1)
        for name in ("Read", "Glob", "Grep", "Write", "Edit", "Bash"):
            self.assertIn(f'"name": "{name}"', p, name)
        for name in ("Task", "AskUserQuestion", "TodoWrite", "ExitPlanMode"):
            self.assertNotIn(f'"name": "{name}"', p, name)

    def test_an_explore_helper_that_tries_to_write_changes_nothing(self):
        self.start(tool_call("Task", description="d", prompt="look"), tool_call("Write", file_path="made.txt", content="x"), report("done"), DONE)
        _, _, events = self.chat()
        self.assertFalse((self.proj / "made.txt").exists())
        self.assertEqual([e for e in events if e["event"] == "permission"], [])

    def test_a_general_helpers_changes_go_through_the_same_gate_and_are_free_inside_the_folder(self):
        self.start(tool_call("Task", description="d", prompt="make", subagent_type="general"), tool_call("Write", file_path="made.txt", content="by the helper\n"), report("made it"), DONE)
        _, _, events = self.chat()
        self.assertEqual((self.proj / "made.txt").read_text(encoding="utf-8"), "by the helper\n")
        self.assertIn("made it", [e for e in events if e["event"] == "result" and e["id"] == [c for c in events if c["event"] == "call"][0]["id"]][0]["text"])

    def test_its_permission_questions_come_to_the_user_on_the_task_call_and_a_no_is_a_no(self):
        outside = self.outside / "b.txt"
        self.start(tool_call("Task", description="d", prompt="read outside"), tool_call("Read", file_path=str(outside)), report("it was refused"), DONE)
        cards = []
        status, _, events = self.chat(on_event=lambda e: e["event"] == "permission" and (cards.append(e), self.answer(e["id"], "deny")))
        self.assertEqual(status, 200)
        task = [e for e in events if e["event"] == "call" and e["name"] == "Task"][0]
        self.assertEqual(len(cards), 1)
        self.assertEqual((cards[0]["tool"], cards[0]["call_id"]), ("Read", task["id"]))             # asked like any other, on the card of the Task
        self.assertIn("outside the project folder", cards[0]["why"])
        step_res = [e for e in events if e["event"] == "step_result"][0]
        self.assertFalse(step_res["ok"])
        self.assertNotIn("secret-ish", json.dumps(events))                                          # nothing of the file was read

    def test_a_deny_rule_of_the_chat_binds_the_helper_too(self):
        self.start(tool_call("Task", description="d", prompt="read a"), tool_call("Read", file_path="a.txt"), report("could not"), DONE)
        _, _, events = self.chat(self.body(agent={"deny": ["Read(a.txt)"]}))
        step_res = [e for e in events if e["event"] == "step_result"][0]
        self.assertFalse(step_res["ok"])
        self.assertIn("denied", step_res["text"])

    def test_helpers_run_one_at_a_time_in_order(self):
        self.start(two_calls({"description": "first", "prompt": "p1"}, {"description": "second", "prompt": "p2"}), report("report one"), report("report two"), DONE)
        _, _, events = self.chat()
        kinds = [(e["event"], e.get("state")) for e in events if e["event"] == "helper"]
        self.assertEqual(kinds, [("helper", "start"), ("helper", "end"), ("helper", "start"), ("helper", "end")])         # the first ends before the second starts
        order = [e["event"] for e in events]
        self.assertLess(order.index("result"), [i for i, e in enumerate(events) if e["event"] == "helper"][2])
        self.assertIn("p1", self.helper_prompt(1))
        self.assertIn("p2", self.helper_prompt(2))
        results = [e["text"] for e in events if e["event"] == "result"]
        self.assertTrue("report one" in results[0] and "report two" in results[1])

    def test_a_helper_stops_after_its_limit_of_rounds(self):
        old = subagent.HELPER_ROUNDS
        subagent.HELPER_ROUNDS = 2
        self.addCleanup(setattr, subagent, "HELPER_ROUNDS", old)
        read = tool_call("Read", file_path="a.txt")
        self.start(tool_call("Task", description="d", prompt="loop"), read, read, read, DONE)
        _, said, events = self.chat()
        (res,) = [e for e in events if e["event"] == "result" and "reached its limit of 2 rounds" in e["text"]]
        self.assertFalse(res["ok"])
        self.assertEqual(len([e for e in events if e["event"] == "step"]), 2)
        self.assertTrue(said.strip().endswith("Done."))                                             # the main answer goes on

    def test_a_long_report_is_cut(self):
        old = subagent.MAX_REPORT
        subagent.MAX_REPORT = 300
        self.addCleanup(setattr, subagent, "MAX_REPORT", old)
        self.start(tool_call("Task", description="d", prompt="p"), report(" ".join(f"w{i}" for i in range(400))), DONE)
        _, _, events = self.chat()
        (res,) = [e for e in events if e["event"] == "result" and "Report of the helper" in e["text"]]
        self.assertIn("the report is cut here", res["text"])

    def test_a_helper_that_says_nothing_is_a_failed_call_and_the_main_answer_goes_on(self):
        self.start(tool_call("Task", description="d", prompt="p"), report(""), DONE)
        status, said, events = self.chat()
        self.assertEqual(status, 200)
        (res,) = [e for e in events if e["event"] == "result" and "finished without a report" in e["text"]]
        self.assertFalse(res["ok"])

    def test_without_helpers_on_a_call_the_model_makes_anyway_does_not_run(self):
        self.start(tool_call("Task", description="d", prompt="p"), DONE, agents=False)
        _, _, events = self.chat()
        self.assertEqual([e for e in events if e["event"] in ("step", "helper")], [])
        self.assertEqual(len(self.engine.prompts), 1)                                               # no second request: no helper ran

    def test_the_setting_is_listed_and_changed_from_this_pc_and_kept_in_the_run_config(self):
        self.start(DONE, agents=False)
        first = self.get("/agent/helpers")
        self.assertEqual((first["on"], first["available"], first["editable"]), (False, True, True))
        with urllib.request.urlopen(self.post("/agent/helpers", {"on": True}), timeout=10) as r:
            self.assertTrue(json.loads(r.read())["on"])
        self.assertEqual(json.loads(self.config.read_text(encoding="utf-8"))["agents"], {"on": True})
        with urllib.request.urlopen(self.post("/agent/helpers", {"on": False}), timeout=10) as r:
            self.assertFalse(json.loads(r.read())["on"])

    def test_the_route_checks_what_it_gets_and_who_asks(self):
        self.start(DONE, agents=False)
        for body in ({"on": "yes"}, {"other": 1}, [], "x"):
            with self.assertRaises(urllib.error.HTTPError) as cm:
                urllib.request.urlopen(self.post("/agent/helpers", body), timeout=10)
            self.assertEqual(cm.exception.code, 400, body)
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(self.post("/agent/helpers", {"on": True}, {"Origin": "http://evil.example"}), timeout=10)
        self.assertEqual(cm.exception.code, 403)
        self.assertEqual(json.loads(self.config.read_text(encoding="utf-8")), {})
        self.svc.config_path = None
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(self.post("/agent/helpers", {"on": True}), timeout=10)
        self.assertEqual(cm.exception.code, 409)


if __name__ == "__main__":
    unittest.main()
