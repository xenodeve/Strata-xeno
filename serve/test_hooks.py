"""Hooks: the user's own commands around the chat's tool calls (serve/hooks.py), tried with real commands in a real shell - one that stops a call, one that feeds back what it found,
one that runs out of time - and through the chat endpoint with a scripted engine."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from serve import agent, hooks, permissions, shell  # noqa: E402
from serve.test_agent_chat import DONE, Fixture  # noqa: E402
from serve.test_harness import tool_call  # noqa: E402

SH = shell.find_shell(prefer="bash")
BASH = SH is not None and SH.kind == "bash"


def cfg(*items, off=()):
    return {"hooks": list(items), "hooks_off": list(off)}


def hook(event, command, **more):
    return {"event": event, "command": command, **more}


class Reading(unittest.TestCase):
    def test_nothing_is_defined_by_default(self):
        self.assertEqual(hooks.load({}), ([], []))
        self.assertEqual(hooks.view({}), {"hooks": [], "problems": []})

    def test_a_hook_has_an_event_a_matcher_a_command_and_a_time_limit(self):
        (h,), problems = hooks.load(cfg(hook("before_tool", "echo hi", matcher="Edit|Write", timeout=5)))
        self.assertEqual((h.event, h.matcher, h.command, h.timeout, h.on, problems), ("before_tool", "Edit|Write", "echo hi", 5, True, []))
        self.assertTrue(h.matches("Edit") and h.matches("Write"))
        self.assertFalse(h.matches("Read") or h.matches("MultiEdit") or h.matches("EditNotebook"))      # the whole name has to match

    def test_no_matcher_or_a_star_is_every_tool(self):
        for m in ({}, {"matcher": ""}, {"matcher": "*"}):
            (h,), _ = hooks.load(cfg(hook("after_tool", "x", **m)))
            self.assertIsNone(h.matcher)
            self.assertTrue(h.matches("Read") and h.matches("Bash"))

    def test_the_default_time_limit_is_thirty_seconds(self):
        (h,), _ = hooks.load(cfg(hook("stop", "x")))
        self.assertEqual(h.timeout, 30)

    def test_a_bad_entry_is_left_out_with_a_reason_and_does_not_take_the_others(self):
        good = hook("prompt", "echo ok")
        got, problems = hooks.load(cfg("text", hook("sometime", "x"), hook("stop", ""), hook("stop", 5), hook("stop", "x", matcher="("), hook("stop", "x", timeout=0), hook("stop", "x", timeout=301),
                                       hook("stop", "x", timeout=True), hook("stop", "x", matcher=3), good))
        self.assertEqual([h.command for h in got], ["echo ok"])
        self.assertEqual(len(problems), 9)
        self.assertTrue(all(p.startswith("hook ") for p in problems))

    def test_hooks_that_is_not_a_list_is_one_problem(self):
        self.assertEqual(hooks.load({"hooks": {"a": 1}})[0], [])
        self.assertEqual(len(hooks.load({"hooks": {"a": 1}})[1]), 1)

    def test_the_same_hook_twice_is_one(self):
        got, _ = hooks.load(cfg(hook("stop", "echo"), hook("stop", "echo")))
        self.assertEqual(len(got), 1)

    def test_only_the_first_fifty_are_used(self):
        got, problems = hooks.load(cfg(*[hook("stop", f"echo {i}") for i in range(60)]))
        self.assertEqual(len(got), 50)
        self.assertEqual(len(problems), 1)

    def test_a_hook_is_off_when_its_id_is_in_hooks_off(self):
        (h,), _ = hooks.load(cfg(hook("stop", "echo")))
        (h2,), _ = hooks.load(cfg(hook("stop", "echo"), off=[h.id]))
        self.assertTrue(h.on)
        self.assertFalse(h2.on)
        self.assertEqual(h.id, h2.id)                                       # the id follows what the hook is, not where it is in the list

    def test_the_switches_a_page_sends_are_checked(self):
        self.assertEqual(hooks.check_off(["h0123abcd", "h0123abcd"]), (["h0123abcd"], None))
        for bad in ("x", [1], ["nope"], ["h" + "0" * 9], {"a": 1}, ["h0123abcd"] * 51, None):
            self.assertIsNone(hooks.check_off(bad)[0], bad)


@unittest.skipUnless(BASH, "needs bash (Git Bash on Windows)")
class Running(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.proj = Path(self._tmp.name).resolve() / "proj"
        self.proj.mkdir()
        self.told: list[dict] = []
        self.cancelled = False

    def runner(self, *items, off=()):
        defined, _ = hooks.load(cfg(*items, off=off))
        return hooks.Runner(defined, SH, str(self.proj), "s1", self.told.append, lambda: self.cancelled)

    def test_exit_code_two_stops_the_call_and_the_output_is_the_reason(self):
        r = self.runner(hook("before_tool", 'echo "no commits to main" >&2; exit 2'))
        self.assertEqual(r.before("Bash", {"command": "git commit"}), "no commits to main")
        self.assertTrue(self.told[0]["blocked"] and self.told[0]["code"] == 2 and not self.told[0]["ok"])

    def test_any_other_exit_code_does_not_stop_the_call_and_is_shown(self):
        r = self.runner(hook("before_tool", "echo trouble; exit 1"))
        self.assertIsNone(r.before("Bash", {"command": "ls"}))
        self.assertEqual((self.told[0]["blocked"], self.told[0]["code"], self.told[0]["text"]), (False, 1, "trouble"))

    def test_a_hook_that_passes_says_so(self):
        r = self.runner(hook("before_tool", "exit 0"))
        self.assertIsNone(r.before("Read", {}))
        self.assertTrue(self.told[0]["ok"])

    def test_the_hook_gets_the_event_as_json_and_runs_in_the_project_folder(self):
        r = self.runner(hook("before_tool", 'cat > seen.json; pwd -W 2>/dev/null > where.txt || pwd > where.txt'))
        r.before("Edit", {"file_path": "a.txt", "old_string": "x", "new_string": "y"})
        seen = json.loads((self.proj / "seen.json").read_text(encoding="utf-8"))
        self.assertEqual((seen["hook_event_name"], seen["tool_name"], seen["tool_input"]["file_path"], seen["session_id"]), ("before_tool", "Edit", "a.txt", "s1"))
        self.assertEqual(os.path.normcase(os.path.realpath(seen["cwd"])), os.path.normcase(str(self.proj)))
        self.assertTrue((self.proj / "where.txt").exists())                        # it ran in the project folder: the files it made are there

    def test_a_hook_that_does_not_read_its_input_is_no_trouble(self):
        r = self.runner(hook("before_tool", "exit 0"))
        for _ in range(3):
            self.assertIsNone(r.before("Write", {"file_path": "a", "content": "x" * 200_000}))

    def test_the_servers_own_settings_are_not_in_a_hooks_environment(self):
        os.environ["STRATA_TEST_HOOK"] = "secret"
        self.addCleanup(os.environ.pop, "STRATA_TEST_HOOK", None)
        r = self.runner(hook("prompt", 'echo "[${STRATA_TEST_HOOK-unset}]"'))
        self.assertIn("[unset]", r.prompt("go"))

    def test_only_the_tools_the_matcher_names_run_the_hook(self):
        r = self.runner(hook("before_tool", "echo ran >> ran.txt", matcher="Edit|Write"))
        r.before("Read", {})
        r.before("Bash", {})
        self.assertFalse((self.proj / "ran.txt").exists())
        r.before("Write", {})
        self.assertTrue((self.proj / "ran.txt").exists())

    def test_a_hook_that_runs_out_of_time_is_stopped_shown_and_does_not_stop_the_call(self):
        r = self.runner(hook("before_tool", "sleep 30; echo late", timeout=1))
        t0 = time.monotonic()
        self.assertIsNone(r.before("Bash", {}))
        self.assertLess(time.monotonic() - t0, 10)
        self.assertTrue(self.told[0]["timeout"] and not self.told[0]["blocked"] and not self.told[0]["ok"])

    def test_a_command_that_cannot_run_is_shown(self):
        r = self.runner(hook("before_tool", "exit 0"))
        r.cwd = str(self.proj / "gone")
        self.assertIsNone(r.before("Bash", {}))
        self.assertTrue(self.told[0]["error"] and not self.told[0]["ok"])

    def test_a_cancelled_request_does_not_wait_for_a_hook(self):
        self.cancelled = True
        r = self.runner(hook("before_tool", "sleep 30"), hook("after_tool", "sleep 30"), hook("prompt", "sleep 30"), hook("stop", "sleep 30"))
        t0 = time.monotonic()
        self.assertIsNone(r.before("Bash", {}))
        self.assertEqual(r.after("Bash", {}, {"content": [{"type": "text", "text": "x"}]}), "")
        self.assertEqual(r.prompt("x"), "")
        r.stop("x")
        self.assertLess(time.monotonic() - t0, 5)

    def test_a_cancel_while_it_runs_ends_it(self):
        r = self.runner(hook("prompt", "sleep 30; echo late"))
        import threading
        threading.Timer(0.6, lambda: setattr(self, "cancelled", True)).start()
        t0 = time.monotonic()
        r.prompt("x")
        self.assertLess(time.monotonic() - t0, 10)

    def test_after_a_call_what_the_hook_prints_is_for_the_model_marked_as_a_hooks(self):
        r = self.runner(hook("after_tool", 'cat > in.json; echo "lint: 2 errors"; exit 1'))
        said = r.after("Edit", {"file_path": "a.py"}, {"content": [{"type": "text", "text": "edited"}], "isError": False})
        self.assertIn("lint: 2 errors", said)
        self.assertIn("exit code 1", said)
        self.assertIn("not an instruction", said)
        seen = json.loads((self.proj / "in.json").read_text(encoding="utf-8"))
        self.assertEqual(seen["tool_response"], {"is_error": False, "text": "edited"})

    def test_a_hook_with_nothing_to_say_adds_nothing(self):
        r = self.runner(hook("after_tool", "exit 0"))
        self.assertEqual(r.after("Edit", {}, {"content": [{"type": "text", "text": "x"}]}), "")

    def test_a_prompt_hooks_output_is_for_the_model_and_gets_the_prompt(self):
        r = self.runner(hook("prompt", 'cat > p.json; echo "branch is main"'))
        said = r.prompt("fix the bug")
        self.assertIn("branch is main", said)
        self.assertEqual(json.loads((self.proj / "p.json").read_text(encoding="utf-8"))["prompt"], "fix the bug")
        self.assertIsNone(self.told[0]["tool"])
        self.assertIn("call_id", self.told[0])                                        # it belongs to the answer, not to a call
        self.assertIsNone(self.told[0]["call_id"])

    def test_a_stop_hooks_output_is_shown_and_not_given_to_the_model(self):
        r = self.runner(hook("stop", "echo finished"))
        self.assertIsNone(r.stop("all done"))
        self.assertEqual(self.told[0]["text"], "finished")
        self.assertEqual(self.told[0]["on"], "stop")

    def test_a_hook_that_is_off_does_not_run(self):
        (h,), _ = hooks.load(cfg(hook("before_tool", "echo ran >> ran.txt; exit 2")))
        r = self.runner(hook("before_tool", "echo ran >> ran.txt; exit 2"), off=[h.id])
        self.assertIsNone(r.before("Bash", {}))
        self.assertFalse((self.proj / "ran.txt").exists())

    def test_no_shell_or_no_folder_means_no_hooks(self):
        defined, _ = hooks.load(cfg(hook("before_tool", "exit 2")))
        self.assertFalse(hooks.Runner(defined, None, str(self.proj), "s", self.told.append))
        self.assertFalse(hooks.Runner(defined, SH, None, "s", self.told.append))
        self.assertFalse(hooks.Runner([], SH, str(self.proj), "s", self.told.append))

    def test_several_hooks_run_in_order_and_the_first_to_stop_wins(self):
        r = self.runner(hook("before_tool", "echo one >> order.txt"), hook("before_tool", "echo two >> order.txt; echo stop two; exit 2"), hook("before_tool", "echo three >> order.txt"))
        self.assertEqual(r.before("Bash", {}), "stop two")
        self.assertEqual((self.proj / "order.txt").read_text().split(), ["one", "two"])


@unittest.skipUnless(BASH, "needs bash (Git Bash on Windows)")
class AtTheGate(unittest.TestCase):
    """A hook can only make a call stricter: it runs after the rules have not denied the call, and nothing it says allows anything."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        base = Path(self._tmp.name).resolve()
        self.proj = base / "proj"
        self.proj.mkdir()
        (self.proj / "a.txt").write_text("hello\n", encoding="utf-8")
        self.out = base / "out"
        self.out.mkdir()
        (self.out / "b.txt").write_text("outside\n", encoding="utf-8")
        self.srv = agent.AgentServer()
        self.addCleanup(self.srv.close)
        self.told: list[dict] = []
        self.asked: list[dict] = []

    def ctx(self, *items, deny=()):
        defined, _ = hooks.load(cfg(*items))
        c = agent.AgentContext(policy=permissions.Policy(cwd=str(self.proj), deny=list(deny)), session="s", ask=lambda req: (self.asked.append(req), "deny")[1])
        c.hooks = hooks.Runner(defined, SH, str(self.proj), "s", self.told.append)
        return c

    def text(self, r):
        return r["content"][0]["text"]

    def test_a_before_hook_that_stops_a_call_keeps_it_from_running(self):
        r = self.srv.call("Read", {"file_path": "a.txt"}, ctx=self.ctx(hook("before_tool", 'echo "reading is closed today" >&2; exit 2', matcher="Read")))
        self.assertTrue(r["isError"])
        self.assertIn("stopped by a hook: reading is closed today", self.text(r))
        self.assertNotIn("hello", self.text(r))

    def test_a_hook_that_says_nothing_lets_the_call_run(self):
        r = self.srv.call("Read", {"file_path": "a.txt"}, ctx=self.ctx(hook("before_tool", "exit 0")))
        self.assertFalse(r.get("isError"))
        self.assertIn("hello", self.text(r))

    def test_an_after_hook_adds_what_it_found_to_the_result(self):
        r = self.srv.call("Read", {"file_path": "a.txt"}, ctx=self.ctx(hook("after_tool", "echo style check failed")))
        self.assertIn("hello", self.text(r))
        self.assertIn("style check failed", self.text(r))

    def test_a_deny_rule_settles_it_before_any_hook_runs(self):
        r = self.srv.call("Read", {"file_path": "a.txt"}, ctx=self.ctx(hook("before_tool", "echo ran >> ../ran.txt"), hook("after_tool", "echo ran >> ../ran.txt"), deny=["Read(a.txt)"]))
        self.assertTrue(r["isError"])
        self.assertIn("denied", self.text(r))
        self.assertFalse((self.proj.parent / "ran.txt").exists())
        self.assertEqual(self.told, [])

    def test_a_hook_that_passes_does_not_allow_what_asks(self):
        target = str(self.out / "b.txt")
        r = self.srv.call("Read", {"file_path": target}, ctx=self.ctx(hook("before_tool", "echo allowed; exit 0")))
        self.assertEqual(len(self.asked), 1)                                   # outside the project folder: the user is still asked
        self.assertTrue(r["isError"])                                          # and said no
        self.assertNotIn("outside", self.text(r).split("did not allow")[0])

    def test_a_hook_does_not_make_a_secret_or_dot_git_free(self):
        (self.proj / ".git").mkdir()
        (self.proj / ".env").write_text("KEY=1\n", encoding="utf-8")
        c = self.ctx(hook("before_tool", "exit 0"), hook("after_tool", "exit 0"))
        self.assertTrue(self.srv.call("Write", {"file_path": ".git/config", "content": "x"}, ctx=c)["isError"])
        self.assertTrue(self.srv.call("Read", {"file_path": ".env"}, ctx=c)["isError"] or bool(self.asked))

    def test_a_blocked_call_is_not_asked_about_first(self):
        r = self.srv.call("Read", {"file_path": str(self.out / "b.txt")}, ctx=self.ctx(hook("before_tool", "echo no; exit 2")))
        self.assertIn("stopped by a hook", self.text(r))
        self.assertEqual(self.asked, [])                                       # a policy says no before the user is bothered

    def test_no_hooks_changes_nothing(self):
        c = agent.AgentContext(policy=permissions.Policy(cwd=str(self.proj)), session="s")
        self.assertIn("hello", self.text(self.srv.call("Read", {"file_path": "a.txt"}, ctx=c)))


class Chat(Fixture):
    """Hooks through the chat endpoint: the run config holds them, the stream says what they did."""

    def setUp(self):
        super().setUp()
        self.config = self.base / "run.json"

    def write(self, *items, off=()):
        self.config.write_text(json.dumps(cfg(*items, off=off)), encoding="utf-8")

    def start(self, *scripts, **kw):
        super().start(*scripts, with_shell=True, **kw)
        self.svc.config_path = str(self.config)

    def get(self, path):
        with urllib.request.urlopen(self.base_url + path, timeout=10) as r:
            return json.loads(r.read())

    def hook_events(self, events):
        return [e for e in events if e["event"] == "hook"]

    @unittest.skipUnless(BASH, "needs bash")
    def test_a_before_hook_stops_the_call_and_the_model_and_the_page_are_told(self):
        self.write(hook("before_tool", 'echo "no reading today" >&2; exit 2', matcher="Read"))
        self.start(tool_call("Read", file_path="a.txt"), DONE)
        status, _, events = self.chat()
        self.assertEqual(status, 200)
        (ev,) = [e for e in self.hook_events(events) if e["on"] == "before_tool"]
        call = [e for e in events if e["event"] == "call"][0]
        self.assertTrue(ev["blocked"])
        self.assertEqual(ev["call_id"], call["id"])                              # tied to the call it stopped
        result = [e for e in events if e["event"] == "result"][0]
        self.assertFalse(result["ok"])
        self.assertIn("no reading today", result["text"])
        self.assertIn("no reading today", self.engine.prompt_text(1))            # the model read why

    @unittest.skipUnless(BASH, "needs bash")
    def test_a_prompt_hooks_output_reaches_the_model_and_a_stop_hook_runs_at_the_end(self):
        self.write(hook("prompt", "echo remember: tabs not spaces"), hook("stop", "echo checked at the end"))
        self.start(DONE)
        status, _, events = self.chat()
        self.assertEqual(status, 200)
        self.assertIn("remember: tabs not spaces", self.engine.prompt_text(0))
        on = {e["on"]: e for e in self.hook_events(events)}
        self.assertEqual(set(on), {"prompt", "stop"})
        self.assertIsNone(on["prompt"]["call_id"])
        self.assertIsNone(on["stop"]["call_id"])
        self.assertEqual(on["stop"]["text"], "checked at the end")
        self.assertLess([e["event"] for e in events].index("hook"), len(events))   # the prompt hook is reported before the answer

    @unittest.skipUnless(BASH, "needs bash")
    def test_an_after_hook_feeds_the_model_and_a_hook_that_fails_does_not_hang_the_answer(self):
        self.write(hook("after_tool", "echo lint found 3 problems; exit 1", matcher="Read"), hook("before_tool", "sleep 30", timeout=1, matcher="Read"))
        self.start(tool_call("Read", file_path="a.txt"), DONE)
        t0 = time.monotonic()
        status, said, events = self.chat()
        self.assertEqual(status, 200)
        self.assertLess(time.monotonic() - t0, 25)
        self.assertTrue(said.strip().endswith("Done."))
        self.assertIn("lint found 3 problems", self.engine.prompt_text(1))
        before = [e for e in self.hook_events(events) if e["on"] == "before_tool"][0]
        self.assertTrue(before["timeout"])
        self.assertTrue([e for e in events if e["event"] == "result"][0]["ok"])      # the call ran: a hook that timed out does not stop it

    @unittest.skipUnless(BASH, "needs bash")
    def test_a_hook_that_is_switched_off_does_not_run(self):
        h = hook("before_tool", "exit 2")
        (hid,) = [x.id for x in hooks.load(cfg(h))[0]]
        self.write(h, off=[hid])
        self.start(tool_call("Read", file_path="a.txt"), DONE)
        _, _, events = self.chat()
        self.assertEqual(self.hook_events(events), [])
        self.assertTrue([e for e in events if e["event"] == "result"][0]["ok"])

    def test_without_hooks_nothing_about_them_is_in_the_stream(self):
        self.start(DONE)
        _, _, events = self.chat()
        self.assertEqual(self.hook_events(events), [])

    @unittest.skipUnless(BASH, "needs bash")
    def test_the_settings_list_them_and_a_switch_is_kept_in_the_run_config(self):
        self.write(hook("before_tool", "echo a", matcher="Bash"), hook("stop", "echo b"))
        self.start(DONE)
        listed = self.get("/agent/hooks")
        self.assertEqual([(h["event"], h["on"]) for h in listed["hooks"]], [("before_tool", True), ("stop", True)])
        self.assertTrue(listed["shell"] and listed["editable"])
        first = listed["hooks"][0]["id"]
        with urllib.request.urlopen(self.post("/agent/hooks", {"off": [first, "h00000000"]}), timeout=10) as r:
            after = json.loads(r.read())
        self.assertEqual([(h["event"], h["on"]) for h in after["hooks"]], [("before_tool", False), ("stop", True)])
        saved = json.loads(self.config.read_text(encoding="utf-8"))
        self.assertEqual(saved["hooks_off"], [first])                          # an id that is no hook is not kept
        self.assertEqual(len(saved["hooks"]), 2)                               # the definitions are untouched
        with urllib.request.urlopen(self.post("/agent/hooks", {"off": []}), timeout=10) as r:
            self.assertTrue(all(h["on"] for h in json.loads(r.read())["hooks"]))

    def test_the_switch_checks_what_it_gets_and_who_asks(self):
        self.write(hook("stop", "echo b"))
        self.start(DONE)
        for body in ({"off": "x"}, {"off": ["nope"]}, {"off": [], "hooks": []}, {"hooks": [{"event": "stop", "command": "evil"}]}, [], {}):
            with self.assertRaises(urllib.error.HTTPError) as cm:
                urllib.request.urlopen(self.post("/agent/hooks", body), timeout=10)
            self.assertEqual(cm.exception.code, 400, body)
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(self.post("/agent/hooks", {"off": []}, {"Origin": "http://evil.example"}), timeout=10)
        self.assertEqual(cm.exception.code, 403)
        self.assertEqual(len(json.loads(self.config.read_text(encoding="utf-8"))["hooks"]), 1)      # no page can write a hook

    def test_without_a_run_config_file_there_is_nowhere_to_keep_a_switch(self):
        self.start(DONE)
        self.svc.config_path = None
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(self.post("/agent/hooks", {"off": []}), timeout=10)
        self.assertEqual(cm.exception.code, 409)
        self.assertFalse(self.get("/agent/hooks")["editable"])


if __name__ == "__main__":
    unittest.main()
