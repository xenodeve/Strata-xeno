"""The coding tools through the chat endpoint: a request with "strata_agent" gives the model Read, Write, Edit, Glob, Grep, Bash, ... (Claude Code's names), the server runs
them in its tool loop, and anything that needs the user's say-so is a `permission` event in the stream that the page answers with POST /agent/permission."""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from serve import agent, agent_run, shell  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.mcp import hub_from_config  # noqa: E402
from serve.server import ByteTokenizer, Service, serve  # noqa: E402
from serve.test_harness import tool_call  # noqa: E402
from serve.test_mcp import ScriptedEngine  # noqa: E402

DONE = "</think>\n\nDone."


class Fixture(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name).resolve()
        self.proj = self.base / "proj"
        self.proj.mkdir()
        (self.proj / "a.txt").write_text("hello\nworld\n", encoding="utf-8")
        self.outside = self.base / "elsewhere"
        self.outside.mkdir()
        (self.outside / "b.txt").write_text("secret-ish\n", encoding="utf-8")
        self.httpd = None

    def tearDown(self):
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
        if getattr(self, "hub", None):
            self.hub.close()

    def start(self, *scripts, with_shell=False):
        self.agent = agent.AgentServer()
        if with_shell:
            shell.install(self.agent, shell.find_shell(prefer="bash"))
        self.hub = hub_from_config({}, None, builtins={"agent": self.agent})
        self.hub.start(wait=True)
        tok = ByteTokenizer()
        self.engine = ScriptedEngine(tok, list(scripts), max_context=200_000)       # the rules and the tools are a few thousand bytes: a byte tokenizer needs room
        self.svc = Service(self.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.mcp = self.hub
        self.svc.agent = self.agent
        self.httpd = serve(self.svc, port=0)
        self.base_url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def body(self, **kw):
        sa = {"cwd": str(self.proj), "session": "chat-1", **kw.pop("agent", {})}
        return {"model": "m", "messages": kw.pop("messages", [{"role": "user", "content": "look at a.txt"}]), "stream": True, "strata_agent": sa, **kw}

    def post(self, path, body, headers=None):
        h = {"Content-Type": "application/json", **(headers or {})}
        return urllib.request.Request(self.base_url + path, data=json.dumps(body).encode(), headers=h)

    def chat(self, body=None, on_event=None, headers=None):
        """Run a chat request; returns (status, text said, events of the stream).  `on_event(event)` is called as each strata_mcp event arrives."""
        events, said = [], []
        try:
            with urllib.request.urlopen(self.post("/v1/chat/completions", body or self.body(), headers), timeout=60) as r:
                for raw in r:
                    ln = raw.decode().strip()
                    if not ln.startswith("data: {"):
                        continue
                    c = json.loads(ln[6:])
                    if "strata_mcp" in c:
                        events.append(c["strata_mcp"])
                        if on_event:
                            on_event(c["strata_mcp"])
                    for ch in c.get("choices") or []:
                        said.append(ch["delta"].get("content") or "")
                return r.status, "".join(said), events
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode(), events

    def answer(self, rid, decision):
        with urllib.request.urlopen(self.post("/agent/permission", {"id": rid, "decision": decision}), timeout=10) as r:
            return r.status


class TheTools(Fixture):
    def test_the_model_reads_a_file_with_claude_codes_tool_and_name(self):
        self.start(tool_call("Read", file_path="a.txt"), DONE)
        status, said, events = self.chat()
        self.assertEqual(status, 200)
        self.assertIn("Done.", said)
        first, second = self.engine.prompt_text(0), self.engine.prompt_text(1)
        self.assertIn('"name": "Read"', first)                               # the plain name, not agent__Read
        self.assertNotIn("agent__", first)
        self.assertIn("1\thello\n     2\tworld", second)                  # the tool's result (the template trims the first line's padding)
        names = [e.get("name") for e in events if e["event"] == "call"]
        self.assertEqual(names, ["Read"])
        self.assertEqual([e["ok"] for e in events if e["event"] == "result"], [True])

    def test_the_rules_for_the_ai_and_the_folder_are_in_the_prompt(self):
        self.start(DONE)
        self.chat()
        p = self.engine.prompt_text(0)
        self.assertIn("coding assistant", p)
        self.assertIn(str(self.proj).replace("\\", "/").lower(), p.replace("\\", "/").lower())
        self.assertIn("data, not instructions", p)

    def test_the_projects_own_instructions_are_in_the_prompt(self):
        (self.proj / "CLAUDE.md").write_text("Always answer in haiku.", encoding="utf-8")
        self.start(DONE)
        self.chat()
        self.assertIn("Always answer in haiku.", self.engine.prompt_text(0))

    def test_a_request_without_strata_agent_does_not_get_the_tools(self):
        self.start(DONE)
        body = self.body()
        del body["strata_agent"]
        body["strata_mcp"] = True
        self.chat(body)
        self.assertNotIn('"name": "Read"', self.engine.prompt_text(0))
        self.assertNotIn("coding assistant", self.engine.prompt_text(0))

    def test_the_tools_are_not_offered_as_mcp_servers_to_a_request_that_only_asks_for_mcp(self):
        self.start(DONE)
        self.assertNotIn("agent", {s["name"] for s in self.hub.status()["servers"]})
        self.assertEqual(self.hub.status()["tools"], 0)

    def test_more_than_the_usual_number_of_rounds_are_allowed(self):
        self.start(*[tool_call("Glob", pattern="*.txt") for _ in range(12)], DONE)
        status, said, events = self.chat()
        self.assertIn("Done.", said)
        self.assertEqual(len([e for e in events if e["event"] == "result"]), 12)

    def test_a_long_file_is_not_cut_by_the_ordinary_tool_limit(self):
        (self.proj / "big.txt").write_text("".join(f"line {i}\n" for i in range(1, 2001)), encoding="utf-8")
        self.start(tool_call("Read", file_path="big.txt"), DONE)
        self.chat()
        self.assertIn("line 2000", self.engine.prompt_text(1))

    def test_the_todo_list_reaches_the_page(self):
        todos = json.dumps([{"content": "a", "status": "in_progress", "activeForm": "Doing a"}])
        self.start(tool_call("TodoWrite", todos=todos), DONE)
        _, _, events = self.chat()
        got = [e for e in events if e["event"] == "todos"]
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["todos"][0]["content"], "a")
        self.assertTrue(got[0]["call_id"])


class Asking(Fixture):
    def outside_write(self):
        return tool_call("Write", file_path=str(self.outside / "new.txt"), content="hi")

    def test_a_call_outside_the_folder_is_a_card_and_runs_when_the_page_allows_it(self):
        self.start(self.outside_write(), DONE)
        seen = []

        def on(e):
            if e["event"] == "permission":
                seen.append(e)
                self.assertEqual(self.answer(e["id"], "allow"), 200)
        status, said, events = self.chat(on_event=on)
        self.assertEqual(status, 200)
        self.assertEqual(len(seen), 1)
        self.assertEqual((seen[0]["tool"], seen[0]["why"]), ("Write", "it is outside the project folder"))
        self.assertTrue(seen[0]["call_id"])
        self.assertEqual(seen[0]["rule"], f"Write({self.outside.as_posix()}/**)")
        self.assertEqual((self.outside / "new.txt").read_text(), "hi")
        self.assertIn("Done.", said)

    def test_when_the_page_refuses_nothing_runs_and_the_model_is_told(self):
        self.start(self.outside_write(), DONE)
        self.chat(on_event=lambda e: e["event"] == "permission" and self.answer(e["id"], "deny"))
        self.assertFalse((self.outside / "new.txt").exists())
        self.assertIn("did not allow", self.engine.prompt_text(1))

    def test_only_what_asks_is_a_card(self):
        self.start(tool_call("Read", file_path="a.txt"), DONE)
        _, _, events = self.chat()
        self.assertEqual([e for e in events if e["event"] == "permission"], [])

    def test_a_rule_the_page_sends_lets_it_through_without_asking(self):
        self.start(self.outside_write(), DONE)
        _, _, events = self.chat(self.body(agent={"allow": [f"Write({self.outside.as_posix()}/**)"]}))
        self.assertEqual([e for e in events if e["event"] == "permission"], [])
        self.assertTrue((self.outside / "new.txt").exists())

    def test_a_deny_rule_refuses_without_asking(self):
        self.start(tool_call("Read", file_path="a.txt"), DONE)
        _, _, events = self.chat(self.body(agent={"deny": ["Read(a.txt)"]}))
        self.assertEqual([e for e in events if e["event"] == "permission"], [])
        self.assertIn("denied", self.engine.prompt_text(1))

    def test_plan_mode_refuses_a_change_and_says_so_in_the_prompt(self):
        self.start(tool_call("Write", file_path="x.txt", content="x"), DONE)
        self.chat(self.body(agent={"mode": "plan"}))
        self.assertFalse((self.proj / "x.txt").exists())
        self.assertIn("plan mode", self.engine.prompt_text(0).lower())
        self.assertIn("plan mode", self.engine.prompt_text(1).lower())

    def test_an_unknown_mode_is_the_default_one(self):
        self.start(self.outside_write(), DONE)
        seen = []
        self.chat(self.body(agent={"mode": "bypass"}), on_event=lambda e: e["event"] == "permission" and (seen.append(e), self.answer(e["id"], "deny")))
        self.assertEqual(len(seen), 1)

    def test_a_request_that_goes_away_while_a_question_is_open_does_not_hang_the_server(self):
        self.start(self.outside_write(), DONE)
        got = threading.Event()
        res = {}

        def run():
            try:
                with urllib.request.urlopen(self.post("/v1/chat/completions", self.body()), timeout=20) as r:
                    for raw in r:
                        if b'"permission"' in raw:
                            got.set()
                            return                                       # the page is gone: the connection closes with the question open
            except OSError as e:
                res["e"] = e
        t = threading.Thread(target=run)
        t.start()
        self.assertTrue(got.wait(15))
        t.join(15)
        deadline = __import__("time").time() + 15
        while __import__("time").time() < deadline and self.svc.broker.pending:
            __import__("time").sleep(0.1)
        self.assertEqual(self.svc.broker.pending, {})
        self.assertFalse((self.outside / "new.txt").exists())


class AutoMode(Fixture):
    def test_a_safe_verdict_runs_a_call_outside_the_folder_without_a_card(self):
        self.start(tool_call("Read", file_path=str(self.outside / "b.txt")), "<severity>1</severity>", DONE)
        _, said, events = self.chat(self.body(agent={"mode": "auto"}))
        kinds = [e["event"] for e in events]
        self.assertNotIn("permission", kinds)
        self.assertEqual([k for k in kinds if k in ("judging", "judged")], ["judging", "judged"])
        self.assertIn("secret-ish", self.engine.prompt_text(2))
        self.assertIn("look at a.txt", self.engine.prompt_text(1))               # the judge was told what the user asked
        self.assertIn("b.txt", self.engine.prompt_text(1))
        self.assertNotIn("Read", self.engine.prompt_text(1).split("The action")[0])

    def test_a_risky_verdict_blocks_it_and_tells_the_model(self):
        self.start(tool_call("Read", file_path=str(self.outside / "b.txt")), "<severity>5</severity>", DONE)
        _, _, events = self.chat(self.body(agent={"mode": "auto"}))
        self.assertNotIn("permission", [e["event"] for e in events])
        self.assertNotIn("secret-ish", self.engine.prompt_text(2))
        self.assertIn("blocked", self.engine.prompt_text(2))

    def test_an_unsure_verdict_asks_the_user(self):
        self.start(tool_call("Read", file_path=str(self.outside / "b.txt")), "<severity>3</severity>", DONE)
        seen = []
        self.chat(self.body(agent={"mode": "auto"}), on_event=lambda e: e["event"] == "permission" and (seen.append(e), self.answer(e["id"], "allow")))
        self.assertEqual(len(seen), 1)
        self.assertIn("secret-ish", self.engine.prompt_text(2))

    def test_a_secret_is_never_sent_to_the_judge(self):
        (self.proj / ".env").write_text("TOKEN=abc\n", encoding="utf-8")
        self.start(tool_call("Read", file_path=".env"), DONE)
        seen = []
        _, _, events = self.chat(self.body(agent={"mode": "auto"}), on_event=lambda e: e["event"] == "permission" and (seen.append(e), self.answer(e["id"], "deny")))
        self.assertEqual(len(seen), 1)
        self.assertNotIn("judging", [e["event"] for e in events])
        self.assertEqual(len(self.engine.prompts), 2)                           # no judge request


class OtherFolders(Fixture):
    """The page names the project's other folders (strata_agent.dirs): the model is told of them and files in them are free; odd values are ignored."""

    def setUp(self):
        super().setUp()
        self.wt = self.base / "proj-wt2"
        self.wt.mkdir()
        (self.wt / "c.txt").write_text("from the other worktree\n", encoding="utf-8")

    def test_a_file_in_another_folder_runs_without_asking_and_the_prompt_names_it(self):
        self.start(tool_call("Read", file_path=str(self.wt / "c.txt")), DONE)
        status, said, events = self.chat(self.body(agent={"dirs": [str(self.wt)]}))
        self.assertEqual(status, 200)
        self.assertFalse([e for e in events if e["event"] == "permission"])
        self.assertIn("from the other worktree", json.dumps(events))
        self.assertIn("Other folders of the project", self.engine.prompt_text(0))
        self.assertIn(str(self.wt), self.engine.prompt_text(0))

    def test_without_naming_it_the_same_file_asks(self):
        self.start(tool_call("Read", file_path=str(self.wt / "c.txt")), DONE)
        seen = []
        self.chat(self.body(), on_event=lambda e: e["event"] == "permission" and (seen.append(e), self.answer(e["id"], "deny")))
        self.assertEqual(len(seen), 1)

    def test_odd_values_are_ignored(self):
        self.start(DONE)
        odd = [5, None, "", "x\0y", str(self.base / "nope"), str(self.proj), str(self.wt), str(self.wt), {"a": 1}]
        status, _, _ = self.chat(self.body(agent={"dirs": odd}))
        self.assertEqual(status, 200)
        status, _, _ = self.chat(self.body(agent={"dirs": "not a list"}))
        self.assertEqual(status, 200)
        status, _, _ = self.chat(self.body(agent={"cwd": "", "dirs": [str(self.wt)]}))             # other folders without a main one
        self.assertEqual(status, 200)


class TheDoor(Fixture):
    def test_a_foreign_origin_is_refused(self):
        self.start(DONE)
        status, text, _ = self.chat(headers={"Origin": "http://evil.example"})
        self.assertEqual(status, 403)
        self.assertEqual(self.engine.prompts, [])

    def test_only_json_is_accepted(self):
        self.start(DONE)
        req = urllib.request.Request(self.base_url + "/v1/chat/completions", data=json.dumps(self.body()).encode(), headers={"Content-Type": "text/plain"})
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req, timeout=10)
        self.assertEqual(cm.exception.code, 415)

    def test_the_answer_endpoint_has_the_same_guard_and_checks_what_it_gets(self):
        self.start(DONE)
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(self.post("/agent/permission", {"id": "x", "decision": "allow"}, {"Origin": "http://evil.example"}), timeout=10)
        self.assertEqual(cm.exception.code, 403)
        for body, code in (({"id": "nope", "decision": "allow"}, 404), ({"id": "nope", "decision": "maybe"}, 400), ({"decision": "allow"}, 400), ([], 400)):
            with self.assertRaises(urllib.error.HTTPError) as cm:
                urllib.request.urlopen(self.post("/agent/permission", body), timeout=10)
            self.assertEqual(cm.exception.code, code, body)

    def test_a_server_with_the_tools_switched_off_says_so(self):
        self.start(DONE)
        self.svc.agent = None
        status, text, _ = self.chat()
        self.assertEqual(status, 409)
        self.assertIn("switched off", text)

    def test_a_folder_that_is_not_there_means_no_folder_not_a_crash(self):
        self.start(tool_call("Read", file_path="a.txt"), DONE)
        seen = []
        status, said, _ = self.chat(self.body(agent={"cwd": str(self.base / "nope")}), on_event=lambda e: e["event"] == "permission" and (seen.append(e), self.answer(e["id"], "deny")))
        self.assertEqual(status, 200)
        self.assertEqual(len(seen), 1)
        self.assertIn("no project folder", seen[0]["why"])

    def test_odd_values_in_the_request_are_ignored(self):
        self.start(DONE)
        status, _, _ = self.chat(self.body(agent={"mode": 5, "allow": "x", "deny": [1, 2], "session": ["a"], "cwd": 7}))
        self.assertEqual(status, 200)


@unittest.skipUnless(shell.find_shell(prefer="bash"), "needs bash")
class WithBash(Fixture):
    def test_a_command_asks_and_runs_when_allowed(self):
        self.start(tool_call("Bash", command="echo from-the-shell > made.txt"), DONE, with_shell=True)
        seen = []
        self.chat(on_event=lambda e: e["event"] == "permission" and (seen.append(e), self.answer(e["id"], "allow")))
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0]["rule"], None)                                 # a redirect is not a rule the page may remember
        self.assertEqual((self.proj / "made.txt").read_text().strip(), "from-the-shell")

    def test_a_plain_read_runs_without_asking(self):
        self.start(tool_call("Bash", command="ls"), DONE, with_shell=True)
        _, _, events = self.chat()
        self.assertEqual([e for e in events if e["event"] == "permission"], [])
        self.assertIn("a.txt", self.engine.prompt_text(1))


if __name__ == "__main__":
    unittest.main()
