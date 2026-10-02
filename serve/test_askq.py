"""AskUserQuestion: the model asks the user a question with choices, the page answers, and the answer goes back to the model as the tool's result (serve/agent.py, serve/agent_run.py)."""
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

from serve import agent, permissions  # noqa: E402
from serve.test_agent_chat import DONE, Fixture  # noqa: E402
from serve.test_harness import tool_call  # noqa: E402

Q = [{"question": "Which library should we use?", "header": "Library", "multiSelect": False,
      "options": [{"label": "requests", "description": "the usual one"}, {"label": "httpx", "description": "async too"}]}]


def run(args, answer):
    ctx = agent.AgentContext(policy=permissions.Policy(cwd=None), session="s", question=lambda qs: answer)
    return agent.AgentServer().call("AskUserQuestion", args, ctx=ctx)


class TheTool(unittest.TestCase):
    def text(self, r):
        return r["content"][0]["text"]

    def test_the_answer_goes_back_in_words_the_model_can_use(self):
        r = run({"questions": Q}, {"answers": {"Which library should we use?": ["httpx"]}})
        self.assertFalse(r["isError"])
        self.assertEqual(self.text(r), 'The user has answered your questions: "Which library should we use?"="httpx". You can now carry on with the answers in mind.')

    def test_several_questions_and_several_choices(self):
        qs = Q + [{"question": "Which checks?", "header": "Checks", "multiSelect": True, "options": [{"label": "lint", "description": "a"}, {"label": "tests", "description": "b"}, {"label": "types", "description": "c"}]}]
        r = run({"questions": qs}, {"answers": {"Which library should we use?": "requests", "Which checks?": ["lint", "tests", "my own: e2e"]}})
        self.assertIn('"Which library should we use?"="requests"', self.text(r))
        self.assertIn('"Which checks?"="lint, tests, my own: e2e"', self.text(r))

    def test_a_question_that_was_not_answered_says_so(self):
        r = run({"questions": Q}, {"answers": {}})
        self.assertIn('="(no answer)"', self.text(r))

    def test_when_the_user_skips_it_the_model_is_told_to_carry_on(self):
        r = run({"questions": Q}, {"answers": None})
        self.assertFalse(r["isError"])
        self.assertIn("did not answer", self.text(r))

    def test_when_the_request_is_cancelled_it_is_an_error(self):
        r = run({"questions": Q}, "cancelled")
        self.assertTrue(r["isError"])
        self.assertIn("cancelled", self.text(r))

    def test_what_the_model_sends_is_checked(self):
        def bad(**changes):
            q = {**Q[0], **changes}
            return run({"questions": [q]}, {"answers": {}})["isError"]
        self.assertTrue(run({}, None)["isError"])
        self.assertTrue(run({"questions": []}, None)["isError"])
        self.assertTrue(run({"questions": Q * 5}, None)["isError"])
        self.assertTrue(bad(header="a header that is too long"))
        self.assertTrue(bad(header=""))
        self.assertTrue(bad(question=""))
        self.assertTrue(bad(question="x" * 401))
        self.assertTrue(bad(options=[{"label": "only one", "description": "d"}]))
        self.assertTrue(bad(options=[{"label": str(i), "description": "d"} for i in range(5)]))
        self.assertTrue(bad(options=[{"label": "same", "description": "a"}, {"label": "same", "description": "b"}]))
        self.assertTrue(bad(options=[{"label": "a"}, {"label": "b", "description": "d"}]))
        self.assertTrue(bad(multiSelect="yes"))
        self.assertTrue(run({"questions": Q + Q}, None)["isError"])                       # the same question twice

    def test_it_needs_no_permission_in_any_mode(self):
        for mode in (None, "plan", "auto"):
            d = permissions.decide("AskUserQuestion", {"questions": Q}, permissions.Policy(cwd=None, mode=mode))
            self.assertEqual(d.kind, "allow", mode)


class WithTheChat(Fixture):
    def answer_question(self, rid, answers):
        req = self.post("/agent/question", {"id": rid, "answers": answers})
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status

    def test_the_page_answers_and_the_model_gets_it(self):
        self.start(tool_call("AskUserQuestion", questions=json.dumps(Q)), DONE)
        seen = []
        status, said, events = self.chat(on_event=lambda e: e["event"] == "question" and (seen.append(e), self.answer_question(e["id"], {"Which library should we use?": ["httpx"]})))
        self.assertEqual(status, 200)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0]["questions"][0]["header"], "Library")
        results = [e for e in events if e["event"] == "result"]
        self.assertIn('"Which library should we use?"="httpx"', results[0]["text"])
        self.assertIn("httpx", self.engine.prompt_text(1))                                # and the model read it

    def test_skipping_it_is_an_answer_too(self):
        self.start(tool_call("AskUserQuestion", questions=json.dumps(Q)), DONE)
        status, _, events = self.chat(on_event=lambda e: e["event"] == "question" and self.answer_question(e["id"], None))
        self.assertEqual(status, 200)
        self.assertIn("did not answer", [e for e in events if e["event"] == "result"][0]["text"])

    def test_the_routes_check_what_they_get_and_keep_the_two_kinds_of_question_apart(self):
        self.start(tool_call("AskUserQuestion", questions=json.dumps(Q)), DONE)
        probes = []

        def on_event(e):
            if e["event"] != "question":
                return
            rid = e["id"]
            for body in ({"id": rid, "answers": "x"}, {"id": rid, "answers": {"q": "not a list"}}, {"id": rid, "answers": {"q": []}}, {"id": rid, "answers": {"q": ["a"] * 9}}, {"answers": {}}, [], {"id": rid, "answers": {"a": ["1"], "b": ["2"], "c": ["3"], "d": ["4"], "e": ["5"]}}):
                with self.assertRaises(urllib.error.HTTPError) as cm:
                    urllib.request.urlopen(self.post("/agent/question", body), timeout=10)
                probes.append(cm.exception.code)
            with self.assertRaises(urllib.error.HTTPError) as cm:                           # a question is not a permission
                urllib.request.urlopen(self.post("/agent/permission", {"id": rid, "decision": "allow"}), timeout=10)
            probes.append(cm.exception.code)
            with self.assertRaises(urllib.error.HTTPError) as cm:                           # and an id that is not waiting
                urllib.request.urlopen(self.post("/agent/question", {"id": "nope", "answers": None}), timeout=10)
            probes.append(cm.exception.code)
            self.answer_question(rid, None)

        self.chat(on_event=on_event)
        self.assertEqual(probes, [400] * 7 + [404, 404])

    def test_a_foreign_origin_is_refused(self):
        self.start(DONE)
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(self.post("/agent/question", {"id": "x", "answers": None}, {"Origin": "http://evil.example"}), timeout=10)
        self.assertEqual(cm.exception.code, 403)

    def test_the_tool_is_offered_and_described_to_the_model(self):
        self.start(DONE)
        self.chat()
        p = self.engine.prompt_text(0)
        self.assertIn("AskUserQuestion", p)

    def test_a_permission_question_cannot_be_answered_as_a_question(self):
        self.start(DONE)
        from serve.agent_run import Broker
        b = Broker()
        b.open("p1")
        self.assertFalse(b.respond("p1", None))
        self.assertTrue(b.answer("p1", "allow"))
        b.open("q1", "question")
        self.assertFalse(b.answer("q1", "allow"))
        self.assertTrue(b.respond("q1", {"a": ["b"]}))
        self.assertFalse(b.respond("q1", None))                                           # once


if __name__ == "__main__":
    unittest.main()
