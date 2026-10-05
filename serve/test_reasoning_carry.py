"""#200: a reply cut at max_tokens while still thinking leaves its thinking for the next turn (a client like
deepseek-harness drops the cut assistant message, so the model planned 80K characters again from zero)."""
import json
import unittest
import urllib.request
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.reasoning_carry import Carry, key_of, mode_of
from serve.server import ByteTokenizer, MockEngine, Service, serve

ROOT = Path(__file__).resolve().parents[1]

BASE = [{"role": "system", "content": "sys"}, {"role": "user", "content": "build a pagoda"}]
CONT = {"role": "user", "content": "ทำต่อ"}


class CarryTest(unittest.TestCase):
    def carry(self, messages=BASE, reasoning="I plan the roof."):
        c = Carry()
        c.settle(key_of(messages), (messages, [1, 2, 3], [7, 8, 9], reasoning))
        return c

    def test_the_next_user_message_gets_the_thinking_before_it(self):
        out, n = self.carry().history(BASE + [CONT])
        self.assertEqual(out[:2], BASE)
        self.assertEqual(out[2], {"role": "assistant", "content": "", "reasoning_content": "I plan the roof."})
        self.assertEqual(out[3], CONT)
        self.assertEqual(n, 3)                          # the written tokens it carries

    def test_an_empty_assistant_message_is_filled(self):
        for empty in ({"role": "assistant", "content": ""}, {"role": "assistant", "content": None},
                      {"role": "assistant", "content": []}):
            out, _ = self.carry().history(BASE + [empty, CONT])
            self.assertEqual(len(out), 4)
            self.assertEqual(out[2]["reasoning_content"], "I plan the roof.")

    def test_a_reply_the_client_kept_is_not_touched(self):
        for kept in ({"role": "assistant", "content": "", "reasoning_content": "mine"},
                     {"role": "assistant", "content": "the answer"},
                     {"role": "assistant", "content": "", "tool_calls": [{"name": "Read"}]}):
            self.assertIsNone(self.carry().history(BASE + [kept, CONT]))

    def test_another_conversation_is_not_touched(self):
        c = self.carry()
        self.assertIsNone(c.history([BASE[0], {"role": "user", "content": "other"}, CONT]))
        self.assertIsNone(c.history(BASE))                         # nothing new after the cut reply
        self.assertIsNone(c.history(BASE + [CONT, CONT]))          # more than one new message

    def test_resume_continues_the_prompt_and_what_was_written(self):
        self.assertEqual(self.carry().resume(BASE + [CONT]), [1, 2, 3, 7, 8, 9])
        self.assertIsNone(self.carry().resume(BASE + [{"role": "assistant", "content": "x"}, CONT]))

    def test_used_once(self):
        c = self.carry()
        self.assertIsNotNone(c.history(BASE + [CONT]))
        self.assertIsNone(c.history(BASE + [CONT]))

    def test_a_chain_of_cuts_keeps_every_thinking(self):
        # review of #200: cut, continue, cut again, continue - the client drops both cut replies; the second carry
        # must still hold the first thinking (the messages the second cut was rendered from)
        c = self.carry()
        rendered, _ = c.history(BASE + [CONT])                  # what the second request rendered
        c.settle(key_of(BASE + [CONT]), (rendered, [1, 2, 3, 4], [5, 6], "then the walls."))
        out, _ = c.history(BASE + [CONT, {"role": "user", "content": "again"}])
        self.assertEqual(out[2]["reasoning_content"], "I plan the roof.")
        self.assertEqual(out[4]["reasoning_content"], "then the walls.")
        self.assertEqual(out[5]["content"], "again")

    def test_a_reply_that_ends_otherwise_forgets_its_conversation(self):
        c = self.carry()
        c.settle(key_of(BASE))                          # the same request, answered in full this time
        self.assertIsNone(c.history(BASE + [CONT]))

    def test_conversations_are_kept_apart(self):
        other = [BASE[0], {"role": "user", "content": "a subagent's task"}]
        c = self.carry()
        c.settle(key_of(other), (other, [4], [5], "the subagent's plan"))
        self.assertEqual(c.history(BASE + [CONT])[0][2]["reasoning_content"], "I plan the roof.")
        self.assertEqual(c.history(other + [CONT])[0][2]["reasoning_content"], "the subagent's plan")

    def test_modes(self):
        self.assertEqual(mode_of({}, "off"), "off")
        self.assertEqual(mode_of({}, "history"), "history")
        self.assertEqual(mode_of({"reasoning_carry": "resume"}, "history"), "resume")
        with self.assertRaises(ValueError):
            mode_of({"reasoning_carry": "sometimes"}, "off")


class PromptLog(MockEngine):
    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts = getattr(self, "prompts", []) + [list(ids)]
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


THINK = "".join("abcdefghijklmnopqrstuvwxyz "[(i * 7919 + i * i * 104729) % 27] for i in range(600))


class OverHttp(unittest.TestCase):
    """The deepseek-harness shape: a reply cut at max_tokens inside its thinking, then the same conversation with the
    cut reply dropped and one new user message."""

    def run_pair(self, mode, first_script, first_max=60, second_messages=None):
        tok = ByteTokenizer()
        eng = PromptLog(tok, [first_script, "the answer"])
        svc = Service(eng, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.reasoning_carry = mode
        httpd = serve(svc, port=0)
        url = f"http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions"

        def post(messages, max_tokens):
            body = {"model": "m", "max_tokens": max_tokens, "messages": messages}
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        try:
            first = post([{"role": "user", "content": "build a pagoda"}], first_max)
            second = post(second_messages or [{"role": "user", "content": "build a pagoda"},
                                              {"role": "user", "content": "continue"}], 4000)
        finally:
            httpd.shutdown()
            httpd.server_close()
        return tok, eng, first, second

    def test_history_puts_the_cut_thinking_back(self):
        tok, eng, first, second = self.run_pair("history", THINK)
        self.assertEqual(first["choices"][0]["finish_reason"], "length")
        cut = first["choices"][0]["message"]["reasoning_content"]
        self.assertTrue(cut)
        self.assertIn(cut.strip(), tok.decode(eng.prompts[1]))
        self.assertIn("continue", tok.decode(eng.prompts[1]))

    def test_resume_generates_on_from_what_was_written(self):
        tok, eng, first, second = self.run_pair("resume", THINK)
        written = tok.encode(first["choices"][0]["message"]["reasoning_content"], parse_special=True)
        self.assertEqual(eng.prompts[1][:len(eng.prompts[0])], eng.prompts[0])
        self.assertEqual(len(eng.prompts[1]), len(eng.prompts[0]) + 60)       # every token the cut reply wrote
        self.assertNotIn("continue", tok.decode(eng.prompts[1]))

    def test_a_finished_thinking_is_not_carried(self):
        tok, eng, first, second = self.run_pair("history", "short thought</think>\n\n" + THINK)
        self.assertEqual(first["choices"][0]["finish_reason"], "length")
        self.assertNotIn("short thought", tok.decode(eng.prompts[1]))

    def test_off_carries_nothing(self):
        tok, eng, first, second = self.run_pair("off", THINK)
        self.assertNotIn(first["choices"][0]["message"]["reasoning_content"].strip(), tok.decode(eng.prompts[1]))


class Chain(unittest.TestCase):
    """Review of #200: cut, continue, cut again, continue - the deepseek-harness loop - and a cut request answered in
    full, over both dialects."""

    def server(self, mode, scripts):
        tok = ByteTokenizer()
        eng = PromptLog(tok, scripts)
        svc = Service(eng, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.reasoning_carry = mode
        httpd = serve(svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        return tok, eng, f"http://127.0.0.1:{httpd.server_address[1]}"

    @staticmethod
    def post(url, body):
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())

    def test_a_chain_of_cuts_keeps_every_thinking(self):
        second = THINK[::-1]
        tok, eng, url = self.server("history", [THINK, second, "the answer"])
        msgs = [{"role": "user", "content": "build a pagoda"}]
        r1 = self.post(url + "/v1/chat/completions", {"model": "m", "max_tokens": 60, "messages": msgs})
        msgs = msgs + [{"role": "user", "content": "continue"}]
        r2 = self.post(url + "/v1/chat/completions", {"model": "m", "max_tokens": 60, "messages": msgs})
        msgs = msgs + [{"role": "user", "content": "continue again"}]
        self.post(url + "/v1/chat/completions", {"model": "m", "max_tokens": 4000, "messages": msgs})
        third = tok.decode(eng.prompts[2])
        self.assertIn(r1["choices"][0]["message"]["reasoning_content"].strip(), third)
        self.assertIn(r2["choices"][0]["message"]["reasoning_content"].strip(), third)
        self.assertIn("continue again", third)

    def test_a_cut_request_answered_in_full_later_is_forgotten(self):
        tok, eng, url = self.server("history", [THINK, "short thought</think>\n\nthe answer", "after"])
        msgs = [{"role": "user", "content": "build a pagoda"}]
        r1 = self.post(url + "/v1/chat/completions", {"model": "m", "max_tokens": 60, "messages": msgs})
        self.post(url + "/v1/chat/completions", {"model": "m", "max_tokens": 4000, "messages": msgs})   # a retry
        self.post(url + "/v1/chat/completions", {"model": "m", "max_tokens": 4000,
                                                 "messages": msgs + [{"role": "user", "content": "continue"}]})
        self.assertNotIn(r1["choices"][0]["message"]["reasoning_content"].strip(), tok.decode(eng.prompts[2]))

    def test_the_anthropic_dialect_carries_too(self):
        tok, eng, url = self.server("history", [THINK, "the answer"])
        body = {"model": "m", "max_tokens": 60, "thinking": {"type": "enabled", "budget_tokens": 100000},
                "messages": [{"role": "user", "content": "build a pagoda"}]}
        r1 = self.post(url + "/v1/messages", body)
        cut = "".join(b.get("thinking", "") for b in r1["content"] if b["type"] == "thinking")
        self.assertTrue(cut)
        body = {**body, "max_tokens": 4000,
                "messages": body["messages"] + [{"role": "user", "content": "continue"}]}
        self.post(url + "/v1/messages", body)
        self.assertIn(cut.strip()[:40], tok.decode(eng.prompts[1]))

if __name__ == "__main__":
    unittest.main()
