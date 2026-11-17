"""#331: the history the web app sends back renders, token for token, like what the server gave the engine while the
answer ran - so the next message continues the engine's cache instead of reading the conversation again.

The shared fixture serve/ui/src/lib/history-thinking.fixture.json holds one answer with a tool round: `api` is what the
page's `apiMessages` sends (checked against the page's code by serve/ui/src/lib/lib.test.ts), `server` is what
`run_with_mcp` appended for the engine.  The page's form goes through the server's own request parsing
(frontend.openai_to_messages) and the shipped chat template."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from serve.frontend import ChatTemplate, openai_to_messages  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service  # noqa: E402

FIXTURE = ROOT / "serve/ui/src/lib/history-thinking.fixture.json"
HEAD = [{"role": "system", "content": "You are an agent."}, {"role": "user", "content": "What does a.txt say?"}]
NEXT = {"role": "user", "content": "Thanks. And b.txt?"}
TOOLS = [{"type": "function", "function": {"name": "Read", "description": "read a file",
                                           "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}}}}]


def service() -> Service:
    tok = ByteTokenizer()
    return Service(MockEngine(tok, "ok", max_context=16384), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))


class HistoryKeepsTheThinking(unittest.TestCase):
    def setUp(self):
        self.fx = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.svc = service()

    def page(self, api):
        messages, _tools, _kw = openai_to_messages({"messages": HEAD + api + [NEXT], "tools": TOOLS})
        return self.svc.encode_prompt(messages, TOOLS, {})

    def test_the_pages_history_renders_as_the_engine_saw_it(self):
        server = self.svc.encode_prompt(HEAD + self.fx["server"] + [NEXT], TOOLS, {})
        self.assertEqual(self.page(self.fx["api"]), server)

    def test_without_the_thinking_it_parts_at_the_first_answer(self):
        """The bug: the history without reasoning_content (as before #331) parts from the engine's tokens right at the
        first answer, so everything after it was read again."""
        server = self.svc.encode_prompt(HEAD + self.fx["server"] + [NEXT], TOOLS, {})
        head = self.svc.encode_prompt(HEAD, TOOLS, {"add_generation_prompt": False})
        old = [{k: v for k, v in m.items() if k != "reasoning_content"} for m in self.fx["api"]]
        ids = self.page(old)
        parted = next(i for i in range(min(len(ids), len(server))) if ids[i] != server[i])
        self.assertLess(parted, len(head) + 40)


if __name__ == "__main__":
    unittest.main()
