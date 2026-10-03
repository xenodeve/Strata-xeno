"""What the user sends while the agent works (serve/runs.py steer, serve/server.py run_with_mcp): the agent reads it at its next step - after the tools of the round that is running, or, when that round
was its last, in the same run instead of ending - so a message sent while the model thinks is not left waiting for the whole answer.  The page shows it as the user's message at that point."""
from __future__ import annotations

import json
import threading
import time
import unittest
import urllib.error
import urllib.request

from serve.test_agent_chat import DONE, Fixture
from serve.test_harness import tool_call

RID = "run-steer-0001"


class Steering(Fixture):
    def steer(self, text, rid=RID, headers=None):
        with urllib.request.urlopen(self.post("/agent/steer", {"id": rid, "text": text}, headers), timeout=10) as r:
            return json.loads(r.read())

    def run_chat(self, on_event, rid=RID):
        """Run the chat request to its end, giving each strata_mcp event to `on_event`; returns (the events, the text said)."""
        events, said = [], []
        req = self.post("/v1/chat/completions", self.body(agent={"run": rid}))
        with urllib.request.urlopen(req, timeout=60) as r:
            for raw in r:
                ln = raw.decode().strip()
                if not ln.startswith("data: {"):
                    continue
                c = json.loads(ln[6:])
                if "strata_mcp" in c:
                    events.append(c["strata_mcp"])
                    on_event(c["strata_mcp"])
                for ch in c.get("choices") or []:
                    said.append(ch["delta"].get("content") or "")
        return events, "".join(said)

    def answer(self, rid, decision="allow"):
        with urllib.request.urlopen(self.post("/agent/permission", {"id": rid, "decision": decision}), timeout=10):
            pass

    def test_a_message_sent_while_a_tool_waits_is_read_at_the_next_step_after_the_results(self):
        outside = str(self.outside / "b.txt")
        self.start(tool_call("Read", file_path=outside), "</think>\n\nI also checked what you asked.")
        sent = []

        def on_event(e):
            if e["event"] == "permission" and not sent:
                sent.append(self.steer("also check the other file please"))
                self.answer(e["id"])

        events, said = self.run_chat(on_event)
        self.assertEqual(sent, [{"ok": True, "found": True}])
        steered = [e for e in events if e["event"] == "steered"]
        self.assertEqual([e["text"] for e in steered], ["also check the other file please"])
        order = [e["event"] for e in events]
        self.assertGreater(order.index("steered"), order.index("result"))                          # after the tool's result, as the next step begins
        self.assertTrue(said.strip().endswith("I also checked what you asked."))
        p = self.engine.prompt_text(1)                                                             # the model's second request holds the tool's result and then the user's message
        self.assertIn("also check the other file please", p)
        self.assertLess(p.index("secret-ish"), p.index("also check the other file please"))

    def test_a_message_sent_while_the_last_round_is_written_goes_on_in_the_same_run(self):
        self.start(DONE, "</think>\n\nSecond reply to the new message.")
        self.engine.delay = 0.03                                                                   # the first answer takes a moment to write
        sent = []

        def on_event(e):
            pass

        def steer_soon():
            time.sleep(0.3)
            sent.append(self.steer("one more thing"))

        threading.Thread(target=steer_soon, daemon=True).start()
        events, said = self.run_chat(on_event)
        self.assertEqual(sent, [{"ok": True, "found": True}])
        self.assertEqual([e["text"] for e in events if e["event"] == "steered"], ["one more thing"])
        self.assertIn("Done.", said)
        self.assertTrue(said.strip().endswith("Second reply to the new message."))                 # one request, two answers
        self.assertEqual(len(self.engine.prompts), 2)
        p = self.engine.prompt_text(1)
        self.assertIn("one more thing", p)
        self.assertLess(p.index("Done."), p.index("one more thing"))                               # the first answer is part of what the model reads, then the message

    def test_several_messages_are_read_in_the_order_they_were_sent(self):
        outside = str(self.outside / "b.txt")
        self.start(tool_call("Read", file_path=outside), "</think>\n\nok")

        def on_event(e):
            if e["event"] == "permission":
                self.steer("first thing")
                self.steer("second thing")
                self.answer(e["id"])

        events, _ = self.run_chat(on_event)
        self.assertEqual([e["text"] for e in events if e["event"] == "steered"], ["first thing", "second thing"])
        p = self.engine.prompt_text(1)
        self.assertLess(p.index("first thing"), p.index("second thing"))

    def test_a_run_that_is_over_or_not_there_does_not_take_a_message(self):
        self.start(DONE)
        self.assertEqual(self.steer("anybody?", rid="run-none-0001"), {"ok": True, "found": False})
        self.run_chat(lambda e: None)
        deadline = time.time() + 5
        while not self.svc.runs.get(RID).done and time.time() < deadline:
            time.sleep(0.05)
        self.assertEqual(self.steer("too late"), {"ok": True, "found": False})

    def test_a_page_that_reads_the_run_again_sees_where_the_message_came_in(self):
        outside = str(self.outside / "b.txt")
        self.start(tool_call("Read", file_path=outside), "</think>\n\nok")
        self.run_chat(lambda e: e["event"] == "permission" and (self.steer("hello again"), self.answer(e["id"])))
        with urllib.request.urlopen(self.base_url + f"/agent/run?id={RID}&from=0", timeout=10) as r:
            kinds = [json.loads(ln[6:]).get("strata_mcp", {}).get("event") for ln in (raw.decode().strip() for raw in r) if ln.startswith("data: {")]
        self.assertIn("steered", kinds)

    def test_the_route_checks_what_it_gets_and_who_asks(self):
        self.start(DONE)
        for body in ({"id": 5, "text": "x"}, {"id": RID}, {"id": RID, "text": ""}, {"id": RID, "text": "   "}, {"id": RID, "text": "x" * 20_001}, [], "x"):
            with self.assertRaises(urllib.error.HTTPError) as cm:
                urllib.request.urlopen(self.post("/agent/steer", body), timeout=10)
            self.assertEqual(cm.exception.code, 400, str(body)[:40])
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(self.post("/agent/steer", {"id": RID, "text": "x"}, {"Origin": "http://evil.example"}), timeout=10)
        self.assertEqual(cm.exception.code, 403)

    def test_a_run_without_a_message_is_unchanged(self):
        self.start(tool_call("Read", file_path="a.txt"), DONE)
        events, said = self.run_chat(lambda e: None)
        self.assertEqual([e for e in events if e["event"] == "steered"], [])
        self.assertTrue(said.strip().endswith("Done."))
        self.assertEqual(len(self.engine.prompts), 2)


if __name__ == "__main__":
    unittest.main()
