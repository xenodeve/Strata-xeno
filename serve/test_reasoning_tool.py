"""serve/test_reasoning_tool.py - `run_python` while thinking and in the reply (serve/reasoning_tool.py, #288), without a GPU or a pack.

    python -B -m unittest serve.test_reasoning_tool -v

The engine is a stand-in like serve/test_mip_adapter.py's: a small program that speaks the engine's protocol, writes down every GEN line it
gets and plays a script of tokens back. Its text is what Gyro-M wrote on the A100 with the route on, in miniature: a call closed while
thinking, already answered by the engine (`<tool_response>` in the thinking), and a call in the reply, which only the server can answer.
Model-written code runs in the kernel through `FakeBox`, as in serve/test_mip_adapter.py; nothing between the server and the kernel is mocked.
"""
from __future__ import annotations

import ast
import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import reasoning_tool as RT  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.sandbox import exec_bridge  # noqa: E402
from serve.sandbox.exec_bridge import ExecBridge  # noqa: E402
from serve.server import ByteTokenizer, Service, StrataEngine, run_with_mcp, sandbox_bridge, serve  # noqa: E402
from serve.test_mip_adapter import FakeBox, sbx, sbxr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TOK = ByteTokenizer()
END = TOK.encode("<|im_end|>", parse_special=True)

CALL = "<tool_call>\n<function=run_python>\n<parameter=code>\nprint(6 * 7)\n</parameter>\n</function>\n</tool_call>"
# The engine's route: the call closed while thinking, and its answer written into the thinking before the model goes on.
IN_THINKING = f"Let me check.\n{CALL}\n<tool_response>\n42\n</tool_response>\nSo it is 42.</think>\n\nIt is 42."
IN_REPLY = f"I will run it.</think>\n\n{CALL}"
AFTER_REPLY = "It printed 42.</think>\n\nThe answer is 42."

ENGINE = r'''
import json, sys
from pathlib import Path
log = Path(sys.argv[sys.argv.index('--log') + 1])
scripts = json.loads(Path(sys.argv[sys.argv.index('--scripts') + 1]).read_text(encoding='utf-8'))
print('READY 65536 stop', flush=True)
n = 0
for line in sys.stdin:
    if line.startswith('QUIT'):
        break
    if line.startswith('GEN'):
        head, _, ids = line.strip().rpartition(' ')
        with log.open('a', encoding='utf-8') as f:
            f.write(json.dumps({'head': head, 'ids': [int(x) for x in ids.split(',') if x]}) + '\n')
        out = scripts[min(n, len(scripts) - 1)]
        n += 1
        for t in out:
            print(f'T {t}', flush=True)
        print(f'DONE {len(out)} {len(ids.split(","))} 0 1 stop', flush=True)
'''

BASH = {"name": "Bash", "description": "Run a shell command", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}}}


def script(text: str) -> list[int]:
    return TOK.encode(text) + END


class Declaration(unittest.TestCase):
    def test_the_words_are_the_benchs(self):
        """The arm that scored 8 of 10 read the bench's TOOL_HINT: the server declares the same function, word for word."""
        tree = ast.parse((ROOT / "tools" / "igc" / "external_bench.py").read_text(encoding="utf-8"))
        hint = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "TOOL_HINT")
        declared = json.loads(hint.split("<tools>\n", 1)[1].split("\n</tools>", 1)[0])
        self.assertEqual(declared, {"type": "function", "function": RT.TOOL})
        bench = (ROOT / "tools" / "igc" / "external_bench.py").read_text(encoding="utf-8")
        for name, value in (("TOOL_ROUNDS", RT.RUNS), ("TOOL_AFTER", RT.AFTER), ("TOOL_OUTPUT", RT.SHOWN)):
            self.assertIn(f"{name}={value}\n", bench)
        self.assertIn(f"TOOL_SPENT={RT.SPENT!r}", bench)

    def test_it_goes_after_the_clients_tools_and_never_over_its_own(self):
        bash = {"name": "Bash", "description": "", "parameters": {}}
        self.assertEqual(RT.declared(None), [RT.TOOL])
        self.assertEqual(RT.declared([bash]), [bash, RT.TOOL])
        own = {"name": "run_python", "description": "the client's", "parameters": {}}
        self.assertEqual(RT.declared([own]), [own])

    def test_the_template_renders_it_in_the_tools_block(self):
        tpl = ChatTemplate(ROOT / "serve" / "chat_template.jinja")
        chat = [{"role": "user", "content": "hi"}]
        self.assertNotIn("<tools>", tpl.render(chat))
        text = tpl.render(chat, tools=RT.declared(None))
        tools = text.split("<tools>", 1)[1].split("</tools>", 1)[0]
        self.assertIn("run_python", tools)
        self.assertIn(RT.DESCRIPTION, tools)

    def test_the_key(self):
        self.assertFalse(RT.on({}))
        self.assertFalse(RT.on({"reasoning_tool": False}))
        self.assertTrue(RT.on({"reasoning_tool": True}))
        with self.assertRaises(ValueError):
            RT.on({"reasoning_tool": "yes"})


class Startup(unittest.TestCase):
    """What the server starts the engine with, or why it does not start."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / "bundle.bin").write_bytes(b"x")
        self.built = self.root / "strata.exe"
        self.built.write_bytes(b"..--igc-sandbox-tool..")
        self.old = self.root / "old.exe"
        self.old.write_bytes(b"..--igc-sandbox..")
        self.args = ["--pack", "p", RT.BUNDLE, "bundle.bin"]
        self.cfg = {"cwd": str(self.root), "reasoning_tool": True}

    def test_a_profile_that_lacks_what_the_route_needs_is_refused(self):
        with self.assertRaisesRegex(ValueError, "learned bundle"):
            RT.startup(self.cfg, ["--pack", "p"], str(self.built))
        with self.assertRaisesRegex(ValueError, "not there"):
            RT.startup(self.cfg, ["--pack", "p", RT.BUNDLE, "gone.bin"], str(self.built))
        with self.assertRaisesRegex(ValueError, "built without"):
            RT.startup(self.cfg, self.args, str(self.old))
        with self.assertRaisesRegex(ValueError, "replicas"):
            RT.startup(self.cfg, self.args, str(self.built), [[0], [1]])

    def test_without_approval_the_engines_interpreter_answers(self):
        with mock.patch.object(exec_bridge, "throwaway", return_value=False):
            extra, box, why = RT.startup(self.cfg, self.args, str(self.built))
        self.assertEqual(extra, list(RT.ENGINE_FLAGS))
        self.assertFalse(box)
        self.assertIn("D1", why)

    def test_flags_the_profile_has_are_not_added_twice(self):
        with mock.patch.object(exec_bridge, "throwaway", return_value=False):
            extra, _, _ = RT.startup(self.cfg, self.args + ["--igc-inline-exec", "--igc-sandbox"], str(self.built))
        self.assertEqual(extra, ["--igc-result-bridge", "--igc-sandbox-tool"])

    def test_approved_with_a_box_real_python_answers_in_the_box(self):
        cfg = {**self.cfg, "sandbox": {"real_python": True}}
        backend = "appcontainer" if os.name == "nt" else "bubblewrap"
        with mock.patch.object(exec_bridge, "throwaway", return_value=False), \
                mock.patch("serve.sandbox.isolation.Isolation.report", return_value={backend: (True, "")}):
            extra, box, _ = RT.startup(cfg, self.args, str(self.built))
        self.assertEqual(extra[-1], "--igc-sandbox-exec")
        self.assertTrue(box)
        with mock.patch.object(exec_bridge, "throwaway", return_value=False), \
                mock.patch("serve.sandbox.isolation.Isolation.report", return_value={backend: (False, "no AppContainer")}):
            extra, box, why = RT.startup(cfg, self.args, str(self.built))
        self.assertNotIn("--igc-sandbox-exec", extra)
        self.assertIn("no isolation box", why)

    def test_a_throwaway_vm_runs_bare_and_batch_slots_run_no_kernel(self):
        with mock.patch.object(exec_bridge, "throwaway", return_value=True):
            extra, box, _ = RT.startup(self.cfg, self.args, str(self.built))
            self.assertIn("--igc-sandbox-exec", extra)
            self.assertFalse(box)
            extra, _, why = RT.startup(self.cfg, self.args + ["--batch", "2"], str(self.built))
            self.assertNotIn("--igc-sandbox-exec", extra)
            self.assertIn("--batch", why)

    def test_the_bridge_takes_the_box(self):
        self.assertIsNone(sandbox_bridge(["--igc-sandbox"], None, FakeBox))
        self.assertIs(sandbox_bridge(["--igc-sandbox-exec"], None, FakeBox).box_factory, FakeBox)


class Kernels(unittest.TestCase):
    """Whose names a call in the reply reads: its own request's, never another's."""

    def bridge(self):
        bridge = ExecBridge(FakeBox)
        self.addCleanup(bridge.close)
        return bridge

    def test_a_call_in_the_reply_reads_the_names_its_thinking_made_in_the_kernel(self):
        bridge, mine = self.bridge(), object()
        bridge.begin(None, mine)
        self.assertTrue(sbxr(bridge.handle(sbx("x = 41\n", fresh=True)))[0])
        bridge.stop()                                            # the reply's call comes after the engine's turn
        self.assertEqual(bridge.run_reply(mine, "print(x + 1)\n"), (True, "42\n"))
        self.assertEqual(bridge.run_reply(mine, "y = x\n"), (True, ""))
        self.assertEqual(bridge.run_reply(mine, "print(y)\n"), (True, "41\n"))     # and its own, between its calls

    def test_another_request_never_reads_them(self):
        bridge, mine, other = self.bridge(), object(), object()
        bridge.begin(None, mine)
        bridge.handle(sbx("x = 41\n", fresh=True))
        bridge.stop()
        ok, text = bridge.run_reply(other, "print(x)\n")
        self.assertFalse(ok)
        self.assertIn("NameError", text)
        bridge.run_reply(mine, "z = 1\n")                        # the kernel is the other request's now: this one gets a fresh one
        bridge.begin(None, other)                                # and the engine's next turn starts its own, as it always did
        ok, text = sbxr(bridge.handle(sbx("print(z)\n", fresh=True)))
        self.assertFalse(ok)
        self.assertIn("NameError", text)

    def test_while_another_request_holds_the_engine_a_call_gets_a_kernel_of_its_own(self):
        bridge, mine, other = self.bridge(), object(), object()
        bridge.begin(None, other)
        bridge.handle(sbx("secret = 1\n", fresh=True))           # the other request is decoding: its kernel is not to be touched
        self.assertEqual(bridge.run_reply(mine, "a = 2\nprint(a)\n"), (True, "2\n"))
        self.assertEqual(bridge.run_reply(mine, "print(a + 1)\n"), (True, "3\n"))
        ok, text = bridge.run_reply(mine, "print(secret)\n")
        self.assertFalse(ok)
        self.assertEqual(sbxr(bridge.handle(sbx("print(secret)\n"))), (True, "1\n"))
        bridge.end(mine)
        self.assertEqual(bridge.side, {})

    def test_the_end_of_a_request_ends_its_kernel(self):
        bridge, mine = self.bridge(), object()
        bridge.begin(None, mine)
        bridge.handle(sbx("x = 1\n", fresh=True))
        bridge.stop()
        bridge.end(mine)
        self.assertIsNone(bridge.kernel)

    def test_a_stop_ends_the_run(self):
        bridge, mine = self.bridge(), object()
        cancel = threading.Event()
        threading.Timer(0.5, cancel.set).start()
        ok, text = bridge.run_reply(mine, "import time\ntime.sleep(30)\n", cancel=cancel)
        self.assertFalse(ok)
        self.assertIn("ended the sandbox", text)
        self.assertEqual(bridge.run_reply(mine, "print(1)\n"), (True, "1\n"))     # the next call starts a kernel again


class Turns(unittest.TestCase):
    """What the model reads back from a call in its reply."""

    def test_no_real_python_says_why(self):
        r = RT.Turn(None).run({"code": "print(1)"})
        self.assertFalse(r["ok"])
        self.assertIn("design D1", r["text"])
        self.assertIn("while thinking", r["text"])

    def test_no_box_says_why(self):
        with mock.patch.object(exec_bridge, "throwaway", return_value=False):
            r = RT.Turn(ExecBridge()).run({"code": "print(1)"})
        self.assertFalse(r["ok"])
        self.assertIn("no isolation box", r["text"])

    def test_no_code_says_what_is_missing(self):
        self.assertEqual(RT.Turn(None).run({})["text"], RT.NO_CODE)

    def test_past_the_runs_a_call_is_told_to_answer(self):
        turn = RT.Turn(None)
        for _ in range(RT.RUNS):
            turn.run({"code": "print(1)"})
        self.assertFalse(turn.over())
        self.assertEqual(turn.run({"code": "print(1)"})["text"], RT.SPENT)
        for _ in range(RT.AFTER - 1):
            turn.run({"code": "print(1)"})
        self.assertTrue(turn.over())

    def test_output_is_cut_where_the_bench_cut_it(self):
        self.assertEqual(RT.shown(""), RT.NOTHING)
        long = RT.shown("x" * (RT.SHOWN + 10))
        self.assertTrue(long.endswith("\n[output cut]"))
        self.assertEqual(len(long), RT.SHOWN + len("\n[output cut]"))


class Served(unittest.TestCase):
    """The whole route but the model, through the HTTP API: the stand-in engine, StrataEngine, Service, run_with_mcp and the kernel."""

    def start(self, scripts, key=True, bridge=True):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.log = root / "gen.jsonl"
        (root / "scripts.json").write_text(json.dumps(scripts), encoding="utf-8")
        py = root / "native.py"
        py.write_text(ENGINE, encoding="utf-8")
        exe = py
        if os.name == "nt":
            exe = root / "native.cmd"
            exe.write_text(f'@"{sys.executable}" "{py}" %*\r\n')
        else:
            py.write_text("#!" + sys.executable + "\n" + ENGINE)
            py.chmod(0o700)
        engine = StrataEngine(str(exe), ["--log", str(self.log), "--scripts", str(root / "scripts.json")])
        self.addCleanup(engine.close)
        if bridge:
            engine.sandbox_exec = ExecBridge(FakeBox)
            self.addCleanup(engine.sandbox_exec.close)
        engine.reasoning_tool = key
        self.svc = Service(engine, TOK, ChatTemplate(ROOT / "serve" / "chat_template.jinja"))
        self.svc.reasoning_tool = key
        httpd = serve(self.svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        self.base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())

    def gens(self):
        lines = [json.loads(x) for x in self.log.read_text(encoding="utf-8").splitlines()]
        return [(g["head"], TOK.decode(g["ids"])) for g in lines]

    def messages(self, **more):
        return self.post("/v1/messages", {"model": "m", "max_tokens": 4000, "messages": [{"role": "user", "content": "what is 6*7?"}],
                                          "tools": [BASH], **more})

    def test_claude_codes_tools_and_run_python_after_them(self):
        self.start([script("ok</think>\n\nok")])
        self.messages()
        (head, prompt), = self.gens()
        self.assertNotIn("igc=0", head)                         # on the route: the engine runs it with MIP
        tools = prompt.split("<tools>", 1)[1].split("</tools>", 1)[0]
        self.assertLess(tools.index('"Bash"'), tools.index('"run_python"'))
        self.assertIn(RT.DESCRIPTION, tools)

    def test_with_no_tools_of_the_clients_it_is_declared_alone(self):
        self.start([script("ok</think>\n\nok")])
        self.post("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        (head, prompt), = self.gens()
        self.assertIn(RT.DESCRIPTION, prompt.split("<tools>", 1)[1])
        self.assertNotIn("igc=0", head)

    def test_with_the_key_off_nothing_changes(self):
        self.start([script(IN_REPLY)], key=False)
        body = self.post("/v1/messages", {"model": "m", "max_tokens": 4000, "messages": [{"role": "user", "content": "x"}],
                                          "tools": [BASH, {"name": "run_python", "description": "the client's",
                                                           "input_schema": {"type": "object", "properties": {"code": {"type": "string"}}}}]})
        (head, prompt), = self.gens()
        self.assertNotIn("igc=", head)
        self.assertNotIn(RT.DESCRIPTION, prompt)
        self.assertEqual([b["name"] for b in body["content"] if b["type"] == "tool_use"], ["run_python"])    # the client's own, to the client
        self.assertEqual(body["stop_reason"], "tool_use")

    def test_a_request_off_the_route_runs_with_no_mip(self):
        self.start([script("ok</think>\n\nok")])
        self.messages(tool_choice={"type": "none"})
        (head, prompt), = self.gens()
        self.assertIn(" igc=0", head)
        self.assertNotIn(RT.DESCRIPTION, prompt)

    def test_a_clients_own_run_python_wins(self):
        self.start([script(IN_REPLY)])
        own = {"name": "run_python", "description": "the client's", "input_schema": {"type": "object", "properties": {"code": {"type": "string"}}}}
        body = self.messages(tools=[BASH, own])
        (head, prompt), = self.gens()
        self.assertIn(" igc=0", head)
        self.assertNotIn(RT.DESCRIPTION, prompt)
        self.assertEqual([b["name"] for b in body["content"] if b["type"] == "tool_use"], ["run_python"])

    def test_a_call_in_the_reply_is_answered_here_and_never_reaches_the_client(self):
        self.start([script(IN_REPLY), script(AFTER_REPLY)])
        body = self.messages()
        self.assertEqual([b["type"] for b in body["content"] if b["type"] == "tool_use"], [])
        self.assertEqual(body["stop_reason"], "end_turn")
        self.assertIn("The answer is 42.", "".join(b.get("text", "") for b in body["content"]))
        (head1, _), (head2, prompt2) = self.gens()
        self.assertNotIn("igc=0", head2)
        self.assertIn("<tool_response>\n42\n</tool_response>", prompt2)        # the kernel ran it, and the model reads what it printed

    def test_the_openai_route_answers_it_too(self):
        self.start([script(IN_REPLY), script(AFTER_REPLY)])
        body = self.post("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}],
                                                  "tools": [{"type": "function", "function": {"name": "Bash", "parameters": {"type": "object"}}}]})
        msg = body["choices"][0]["message"]
        self.assertNotIn("tool_calls", msg)
        self.assertEqual(msg["content"], "The answer is 42.")
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")

    def test_with_no_real_python_the_reply_reads_why(self):
        self.start([script(IN_REPLY), script(AFTER_REPLY)], bridge=False)
        self.messages()
        (_, _), (_, prompt2) = self.gens()
        self.assertIn(RT.NO_PYTHON, prompt2)

    def test_a_call_made_while_thinking_is_left_to_the_engine(self):
        self.start([script(IN_THINKING)])
        body = self.messages()
        self.assertEqual(len(self.gens()), 1)                    # no second round: the engine answered it inside the thinking
        self.assertEqual([b for b in body["content"] if b["type"] == "tool_use"], [])
        thinking = "".join(b.get("thinking", "") for b in body["content"] if b["type"] == "thinking")
        self.assertIn(CALL, thinking)
        self.assertIn("<tool_response>\n42\n</tool_response>", thinking)
        self.assertEqual("".join(b.get("text", "") for b in body["content"] if b["type"] == "text"), "It is 42.")


class Status(unittest.TestCase):
    """The events the page's status reads (serve/ui/src/lib/status.ts), sent as each step begins."""

    def run_events(self, scripts, bridge=None):
        from serve.test_mcp import ScriptedEngine
        engine = ScriptedEngine(TOK, scripts)
        engine.sandbox_exec = bridge
        svc = Service(engine, TOK, ChatTemplate(ROOT / "serve" / "chat_template.jinja"))
        svc.reasoning_tool = True
        rtool = RT.Turn(bridge)
        tools = RT.declared(None)
        messages = [{"role": "user", "content": "hi"}]
        ids, thinking, max_new = svc.prepare(messages, tools, {}, 4000)
        return [x for kind, x in run_with_mcp(svc, None, messages, tools, {}, ids, thinking, max_new, 4000, {}, threading.Event(), set(),
                                              rtool=rtool) if kind == "mcp"]

    def test_a_call_while_thinking_says_when_it_begins_and_ends(self):
        events = self.run_events([IN_THINKING])
        self.assertEqual(events, [{"event": "reasoning_tool", "phase": "thinking", "state": "running"},
                                  {"event": "reasoning_tool", "phase": "thinking", "state": "done"}])

    def test_a_call_in_the_reply_is_a_tool_step_of_its_own(self):
        events = self.run_events([IN_REPLY, AFTER_REPLY])
        self.assertEqual([e["event"] for e in events], ["start", "call", "result"])
        self.assertEqual(events[1]["name"], "run_python")
        self.assertEqual(events[1]["arguments"], {"code": "print(6 * 7)"})
        self.assertFalse(events[2]["ok"])
        self.assertEqual(events[2]["text"], RT.NO_PYTHON)

    def test_a_request_off_the_route_sends_none_of_them(self):
        from serve.test_mcp import ScriptedEngine
        svc = Service(ScriptedEngine(TOK, [IN_THINKING]), TOK, ChatTemplate(ROOT / "serve" / "chat_template.jinja"))
        out = list(svc.run(TOK.encode("x"), True, RT.declared(None), 4000, {}, threading.Event()))
        self.assertEqual([k for k, _ in out if k not in ("event", "done", "ping")], [])


if __name__ == "__main__":
    unittest.main()
