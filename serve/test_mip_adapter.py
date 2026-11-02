"""serve/test_mip_adapter.py - MCP tools from a program in the reasoning (serve/mip/, serve/sandbox/), against the fake MCP
server in serve/mcp_fake_server.py (no GPU, no pack, no MCP SDK).

    python -m unittest serve.test_mip_adapter -v

Model-written code never runs bare on this PC, and `throwaway()` is false on Windows, so the blocks here go through
`FakeBox`, which starts the kernel on programs this file wrote.  A path counts as run when its output reads right and
the fake server's log shows what reached it; nothing between the kernel and the server is mocked.
"""
from __future__ import annotations

import base64
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import mcp_fake_server as fake  # noqa: E402
from serve import permissions  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.mcp import McpError, McpHub  # noqa: E402
from serve.mip.core import MIPInvocation, MIPResult  # noqa: E402
from serve.mip.mcp_adapter import CALLS_PER_BLOCK, CALLS_PER_TURN, McpAdapter, check_arguments  # noqa: E402
from serve.sandbox.exec_bridge import ExecBridge, answer, kernel_env  # noqa: E402
from serve.server import ByteTokenizer, Service, StrataEngine, run_with_mcp, sandbox_bridge  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FAKE = str(ROOT / "serve" / "mcp_fake_server.py")


def stdio(*args):
    return {"command": sys.executable, "args": [FAKE, *args]}


def sbx(code, fresh=False):
    return f"SBX {int(fresh)} {base64.b64encode(code.encode('utf-8')).decode('ascii')}\n"


def sbxr(answer):
    """(ok, the text the reasoning reads) of an SBXR line."""
    parts = answer.split()
    return parts[1] == "1", base64.b64decode(parts[2]).decode("utf-8") if len(parts) > 2 else ""


class FakeBox:
    """Stands where the isolation box will: it starts the kernel, for the programs of this file."""
    started = 0

    def __init__(self):
        self.workdir = tempfile.mkdtemp(prefix="fakebox-")   # a box's own folder: the kernel is copied in and started from there

    def popen(self, argv):
        FakeBox.started += 1
        argv = [sys.executable if argv[0] == "python" else argv[0], *argv[1:]]
        return subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, cwd=self.workdir,
                                encoding="utf-8", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    def kill(self):
        shutil.rmtree(self.workdir, ignore_errors=True)


class WithServers(unittest.TestCase):
    """The fake MCP server, with its log of what reached it."""
    settings = {"timeout_s": 1.0, "max_result_chars": 3000}

    @classmethod
    def setUpClass(cls):
        cls.logfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl")
        cls.logfile.close()
        os.environ["FAKE_MCP_LOG"] = cls.logfile.name
        cls.hub = McpHub({"fake": stdio()}, cls.settings)
        cls.hub.start(wait=True)

    @classmethod
    def tearDownClass(cls):
        cls.hub.close()
        del os.environ["FAKE_MCP_LOG"]
        os.unlink(cls.logfile.name)

    def setUp(self):
        open(self.logfile.name, "w").close()

    def reached(self):
        """The calls the server saw, by its own log."""
        lines = Path(self.logfile.name).read_text(encoding="utf-8").splitlines()
        return [x for x in map(json.loads, lines) if "call" in x]


# ------------------------------------------------------------------------------------------------ the envelope
class Envelope(unittest.TestCase):
    def test_an_invocation_survives_its_json_form(self):
        call = MIPInvocation("mcp", "fake.echo", {"text": "a"}, argument_provenance="model_program")
        back = MIPInvocation.from_json(json.loads(json.dumps(call.to_json())))
        self.assertEqual(back, call)
        self.assertEqual(back.target_kind, "mcp")
        self.assertTrue(back.invocation_id)

    def test_a_result_survives_its_json_form(self):
        result = MIPResult("id-1", "success", {"a": [1, 2]}, result_provenance={"server": "fake"})
        self.assertEqual(MIPResult.from_json(json.loads(json.dumps(result.to_json()))), result)

    def test_every_status_of_the_contract_is_accepted(self):
        for status in ("success", "refused", "unsupported", "cancelled", "fault"):
            self.assertEqual(MIPResult("x", status).status, status)

    def test_a_status_or_target_outside_the_contract_is_refused(self):
        with self.assertRaises(ValueError):
            MIPResult("x", "ok")
        with self.assertRaises(ValueError):
            MIPInvocation("rest", "fake.echo")


# ------------------------------------------------------------------------------------------------ the arguments
class Arguments(unittest.TestCase):
    schema = {"type": "object", "required": ["q"],
              "properties": {"q": {"type": "string"}, "n": {"type": "integer"}, "x": {"type": "number"}, "on": {"type": "boolean"},
                             "pick": {"enum": ["a", 1]}, "both": {"type": ["string", "null"]},
                             "tags": {"type": "array", "items": {"type": "string"}},
                             "deep": {"type": "object", "required": ["k"], "properties": {"k": {"type": "integer"}}}}}

    def why(self, **arguments):
        return check_arguments(self.schema, {"q": "a", **arguments})

    def test_good_arguments_pass(self):
        self.assertIsNone(self.why(n=3, x=1.5, on=True, pick="a", both=None, tags=["a"], deep={"k": 1}, other=object()))

    def test_a_missing_required_argument(self):
        self.assertIn("'q'", check_arguments(self.schema, {}))

    def test_a_wrong_type(self):
        self.assertIn("n", self.why(n="3"))
        self.assertIn("q", check_arguments(self.schema, {"q": 1}))

    def test_a_bool_is_not_a_number(self):
        self.assertIsNotNone(self.why(n=True))
        self.assertIsNotNone(self.why(x=False))

    def test_a_whole_float_is_an_integer_and_a_fraction_is_not(self):
        self.assertIsNone(self.why(n=2.0))
        self.assertIsNotNone(self.why(n=2.5))

    def test_a_value_outside_the_enum(self):
        self.assertIn("pick", self.why(pick="b"))
        self.assertIsNotNone(self.why(pick=True))            # True == 1 in Python, not in JSON
        self.assertIsNone(self.why(pick=1))

    def test_a_type_can_be_a_list(self):
        self.assertIsNone(self.why(both="a"))
        self.assertIsNotNone(self.why(both=1))

    def test_inside_objects_and_arrays(self):
        self.assertIn("tags[1]", self.why(tags=["a", 2]))
        self.assertIn("deep", self.why(deep={}))
        self.assertIn("deep.k", self.why(deep={"k": "a"}))

    def test_additional_properties_false_is_kept(self):
        self.assertIn("other", check_arguments({"type": "object", "additionalProperties": False, "properties": {}}, {"other": 1}))

    def test_what_the_schema_does_not_say_stays_with_the_server(self):
        self.assertIsNone(check_arguments({}, {"a": 1}))
        self.assertIsNone(check_arguments(None, {"a": 1}))
        self.assertIsNone(check_arguments({"type": "object", "properties": {"a": {"anyOf": [{"type": "string"}]}}}, {"a": 1}))


# ------------------------------------------------------------------------------------------------ the adapter
class Adapter(WithServers):
    def adapter(self, **kw):
        return McpAdapter(self.hub, **kw)

    def call(self, tool, adapter=None, **arguments):
        return (adapter or self.adapter()).invoke(MIPInvocation("mcp", f"fake.{tool}", arguments))

    def test_discovery_finds_by_word_and_names_the_call(self):
        result = self.adapter().search("weather")
        self.assertEqual(result.status, "success")
        self.assertEqual(len(result.typed_content), 1)
        line = result.typed_content[0]
        self.assertIn("mcp.fake.weather(city: str, units: 'c' | 'f' = ..., days: int = ..., tags: list = ...)", line)
        self.assertIn("The weather of a city", line)

    def test_discovery_finds_by_description_and_any_case(self):
        self.assertIn("mcp.fake.add(", self.adapter().search("INTEGERS").typed_content[0])

    def test_discovery_lists_only_what_matches(self):
        self.assertEqual(self.adapter().search("qqq").typed_content, [])
        self.assertEqual(len(self.adapter().search("echo").typed_content), 1)

    def test_discovery_names_the_tool_path_of_a_tool_it_will_not_run(self):
        self.assertIn("fake__save", self.adapter().search("save").typed_content[0])

    def test_a_call_returns_the_structured_result(self):
        result = self.call("weather", city="bkk", units="f", days=2)
        self.assertEqual(result.status, "success")
        self.assertEqual(result.typed_content["temp"], 31)
        self.assertEqual(result.typed_content["units"], "f")
        self.assertEqual(self.reached(), [{"call": "weather", "arguments": {"city": "bkk", "units": "f", "days": 2}}])
        self.assertEqual(result.result_provenance["server"], "fake")
        self.assertEqual(result.result_provenance["tool"], "weather")

    def test_a_call_with_no_structured_result_returns_its_text(self):
        result = self.call("add", a=2, b=3)
        self.assertEqual((result.status, result.typed_content), ("success", "5"))

    def test_a_missing_required_argument_is_refused_before_the_server(self):
        result = self.call("weather")
        self.assertEqual(result.status, "refused")
        self.assertIn("'city'", result.typed_content)
        self.assertEqual(self.reached(), [])

    def test_a_wrong_type_is_refused_before_the_server(self):
        result = self.call("weather", city=5)
        self.assertEqual(result.status, "refused")
        self.assertIn("city", result.typed_content)
        self.assertEqual(self.reached(), [])

    def test_a_value_outside_the_enum_is_refused_before_the_server(self):
        result = self.call("weather", city="a", units="k")
        self.assertEqual(result.status, "refused")
        self.assertIn("units", result.typed_content)
        self.assertEqual(self.reached(), [])

    def test_an_unknown_argument_is_refused_where_the_schema_says_so(self):
        self.assertEqual(self.call("weather", city="a", wind=1).status, "refused")
        self.assertEqual(self.reached(), [])

    def test_a_number_json_has_no_form_for_is_refused(self):
        result = self.call("sleep", seconds=float("nan"))
        self.assertEqual(result.status, "refused")
        self.assertEqual(self.reached(), [])

    def test_arguments_that_are_not_an_object_are_refused(self):
        answer = self.adapter().answer({"op": "call", "server": "fake", "tool": "add", "arguments": [1, 2]})
        self.assertEqual(answer["status"], "refused")
        self.assertEqual(self.reached(), [])

    def test_a_tool_without_read_only_is_refused_and_named_on_the_tool_path(self):
        for tool in ("save", "die"):                         # readOnlyHint false, and no annotations at all
            result = self.call(tool)
            self.assertEqual(result.status, "refused")
            self.assertIn(f"fake__{tool}", result.typed_content)
            self.assertIn("readOnlyHint", result.typed_content)
        self.assertEqual(self.reached(), [])
        self.assertEqual(self.call("add", a=1, b=1).status, "success")      # die did not run: the server is alive

    def test_a_hint_that_is_only_truthy_is_no_hint(self):
        hub = McpHub({}, builtins={"odd": Stub({"annotations": {"readOnlyHint": "true"}})})
        result = McpAdapter(hub).invoke(MIPInvocation("mcp", "odd.t", {}))
        self.assertEqual(result.status, "refused")

    def test_an_unknown_tool_is_unsupported(self):
        self.assertEqual(self.call("nope").status, "unsupported")
        result = self.adapter().invoke(MIPInvocation("mcp", "nope.echo", {"text": "a"}))
        self.assertEqual(result.status, "unsupported")
        self.assertIn("mcp.tools", result.typed_content)

    def test_a_tool_error_is_a_fault_with_its_words(self):
        result = self.call("fail")
        self.assertEqual(result.status, "fault")
        self.assertIn("it failed on purpose", result.typed_content)

    def test_a_timeout_is_a_fault(self):
        start = time.monotonic()
        result = self.call("sleep", seconds=5)
        self.assertEqual(result.status, "fault")
        self.assertIn("timed out", result.typed_content)
        self.assertLess(time.monotonic() - start, 3)

    def test_a_cancelled_request_runs_nothing(self):
        cancel = threading.Event()
        cancel.set()
        self.assertEqual(self.call("echo", self.adapter(cancel=cancel), text="a").status, "cancelled")
        self.assertEqual(self.reached(), [])

    def test_stop_during_a_call_is_cancelled(self):
        cancel = threading.Event()
        threading.Timer(0.2, cancel.set).start()
        self.assertEqual(self.call("sleep", self.adapter(cancel=cancel), seconds=5).status, "cancelled")

    def test_an_oversized_text_is_cut_with_a_note(self):
        result = self.call("big", n=5000)
        self.assertEqual(result.status, "success")
        self.assertIn("truncated: the tool returned 5,000 characters", result.typed_content)
        self.assertLess(len(result.typed_content), 3200)
        self.assertTrue(result.result_provenance["truncated"])
        self.assertEqual(result.result_provenance["chars"], 5000)

    def test_an_oversized_structured_result_comes_back_as_cut_text(self):
        result = self.call("big", n=5000, structured=True)
        self.assertIsInstance(result.typed_content, str)     # half a dict cannot be read as one
        self.assertIn("truncated: the tool returned", result.typed_content)
        self.assertTrue(result.result_provenance["truncated"])

    def test_the_block_budget(self):
        adapter = self.adapter()
        adapter.begin_block(60)
        for _ in range(CALLS_PER_BLOCK):
            self.assertEqual(self.call("add", adapter, a=1, b=1).status, "success")
        result = self.call("add", adapter, a=1, b=1)
        self.assertEqual(result.status, "refused")
        self.assertIn(str(CALLS_PER_BLOCK), result.typed_content)
        self.assertEqual(len(self.reached()), CALLS_PER_BLOCK)
        adapter.begin_block(60)
        self.assertEqual(self.call("add", adapter, a=1, b=1).status, "success")

    def test_the_turn_budget(self):
        adapter = self.adapter()
        for _ in range(CALLS_PER_TURN // CALLS_PER_BLOCK):
            adapter.begin_block(60)
            for _ in range(CALLS_PER_BLOCK):
                self.assertEqual(self.call("add", adapter, a=1, b=1).status, "success")
        adapter.begin_block(60)
        result = self.call("add", adapter, a=1, b=1)
        self.assertEqual(result.status, "refused")
        self.assertIn("turn", result.typed_content)
        adapter.begin_turn()
        adapter.begin_block(60)
        self.assertEqual(self.call("add", adapter, a=1, b=1).status, "success")

    def test_a_refused_call_does_not_spend_the_budget(self):
        adapter = self.adapter()
        adapter.begin_block(60)
        for _ in range(CALLS_PER_BLOCK + 1):
            self.assertEqual(self.call("save", adapter).status, "refused")
        self.assertEqual(self.call("add", adapter, a=1, b=1).status, "success")

    def test_a_block_out_of_time_calls_nothing(self):
        adapter = self.adapter()
        adapter.begin_block(0)
        self.assertEqual(self.call("add", adapter, a=1, b=1).status, "refused")
        self.assertEqual(self.reached(), [])

    def test_a_call_waits_no_longer_than_the_block_has_left(self):
        adapter = self.adapter()
        adapter.begin_block(0.2)
        start = time.monotonic()
        self.assertEqual(self.call("sleep", adapter, seconds=5).status, "fault")
        self.assertLess(time.monotonic() - start, 0.9)

    def test_only_the_tools_the_request_offered(self):
        adapter = self.adapter(names={"fake__echo"})
        self.assertEqual(len(adapter.search("").typed_content), 1)
        self.assertEqual(self.call("echo", adapter, text="a").status, "success")
        self.assertEqual(self.call("add", adapter, a=1, b=1).status, "unsupported")

    def test_a_deny_rule_of_the_chat_holds(self):
        policy = permissions.Policy(cwd=None, deny=["fake__echo"])
        result = self.call("echo", self.adapter(policy=policy), text="a")
        self.assertEqual(result.status, "refused")
        self.assertIn("fake__echo", result.typed_content)
        self.assertEqual(self.reached(), [])
        self.assertEqual(self.call("add", self.adapter(policy=policy), a=1, b=1).status, "success")

    def test_a_junk_request_is_answered(self):
        adapter = self.adapter()
        for junk in (5, None, [], {}, {"op": "x"}, {"op": "call"}, {"op": "call", "server": 5, "tool": []}):
            self.assertIn(adapter.answer(junk)["status"], ("unsupported", "refused"), junk)


class Stub:
    """A server by the hub's interface, to give a tool an odd listing or a failure a real server cannot be made to give."""
    kind, name, status, error, info = "stdio", "odd", "ready", None, {}

    def __init__(self, extra=None, failure=None):
        self.tools = [{"name": "t", "inputSchema": {"type": "object"}, "annotations": {"readOnlyHint": True}, **(extra or {})}]
        self.failure = failure

    def call(self, tool, arguments, timeout, cancel=None):
        raise self.failure

    def start(self):
        pass

    def close(self):
        pass


class Names(WithServers):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.named = McpHub({"my-server": stdio(), "tools": stdio(), "2x": stdio()}, cls.settings)
        cls.named.start(wait=True)

    @classmethod
    def tearDownClass(cls):
        cls.named.close()
        super().tearDownClass()

    def test_a_name_python_cannot_write_gets_one_it_can(self):
        result = McpAdapter(self.named).invoke(MIPInvocation("mcp", "my_server.add", {"a": 1, "b": 2}))
        self.assertEqual(result.typed_content, "3")
        self.assertIn("mcp._2x.add(", "\n".join(McpAdapter(self.named, names={"2x__add"}).search("add").typed_content))

    def test_a_server_called_tools_does_not_hide_the_search(self):
        adapter = McpAdapter(self.named, names={"tools__add"})
        self.assertIn("mcp.tools_.add(", adapter.search("add").typed_content[0])
        self.assertEqual(adapter.invoke(MIPInvocation("mcp", "tools_.add", {"a": 1, "b": 1})).status, "success")


class Reach(unittest.TestCase):
    def test_the_built_in_servers_are_not_reached(self):
        builtin = Stub()
        builtin.kind, builtin.name = "builtin", "skills"
        hub = McpHub({}, builtins={"skills": builtin})
        adapter = McpAdapter(hub)
        self.assertEqual(adapter.search("").typed_content, [])
        self.assertEqual(adapter.invoke(MIPInvocation("mcp", "skills.t", {})).status, "unsupported")

    def test_arguments_that_are_not_an_object_never_leave_even_with_an_empty_schema(self):
        hub = McpHub({}, builtins={"odd": Stub({"inputSchema": {}}, failure=McpError("it ran"))})
        self.assertEqual(McpAdapter(hub).answer({"op": "call", "server": "odd", "tool": "t", "arguments": [1, 2]})["status"], "refused")

    def test_a_connection_failure_says_nothing_of_the_address(self):
        secret = "https://example.invalid/mcp?key=SECRET-TOKEN"
        hub = McpHub({}, builtins={"odd": Stub(failure=McpError(f"could not reach {secret}"))})
        result = McpAdapter(hub).invoke(MIPInvocation("mcp", "odd.t", {}))
        self.assertEqual(result.status, "fault")
        self.assertNotIn("SECRET-TOKEN", json.dumps(result.to_json()))

    def test_a_failure_nobody_expected_is_still_an_answer(self):
        hub = McpHub({}, builtins={"odd": Stub(failure=RuntimeError("SECRET-TOKEN"))})
        got = answer(McpAdapter(hub), {"op": "call", "server": "odd", "tool": "t", "arguments": {}})
        self.assertEqual(got["status"], "fault")
        self.assertNotIn("SECRET-TOKEN", json.dumps(got))

    def test_a_kernel_with_no_box_gets_no_key_from_the_environment(self):
        with mock.patch.dict(os.environ, {"STRATA_API_KEY": "SECRET-TOKEN", "PATH": "x"}):
            self.assertEqual(kernel_env()["PATH"], "x")
            self.assertNotIn("SECRET-TOKEN", json.dumps(kernel_env()))


# ------------------------------------------------------------------------------------------------ blocks, in the kernel
class Blocks(WithServers):
    """Programs in the real kernel, through the real pipe, to the real fake server."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.bridge = ExecBridge(FakeBox)
        cls.bridge.begin(McpAdapter(cls.hub))
        ok, text = sbxr(cls.bridge.handle(sbx("print('up')\n", fresh=True)))
        assert ok and text == "up\n", text

    @classmethod
    def tearDownClass(cls):
        cls.bridge.close()
        super().tearDownClass()

    def setUp(self):
        super().setUp()
        self.bridge.begin(McpAdapter(self.hub))

    def run_block(self, code):
        return sbxr(self.bridge.handle(sbx(code)))

    def test_a_block_calls_a_tool_picks_a_field_and_prints_it(self):
        ok, text = self.run_block("w = mcp.fake.weather(city='bkk')\nprint(w['temp'] + 1)\n")
        self.assertTrue(ok, text)
        self.assertEqual(text, "32\n")
        self.assertEqual(self.reached(), [{"call": "weather", "arguments": {"city": "bkk"}}])

    def test_what_the_program_does_not_print_stays_out(self):
        ok, text = self.run_block("w = mcp.fake.weather(city='bkk')\nkept = len(w['report'])\n")
        self.assertTrue(ok, text)
        self.assertEqual(text, "")
        self.assertEqual(len(self.reached()), 1)             # the tool ran, and its 1,700 characters were not read
        ok, text = self.run_block("print(kept)\n")
        self.assertEqual(text, f"{len(fake.REPORT)}\n")      # the name stays for the next block of the turn

    def test_a_text_result_is_a_str(self):
        self.assertEqual(self.run_block("print(repr(mcp.fake.add(a=2, b=3)))\n")[1], "'5'\n")

    def test_a_structured_result_is_a_dict(self):
        self.assertEqual(self.run_block("print(type(mcp.fake.weather(city='a')).__name__)\n")[1], "dict\n")

    def test_tools_finds_on_demand(self):
        ok, text = self.run_block("for line in mcp.tools('weather'):\n    print(line)\n")
        self.assertTrue(ok, text)
        self.assertIn("mcp.fake.weather(city: str", text)
        self.assertNotIn("echo", text)
        self.assertEqual(self.reached(), [])                 # the search reads the list the hub already has

    def test_text_outside_ascii_goes_both_ways(self):
        self.assertEqual(self.run_block("print(mcp.fake.echo(text='สวัสดี'))\n")[1], "สวัสดี\n")
        self.assertEqual(self.reached()[0]["arguments"], {"text": "สวัสดี"})

    def test_a_refused_tool_is_the_blocks_error_with_the_reason(self):
        ok, text = self.run_block("x = 1\nmcp.fake.save(text='a')\n")
        self.assertFalse(ok)
        self.assertIn("refused", text)
        self.assertIn("fake__save", text)
        self.assertIn("line 2", text)                        # the model's line, not the kernel's
        self.assertEqual(self.reached(), [])

    def test_a_validation_failure_is_the_blocks_error(self):
        ok, text = self.run_block("mcp.fake.weather()\n")
        self.assertFalse(ok)
        self.assertIn("'city'", text)

    def test_a_program_can_catch_a_failure_and_go_on(self):
        ok, text = self.run_block("try:\n    mcp.fake.fail()\nexcept Exception as e:\n    print('caught:', e)\nprint('on')\n")
        self.assertTrue(ok, text)
        self.assertIn("caught: fault: it failed on purpose", text)
        self.assertTrue(text.endswith("on\n"))

    def test_a_tool_error_ends_the_block_with_its_words(self):
        ok, text = self.run_block("mcp.fake.fail()\n")
        self.assertFalse(ok)
        self.assertIn("it failed on purpose", text)

    def test_arguments_must_be_named(self):
        ok, text = self.run_block("mcp.fake.echo('a')\n")
        self.assertFalse(ok)
        self.assertIn("mcp.fake.echo(arg=value)", text)
        self.assertEqual(self.reached(), [])

    def test_a_parameter_called_self_works(self):
        ok, text = self.run_block("print(mcp.fake.echo(text='a', self=1))\n")
        self.assertTrue(ok, text)
        self.assertEqual(text, "a\n")

    def test_the_block_budget_in_the_kernel(self):
        ok, text = self.run_block(f"for i in range({CALLS_PER_BLOCK + 1}):\n    mcp.fake.add(a=i, b=1)\n")
        self.assertFalse(ok)
        self.assertIn("refused", text)
        self.assertEqual(len(self.reached()), CALLS_PER_BLOCK)
        self.assertTrue(self.run_block("print(mcp.fake.add(a=1, b=1))\n")[0])        # a new block has its own

    def test_a_program_cannot_talk_past_the_gate(self):
        forged = ("import json, sys\n"
                  "sys.__stdout__.write(json.dumps({'mcp': {'op': 'call', 'server': 'fake', 'tool': 'save', 'arguments': {}}}) + '\\n')\n"
                  "sys.__stdout__.flush()\n"
                  "print(json.loads(sys.__stdin__.readline())['mcp']['status'])\n")
        ok, text = self.run_block(forged)
        self.assertTrue(ok, text)
        self.assertEqual(text, "refused\n")
        self.assertEqual(self.reached(), [])

    def test_calls_from_threads_each_get_their_own_answer(self):
        code = ("import threading\nout, wrong = {}, []\ntogether = threading.Barrier(4)\n"
                "def go(i):\n    together.wait()\n    for _ in range(2):\n        out[i] = mcp.fake.echo(text=str(i))\n        wrong.append(out[i] != str(i))\n"
                "ts = [threading.Thread(target=go, args=(i,)) for i in range(4)]\n"
                "[t.start() for t in ts]\n[t.join() for t in ts]\n"
                "print(sorted(out.items()), any(wrong))\n")
        ok, text = self.run_block(code)
        self.assertTrue(ok, text)
        self.assertEqual(text, f"{[(i, str(i)) for i in range(4)]} False\n")

    def test_a_request_with_no_mcp_servers_reaches_nothing(self):
        self.bridge.begin(None)
        ok, text = self.run_block("mcp.fake.add(a=1, b=1)\n")
        self.assertFalse(ok)
        self.assertIn("no MCP servers", text)
        self.assertEqual(self.reached(), [])
        self.assertEqual(self.run_block("print(1 + 1)\n"), (True, "2\n"))            # and a plain block runs as ever

    def test_a_name_of_the_program_can_stand_for_mcp(self):
        self.assertEqual(self.run_block("mcp2 = mcp\nprint(mcp2.fake.add(a=1, b=1))\n")[1], "2\n")


class Pipe(unittest.TestCase):
    def test_a_thread_never_reads_the_answer_of_another(self):
        """The answers carry no id.  Without the lock the first thread, held after it wrote, loses its answer to the second: an order
        a real run reaches only by chance, forced here with a stand-in for the pipe."""
        from serve.sandbox import kernel_main as kernel
        answers, holds = queue.Queue(), iter([0.1])

        class Up:
            def write(self, line):
                query = json.loads(line)["mcp"]["query"]
                answers.put(json.dumps({"mcp": {"status": "success", "typed_content": query}}) + "\n")

            def flush(self):
                time.sleep(next(holds, 0))

        class Down:
            def readline(self):
                return answers.get(timeout=5)

        got = {}
        with mock.patch.object(kernel, "REAL_OUT", Up()), mock.patch.object(kernel, "REAL_IN", Down()):
            threads = [threading.Thread(target=lambda q=q: got.update({q: kernel.Mcp().tools(q)})) for q in ("a", "b")]
            threads[0].start()
            time.sleep(0.02)
            threads[1].start()
            for t in threads:
                t.join(5)
        self.assertEqual(got, {"a": "a", "b": "b"})


class KernelLife(WithServers):
    """Each with its own kernel: what ends a kernel, and what a fresh one starts."""

    def bridge(self):
        bridge = ExecBridge(FakeBox)
        self.addCleanup(bridge.close)
        return bridge

    def test_a_fresh_kernel_is_a_new_turn(self):
        bridge = self.bridge()
        bridge.begin(McpAdapter(self.hub))
        loop = f"for i in range({CALLS_PER_BLOCK}):\n    mcp.fake.add(a=i, b=1)\n"
        for i in range(CALLS_PER_TURN // CALLS_PER_BLOCK):
            self.assertTrue(sbxr(bridge.handle(sbx(loop, fresh=i == 0)))[0])
        ok, text = sbxr(bridge.handle(sbx("mcp.fake.add(a=1, b=1)\n")))
        self.assertFalse(ok)
        self.assertIn("turn", text)
        self.assertEqual(sbxr(bridge.handle(sbx("print(mcp.fake.add(a=1, b=1))\n", fresh=True))), (True, "2\n"))

    def test_a_kernel_that_dies_during_a_block_says_so(self):
        bridge = self.bridge()
        bridge.begin(McpAdapter(self.hub))
        ok, text = sbxr(bridge.handle(sbx("import os\nmcp.fake.add(a=1, b=1)\nos._exit(1)\n", fresh=True)))
        self.assertFalse(ok)
        self.assertIn("ended the sandbox", text)

    def test_no_box_runs_nothing(self):
        bridge = ExecBridge()                                # no isolation box, on a PC that is not thrown away
        bridge.begin(McpAdapter(self.hub))
        ok, text = sbxr(bridge.handle(sbx("print(mcp.fake.add(a=1, b=1))\n", fresh=True)))
        self.assertFalse(ok)
        self.assertIn("no isolation box", text)
        self.assertEqual(self.reached(), [])

    def test_a_block_the_engine_stopped_waiting_for_gets_no_answer(self):
        bridge = self.bridge()
        bridge.begin(None)
        bridge.handle(sbx("x = 1\n", fresh=True))             # the kernel is up
        bridge.begin(None)
        got = []
        worker = threading.Thread(target=lambda: got.append(bridge.handle(sbx("import time\ntime.sleep(0.5)\n"))))
        worker.start()
        time.sleep(0.1)
        bridge.stop()
        worker.join(5)
        self.assertEqual(got, [""])

    def test_a_block_that_outlives_its_request_keeps_its_own_reach(self):
        bridge = self.bridge()
        cancel = threading.Event()
        bridge.begin(McpAdapter(self.hub, cancel=cancel))
        bridge.handle(sbx("x = 1\n", fresh=True))
        got = []
        code = "import time\ntime.sleep(0.5)\ntry:\n    mcp.fake.add(a=1, b=1)\nexcept Exception as e:\n    print(e)\n"
        worker = threading.Thread(target=lambda: got.append(bridge.handle(sbx(code))))
        worker.start()
        time.sleep(0.2)
        cancel.set()
        bridge.begin(McpAdapter(self.hub))                    # the next request: a late block must not borrow its reach
        worker.join(5)
        self.assertEqual(got, [""])
        self.assertEqual(self.reached(), [])


# ------------------------------------------------------------------------------------------------ the server
ENGINE = r'''
import base64, sys
from pathlib import Path
out, block = Path(sys.argv[sys.argv.index('--out') + 1]), Path(sys.argv[sys.argv.index('--code') + 1])
print('READY 4096 stop', flush=True)
for line in sys.stdin:
    if line.startswith('QUIT'):
        break
    if line.startswith('GEN'):
        print('T 90', flush=True)
        print('SBX 1 ' + base64.b64encode(block.read_bytes()).decode('ascii'), flush=True)
        while True:
            line = sys.stdin.readline()
            if line.startswith('SBXR ') or line.startswith('STOP') or not line:
                break
        with out.open('a', encoding='utf-8') as f:
            f.write(line)
        print('DONE 1 1 0 1 stop', flush=True)
    elif line.startswith('SBXR'):
        with out.open('a', encoding='utf-8') as f:
            f.write('LATE ' + line)
'''


class Served(WithServers):
    """The whole route but the model: a stand-in engine prints SBX, StrataEngine's pump and ExecBridge answer through the kernel."""

    def setUp(self):
        super().setUp()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.out, self.code = self.root / "seen.txt", self.root / "block.py"
        py = self.root / "native.py"
        py.write_text(ENGINE, encoding="utf-8")
        script = py
        if os.name == "nt":                                  # no shebangs on Windows: a .cmd runs the same fake engine
            script = self.root / "native.cmd"
            script.write_text(f'@"{sys.executable}" "{py}" %*\r\n')
        else:
            py.write_text("#!" + sys.executable + "\n" + ENGINE)
            py.chmod(0o700)
        self.engine = StrataEngine(str(script), ["--out", str(self.out), "--code", str(self.code)])
        self.addCleanup(self.engine.close)
        self.engine.sandbox_exec = self.bridge = ExecBridge(FakeBox)
        self.addCleanup(self.bridge.close)
        self.svc = Service(self.engine, ByteTokenizer(), ChatTemplate(ROOT / "serve" / "chat_template.jinja"))

    def request(self, offered):
        """One answer through the tool loop, with these tools offered -> what the engine got back for its block."""
        before = len(self.seen())
        list(run_with_mcp(self.svc, self.hub, [{"role": "user", "content": "hi"}], None, {}, [1], False, 8, 8, {},
                          threading.Event(), offered))
        return self.seen()[before]

    def seen(self):
        return sbxr_lines(self.out)

    def test_a_block_reaches_the_tool_its_request_offered(self):
        self.code.write_text("w = mcp.fake.weather(city='bkk')\nprint(w['temp'])\n", encoding="utf-8")
        self.assertEqual(self.request({"fake__weather"}), (True, "31\n"))
        self.assertEqual(self.reached(), [{"call": "weather", "arguments": {"city": "bkk"}}])

    def test_a_block_does_not_reach_a_tool_its_request_did_not_offer(self):
        self.code.write_text("print(mcp.fake.weather(city='bkk')['temp'])\n", encoding="utf-8")
        ok, text = self.request({"fake__echo"})
        self.assertFalse(ok)
        self.assertIn("unsupported", text)
        self.assertEqual(self.reached(), [])

    def test_a_request_outside_the_tool_loop_reaches_nothing(self):
        self.code.write_text("print(mcp.fake.add(a=1, b=1))\n", encoding="utf-8")
        self.assertEqual(self.request({"fake__add"}), (True, "2\n"))
        list(self.svc.run([1], False, [], 8, {}, threading.Event()))      # after a request with tools, one without
        ok, text = self.seen()[-1]
        self.assertFalse(ok)
        self.assertIn("no MCP servers", text)
        self.assertEqual(len(self.reached()), 1)

    def test_a_hub_replaced_since_the_start_is_the_one_used(self):
        self.code.write_text("print(mcp.fake.add(a=1, b=1))\n", encoding="utf-8")
        hub = McpHub({"fake": stdio()}, self.settings)
        hub.start(wait=True)
        self.addCleanup(hub.close)
        before = len(self.seen())
        list(run_with_mcp(self.svc, hub, [{"role": "user", "content": "hi"}], None, {}, [1], False, 8, 8, {}, threading.Event(),
                          {"fake__add"}))
        self.assertEqual(self.seen()[before], (True, "2\n"))

    def test_an_answer_after_stop_never_reaches_the_engine(self):
        self.code.write_text("import time\ntime.sleep(1)\nprint('late')\n", encoding="utf-8")
        done = threading.Event()
        handle = self.bridge.handle

        def watched(line):
            try:
                return handle(line)
            finally:
                done.set()
        self.bridge.handle = watched
        gen = self.engine.generate([1], 8, {}, threading.Event())
        self.assertEqual(next(gen), 90)
        gen.close()                                          # the client went away: STOP, while the block runs
        self.assertTrue(done.wait(20))
        time.sleep(0.3)
        self.assertNotIn("LATE", self.out.read_text(encoding="utf-8"))


    def test_nothing_is_answered_once_a_request_has_finished(self):
        self.code.write_text("print(1)\n", encoding="utf-8")
        self.assertEqual(self.request({"fake__add"}), (True, "1\n"))
        self.assertEqual(self.bridge.handle(sbx("print(2)\n")), "")     # past the engine's wait, a block that comes back late


def sbxr_lines(path):
    """The answers a stand-in engine wrote down, in order."""
    if not path.exists():
        return []
    return [sbxr(x) for x in path.read_text(encoding="utf-8").splitlines() if x.startswith("SBXR")]


class Wiring(unittest.TestCase):
    def test_a_bridge_only_for_an_engine_that_asked_for_one(self):
        self.assertIsNone(sandbox_bridge(["--max-context", "4096"], None))
        self.assertIsInstance(sandbox_bridge(["--igc-sandbox-exec"], None), ExecBridge)
        self.assertIsNone(sandbox_bridge(["--igc-sandbox-exec"], [[0], [1]]))    # one kernel, one request: not across replicas


if __name__ == "__main__":
    unittest.main()
