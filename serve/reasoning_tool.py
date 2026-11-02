"""serve/reasoning_tool.py - `run_python` in the model's own tool format, answered while it thinks and in its reply (#288).

On the A100, declaring `run_python` in the template's `<tools>` block made Gyro-M test its own hypotheses unforced: it ran its program on the
examples, set a brute force against it on random inputs, walked a grid as an independent oracle and timed large inputs, and it calls the tool
both while thinking and in its reply (`tools/igc/external_bench.py --tool-loop`, lt1 and lt2). This module brings that route to a served
request, opt-in behind the run config's `"reasoning_tool": true`:

- the engine runs a call made while thinking (`--igc-sandbox-tool`, include/strata/program/igc_learned_diagnostic.hpp) and writes its
  `<tool_response>` into the thinking. The server starts the engine with what that route needs (`startup`), and refuses to start when the
  profile cannot give it, rather than run half of it;
- the declaration (`TOOL`) goes into the tools of a request on the route, beside the client's own;
- a call in the reply is the server's, never the client's: `Turn.run` answers it and the turn goes on (`run_with_mcp`).

Code the model wrote runs as real Python only in an isolation box (serve/sandbox/isolation.py) the developer has approved (design D1:
`"sandbox": {"real_python": true}`), or on a throwaway VM (`exec_bridge.throwaway`). Without either, the engine's own interpreter answers the
calls made while thinking, a program it refuses is told why, and a call in the reply is told that nothing ran there.

With the key off nothing here runs: the server reads `on(cfg)` and goes on as before.
"""
from __future__ import annotations

import os
import threading
import time

NAME = "run_python"
# The same words as the bench's TOOL_HINT (tools/igc/external_bench.py): the arm that scored 8 of 10 read exactly these.
DESCRIPTION = ("Run Python and read what it prints. Names stay between calls. Test an idea on small cases, compare with a brute force, "
               "or run your program on the examples, as often as it helps, before you answer.")
TOOL = {"name": NAME, "description": DESCRIPTION,
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}

RUNS = 24            # calls in the reply that run, per request (the bench's TOOL_ROUNDS)
AFTER = 4            # calls past those that are answered with SPENT before the turn ends (TOOL_AFTER)
SPENT = "[no more runs: finish the thinking and write the answer]"
SHOWN = 4096         # characters of a run's output the model reads (TOOL_OUTPUT)
NOTHING = "[it ran and printed nothing]"
NO_CODE = "[run_python needs `code`: the program, as a string]"
NO_PYTHON = ("[run_python ran nothing: here a call runs only while thinking, before </think>, in the engine's own interpreter. A call in the "
             "reply needs real Python, which this PC has not approved (design D1).]")

# What --igc-sandbox needs, and the route on top of it (src/program/generate.cpp, the --igc-sandbox checks). --igc-inline-exec also sets
# the result bridge; the bridge is named too, so the args say what runs.
ENGINE_FLAGS = ("--igc-inline-exec", "--igc-result-bridge", "--igc-sandbox", "--igc-sandbox-tool")
BUNDLE = "--igc-learned-diagnostic-bundle"
BOX_MEMORY = 2 << 30
BOX_SECONDS = 3600.0


def on(cfg) -> bool:
    """The run config's `"reasoning_tool"`: true turns the route on; absent or false leaves the server as it was. Anything else is refused."""
    value = cfg.get("reasoning_tool") if isinstance(cfg, dict) else None
    if value is None or value is False:
        return False
    if value is not True:
        raise ValueError(f'"reasoning_tool" must be true or false, not {value!r}')
    return True


def declared(tools):
    """The request's tools with `run_python` after them. A client that brings its own `run_python` keeps it: its call then goes to it."""
    tools = list(tools or [])
    if any(isinstance(t, dict) and t.get("name") == NAME for t in tools):
        return tools
    return tools + [dict(TOOL)]


def exec_plan(cfg, args) -> tuple[bool, str]:
    """Whether the engine gets --igc-sandbox-exec, which sends what its interpreter refuses to real Python, and why (or why not)."""
    from serve.sandbox.exec_bridge import throwaway
    if "--batch" in args or "--slots" in args:
        return False, "the engine runs no kernel for --batch slots, so its own interpreter answers"
    if throwaway():
        return True, "a throwaway VM (STRATA_SANDBOX_THROWAWAY=1): real Python runs bare"
    sandbox = cfg.get("sandbox") if isinstance(cfg.get("sandbox"), dict) else {}
    if sandbox.get("real_python") is not True:
        return False, "real Python is not approved on this PC (design D1: \"sandbox\": {\"real_python\": true}), so the engine's own interpreter answers"
    from serve.sandbox.isolation import Isolation
    report = Isolation.report()
    backend = "appcontainer" if os.name == "nt" else "bubblewrap"
    ok, why = report.get(backend, (False, "no backend"))
    if not ok:
        return False, f"no isolation box on this PC ({backend}: {why}), so the engine's own interpreter answers"
    return True, f"real Python in an isolation box ({backend})"


def startup(cfg, args, binary: str, groups=None) -> tuple[list[str], bool, str]:
    """The engine args the route adds to `args`, whether the kernel goes in an isolation box (`box_factory`; false on a throwaway VM and
    where real Python does not run), and a line that says what answers. ValueError, with what to change, when the profile cannot run the
    whole route: the server then refuses to start."""
    if groups:
        raise ValueError('"reasoning_tool": one engine holds one sandbox session; it does not run with replicas')
    if BUNDLE not in args[:-1]:
        raise ValueError(f'"reasoning_tool" needs the learned bundle: the engine runs --igc-sandbox only with {BUNDLE} <path> in the '
                         "profile's args")
    bundle = args[args.index(BUNDLE) + 1]
    where = bundle if os.path.isabs(bundle) else os.path.join(cfg.get("cwd") or ".", bundle)
    if not os.path.exists(where):
        raise ValueError(f'"reasoning_tool": the learned bundle {bundle} is not there ({os.path.abspath(where)})')
    try:
        with open(binary, "rb") as f:
            built = b"--igc-sandbox-tool" in f.read()
    except OSError as e:
        raise ValueError(f'"reasoning_tool": the engine {binary} could not be read ({e})') from None
    if not built:
        raise ValueError('"reasoning_tool": this engine was built without the reasoning tool (no --igc-sandbox-tool); build it with '
                         "STRATA_BUILD_IGC_HEAD_PROBE=ON")
    extra = [flag for flag in ENGINE_FLAGS if flag not in args]
    real, why = exec_plan(cfg, args)
    if real and "--igc-sandbox-exec" not in args:
        extra.append("--igc-sandbox-exec")
    elif not real and "--igc-sandbox-exec" in args:
        why += "; the profile's own --igc-sandbox-exec stays, and each block it sends is told that no box runs it"
    from serve.sandbox.exec_bridge import throwaway
    return extra, real and not throwaway(), why


def box_factory():
    """What starts the kernel's isolation box: no project, no network, its own folder and the box's limits (serve/sandbox/isolation.py)."""
    from serve.sandbox.isolation import Isolation, Policy

    def start():
        return Isolation.start(Policy(memory_bytes=BOX_MEMORY, seconds=BOX_SECONDS))
    return start


def shown(text: str) -> str:
    text = text if text and text.strip() else NOTHING
    return text if len(text) <= SHOWN else text[:SHOWN] + "\n[output cut]"


class Turn:
    """One request's `run_python`: its calls in the reply, and the mark that puts the request on the route (`_rtool` in its sampling, so
    the engine runs it with MIP and the bridge knows whose kernel is whose). `bridge` is the engine's ExecBridge, None when there is no real
    Python; `mcp` what a program may reach of the request's MCP tools (serve/mip/mcp_adapter.py), as in the thinking."""

    def __init__(self, bridge=None, mcp=None):
        self.bridge, self.mcp = bridge, mcp
        self.calls = 0
        self.lock = threading.Lock()

    def over(self) -> bool:
        """Past the runs and the answers that say to finish: the next call ends the turn, as the bench's loop did."""
        return self.calls >= RUNS + AFTER

    def run(self, arguments, cancel=None) -> dict:
        """One call in the reply -> {"ok", "text" (what the model reads), "chars", "truncated", "ms"}, as McpHub.call gives."""
        t0 = time.monotonic()
        code = arguments.get("code") if isinstance(arguments, dict) else None
        with self.lock:
            self.calls += 1
            n = self.calls
        if not isinstance(code, str) or not code.strip():
            ok, text = False, NO_CODE
        elif n > RUNS:
            ok, text = False, SPENT
        elif self.bridge is None:
            ok, text = False, NO_PYTHON
        else:
            ok, text = self.bridge.run_reply(self, code, self.mcp, cancel)
            text = shown(text)
        return {"ok": ok, "text": text, "chars": len(text), "truncated": text.endswith("\n[output cut]"), "ms": int((time.monotonic() - t0) * 1000)}

    def close(self):
        if self.bridge is not None:
            self.bridge.end(self)
