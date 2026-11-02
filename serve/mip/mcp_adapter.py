"""serve/mip/mcp_adapter.py - a program in the reasoning reaches the MCP servers of serve/mcp.py: milestone M2 of the Roadmap.

The servers see an ordinary MCP client, because this is one: it goes through `McpHub`'s routes and each server's `call`, and
it assembles the JSON-RPC itself, so the model writes `mcp.<server>.<tool>(arg=value)` and never a tool call's syntax.  Not
`McpHub.call`: that returns text, and a program needs `structuredContent` to pick a field from.

One `McpAdapter` is the reach of one request: the hub as it is now (a config change replaces the hub), the tools the request
offered the model, and the chat's rules.  A request with no MCP tools has no adapter at all.  The program is the model's and
runs in a box, so nothing it sends is trusted: the name, the arguments and the budget are all checked here.

A call from the reasoning is not an authorization.  A tool runs only when its server marks it `readOnlyHint: true`, and any
other tool is refused with the name it has on the tool path, where the user sees it run.  The hint is the server's own word:
it says what the tool is for, it does not prove it (MCP calls its annotations untrusted).

What reaches the program is a result or a reason in words, never a transport's error text: those can quote an address or a
log line, and a token can sit in either.  They go to the server's own log.
"""
from __future__ import annotations

import json
import keyword
import re
import time
from dataclasses import dataclass

from serve import permissions, timeline
from serve.mcp import McpCancelled, McpError, McpRpcError, McpTimeout, cut_text, result_text
from serve.mip.core import MIPInvocation, MIPResult

CALLS_PER_BLOCK = 8          # the tool path's limit for a whole answer (DEFAULTS["max_rounds"]): a block that wants more is looping
CALLS_PER_TURN = 32          # four full blocks; decoding waits through every call, and past this a reasoning is polling, not checking
TOOLS_SHOWN = 8              # about what fits in the 1,024 bytes of a block's output
TEXT_SHOWN = 80              # of a tool's description: enough to choose by, since each byte printed is a forced token
REASON_CHARS = 400           # a reason reaches the model through the same 1,024 bytes
RESERVED = {"tools"}         # `mcp.tools` is the search: a server of that name is `mcp.tools_`
PY_TYPES = {"string": "str", "integer": "int", "number": "float", "boolean": "bool", "array": "list", "object": "dict", "null": "None"}


# ------------------------------------------------------------------------------------------------ the arguments
def _same(a, b) -> bool:
    return isinstance(a, bool) == isinstance(b, bool) and a == b       # True == 1 in Python, not in JSON


def _is(value, kind) -> bool:
    if kind == "string":
        return isinstance(value, str)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "null":
        return value is None
    if kind == "array":
        return isinstance(value, list)
    if kind == "object":
        return isinstance(value, dict)
    if kind in ("integer", "number"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        return kind == "number" or isinstance(value, int) or value.is_integer()
    return True                  # a type word JSON Schema does not have is the server's to judge


def check_arguments(schema, value, where: str = "") -> str | None:
    """The first way `value` breaks the tool's inputSchema, None when it fits.  It reads `type`, `enum`, `required`, `properties`,
    `items` and `additionalProperties: false`; the rest of JSON Schema (anyOf, $ref, minimum) stays with the server."""
    if not isinstance(schema, dict):
        return None
    name = where or "the arguments"
    kinds = schema.get("type")
    if kinds is not None:
        kinds = kinds if isinstance(kinds, list) else [kinds]
        if not any(_is(value, k) for k in kinds):
            return f"{name} must be {' or '.join(str(k) for k in kinds)}, not {type(value).__name__}"
    if isinstance(schema.get("enum"), list) and not any(_same(value, x) for x in schema["enum"]):
        return f"{name} must be one of {schema['enum']!r}, not {value!r}"
    if isinstance(value, dict):
        for key in schema.get("required") or []:
            if key not in value:
                return f"{where + ' is ' if where else ''}missing the required argument {key!r}"
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        for key, item in value.items():
            if key not in props and schema.get("additionalProperties") is False:
                return f"{key!r} is not an argument of this tool: it takes {sorted(props)}"
            why = check_arguments(props.get(key), item, f"{where}.{key}" if where else key)
            if why:
                return why
    if isinstance(value, list):
        for i, item in enumerate(value):
            why = check_arguments(schema.get("items"), item, f"{name}[{i}]")
            if why:
                return why
    return None


# ------------------------------------------------------------------------------------------------ names and signatures
def _identifier(name: str, taken, reserved=()) -> str:
    """A name Python can write after a dot, different from those `taken`."""
    base = re.sub(r"\W", "_", name, flags=re.ASCII) or "x"
    if base[0].isdigit():
        base = "_" + base
    if keyword.iskeyword(base) or base in reserved:
        base += "_"
    text, n = base, 2
    while text in taken:
        text, n = f"{base}_{n}", n + 1
    return text


def _kind(schema) -> str:
    if not isinstance(schema, dict):
        return "object"
    if isinstance(schema.get("enum"), list) and schema["enum"]:
        return " | ".join(repr(x) for x in schema["enum"])
    kinds = schema.get("type") if isinstance(schema.get("type"), list) else [schema.get("type")]
    return " | ".join(PY_TYPES.get(k, "object") for k in kinds)


@dataclass
class _Tool:
    capability: str              # <server>.<tool> as the program writes it
    route: str                   # its name on the tool path
    server: object
    name: str                    # its own name, on its server
    spec: dict                   # as the server listed it

    @property
    def read_only(self) -> bool:
        notes = self.spec.get("annotations")
        return isinstance(notes, dict) and notes.get("readOnlyHint") is True    # the spec's default is false, and "true" is no answer

    @property
    def schema(self):
        return self.spec.get("inputSchema")

    def signature(self) -> str:
        schema = self.schema if isinstance(self.schema, dict) else {}
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        required = [k for k in schema.get("required") or [] if k in props]
        parts = [f"{k}: {_kind(props[k])}" for k in required]
        parts += [f"{k}: {_kind(v)} = ..." for k, v in props.items() if k not in required]
        return f"mcp.{self.capability}({', '.join(parts)})"

    def line(self) -> str:
        about = " ".join(str(self.spec.get("description") or self.spec.get("title") or "").split())[:TEXT_SHOWN]
        if not self.read_only:
            about = f"not from here, it can change things: call the tool {self.route}. {about}"
        return f"{self.signature()}  # {about}" if about else self.signature()


# ------------------------------------------------------------------------------------------------ the adapter
class McpAdapter:
    def __init__(self, hub, names=None, policy=None, cancel=None):
        """`hub`: serve/mcp.py's McpHub.  `names`: the tools this request offered, by the names of the tool path (None: every
        MCP tool of the hub).  `policy`: the chat's serve/permissions.py rules, when it has them.  `cancel`: the chat's stop."""
        self.hub, self.names, self.policy, self.cancel = hub, names, policy, cancel
        self.turn_calls = self.block_calls = 0
        self.deadline = None

    def begin_turn(self):
        self.turn_calls = 0

    def begin_block(self, seconds: float):
        self.block_calls = 0
        self.deadline = time.monotonic() + seconds

    def tools(self) -> dict[str, _Tool]:
        """<server>.<tool> -> what this request may reach.  The hub's built-in servers are not MCP servers: the coding tools have
        their own gate and need a chat's context, and the skills have their own call."""
        out, servers, tools = {}, {}, {}
        for route, (server, name) in self.hub.routes().items():
            if getattr(server, "kind", "") == "builtin" or (self.names is not None and route not in self.names):
                continue
            if server.name not in servers:
                servers[server.name] = _identifier(server.name, servers.values(), RESERVED)
                tools[server.name] = set()
            py = _identifier(name, tools[server.name])
            tools[server.name].add(py)
            capability = f"{servers[server.name]}.{py}"
            out[capability] = _Tool(capability, route, server, name, next(x for x in server.tools if x["name"] == name))
        return out

    def search(self, query) -> MIPResult:
        """The tools that match, best first, each as a line with its signature, so a program finds a tool on demand and no list
        of them has to sit in the prompt.  A name counts double a word of the description."""
        terms = str(query or "").lower().split()
        scored = []
        for tool in self.tools().values():
            head = f"{tool.capability} {tool.route}".lower()
            body = " ".join([str(tool.spec.get("description") or ""), tool.signature()]).lower()
            score = sum(2 if t in head else 1 if t in body else 0 for t in terms)
            if score or not terms:
                scored.append((-score, tool.capability, tool))
        scored.sort(key=lambda x: x[:2])
        lines = [tool.line() for _, _, tool in scored[:TOOLS_SHOWN]]
        if len(scored) > TOOLS_SHOWN:
            lines.append(f"... and {len(scored) - TOOLS_SHOWN} more: narrow the query")
        return MIPResult("search", "success", lines, result_provenance={"found": len(scored)})

    def _spent(self) -> str | None:
        if self.block_calls >= CALLS_PER_BLOCK:
            return f"this block has made its {CALLS_PER_BLOCK} tool calls: use what they gave, or go on in a new block"
        if self.turn_calls >= CALLS_PER_TURN:
            return f"this turn has made its {CALLS_PER_TURN} tool calls: answer from what they gave"
        if self.deadline is not None and self.deadline - time.monotonic() <= 0:
            return "this block has used its time for tools"
        return None

    def invoke(self, invocation: MIPInvocation) -> MIPResult:
        """Everything is decided here, on the server's side of the box: a program can write anything down its pipe."""
        def result(status, content, **provenance):
            if status != "success":
                content = " ".join(str(content).split())[:REASON_CHARS]     # one line: the block's error shows the last line of it
            return MIPResult(invocation.invocation_id, status, content,
                             result_provenance={"capability": invocation.capability_id, **provenance})

        if self.cancel is not None and self.cancel.is_set():
            return result("cancelled", "the request was stopped")
        tool = self.tools().get(invocation.capability_id)
        if tool is None:
            return result("unsupported", f"no tool {invocation.capability_id} here: mcp.tools(\"words\") finds them")
        arguments = invocation.typed_arguments
        if self.policy is not None:
            decision = permissions.decide(tool.route, arguments, self.policy)
            if decision.kind == "deny":
                return result("refused", decision.why)
        if not tool.read_only:
            return result("refused", f"{tool.capability} does not say readOnlyHint: true, and a call from the reasoning authorizes "
                                     f"nothing: end the thinking and call the tool {tool.route}")
        if not isinstance(arguments, dict):
            return result("refused", "the arguments must be an object of names and values")
        why = check_arguments(tool.schema, arguments)
        if why is None:
            try:
                json.dumps(arguments, allow_nan=False)
            except ValueError:
                why = "the arguments hold a number JSON has no form for (nan or inf)"
        if why:
            return result("refused", f"{tool.capability}: {why}. It is {tool.signature()}")
        spent = self._spent()
        if spent:
            return result("refused", spent)
        self.block_calls += 1
        self.turn_calls += 1
        t0 = time.monotonic()
        with timeline.span("mip.call", self.turn_calls):
            status, content, extra = self._run(tool, arguments)
        ms = int((time.monotonic() - t0) * 1000)
        print(f"[strata] mip {tool.capability}: {status} in {ms / 1000:.1f} s", flush=True)
        return result(status, content, server=tool.server.name, tool=tool.name, ms=ms, **extra)

    def _run(self, tool: _Tool, arguments: dict) -> tuple[str, object, dict]:
        """(status, content, what to record of it)"""
        timeout = float(self.hub.settings["timeout_s"])
        if self.deadline is not None:
            timeout = min(timeout, self.deadline - time.monotonic())
        try:
            res = tool.server.call(tool.name, arguments, timeout, self.cancel)
        except McpCancelled:
            return "cancelled", "the request was stopped", {}
        except (McpTimeout, McpRpcError) as e:                     # the deadline's words, or the server's: neither quotes the connection
            return "fault", str(e), {}
        except McpError as e:
            print(f"[strata] mip {tool.capability}: {e}", flush=True)
            return "fault", f"{tool.server.name} did not answer: it stopped, or cannot be reached", {}
        text = result_text(res)
        if res.get("isError"):
            return "fault", text or "the tool reported an error", {}
        value = res.get("structuredContent")
        structured = isinstance(value, (dict, list))
        if structured:
            text = json.dumps(value, ensure_ascii=False)
        cap = int(self.hub.settings["max_result_chars"])
        over = cap > 0 and len(text) > cap
        return "success", value if structured and not over else cut_text(text, cap), {"chars": len(text), "truncated": over}       # half a dict cannot be read as one

    def answer(self, request) -> dict:
        """One request from the kernel -> its result in the JSON form."""
        if not isinstance(request, dict):
            return MIPResult("?", "unsupported", "not a request").to_json()
        if request.get("op") == "tools":
            return self.search(request.get("query")).to_json()
        if request.get("op") != "call":
            return MIPResult("?", "unsupported", "not a request").to_json()
        capability = f"{request.get('server')}.{request.get('tool')}"
        return self.invoke(MIPInvocation("mcp", capability, request.get("arguments"), adapter_binding="mcp",
                                         argument_provenance="model_program")).to_json()
