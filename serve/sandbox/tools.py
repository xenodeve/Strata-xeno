"""serve/sandbox/tools.py - the sandbox as a built-in server of the MCP hub (the design's first route, 4.5), with the names of Serena for what reads and changes by symbol (4.9).

Everything but `sandbox_apply` acts on the copy, so nothing here asks anyone: the operating system's wall is what holds, not a question (design 12). `sandbox_apply` is the one way to the real project: the change that was tested goes
through the rules of the chat, the user's answer and the prompt's checkpoint, the model does not type it again, and a file that moved in the real project stops it. A file changed on both sides is shown by `sandbox_conflict`, which
reads the real project and nothing else does, and settled by `sandbox_resolve`.

Every result starts with the same line (I5), a refusal too, so the hub is told not to put anything in front of it. The tools are named as they are written here (`plain_names`): `output.py` tells the model to call `sandbox_read`, and
that is the name it has to see."""
from __future__ import annotations

import collections
import os
import re
import threading
from dataclasses import replace

from serve import agent, permissions
from serve.sandbox import session as session_mod, testrun
from serve.sandbox.isolation import IsolationError
from serve.sandbox.output import OUTPUT_DIR, conflict_lines, header
from serve.sandbox.search import clean_path
from serve.sandbox.session import Refused, SandboxSession
from serve.sandbox.workspace import ApplyRefused, ConflictMoved, Plan, WorkspaceError

MAX_SESSIONS = 16                      # chats with a copy on disk; past it the one used longest ago closes, and its change goes with it
MAX_CARD_DIFF = 40_000                 # characters of the diff on the question to the user
BOX_LIBRARIES = "sympy, numpy, mpmath, networkx, pytest"
STILL_RUNNING = "ยังรันอยู่"
CHANGES_COPY ={"sandbox_run", "sandbox_edit", "sandbox_resolve", "sandbox_apply", "replace_body", "insert_before", "insert_after"}     # none of these while tests run: the tree under them has to stand still

_NO_INPUT = re.compile(r"\bEOFError\b")
_NO_MODULE = re.compile(r"No module named '?([\w.]+)")


def _schema(name: str, description: str, properties: dict, required: list) -> dict:
    return {"name": name, "description": description, "inputSchema": {"type": "object", "properties": properties, "required": required}}


_S = {"type": "string"}
_I = {"type": "integer"}
_B = {"type": "boolean"}
TOOLS = [
    _schema("sandbox_run", "Run one command or some Python in the sandbox's copy of the project. It has no network and cannot reach the real project. Each call is a new process.",
            {"command": {**_S, "description": "one program and its arguments, such as python script.py: no shell, so no | or &&"}, "code": {**_S, "description": "Python to run instead, with the project importable"},
             "stdin": {**_S, "description": "what the program reads as input"}}, []),
    _schema("sandbox_read", "Read a piece of a file of the copy. The result says where to go on.", {"path": _S, "offset": {**_I, "description": "byte to start at"}, "limit": _I}, ["path"]),
    _schema("sandbox_grep", "Search the copy's files with a regular expression.", {"pattern": _S, "glob": {**_S, "description": "only files like this, such as *.py"}}, ["pattern"]),
    _schema("sandbox_edit", "Change a file of the copy: replace old with new (old must appear once unless replace_all; an empty old makes a new file), or delete it, or rename it.",
            {"path": _S, "old": _S, "new": _S, "replace_all": _B, "delete": _B, "rename_to": _S}, ["path"]),
    _schema("sandbox_test", "Run the project's tests on the copy and the same tests on the project before your change, and say which fail because of it. Call it again with no arguments when it says it is still running.",
            {"command": {**_S, "description": "the test command; the last one, or pytest, when left out"}, "simulated": {**_S, "description": "what the test run replaced, such as: no GPU, fake model"}}, []),
    _schema("sandbox_diff", "Show what the copy changes against the project, and where this sandbox stands.", {}, []),
    _schema("sandbox_apply", "Ask to put the tested change into the real project. You do not type it again. It stops when a file moved in the real project.",
            {"untested": {**_B, "description": "ask the user to take a change with no passing test"}}, []),
    _schema("sandbox_conflict", "Show a file changed on both sides: the real project's change and the copy's, each against the base. With no path, list such files.", {"path": _S}, []),
    _schema("sandbox_resolve", "Settle a file changed on both sides. take=real puts the real file in the copy. take=copy keeps your change on top of the real file; where both changed the same lines yours win. Run sandbox_test again after it.",
            {"path": _S, "take": {"type": "string", "enum": ["real", "copy"]}, "seen": {**_S, "description": "the seen value sandbox_conflict gave"}}, ["path", "take", "seen"]),
    _schema("tree", "List the copy's folders and files.", {"path": _S, "depth": _I}, []),
    _schema("find_file", "Find files of the copy by a glob.", {"glob": _S}, ["glob"]),
    _schema("search", "Search the copy's files with a regular expression.", {"pattern": _S, "glob": _S}, ["pattern"]),
    _schema("outline", "List the symbols of a file of the copy.", {"path": _S, "depth": _I}, ["path"]),
    _schema("find_symbol", "Find a symbol of the copy, as name, Class/name or the ref an answer gave. body gives its text.", {"name": _S, "body": _B}, ["name"]),
    _schema("replace_body", "Replace a whole symbol of the copy: what find_symbol with body shows.", {"symbol": _S, "text": _S}, ["symbol", "text"]),
    _schema("insert_before", "Put lines above a symbol of the copy.", {"symbol": _S, "text": _S}, ["symbol", "text"]),
    _schema("insert_after", "Put lines below a symbol of the copy.", {"symbol": _S, "text": _S}, ["symbol", "text"]),
]


def _ok(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": False}


def _err(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": True}


def _bare(text: str) -> dict:
    """A refusal from before there is a copy."""
    return _err(f"{header()}\n{text}\n")


def _whole(value, name: str, low: int, default):
    if value is None:
        return default
    number, why = agent._int(value, name, low)
    if why:
        raise Refused(why)
    return number


def _text(value, name: str, default=None) -> str:
    if value is None and default is not None:
        return default
    if not isinstance(value, str):
        raise Refused(f"{name} must be text")
    return value


def _near(found: dict) -> list[str]:
    return [f"ใกล้เคียง: {', '.join(found['near'])}"] if found.get("near") else []


class SandboxServer:
    """Looks like an McpServer to the hub (name, kind, status, tools, call, start, close) but has no process and no transport."""

    name = "sandbox"
    kind = "builtin"
    wants_context = True                                        # the hub passes the request's AgentContext to call()
    plain_names = True
    fixed_first_line = True

    def __init__(self, conf: dict, root=None, log_dir=None, start=None):
        self.conf, self.root, self.log_dir, self._start = conf, root, log_dir, start
        self.status = "ready"
        self.error = None
        self.info = {"name": "sandbox", "title": "A copy of the project to try changes in"}
        self.transport = None
        self.lock = threading.Lock()
        self.last_start = 0.0
        self.tools = list(TOOLS)
        self._sessions: collections.OrderedDict[tuple, SandboxSession] = collections.OrderedDict()
        self._applying = threading.Lock()                       # one apply at a time in this server: two chats on one project meet here, and the second finds the first's files
        self._swept = False
        self._run = {"sandbox_run": self._run_code, "sandbox_read": self._read, "sandbox_grep": self._grep, "search": self._grep, "sandbox_edit": self._edit, "sandbox_test": self._test, "sandbox_diff": self._diff,
                     "sandbox_apply": self._apply, "sandbox_conflict": self._conflict, "sandbox_resolve": self._resolve, "tree": self._tree, "find_file": self._find_file, "outline": self._outline,
                     "find_symbol": self._find_symbol, "replace_body": self._by_symbol, "insert_before": self._by_symbol, "insert_after": self._by_symbol}

    # -------------------------------------------------------------------------------------------- the hub's side
    def start(self) -> bool:
        return True

    def close(self) -> None:
        with self.lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for s in sessions:
            s.close()

    def forget(self, chat: str) -> None:
        """The chat was deleted: its copy, what runs in it and its record go with it."""
        with self.lock:
            gone = [k for k in self._sessions if k[0] == chat]
            sessions = [self._sessions.pop(k) for k in gone]
        for s in sessions:
            s.close()
        if self.log_dir:
            session_mod.forget_log(self.log_dir, chat)

    def _session(self, ctx) -> SandboxSession:
        key = (ctx.session, os.path.normcase(os.path.realpath(ctx.policy.cwd)))
        old = None
        with self.lock:
            s = self._sessions.get(key)
            if s is None or s.closed:
                kw = {"start": self._start} if self._start else {}
                s = self._sessions[key] = SandboxSession(ctx.session, os.path.realpath(ctx.policy.cwd), self.conf, self.root, self.log_dir, **kw)
                if len(self._sessions) > MAX_SESSIONS:
                    old = self._sessions.popitem(last=False)[1]
            self._sessions.move_to_end(key)
        if old is not None:
            old.close()
        return s

    def _sweep_once(self) -> None:
        """After the first copy is made, so that the two never both make the scratch root."""
        with self.lock:
            sweep, self._swept = not self._swept, True
        if sweep:
            threading.Thread(target=session_mod.sweep, args=(self.root,), daemon=True, name="sandbox-sweep").start()

    def call(self, tool: str, arguments: dict, timeout: float = 0, cancel=None, ctx=None) -> dict:
        if ctx is None:
            return _bare("the sandbox runs only inside a chat that says which project folder and rules apply")
        run = self._run.get(tool)
        if run is None:
            return _bare(f"there is no tool named {tool!r} in the sandbox")
        if not ctx.policy.cwd:
            return _bare("this chat has no project folder, so there is nothing to copy")
        s = self._session(ctx)
        try:
            s.open(ctx.emit)
            self._sweep_once()
            s.begin(ctx)
            if tool in CHANGES_COPY and s.busy():
                raise Refused(f"[{STILL_RUNNING}] tests are running on the copy: call sandbox_test to wait for them")
            return run(s, arguments if isinstance(arguments, dict) else {}, ctx, tool)
        except (Refused, WorkspaceError, IsolationError, OSError, ValueError) as e:
            return self._fail(s, str(e))

    @staticmethod
    def _fail(s: SandboxSession, text: str) -> dict:
        return _err(s.render(text)) if s.out is not None else _bare(text)

    # -------------------------------------------------------------------------------------------- reading and finding
    def _read(self, s: SandboxSession, a: dict, ctx, _tool) -> dict:
        path = _text(a.get("path"), "path")
        offset = _whole(a.get("offset"), "offset", 0, 0)
        room = max(64, min(s.out.per_result, s.out.left) - len(header().encode()) - 96)     # the header and the line that says where to go on: a read that fits is not cut
        got = s.search.read(path, offset, min(_whole(a.get("limit"), "limit", 1, room), room))
        if not got["ok"]:
            left_out = s.ws.excluded.get(clean_path(path))
            return self._fail(s, "\n".join([got["error"], *([f"left out of the copy: {left_out}"] if left_out else []), *_near(got)]))
        if "kind" in got:
            return _ok(s.render(f"{got['path']}: {got['size']:,} bytes, {got['kind']}: not shown"))
        body = got["text"]
        if got.get("next") is not None:
            body += f"\n[ต่อ: sandbox_read(\"{got['path']}\", offset={got['next']}) จาก {got['size']:,} ไบต์]"
        return _ok(s.render(body))

    def _grep(self, s: SandboxSession, a: dict, ctx, _tool) -> dict:
        got = s.search.search(_text(a.get("pattern"), "pattern"), a.get("glob") if isinstance(a.get("glob"), str) else None)
        if not got["ok"]:
            return self._fail(s, got["error"])
        rows = [f"{h['path']}:{h['line']}: {h['text']}" for h in got["hits"]]
        skipped = got["skipped"]
        if got["truncated"]:
            rows.append(f"[more than {len(rows)} matches: narrow it with glob]")
        if skipped["generated"] or skipped["binary"]:
            rows.append(f"[not searched: {skipped['generated']} generated, {skipped['binary']} binary]")
        return _ok(s.render("\n".join(rows) or "no match"))

    def _tree(self, s: SandboxSession, a: dict, ctx, _tool) -> dict:
        got = s.search.tree(_text(a.get("path"), "path", "."), _whole(a.get("depth"), "depth", 1, 2))
        return _ok(s.render(got["text"])) if got["ok"] else self._fail(s, "\n".join([got["error"], *_near(got)]))

    def _find_file(self, s: SandboxSession, a: dict, ctx, _tool) -> dict:
        got = s.search.find_file(_text(a.get("glob"), "glob"))
        return _ok(s.render("\n".join(got["paths"]) or "no file like that in the copy"))

    def _outline(self, s: SandboxSession, a: dict, ctx, _tool) -> dict:
        got = s.symbols.outline(_text(a.get("path"), "path"), _whole(a.get("depth"), "depth", 1, None))
        if not got["ok"]:
            return self._fail(s, "\n".join([got["error"], *_near(got)]))
        return _ok(s.render("\n".join(f"{r['line']}-{r['end']} {r['kind']} {r['name']}" for r in got["symbols"]) or "no symbols"))

    @staticmethod
    def _symbol_rows(rows: list[dict]) -> list[str]:
        return [f"{r['ref']} {r['kind']}" + (f"\n{r['body']}" if "body" in r else "") for r in rows]

    def _find_symbol(self, s: SandboxSession, a: dict, ctx, _tool) -> dict:
        got = s.symbols.find_symbol(_text(a.get("name"), "name"), a.get("body") is True)
        if not got["ok"]:
            return self._fail(s, got["error"])
        hits = [f"{h['path']}:{h['line']}: {h['text']}  [text, not a confirmed symbol]" for h in got["text_hits"]]
        return _ok(s.render("\n".join([*self._symbol_rows(got["matches"]), *hits, *([got["note"]] if got.get("note") else [])])))

    # -------------------------------------------------------------------------------------------- changing the copy
    def _by_symbol(self, s: SandboxSession, a: dict, ctx, tool: str) -> dict:
        symbol, text = _text(a.get("symbol"), "symbol"), _text(a.get("text"), "text")
        got = getattr(s.symbols, tool)(symbol, text)
        if not got["ok"]:
            return self._fail(s, "\n".join([got["error"], *self._symbol_rows(got.get("matches", []))]))
        return _ok(s.render(f"{got['path']}: changed at line {got['line']}", touched=True))

    def _edit(self, s: SandboxSession, a: dict, ctx, tool) -> dict:
        rel = clean_path(_text(a.get("path"), "path"))
        try:
            return self._edit_file(s, a, rel)
        except FileNotFoundError:
            near = s.search.near(rel)
            return self._fail(s, "\n".join([f"no such file in the copy: {rel}", *([f"ใกล้เคียง: {', '.join(near)}"] if near else [])]))

    def _edit_file(self, s: SandboxSession, a: dict, rel: str) -> dict:
        if a.get("delete") is True:
            s.ws.delete(rel)
            return _ok(s.render(f"{rel}: deleted in the copy", touched=True))
        if a.get("rename_to") is not None:
            new = clean_path(_text(a.get("rename_to"), "rename_to"))
            s.ws.rename(rel, new)
            return _ok(s.render(f"{rel} -> {new} in the copy", touched=True))
        old, new = _text(a.get("old"), "old"), _text(a.get("new"), "new")
        if old == "":
            if os.path.lexists(os.path.join(s.ws.copy, *rel.split("/"))):
                return self._fail(s, f"{rel} is already in the copy: give the text to replace")
            s.ws.write_bytes(rel, new.encode("utf-8"))
            return _ok(s.render(f"{rel}: made in the copy", touched=True))
        if old == new:
            return self._fail(s, "old and new are the same")
        raw = s.ws.read_bytes(rel, s.ws.max_file + 1)
        find, put = old.encode("utf-8"), new.encode("utf-8")
        if b"\r\n" in raw:                                                    # the model writes \n; the file keeps its own line ends
            find, put = (x.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n") for x in (find, put))
        count = raw.count(find)
        if count == 0:
            return self._fail(s, f"{rel}: that text is not in the file")
        if count > 1 and a.get("replace_all") is not True:
            return self._fail(s, f"{rel}: that text is in the file {count} times: give more of it, or set replace_all")
        s.ws.write_bytes(rel, raw.replace(find, put))
        return _ok(s.render(f"{rel}: {count} {'place' if count == 1 else 'places'} changed", touched=True))

    # -------------------------------------------------------------------------------------------- running
    def _run_code(self, s: SandboxSession, a: dict, ctx, _tool) -> dict:
        s.may_run()
        code, command = a.get("code"), a.get("command")
        if isinstance(code, str) == isinstance(command, str):
            raise Refused("give one of command or code")
        lines = s.fresh()
        if isinstance(code, str):
            rel = f"{OUTPUT_DIR}/run.py"
            s.ws.write_bytes(rel, code.encode("utf-8"))
            argv = ["python", "-c", f"import runpy; runpy.run_path({rel!r}, run_name='__main__')"]        # `-c`, for the copy's root on the import path: a script would have its own folder there
        else:
            argv = testrun.prepared(testrun.split_command(command))[0]
        ctx.emit({"event": "sandbox", "step": "trying"})
        stdin = a.get("stdin") if isinstance(a.get("stdin"), str) else None
        r = s.run_in_copy(argv, stdin, s.conf["run_seconds"], ctx.cancel.is_set)
        out = r.stdout.decode("utf-8", "replace")
        err = r.stderr.decode("utf-8", "replace")
        lines.append(out + ("" if not out or out.endswith("\n") or not err else "\n") + err)
        if r.killed:
            lines.append(f"[ถูกหยุด: {session_mod.STOPPED_BY.get(r.killed, r.killed)}]")
        missing = _NO_MODULE.search(err)
        if missing:
            lines.append(f"[box ไม่มี {missing[1]} และไม่มี pip หรือ network ใน box มี: {BOX_LIBRARIES}]")
        if _NO_INPUT.search(err) and not stdin:
            lines.append("[โปรแกรมอ่าน stdin แต่ไม่มีอะไรให้อ่าน: ส่งผ่าน stdin]")
        s.note("run", command=command, code=code, exit_code=r.exit_code, killed=r.killed)
        return _ok(s.render("\n".join(x for x in lines if x), r.exit_code, r.wall_ms / 1000, touched=True))

    def _test(self, s: SandboxSession, a: dict, ctx, _tool) -> dict:
        s.may_run()
        simulated = a.get("simulated") if isinstance(a.get("simulated"), str) and a["simulated"].strip() else None
        lines = s.fresh()
        done = s.test(a.get("command") if isinstance(a.get("command"), str) else None, simulated, ctx.emit, ctx.cancel)
        if done is None:
            return _ok(s.render("\n".join([*lines, f"[{STILL_RUNNING}] call sandbox_test again to wait for the result. Reading works meanwhile; changing the copy does not."])))
        return _ok(s.render("\n".join([*lines, *done.lines]), done.exit_code, done.seconds, touched=True))

    def _diff(self, s: SandboxSession, a: dict, ctx, _tool) -> dict:
        lines = [*s.fresh(), s.summary()]
        stat = s.ws.diff_stat()
        if stat["files"]:
            lines.append(f"{stat['files']} ไฟล์, +{stat['added']} -{stat['removed']}")
            lines.append(s.ws.diff())
        return _ok(s.render("\n".join(lines)))

    # -------------------------------------------------------------------------------------------- a file changed on both sides
    def _conflict(self, s: SandboxSession, a: dict, ctx, _tool) -> dict:
        lines = s.fresh()
        if a.get("path") is None:
            return _ok(s.render("\n".join(lines) or conflict_lines(sorted(s.ws.conflicts)) or "no file is changed on both sides"))
        view = s.ws.conflict(clean_path(_text(a.get("path"), "path")))
        if view is None:
            return _ok(s.render(f"{a['path']}: it is in no conflict now"))
        s.note("conflict", path=view["path"], seen=view["seen"])
        real = f"ของจริงเทียบ base: ไม่แสดง เพราะ {view['hidden']}" if view["hidden"] else f"ของจริงเทียบ base:\n{view['real']}"
        return _ok(s.render("\n".join([conflict_lines([view["path"]]), f"seen: {view['seen']}", real, f"สำเนาเทียบ base:\n{view['copy']}"])))

    def _resolve(self, s: SandboxSession, a: dict, ctx, _tool) -> dict:
        rel = clean_path(_text(a.get("path"), "path"))
        try:
            with s.lock:
                done = s.ws.resolve(rel, _text(a.get("take"), "take"), _text(a.get("seen"), "seen"))
        except ConflictMoved as e:
            return self._fail(s, f"[เปลี่ยนในของจริงอีกครั้ง: {rel}]\n{e}. Look again with sandbox_conflict before you decide.")
        s.note("resolve", **done)
        if done["take"] == "real":
            said = f"{rel}: the copy now holds the real file; the copy's change to it is gone"
        elif done["overlap"]:
            said = f"{rel}: your change now sits on the real file; both sides changed the same lines, and yours were kept there"
        elif done["merged"]:
            said = f"{rel}: your change now sits on the real file, with the real side's change kept"
        else:
            said = f"{rel}: your version stays, now measured against the real file"
        return _ok(s.render(said + "\nsandbox_test again: this is not the change that was tested", touched=True))

    # -------------------------------------------------------------------------------------------- to the real project
    @staticmethod
    def _standing(s: SandboxSession, plan: Plan) -> tuple[bool, str]:
        """Whether the plan's test record lets it through, and in words what it stands on."""
        t = plan.tested
        if t is None:
            return False, "no test ran on exactly this change"
        if t.get("project") != s.ws.base_id():
            return False, f"the project moved after this change was tested ({t['summary']})"
        return t.get("verdict") in testrun.PASSES, t["summary"]

    @staticmethod
    def _card(s: SandboxSession, plan: Plan, standing: str, good: bool, judgeable: bool) -> dict:
        stat = s.ws.diff_stat()
        diff = s.ws.diff()
        if len(diff) > MAX_CARD_DIFF:
            diff = diff[:MAX_CARD_DIFF] + "\n[the diff is cut here]"
        rows = [f"{w.path} ({'delete' if w.action == 'delete' else 'write'})" for w in plan.writes]
        tests = testrun.touches_tests([w.path for w in plan.writes])
        notes = [*([f"[จำลอง: {plan.tested['simulated']}]"] if plan.tested and plan.tested.get("simulated") else []),
                 *([f"[การแก้นี้แก้ไฟล์ test ด้วย: {', '.join(tests[:4])}]"] if tests else []),
                 *([f"คำสั่งที่ใช้ test: {plan.tested['command']} (ผ่านแค่คำสั่งนี้ ไม่ได้แปลว่าถูก)"] if plan.tested else [])]
        arguments = {"files": rows, "stat": f"{stat['files']} ไฟล์, +{stat['added']} -{stat['removed']}", "tested": standing, "notes": notes, "diff": diff}
        why = "it changes the real project" if good else "it changes the real project, and it did not pass a test"
        return {"tool": "sandbox_apply", "arguments": arguments, "why": why, "danger": False, "judgeable": judgeable, "rule": None}

    @staticmethod
    def _after(scope, path: str) -> None:
        try:
            scope.after(path)
        except Exception:  # noqa: BLE001 - what the way back could not note must not undo a change the user allowed
            pass

    def _write(self, s: SandboxSession, plan: Plan, scope) -> tuple[list, str | None]:
        """Writes the plan a file at a time, each one checked against what the project must hold right before it. (what was done, why it stopped)."""
        done = []
        try:
            for w in plan.writes:
                s.ws.recheck(replace(plan, writes=[w]))
                if w.action == "write":
                    os.makedirs(os.path.dirname(w.full), exist_ok=True)
                    agent.write_bytes_atomic(w.full, w.data)
                else:
                    os.remove(w.full)
                done.append(w)
                self._after(scope, w.full)
        except (OSError, ApplyRefused) as e:
            return done, str(e)
        return done, None

    def _undo(self, s: SandboxSession, done: list, scope) -> list[str]:
        """Puts back what a plan that stopped half way wrote. The kept bytes are what the project held: each step was checked against them. What could not be put back is named."""
        left = []
        for w in reversed(done):
            try:
                old = s.ws.base_bytes(w.path)
                if old is None:
                    os.remove(w.full)
                else:
                    os.makedirs(os.path.dirname(w.full), exist_ok=True)
                    agent.write_bytes_atomic(w.full, old)
                self._after(scope, w.full)
            except OSError:
                left.append(w.path)
        return left

    def _hooks_before(self, plan: Plan, ctx) -> str | None:
        """The user's own commands see each file as they see a Write: a hook that guards a file holds for this route as well."""
        hooks = ctx.hooks
        if not hooks:
            return None
        stopped = hooks.before("sandbox_apply", {"files": [w.path for w in plan.writes]})
        for w in plan.writes:
            if not stopped and w.action == "write":
                stopped = hooks.before("Write", {"file_path": w.full, "content": w.data.decode("utf-8", "replace")})
        return stopped

    def _apply(self, s: SandboxSession, a: dict, ctx, _tool) -> dict:
        lines = s.fresh()
        try:
            with s.lock:
                plan = s.ws.apply_plan()
        except ApplyRefused as e:
            why = [f"{r['path']}: {r['why']}" for r in e.reasons]
            return self._fail(s, "\n".join([*lines, "the change cannot go to the real project, and nothing was written", *why, "for a file that changed there: sandbox_conflict, then sandbox_resolve"]))
        if not plan.writes:
            return _ok(s.render("\n".join([*lines, "the copy changes nothing, so there is nothing to apply"])))
        paths = [w.full for w in plan.writes]
        decision = permissions.decide_apply(paths, ctx.policy)
        if decision.kind == "deny":
            return self._fail(s, f"sandbox_apply is denied: {decision.why}")
        good, standing = self._standing(s, plan)
        if not good and a.get("untested") is not True:
            return self._fail(s, "\n".join([*lines, f"not applied: {standing}", "run sandbox_test, or call sandbox_apply with untested=true to ask the user to take it as it is"]))
        if ctx.checkpoint is None:
            return self._fail(s, "not applied: this prompt has no checkpoint, so the change would have no way back (design I6)")
        stopped = self._hooks_before(plan, ctx)
        if stopped:
            return self._fail(s, f"sandbox_apply was stopped by a hook: {stopped}")
        card = self._card(s, plan, standing, good, decision.judgeable and good)
        if decision.kind == "ask" or not good:
            answer = ctx.ask(card)
            if isinstance(answer, str) and answer.startswith("blocked:"):
                return self._fail(s, f"sandbox_apply was blocked: {answer[8:].strip()}. Do not retry it; ask the user.")
            if answer == "cancelled":
                return self._fail(s, "the request was cancelled before the user answered")
            if answer not in ("allow", "allow_chat"):
                return self._fail(s, "the user did not allow this change, so nothing was written. Ask them what they want instead.")
        ctx.emit({"event": "sandbox", "step": "applying"})
        with self._applying:                                                  # after the answer, which can take minutes: this is the moment the project is checked for what happened meanwhile
            try:
                s.ws.recheck(plan)
            except ApplyRefused as e:
                return self._fail(s, "\n".join(["the real project moved while the question was open, and nothing was written", *[f"{r['path']}: {r['why']}" for r in e.reasons], "sandbox_conflict shows it"]))
            why = ctx.checkpoint.before_all(paths)
            if why:
                return self._fail(s, f"not applied: a way back could not be kept, and nothing was written ({why})")
            done, stopped = self._write(s, plan, ctx.checkpoint)
            if stopped:
                left = self._undo(s, done, ctx.checkpoint)
                s.ws.applied(plan, [w for w in done if w.path in left])
                return self._fail(s, f"the change stopped half way: {stopped}. " + (f"These could not be put back: {', '.join(left)}. Rewind this prompt." if left else "What was written was put back."))
            with s.lock:
                s.ws.applied(plan)
                try:
                    s.ws.sync()
                except WorkspaceError:
                    pass                                                      # written and noted; the next command tries again
        s.note("apply", files=[w.path for w in plan.writes], delta=plan.delta, tested=standing, passed=good, diff=card["arguments"]["diff"])
        said = self._hooks_after(plan, ctx)
        body = [f"นำขึ้น project จริงแล้ว {len(plan.writes)} ไฟล์: {', '.join(w.path for w in plan.writes[:6])}", f"ทดสอบ: {standing}", "ย้อนกลับได้ด้วย rewind ของ prompt นี้"]
        return _ok(s.render("\n".join([*body, *([said] if said else [])])))

    @staticmethod
    def _hooks_after(plan: Plan, ctx) -> str:
        hooks = ctx.hooks
        if not hooks:
            return ""
        told = [hooks.after("Write", {"file_path": w.full}, _ok("written")) for w in plan.writes if w.action == "write"]
        return "\n\n".join(x for x in told if x)
