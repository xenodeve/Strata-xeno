"""The sandbox's tool layer (serve/sandbox/session.py, tools.py, testrun.py): what a model sees, how a change gets to the real project, and how a file changed on both sides is settled. Every repository here is made by the test,
and the code that runs in a box is the test's own, never a model's.

Design cases (docs/plans/2026-11-02-sandbox-workspace-design.md, section 5) this layer owns and these tests cover:
  3 a library the box lacks, 4 a program that reads stdin, 8 a file read under the budget, 9 big and binary files, 15 the working tree, 16 secrets, 18 change then test, 19 delete and rename, 20 a very large change, 21 a test that writes files,
  22 tests failing before the change, 23 flaky tests, 24 slow tests, 26 no network, 30 a process that hangs, 31 a change to the tests, 32 the user changes a file meanwhile, 34 a branch switch, 35 two chats on one project, 37 the user's Stop,
  38 a new request keeps the change, 39 where the sandbox stands, 41 the first line, 42 a path that does not exist, 44 output that floods, 47 a pass names what passed, 48 a box that cannot start, 49 the check when a box starts,
  54 a network project, 56 a language with no grammar, 58 a name that fits several symbols.
Covered elsewhere: 10-14, 17, 33 (at the next command, not at once), 50, 51, 55, 57, 59, 60, 62 by test_sandbox_workspace and tests/sandbox.
Not covered: 1, 2, 5-7, 25, 27-29, 36 (only the sweep), 40, 43, 45, 46, 52, 53, 61: other layers, or not built."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from serve import agent, checkpoints, permissions  # noqa: E402
from serve.mcp import McpHub  # noqa: E402
from serve.sandbox import session as session_mod, testrun  # noqa: E402
from serve.sandbox.isolation import Isolation, IsolationError, RunResult  # noqa: E402
from serve.sandbox.output import CONFLICT, HEADER  # noqa: E402
from serve.sandbox.session import SandboxSession  # noqa: E402
from serve.sandbox.tools import TOOLS, SandboxServer  # noqa: E402
from serve.test_sandbox_workspace import Base, everything, junction  # noqa: E402

WINDOWS = os.name == "nt"
OFF = session_mod.settings({})
ON = {**OFF, "on": True, "real_python": True}


def no_box(*_a, **_k):
    raise AssertionError("a box was started")


class Case(Base):
    conf = OFF
    start = staticmethod(no_box)
    box_root = None

    def setUp(self):
        super().setUp()
        self.events: list[dict] = []
        self.logs = self.base / "logs"
        self.store = checkpoints.Checkpoints(self.base / "cp")
        self.server = SandboxServer(self.conf, self.box_root or self.base / "sandbox-root", self.logs, start=self.start)
        self.addCleanup(self.server.close)
        self.asked: list[dict] = []
        self.answer = "allow"
        self.mode = None
        self.allow: list[str] = []
        self.deny: list[str] = []
        self.ctx = self.context()

    def context(self, chat: str = "chat1", checkpoint: bool = True, cwd=None, **kw) -> agent.AgentContext:
        def ask(card: dict) -> str:
            self.asked.append(card)
            return self.answer(card) if callable(self.answer) else self.answer
        policy = permissions.Policy(cwd=str(cwd or self.proj), mode=self.mode, allow=self.allow, deny=self.deny)
        return agent.AgentContext(policy=policy, session=chat, ask=ask, emit=self.events.append, checkpoint=self.store.scope(chat, "p1") if checkpoint else None, **kw)

    def result(self, tool: str, ctx=None, **args) -> dict:
        return self.server.call(tool, args, 0, None, ctx or self.ctx)

    def call(self, tool: str, ctx=None, **args) -> str:
        return self.result(tool, ctx, **args)["content"][0]["text"]

    def session(self, chat: str = "chat1") -> SandboxSession:
        return next(s for k, s in self.server._sessions.items() if k[0] == chat)

    def steps(self) -> list[str]:
        return [e["step"] for e in self.events if e.get("event") == "sandbox"]

    def passed(self, verdict: str = "passed", simulated: str | None = None, chat: str = "chat1") -> dict:
        s = self.session(chat)
        record = s.ws.tested_record("python -m pytest -q", 0, "tests: 1 ผ่าน (ก่อนแก้ 1 ผ่าน)", simulated)
        record["verdict"] = verdict
        return record

    def changed_in_copy(self, data: bytes = b"copy\ntwo\nthree\n", rel: str = "a.txt") -> None:
        self.call("sandbox_diff")                                             # any call: it opens the copy
        self.box(self.session().ws, rel, data)

    def conflicted(self, copy: bytes = b"copy\ntwo\nthree\n", real: bytes = b"real\ntwo\nthree\n", rel: str = "a.txt") -> None:
        self.changed_in_copy(copy, rel)
        self.put(rel, real)

    def seen(self, rel: str = "a.txt") -> str:
        return next(x[6:] for x in self.call("sandbox_conflict", path=rel).splitlines() if x.startswith("seen: "))


class TheFirstLine(Case):
    def test_every_tool_starts_with_it_whatever_happened(self):
        calls = {"sandbox_run": {}, "sandbox_read": {"path": "a.txt"}, "sandbox_grep": {"pattern": "one"}, "sandbox_edit": {"path": "a.txt", "old": "one", "new": "1"}, "sandbox_test": {}, "sandbox_diff": {},
                 "sandbox_apply": {}, "sandbox_conflict": {}, "sandbox_resolve": {"path": "a.txt", "take": "real", "seen": "x"}, "tree": {}, "find_file": {"glob": "*.txt"}, "search": {"pattern": "one"},
                 "outline": {"path": "src/b.py"}, "find_symbol": {"name": "f"}, "replace_body": {"symbol": "f", "text": "def f():\n    return 2"},
                 "insert_before": {"symbol": "f", "text": "# x"}, "insert_after": {"symbol": "f", "text": "# y"}}
        self.assertEqual(set(calls), {t["name"] for t in TOOLS}, "every tool is in this test")
        for tool, args in calls.items():
            for bad in (False, True):
                got = self.call(tool, **({"path": 1, "pattern": "(", "glob": 1, "name": "", "symbol": ""} if bad else args))
                self.assertTrue(got.startswith(HEADER), f"{tool}: {got[:80]!r}")

    def test_the_hub_puts_nothing_in_front_of_it_and_keeps_the_names(self):
        hub = McpHub({}, None, {"sandbox": self.server})
        self.assertIn("sandbox_read", hub.routes())
        self.assertIn("find_symbol", hub.routes())
        ok = hub.call("sandbox_read", {"path": "a.txt"}, None, self.ctx)
        refused = hub.call("sandbox_read", {"path": "nope.txt"}, None, self.ctx)
        self.assertTrue(ok["ok"])
        self.assertFalse(refused["ok"])
        self.assertTrue(refused["text"].startswith(HEADER), refused["text"][:60])

    def test_a_call_from_outside_a_chat_or_with_no_folder_is_refused(self):
        for got in (self.server.call("sandbox_diff", {}, 0, None, None), self.server.call("sandbox_diff", {}, 0, None, agent.AgentContext(policy=permissions.Policy(cwd=None), session="x"))):
            self.assertTrue(got["isError"])
            self.assertTrue(got["content"][0]["text"].startswith(HEADER))
        self.assertEqual(len(self.server._sessions), 0)

    def test_a_folder_that_is_not_a_repository_says_so(self):
        got = self.call("sandbox_diff", ctx=self.context(cwd=self.outside))
        self.assertTrue(got.startswith(HEADER))
        self.assertIn("could not start", got)

    @unittest.skipUnless(WINDOWS, "a Windows path")
    def test_a_project_on_a_network_path_is_refused(self):                    # design case 54
        got = self.call("sandbox_diff", ctx=self.context(cwd="\\\\server\\share\\proj"))
        self.assertIn("network path", got)


class ALazySession(Case):
    def test_nothing_exists_until_the_first_call(self):
        self.assertEqual(len(self.server._sessions), 0)
        self.assertFalse((self.base / "sandbox-root").exists(), "no folder either")
        self.call("sandbox_diff")
        self.assertEqual(len(self.server._sessions), 1)
        self.assertTrue(os.path.isdir(self.session().ws.copy))

    def test_one_session_for_a_chat_and_a_folder(self):
        self.call("sandbox_diff")
        self.call("sandbox_diff", ctx=self.context())                         # a new request of the same chat (design case 38): the same copy
        self.assertEqual(len(self.server._sessions), 1)
        self.call("sandbox_diff", ctx=self.context("chat2"))
        self.assertEqual(len(self.server._sessions), 2)

    def test_a_new_request_keeps_the_change(self):                            # design case 38
        self.call("sandbox_edit", path="a.txt", old="one", new="1")
        self.assertIn("a.txt", self.call("sandbox_diff", ctx=self.context()))

    def test_it_closes_with_the_chat(self):
        self.call("sandbox_edit", path="a.txt", old="one", new="1")
        s = self.session()
        s.note("run", command="x")
        copy = s.ws.root
        self.assertTrue(list(self.logs.iterdir()))
        self.server.forget("chat1")
        self.assertFalse(os.path.exists(copy))
        self.assertEqual(len(self.server._sessions), 0)
        self.assertFalse(list(self.logs.iterdir()), "its record goes with it")
        self.assertNotIn("a.txt", self.call("sandbox_diff"), "the next call starts from the project")

    def test_the_server_closing_closes_them_all(self):
        self.call("sandbox_diff")
        root = self.session().ws.root
        self.server.close()
        self.assertFalse(os.path.exists(root))

    def test_past_the_limit_the_one_used_longest_ago_goes(self):
        with mock.patch("serve.sandbox.tools.MAX_SESSIONS", 2):
            for chat in ("a", "b", "c"):
                self.call("sandbox_diff", ctx=self.context(chat))
        self.assertEqual([k[0] for k in self.server._sessions], ["b", "c"])

    def test_a_copy_a_dead_server_left_is_swept_after_its_time(self):
        self.call("sandbox_diff")
        root = Path(self.session().ws.root)
        old = root.parent / "ws-left"
        (old / "copy").mkdir(parents=True)
        (old / "copy" / "x").write_text("x")
        session_mod.sweep(self.base / "sandbox-root", keep=3600)
        self.assertTrue(old.exists(), "it may still be in use")
        ages = time.time() - 7200
        os.utime(old / "copy", (ages, ages))
        os.utime(old, (ages, ages))
        session_mod.sweep(self.base / "sandbox-root", keep=3600)
        self.assertFalse(old.exists())
        self.assertTrue(root.exists(), "this server's own stays")

    def test_a_copy_is_swept_by_whether_its_server_lives_not_by_its_age(self):
        self.call("sandbox_diff")
        root = Path(self.session().ws.root)
        self.assertTrue(root.name.startswith(f"ws-{os.getpid()}-"))
        ended = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True, creationflags=0x08000000 if WINDOWS else 0)
        dead = root.parent / f"ws-{int(ended.stdout)}-left"
        (dead / "copy").mkdir(parents=True)
        long_ago = time.time() - 30 * 86400
        for p in (root, root / "copy", root / "base.pack"):
            os.utime(p, (long_ago, long_ago))                                 # a chat that has not touched its copy for a month: its server lives, so it stays
        session_mod.sweep(self.base / "sandbox-root", keep=3600)
        self.assertFalse(dead.exists(), "a dead server's copy goes at any age")
        self.assertTrue(root.exists())
        self.assertIn("one", self.call("sandbox_read", path="a.txt"))


class Off(Case):
    """The key for real Python is off by default (design D1): no box starts at all, and the rest works."""

    def test_the_settings_are_off_unless_they_say_so(self):
        self.assertFalse(session_mod.settings({})["on"])
        self.assertFalse(session_mod.settings({"sandbox": {"on": 1, "real_python": "yes"}})["real_python"])
        self.assertTrue(session_mod.settings({"sandbox": {"on": True}})["on"])
        self.assertFalse(session_mod.settings({"sandbox": {"on": True}})["real_python"])
        self.assertTrue(session_mod.settings({"sandbox": {"real_python": True}})["real_python"])
        self.assertEqual(session_mod.settings({"sandbox": "on"}), OFF)
        self.assertEqual(session_mod.settings({"sandbox": {"result_bytes": -3}})["result_bytes"], OFF["result_bytes"])

    def test_run_and_test_refuse_and_say_why(self):
        for tool, args in (("sandbox_run", {"command": "python -V"}), ("sandbox_run", {"code": "print(1)"}), ("sandbox_test", {"command": "python -m pytest"})):
            got = self.result(tool, **args)
            self.assertTrue(got["isError"])
            self.assertIn("design D1", got["content"][0]["text"])
            self.assertIn("real_python", got["content"][0]["text"])

    def test_the_rest_works_on_the_copy(self):
        self.assertIn("one", self.call("sandbox_read", path="a.txt"))
        self.assertIn("a.txt:1: one", self.call("sandbox_grep", pattern="one"))
        self.assertIn("1 place changed", self.call("sandbox_edit", path="a.txt", old="one", new="uno"))
        self.assertIn("+uno", self.call("sandbox_diff"))
        self.assertIn("function f", self.call("outline", path="src/b.py"))

    def test_a_throwaway_vm_stays_as_it_was(self):
        self.call("sandbox_diff")
        with mock.patch.object(session_mod, "throwaway", lambda: True):
            self.session().may_run()                                          # it does not raise
        with mock.patch.object(session_mod, "throwaway", lambda: False):
            with self.assertRaises(session_mod.Refused):
                self.session().may_run()


class Reading(Case):
    def test_a_piece_and_how_to_go_on(self):                                  # design case 8
        self.put("long.txt", b"line of text\n" * 400)
        got = self.call("sandbox_read", path="long.txt")
        self.assertLessEqual(len(got.encode()), 1024)
        self.assertIn("sandbox_read(\"long.txt\", offset=", got)
        self.assertNotIn("[cut", got, "a read that fits its budget is not cut and saved")

    def test_reading_on_reaches_the_end(self):
        data = "".join(f"{i}\n" for i in range(600))
        self.put("long.txt", data.encode())
        self.call("sandbox_diff")
        out, offset = "", 0
        for _ in range(30):
            self.session().out.new_turn()
            body = self.call("sandbox_read", path="long.txt", offset=offset).split("\n", 1)[1]
            piece, _, more = body.partition("\n[ต่อ: ")
            out += piece
            if not more:
                break
            offset = int(more.split("offset=")[1].split(")")[0])
        self.assertEqual(out, data)

    def test_binary_and_big_files_give_their_size(self):                      # design case 9
        self.put("blob.bin", b"\0" * 2048)
        got = self.call("sandbox_read", path="blob.bin")
        self.assertIn("2,048 bytes, binary", got)
        self.assertNotIn("\0", got)

    def test_a_path_that_does_not_exist_gets_what_is_near(self):              # design case 42
        got = self.result("sandbox_read", path="b.py")
        self.assertTrue(got["isError"])
        self.assertIn("src/b.py", got["content"][0]["text"])

    def test_what_the_copy_left_out_says_why(self):                           # design case 16
        self.put(".env", b"TOKEN=secret-value\n")
        got = self.call("sandbox_read", path=".env")
        self.assertIn("left out of the copy: it looks like a secret", got)
        self.assertNotIn("secret-value", got)

    def test_the_working_tree_is_what_is_read(self):                          # design case 15
        self.put("a.txt", b"edited and not committed\n")
        self.assertIn("edited and not committed", self.call("sandbox_read", path="a.txt"))

    def test_search_answers_as_sandbox_grep_does(self):
        self.assertEqual(self.call("search", pattern="two", glob="*.txt"), self.call("sandbox_grep", pattern="two", glob="*.txt"))

    def test_a_bad_pattern_is_said(self):
        self.assertIn("not a valid pattern", self.call("sandbox_grep", pattern="("))
        self.assertIn("repeat inside a repeat", self.call("sandbox_grep", pattern="(a+)+b"))

    def test_tree_and_find_file(self):
        self.assertIn("src/ (1 files)", self.call("tree", depth=1))
        self.assertIn("src/b.py", self.call("find_file", glob="*.py"))
        self.assertIn("no file like that", self.call("find_file", glob="*.nothing"))

    def test_a_language_with_no_grammar_says_text_only(self):                 # design case 56
        self.put("notes.xyz", b"some words\n")
        self.assertIn("Only text search", self.call("outline", path="notes.xyz"))

    def test_a_name_that_fits_two_symbols_changes_nothing(self):              # design case 58
        self.put("src/c.py", b"def f():\n    return 3\n")
        before = self.call("find_symbol", name="f")
        self.assertIn("src/b.py:f@1", before)
        self.assertIn("src/c.py:f@1", before)
        got = self.result("replace_body", symbol="f", text="def f():\n    return 9")
        self.assertTrue(got["isError"])
        self.assertIn("nothing was changed", got["content"][0]["text"])
        self.assertIn("src/b.py:f@1", got["content"][0]["text"])
        self.assertIn("changed at line 1", self.call("replace_body", symbol="src/b.py:f@1", text="def f():\n    return 9"))
        self.assertIn("return 9", self.call("find_symbol", name="src/b.py:f", body=True))


class TheBudget(Case):
    def test_a_long_result_is_cut_to_the_budget_and_the_rest_is_saved(self):  # design case 44
        self.put("long.txt", b"needle in a line of text\n" * 200)
        got = self.call("sandbox_grep", pattern="needle")
        self.assertLessEqual(len(got.encode()), 1024)
        saved = got.split('sandbox_read("')[1].split('"')[0]
        self.assertTrue(saved.startswith(".strata-sandbox/output/"))
        self.assertIn("needle", self.call("sandbox_read", path=saved))

    def test_a_turn_runs_out_and_the_next_round_has_its_own(self):
        self.put("long.txt", b"x" * 900 + b"\n")
        ctx = self.context()
        spent = [self.call("sandbox_read", ctx=ctx, path="long.txt") for _ in range(24)]
        self.assertIn("หมดงบการอ่าน", spent[-1])
        self.assertTrue(spent[-1].startswith(HEADER))
        ctx.round += 1
        self.assertNotIn("หมดงบการอ่าน", self.call("sandbox_read", ctx=ctx, path="long.txt"))

    def test_the_run_config_can_set_it(self):
        server = SandboxServer({**OFF, "result_bytes": 400}, self.base / "sandbox-root", start=no_box)
        self.addCleanup(server.close)
        self.put("long.txt", b"needle text\n" * 200)
        self.assertLessEqual(len(server.call("sandbox_grep", {"pattern": "needle"}, 0, None, self.ctx)["content"][0]["text"].encode()), 400)


class Editing(Case):
    def setUp(self):
        super().setUp()
        self.before = everything(self.proj)

    def tearDown(self):
        self.assertEqual(everything(self.proj), self.before, "the real project was not touched")
        super().tearDown()

    def test_a_change_stays_in_the_copy_and_says_so(self):
        got = self.call("sandbox_edit", path="a.txt", old="two", new="2")
        self.assertIn("แก้แล้ว 1 ไฟล์ในสำเนา ยังไม่แตะ project จริง", got)
        self.assertEqual(self.in_copy(self.session().ws, "a.txt"), b"one\n2\nthree\n")

    def test_the_text_has_to_be_there_once(self):
        self.assertIn("not in the file", self.call("sandbox_edit", path="a.txt", old="nine", new="9"))
        self.call("sandbox_edit", path="a.txt", old="three", new="one")
        self.assertIn("2 times", self.call("sandbox_edit", path="a.txt", old="one", new="1"))
        self.assertIn("2 places", self.call("sandbox_edit", path="a.txt", old="one", new="1", replace_all=True))

    def test_crlf_is_kept_and_other_bytes_too(self):                          # design case 14
        self.call("sandbox_diff")
        ws = self.session().ws
        self.box(ws, "w.txt", b"one\r\ntwo\xff\r\nthree\r\n")
        self.call("sandbox_edit", path="w.txt", old="one\nt", new="1\nT")
        self.assertEqual(self.in_copy(ws, "w.txt"), b"1\r\nTwo\xff\r\nthree\r\n")

    def test_a_new_file_and_one_that_exists(self):
        self.assertIn("made in the copy", self.call("sandbox_edit", path="new/n.txt", old="", new="hello\n"))
        self.assertIn("already in the copy", self.call("sandbox_edit", path="a.txt", old="", new="x"))
        self.assertEqual(self.status_of(self.session().ws)["new/n.txt"], "A")

    def test_delete_and_rename_show_in_the_diff(self):                        # design case 19
        self.call("sandbox_edit", path="a.txt", delete=True)
        self.call("sandbox_edit", path="src/b.py", rename_to="src/c.py")
        self.assertEqual(self.status_of(self.session().ws), {"a.txt": "D", "src/c.py": "R"})
        self.assertIn("rename from src/b.py", self.call("sandbox_diff"))

    def test_a_file_that_is_not_there(self):                                  # design case 42
        for args in ({"path": "b.py", "old": "x", "new": "y"}, {"path": "b.py", "delete": True}, {"path": "b.py", "rename_to": "c.py"}):
            got = self.result("sandbox_edit", **args)
            self.assertTrue(got["isError"])
            self.assertIn("src/b.py", got["content"][0]["text"])

    def test_a_path_out_of_the_copy_is_refused(self):
        for bad in ("../outside.txt", ".git/config", "a/../../x"):
            self.assertTrue(self.result("sandbox_edit", path=bad, old="", new="x")["isError"], bad)

    def test_rename_onto_a_file_is_refused(self):
        self.assertIn("there already", self.call("sandbox_edit", path="a.txt", rename_to="src/b.py"))

    def test_symbols_change_the_copy_only(self):
        self.assertIn("changed at line 1", self.call("insert_before", symbol="f", text="# above"))
        self.assertIn("changed at line", self.call("insert_after", symbol="f", text="# below"))
        got = self.in_copy(self.session().ws, "src/b.py")
        self.assertTrue(got.startswith(b"# above\ndef f"), got)
        self.assertTrue(got.endswith(b"# below\n"), got)

    def test_a_folder_is_not_deleted(self):
        self.assertIn("it is a folder", self.call("sandbox_edit", path="src", delete=True))

    def test_where_it_stands_is_in_the_diff(self):                            # design case 39
        self.assertIn("ไม่มีการแก้ในสำเนา", self.call("sandbox_diff"))
        self.call("sandbox_edit", path="a.txt", old="one", new="1")
        self.passed()
        got = self.call("sandbox_diff")
        self.assertIn("แก้แล้ว 1 ไฟล์", got)
        self.assertIn("test ล่าสุด: python -m pytest -q", got)
        self.assertIn("1 ไฟล์, +1 -1", got)

    def test_a_very_large_change_is_counted_and_not_printed(self):            # design case 20
        self.call("sandbox_diff")
        for i in range(30):
            self.box(self.session().ws, f"many/{i}.txt", b"text\n" * 50)
        got = self.call("sandbox_diff")
        self.assertIn("30 ไฟล์, +1500 -0", got)
        self.assertLessEqual(len(got.encode()), 1024)


class Applying(Case):
    def edited(self) -> None:
        self.call("sandbox_edit", path="a.txt", old="one", new="uno")

    def test_a_tested_change_goes_in_after_the_question(self):                # design case 18
        self.edited()
        self.passed()
        got = self.result("sandbox_apply")
        self.assertFalse(got["isError"], got)
        self.assertEqual(self.real("a.txt"), b"uno\ntwo\nthree\n")
        self.assertEqual(len(self.asked), 1)
        card = self.asked[0]
        self.assertEqual(card["tool"], "sandbox_apply")
        self.assertIn("+uno", card["arguments"]["diff"])
        self.assertIn("1 ผ่าน", card["arguments"]["tested"])
        self.assertFalse(card["judgeable"], "not in the default mode")
        self.assertEqual(self.steps()[-1], "applying")
        self.assertIn("นำขึ้น project จริงแล้ว 1 ไฟล์", got["content"][0]["text"])

    def test_afterwards_the_copy_and_the_project_agree(self):
        self.edited()
        self.passed()
        self.call("sandbox_apply")
        self.assertEqual(self.session().ws.changed(), [])
        self.assertIn("nothing to apply", self.call("sandbox_apply"))

    def test_the_bytes_go_as_they_are(self):
        self.call("sandbox_diff")
        data = b"\xff\xfe binary \r\n and \0 zero\n"
        self.box(self.session().ws, "blob.dat", data)
        self.passed()
        self.call("sandbox_apply")
        self.assertEqual(self.real("blob.dat"), data)

    def test_a_delete_and_a_rename_go_too(self):                              # design case 19
        self.call("sandbox_edit", path="a.txt", delete=True)
        self.call("sandbox_edit", path="src/b.py", rename_to="src/deep/c.py")
        self.passed()
        got = self.result("sandbox_apply")
        self.assertFalse(got["isError"], got)
        self.assertFalse((self.proj / "a.txt").exists())
        self.assertFalse((self.proj / "src/b.py").exists())
        self.assertEqual(self.real("src/deep/c.py"), b"def f():\n    return 1\n")

    def test_the_way_back_works(self):
        self.edited()
        self.call("sandbox_edit", path="new.txt", old="", new="made\n")
        self.passed()
        self.call("sandbox_apply")
        got = self.store.apply("chat1", "p1")
        self.assertEqual((got["restored"], got["deleted"], got["kept"]), (1, 1, []))
        self.assertEqual(self.real("a.txt"), b"one\ntwo\nthree\n")
        self.assertFalse((self.proj / "new.txt").exists())

    def test_no_test_record_stops_it_without_a_question(self):
        self.edited()
        got = self.result("sandbox_apply")
        self.assertTrue(got["isError"])
        self.assertIn("no test ran", got["content"][0]["text"])
        self.assertIn("untested=true", got["content"][0]["text"])
        self.assertEqual(self.asked, [])
        self.assertEqual(self.real("a.txt"), b"one\ntwo\nthree\n")

    def test_a_record_of_another_delta_does_not_count(self):
        self.edited()
        self.passed()
        self.call("sandbox_edit", path="a.txt", old="uno", new="ein")
        self.assertIn("no test ran", self.call("sandbox_apply"))

    def test_a_failing_record_stops_it_too(self):
        self.edited()
        self.passed("broke")
        self.assertIn("not applied", self.call("sandbox_apply"))
        self.assertEqual(self.real("a.txt"), b"one\ntwo\nthree\n")

    def test_a_record_from_before_the_project_moved_does_not_count(self):
        self.edited()
        self.passed()
        self.put("src/b.py", b"def f():\n    return 2\n")                     # a file the change does not touch: it follows in silence, so what was tested is not what would go
        got = self.call("sandbox_apply")
        self.assertIn("the project moved after this change was tested", got)
        self.assertEqual(self.real("a.txt"), b"one\ntwo\nthree\n")

    def test_the_user_may_take_it_untested(self):
        self.edited()
        got = self.result("sandbox_apply", untested=True)
        self.assertFalse(got["isError"], got)
        self.assertEqual(self.real("a.txt"), b"uno\ntwo\nthree\n")
        self.assertEqual(self.asked[0]["why"], "it changes the real project, and it did not pass a test")

    def test_auto_mode_may_settle_a_tested_change_and_never_an_untested_one(self):
        self.mode = "auto"
        ctx = self.context()
        self.call("sandbox_edit", ctx=ctx, path="a.txt", old="one", new="uno")
        self.result("sandbox_apply", ctx=ctx, untested=True)
        self.assertFalse(self.asked[-1]["judgeable"])
        self.call("sandbox_edit", ctx=ctx, path="a.txt", old="uno", new="ein")
        self.passed()
        self.result("sandbox_apply", ctx=ctx)
        self.assertTrue(self.asked[-1]["judgeable"])

    def test_the_user_saying_no_writes_nothing(self):
        self.edited()
        self.passed()
        for answer, words in (("deny", "did not allow"), ("cancelled", "cancelled"), ("blocked: too risky", "too risky")):
            self.answer = answer
            got = self.result("sandbox_apply")
            self.assertTrue(got["isError"])
            self.assertIn(words, got["content"][0]["text"])
            self.assertEqual(self.real("a.txt"), b"one\ntwo\nthree\n")

    def test_plan_mode_and_a_deny_rule_stop_it(self):
        self.edited()
        self.passed()
        self.mode = "plan"
        self.assertIn("plan mode", self.call("sandbox_apply", ctx=self.context()))
        self.mode = None
        self.deny.append("sandbox_apply")
        self.assertIn("denied", self.call("sandbox_apply", ctx=self.context()))
        self.deny[:] = [f"Write({self.proj.as_posix()}/a.txt)"]
        self.assertIn("denied", self.call("sandbox_apply", ctx=self.context()))
        self.assertEqual(self.asked, [])
        self.assertEqual(self.real("a.txt"), b"one\ntwo\nthree\n")

    def test_an_allow_rule_takes_the_question_away_and_not_for_an_untested_change(self):
        self.allow.append("sandbox_apply")
        ctx = self.context()
        self.call("sandbox_edit", ctx=ctx, path="a.txt", old="one", new="uno")
        self.result("sandbox_apply", ctx=ctx, untested=True)
        self.assertEqual(len(self.asked), 1)
        self.call("sandbox_edit", ctx=ctx, path="a.txt", old="uno", new="ein")
        self.passed()
        got = self.result("sandbox_apply", ctx=ctx)
        self.assertFalse(got["isError"], got)
        self.assertEqual(len(self.asked), 1)

    def test_an_allow_rule_does_not_silence_a_file_the_rules_ask_about(self):
        self.allow.append("sandbox_apply")
        ctx = self.context()
        ctx.policy.protected = [str(self.proj / "config.json")]
        self.call("sandbox_edit", ctx=ctx, path="config.json", old="", new="{}\n")
        self.passed()
        self.result("sandbox_apply", ctx=ctx)
        self.assertEqual(len(self.asked), 1)

    def test_the_servers_own_config_is_never_settled_by_a_judge(self):
        self.mode = "auto"
        ctx = self.context()
        ctx.policy.protected = [str(self.proj / "config.json")]
        self.call("sandbox_edit", ctx=ctx, path="config.json", old="", new="{}\n")
        self.passed()
        self.result("sandbox_apply", ctx=ctx)
        self.assertFalse(self.asked[-1]["judgeable"])

    def test_no_checkpoint_no_change(self):                                   # I6
        ctx = self.context(checkpoint=False)
        self.call("sandbox_edit", ctx=ctx, path="a.txt", old="one", new="uno")
        self.passed()
        got = self.call("sandbox_apply", ctx=ctx)
        self.assertIn("no checkpoint", got)
        self.assertEqual(self.asked, [])
        self.assertEqual(self.real("a.txt"), b"one\ntwo\nthree\n")

    def test_a_way_back_that_cannot_be_kept_stops_it_before_the_first_write(self):
        self.call("sandbox_diff")
        for rel in (".gitignore", "a.txt", "src/b.py"):                         # files that exist, so the way back for each is a kept copy of it
            self.box(self.session().ws, rel, b"changed\n")
        self.passed()
        with mock.patch.object(checkpoints, "MAX_FILES", 2):
            got = self.call("sandbox_apply")
        self.assertIn("a way back could not be kept", got)
        self.assertEqual(self.real("a.txt"), b"one\ntwo\nthree\n")
        self.assertEqual(self.store.listing("chat1"), [], "and none is left half kept")
        self.assertEqual([p for p in (self.base / "cp").rglob("*") if p.is_file()], [], "not even its copies of the files")

    def test_a_file_that_moves_while_the_question_is_open_stops_it(self):
        self.edited()
        self.passed()

        def meanwhile(_card):
            self.put("a.txt", b"the user got there first\n")
            return "allow"
        self.answer = meanwhile
        got = self.result("sandbox_apply")
        self.assertTrue(got["isError"])
        self.assertIn("moved while the question was open", got["content"][0]["text"])
        self.assertEqual(self.real("a.txt"), b"the user got there first\n")

    def test_each_step_is_checked_right_before_it(self):
        self.call("sandbox_diff")
        ws = self.session().ws
        self.box(ws, "a.txt", b"uno\n")
        self.box(ws, "src/b.py", b"# b\n")
        self.passed()
        real_write = agent.write_bytes_atomic

        def one_then_move(path, data):
            real_write(path, data)
            self.put("src/b.py", b"moved between the steps\n")
        with mock.patch.object(agent, "write_bytes_atomic", one_then_move):
            got = self.call("sandbox_apply")
        self.assertIn("put back", got)
        self.assertEqual(self.real("src/b.py"), b"moved between the steps\n")
        self.assertEqual(self.real("a.txt"), b"one\ntwo\nthree\n")

    def test_a_write_that_fails_half_way_puts_back_what_was_written(self):
        self.call("sandbox_edit", path="new.txt", old="", new="made\n")
        self.call("sandbox_edit", path="a.txt", old="one", new="uno")
        self.call("sandbox_edit", path="src/b.py", old="return 1", new="return 2")
        self.passed()
        real_write = agent.write_bytes_atomic
        calls = []

        def fails_on_the_third(path, data):
            calls.append(path)
            if len(calls) == 3:
                raise OSError("the disk is full")
            real_write(path, data)
        with mock.patch.object(agent, "write_bytes_atomic", fails_on_the_third):
            got = self.result("sandbox_apply")
        self.assertTrue(got["isError"])
        self.assertIn("the disk is full", got["content"][0]["text"])
        self.assertIn("What was written was put back", got["content"][0]["text"])
        self.assertEqual(self.real("a.txt"), b"one\ntwo\nthree\n")
        self.assertFalse((self.proj / "new.txt").exists())
        self.assertEqual(len(self.session().ws.changed()), 3, "the change is still in the copy")

    def test_what_could_not_be_put_back_is_named(self):
        self.call("sandbox_edit", path="a.txt", old="one", new="uno")
        self.call("sandbox_edit", path="src/b.py", old="return 1", new="return 2")
        self.passed()
        real_write = agent.write_bytes_atomic
        calls = []

        def everything_fails_after_the_first(path, data):
            calls.append(path)
            if len(calls) > 1:
                raise OSError("locked")
            real_write(path, data)
        with mock.patch.object(agent, "write_bytes_atomic", everything_fails_after_the_first):
            got = self.call("sandbox_apply")
        self.assertIn("could not be put back: a.txt", got)
        self.assertIn("Rewind", got)
        self.assertNotIn("a.txt", {r["path"] for r in self.session().ws.changed()}, "the file the project now holds is the base")

    def test_a_hook_can_stop_it_before_the_question(self):
        class Hooks:
            seen = []

            def before(self, tool, args):
                self.seen.append((tool, args.get("file_path")))
                return "not this file" if tool == "Write" else None

            def after(self, *_a):
                return ""
        ctx = self.context()
        ctx.hooks = Hooks()
        self.call("sandbox_edit", ctx=ctx, path="a.txt", old="one", new="uno")
        self.passed()
        got = self.call("sandbox_apply", ctx=ctx)
        self.assertIn("not this file", got)
        self.assertEqual(self.asked, [])
        self.assertEqual(Hooks.seen, [("sandbox_apply", None), ("Write", str(self.proj / "a.txt"))])

    def test_a_formatter_hook_on_the_written_file_is_no_conflict(self):
        proj = self.proj

        class Hooks:
            def before(self, *_a):
                return None

            def after(self, tool, args, _result):
                Path(args["file_path"]).write_bytes(b"formatted\n")
                return "formatted it"
        ctx = self.context()
        ctx.hooks = Hooks()
        self.call("sandbox_edit", ctx=ctx, path="a.txt", old="one", new="uno")
        self.passed()
        self.assertIn("formatted it", self.call("sandbox_apply", ctx=ctx))
        self.assertEqual((proj / "a.txt").read_bytes(), b"formatted\n")
        self.call("sandbox_diff", ctx=ctx)
        ws = self.session().ws
        self.assertEqual(ws.conflicts, set())
        self.assertEqual(self.in_copy(ws, "a.txt"), b"formatted\n", "the copy follows")

    def test_an_edit_that_lands_right_after_the_write_is_followed_and_is_no_conflict(self):
        self.edited()
        self.passed()
        real_write = agent.write_bytes_atomic

        def and_then_the_user(path, data):
            real_write(path, data)
            Path(path).write_bytes(b"the user's edit, a moment later\n")       # before the sync that follows the write
        with mock.patch.object(agent, "write_bytes_atomic", and_then_the_user):
            self.assertIn("นำขึ้น project จริงแล้ว", self.call("sandbox_apply"))
        got = self.call("sandbox_diff")
        self.assertNotIn("เปลี่ยนในของจริงด้วย", got)
        self.assertEqual(self.in_copy(self.session().ws, "a.txt"), b"the user's edit, a moment later\n")

    def test_a_change_to_the_tests_is_on_the_question(self):                  # design case 31
        self.call("sandbox_edit", path="tests/test_x.py", old="", new="def test_x():\n    pass\n")
        self.passed()
        self.call("sandbox_apply")
        self.assertIn("tests/test_x.py", " ".join(self.asked[0]["arguments"]["notes"]))

    def test_a_simulated_pass_says_so_to_the_end(self):                       # design 12
        self.edited()
        self.passed(simulated="no GPU, fake model")
        self.call("sandbox_apply")
        self.assertIn("[จำลอง: no GPU, fake model]", self.asked[0]["arguments"]["notes"])

    def test_a_pass_names_what_passed_and_does_not_call_it_right(self):       # design case 47
        self.edited()
        self.passed()
        self.call("sandbox_apply")
        self.assertIn("python -m pytest -q", " ".join(self.asked[0]["arguments"]["notes"]))
        self.assertIn("ไม่ได้แปลว่าถูก", " ".join(self.asked[0]["arguments"]["notes"]))

    def test_a_link_the_box_planted_is_not_offered(self):
        self.call("sandbox_diff")
        ws = self.session().ws
        junction(self.outside, Path(ws.copy) / "planted")
        self.passed()
        got = self.result("sandbox_apply")
        self.assertTrue(got["isError"])
        self.assertIn("link", got["content"][0]["text"])
        self.assertEqual(self.asked, [])

    def test_two_chats_on_one_project_meet_at_apply(self):                    # design case 35
        first, second = self.context("one"), self.context("two")
        self.call("sandbox_edit", ctx=first, path="a.txt", old="one", new="first")
        self.call("sandbox_edit", ctx=second, path="a.txt", old="one", new="second")
        self.passed(chat="one")
        self.passed(chat="two")
        self.assertIn("นำขึ้น project จริงแล้ว", self.call("sandbox_apply", ctx=first))
        got = self.result("sandbox_apply", ctx=second)
        self.assertTrue(got["isError"])
        self.assertIn(CONFLICT.format(path="a.txt"), got["content"][0]["text"])
        self.assertEqual(self.real("a.txt"), b"first\ntwo\nthree\n")

    def test_a_branch_switch_is_said_and_the_change_stays(self):              # design case 34
        self.edited()
        self.git("checkout", "-qb", "other")
        self.put("src/b.py", b"def f():\n    return 7\n")
        self.commit("on the other branch")
        got = self.call("sandbox_diff")
        self.assertIn("branch หรือ commit ของจริงเปลี่ยน", got)
        self.assertIn("+uno", got)

    def test_the_user_changing_a_file_the_copy_did_not_touch_follows_in_silence(self):      # design case 32
        self.edited()
        self.put("src/b.py", b"def f():\n    return 7\n")
        got = self.call("sandbox_diff")
        self.assertNotIn("เปลี่ยนในของจริงด้วย", got)
        self.assertEqual(self.in_copy(self.session().ws, "src/b.py"), b"def f():\n    return 7\n")

    def test_each_apply_is_in_the_record_with_its_diff(self):                 # design 4.8
        self.edited()
        self.passed()
        self.call("sandbox_apply")
        rows = [json.loads(x) for x in next(self.logs.iterdir()).read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rows[-1]["kind"], "apply")
        self.assertEqual(rows[-1]["files"], ["a.txt"])
        self.assertIn("+uno", rows[-1]["diff"])


class Conflicts(Case):
    """Design 4.4: a file changed on both sides stays until the model decides. These are the two ways to decide, and a third change arriving while it does."""

    def test_it_is_said_and_apply_refuses(self):                              # design case 32
        self.conflicted()
        self.passed()
        got = self.result("sandbox_apply")
        self.assertTrue(got["isError"])
        self.assertIn(CONFLICT.format(path="a.txt"), got["content"][0]["text"])
        self.assertIn("sandbox_conflict", got["content"][0]["text"])
        self.assertEqual(self.real("a.txt"), b"real\ntwo\nthree\n")

    def test_the_list(self):
        self.assertIn("no file is changed on both sides", self.call("sandbox_conflict"))
        self.conflicted()
        self.assertIn(CONFLICT.format(path="a.txt"), self.call("sandbox_conflict"))

    def test_the_real_side_is_shown_against_the_base_and_nothing_changes(self):
        self.conflicted()
        before = everything(self.proj)
        ws = self.session().ws
        kept = self.in_copy(ws, "a.txt")
        got = self.call("sandbox_conflict", path="a.txt")
        self.assertIn("-one", got)
        self.assertIn("+real", got)
        self.assertIn("+copy", got)
        self.assertEqual(everything(self.proj), before, "a read")
        self.assertEqual(self.in_copy(ws, "a.txt"), kept)

    def test_a_long_side_is_saved_and_can_be_read_on(self):
        self.conflicted(b"copy " * 500 + b"\ntwo\nthree\n", b"real " * 500 + b"\ntwo\nthree\n")
        got = self.call("sandbox_conflict", path="a.txt")
        self.assertIn("seen: ", got)
        self.assertLessEqual(len(got.encode()), 1024)
        self.assertIn(".strata-sandbox/output/", got)

    def test_what_is_in_no_conflict_says_so(self):
        self.call("sandbox_edit", path="a.txt", old="one", new="uno")
        self.assertIn("in no conflict now", self.call("sandbox_conflict", path="a.txt"))
        seen = hashlib.sha256(self.real("a.txt")).hexdigest()[:16]               # the right token: only the conflict itself is missing
        got = self.result("sandbox_resolve", path="a.txt", take="real", seen=seen)
        self.assertTrue(got["isError"])
        self.assertIn("in no conflict now", got["content"][0]["text"])
        self.assertEqual(self.in_copy(self.session().ws, "a.txt"), b"uno\ntwo\nthree\n", "the copy's change stays")

    def test_a_secret_stays_out_even_as_a_hash(self):                         # design case 16
        self.call("sandbox_diff")
        self.box(self.session().ws, "src/token.key", b"copy side\n")
        self.put("src/token.key", b"the secret value\n")
        got = self.call("sandbox_conflict", path="src/token.key")
        self.assertNotIn("the secret value", got)
        self.assertNotIn("ของจริงเทียบ base:\n", got)
        self.assertIn("seen: hidden", got, "no hash of it either")
        self.assertIn("only take=real", self.call("sandbox_resolve", path="src/token.key", take="copy", seen="hidden"), "a file nobody may see cannot be merged onto")
        self.call("sandbox_resolve", path="src/token.key", take="real", seen="hidden")
        self.assertIsNone(self.in_copy(self.session().ws, "src/token.key"), "taking it leaves it out of the copy")

    def test_a_path_out_of_the_project_is_refused(self):
        self.call("sandbox_diff")
        for bad in ("../outside/stolen.txt", ".git/config"):
            self.assertTrue(self.result("sandbox_conflict", path=bad)["isError"], bad)
            self.assertTrue(self.result("sandbox_resolve", path=bad, take="real", seen="none")["isError"], bad)

    def test_taking_the_real_side(self):
        self.conflicted()
        got = self.call("sandbox_resolve", path="a.txt", take="real", seen=self.seen())
        self.assertIn("the copy's change to it is gone", got)
        ws = self.session().ws
        self.assertEqual(self.in_copy(ws, "a.txt"), b"real\ntwo\nthree\n")
        self.assertEqual(ws.changed(), [])
        self.assertEqual(ws.conflicts, set())
        self.assertEqual(self.real("a.txt"), b"real\ntwo\nthree\n")

    def test_keeping_the_copy_applies_cleanly(self):
        self.conflicted(b"copy\ntwo\nthree\n", b"real\ntwo\nthree\n")
        got = self.call("sandbox_resolve", path="a.txt", take="copy", seen=self.seen())
        self.assertIn("yours were kept", got)
        ws = self.session().ws
        self.assertEqual(self.in_copy(ws, "a.txt"), b"copy\ntwo\nthree\n")
        self.assertEqual(ws.apply_plan().writes[0].base, ws.base["a.txt"].hash, "measured against the real file now")
        self.passed()
        got = self.result("sandbox_apply")
        self.assertFalse(got["isError"], got)
        self.assertEqual(self.real("a.txt"), b"copy\ntwo\nthree\n")
        self.assertEqual(ws.changed(), [])

    def test_the_diff_the_user_sees_is_against_the_real_file(self):
        self.conflicted()
        self.call("sandbox_resolve", path="a.txt", take="copy", seen=self.seen())
        self.passed()
        self.call("sandbox_apply")
        self.assertIn("-real", self.asked[0]["arguments"]["diff"])

    def test_keeping_the_copy_keeps_what_the_real_side_did_elsewhere(self):
        lines = [b"line %d\n" % i for i in range(20)]
        self.put("long.txt", b"".join(lines))
        self.commit()
        copy, real = list(lines), list(lines)
        copy[2], real[17] = b"changed in the copy\n", b"changed by the user\n"
        self.conflicted(b"".join(copy), b"".join(real), "long.txt")
        got = self.call("sandbox_resolve", path="long.txt", take="copy", seen=self.seen("long.txt"))
        self.assertIn("real side's change kept", got)
        now = self.in_copy(self.session().ws, "long.txt")
        self.assertIn(b"changed in the copy", now)
        self.assertIn(b"changed by the user", now, "a change of the user's is not lost")
        self.passed()
        self.call("sandbox_apply")
        self.assertEqual(self.real("long.txt"), now)

    def test_line_ends_survive_the_merge(self):
        self.put("w.txt", b"a\r\nb\r\nc\r\nd\r\ne\r\nf\r\n")
        self.commit()
        self.conflicted(b"A\r\nb\r\nc\r\nd\r\ne\r\nf\r\n", b"a\r\nb\r\nc\r\nd\r\ne\r\nF\r\n", "w.txt")
        self.call("sandbox_resolve", path="w.txt", take="copy", seen=self.seen("w.txt"))
        self.assertEqual(self.in_copy(self.session().ws, "w.txt"), b"A\r\nb\r\nc\r\nd\r\ne\r\nF\r\n")

    def test_a_binary_file_keeps_the_copys_bytes(self):
        self.put("blob.bin", b"\0base\0")
        self.commit()
        self.conflicted(b"\0copy\0", b"\0real\0", "blob.bin")
        self.call("sandbox_resolve", path="blob.bin", take="copy", seen=self.seen("blob.bin"))
        self.assertEqual(self.in_copy(self.session().ws, "blob.bin"), b"\0copy\0")
        self.passed()
        self.assertFalse(self.result("sandbox_apply")["isError"])
        self.assertEqual(self.real("blob.bin"), b"\0copy\0")

    def test_the_copy_deleted_it_and_the_real_side_changed_it(self):
        self.call("sandbox_edit", path="a.txt", delete=True)
        self.put("a.txt", b"real\n")
        self.call("sandbox_resolve", path="a.txt", take="copy", seen=self.seen())
        self.passed()
        self.assertFalse(self.result("sandbox_apply")["isError"])
        self.assertFalse((self.proj / "a.txt").exists())

    def test_the_real_side_deleted_it_and_the_copy_changed_it(self):
        self.call("sandbox_edit", path="a.txt", old="one", new="uno")
        (self.proj / "a.txt").unlink()
        seen = self.seen()
        self.assertEqual(seen, "none")
        self.call("sandbox_resolve", path="a.txt", take="copy", seen=seen)
        self.passed()
        self.assertFalse(self.result("sandbox_apply")["isError"])
        self.assertEqual(self.real("a.txt"), b"uno\ntwo\nthree\n")

    def test_taking_a_deletion(self):
        self.call("sandbox_edit", path="a.txt", old="one", new="uno")
        (self.proj / "a.txt").unlink()
        self.call("sandbox_resolve", path="a.txt", take="real", seen=self.seen())
        ws = self.session().ws
        self.assertIsNone(self.in_copy(ws, "a.txt"))
        self.assertEqual(ws.changed(), [])

    def test_both_made_the_same_new_file(self):
        self.conflicted(b"from the copy\n", b"from the user\n", "new.txt")
        self.call("sandbox_resolve", path="new.txt", take="copy", seen=self.seen("new.txt"))
        self.passed()
        self.assertFalse(self.result("sandbox_apply")["isError"])
        self.assertEqual(self.real("new.txt"), b"from the copy\n")

    def test_a_third_change_while_deciding_settles_nothing(self):
        for take in ("copy", "real"):
            with self.subTest(take):
                self.conflicted()
                seen = self.seen()
                self.put("a.txt", b"third\ntwo\nthree\n")
                ws = self.session().ws
                kept, base = self.in_copy(ws, "a.txt"), ws.base["a.txt"]
                got = self.result("sandbox_resolve", path="a.txt", take=take, seen=seen)
                self.assertTrue(got["isError"])
                self.assertIn("เปลี่ยนในของจริงอีกครั้ง: a.txt", got["content"][0]["text"])
                self.assertIn("sandbox_conflict", got["content"][0]["text"])
                self.assertEqual(self.in_copy(ws, "a.txt"), kept, "the copy stands")
                self.assertEqual(ws.base["a.txt"], base, "the base stands")
                now = self.seen()
                self.assertNotEqual(now, seen)
                self.assertIn("third", self.call("sandbox_conflict", path="a.txt"))
                self.assertFalse(self.result("sandbox_resolve", path="a.txt", take=take, seen=now)["isError"])
                self.server.forget("chat1")
                self.git("checkout", "-q", "a.txt")

    def test_a_third_change_after_keeping_the_copy_is_a_conflict_again(self):
        self.conflicted()
        self.call("sandbox_resolve", path="a.txt", take="copy", seen=self.seen())
        self.put("a.txt", b"third\ntwo\nthree\n")
        self.passed()
        got = self.result("sandbox_apply")
        self.assertTrue(got["isError"])
        self.assertIn(CONFLICT.format(path="a.txt"), got["content"][0]["text"])
        self.assertEqual(self.real("a.txt"), b"third\ntwo\nthree\n")
        self.call("sandbox_resolve", path="a.txt", take="copy", seen=self.seen())
        self.passed()
        self.assertFalse(self.result("sandbox_apply")["isError"])

    def test_a_third_change_after_taking_the_real_side_just_follows(self):
        self.conflicted()
        self.call("sandbox_resolve", path="a.txt", take="real", seen=self.seen())
        self.put("a.txt", b"third\ntwo\nthree\n")
        self.assertNotIn("เปลี่ยนในของจริงด้วย", self.call("sandbox_diff"))
        self.assertEqual(self.in_copy(self.session().ws, "a.txt"), b"third\ntwo\nthree\n")

    def test_a_third_change_while_the_user_answers_writes_nothing(self):
        self.conflicted()
        self.call("sandbox_resolve", path="a.txt", take="copy", seen=self.seen())
        self.passed()

        def meanwhile(_card):
            self.put("a.txt", b"third\ntwo\nthree\n")
            return "allow"
        self.answer = meanwhile
        got = self.call("sandbox_apply")
        self.assertIn("moved while the question was open", got)
        self.assertEqual(self.real("a.txt"), b"third\ntwo\nthree\n")

    def test_a_test_record_does_not_carry_over(self):
        self.conflicted()
        self.passed()
        self.call("sandbox_resolve", path="a.txt", take="copy", seen=self.seen())
        self.assertIn("no test ran", self.call("sandbox_apply"))

    def test_one_file_at_a_time(self):
        self.conflicted()
        self.box(self.session().ws, "src/b.py", b"# copy\n")
        self.put("src/b.py", b"# real\n")
        self.call("sandbox_resolve", path="a.txt", take="copy", seen=self.seen())
        self.assertIn(CONFLICT.format(path="src/b.py"), self.call("sandbox_apply"))
        self.call("sandbox_resolve", path="src/b.py", take="real", seen=self.seen("src/b.py"))
        self.passed()
        self.assertFalse(self.result("sandbox_apply")["isError"])

    def test_a_link_in_the_copy_can_only_be_replaced(self):
        self.put("gone.txt", b"base\n")
        self.commit()
        self.call("sandbox_diff")
        ws = self.session().ws
        outside = everything(self.outside)
        os.remove(Path(ws.copy) / "gone.txt")
        junction(self.outside, Path(ws.copy) / "gone.txt")
        self.put("gone.txt", b"real\n")
        seen = self.seen("gone.txt")
        self.assertIn("a link", self.call("sandbox_resolve", path="gone.txt", take="copy", seen=seen))
        self.call("sandbox_resolve", path="gone.txt", take="real", seen=seen)
        self.assertEqual(self.in_copy(ws, "gone.txt"), b"real\n")
        self.assertEqual(everything(self.outside), outside)

    def test_a_link_where_both_sides_made_a_new_file_is_not_kept(self):
        self.call("sandbox_diff")
        ws = self.session().ws
        junction(self.outside, Path(ws.copy) / "new.txt")
        self.put("new.txt", b"the user's new file\n")
        seen = self.seen("new.txt")
        got = self.result("sandbox_resolve", path="new.txt", take="copy", seen=seen)
        self.assertTrue(got["isError"])
        self.assertIn("a link", got["content"][0]["text"])
        self.assertNotIn("new.txt", ws.base, "nothing was settled")

    def test_a_folder_on_the_way_that_is_a_link_is_refused(self):
        self.call("sandbox_diff")
        ws = self.session().ws
        self.put("src/b.py", b"# real\n")
        (Path(ws.copy) / "src" / "b.py").unlink()
        os.rmdir(Path(ws.copy) / "src")
        junction(self.outside, Path(ws.copy) / "src")
        got = self.result("sandbox_resolve", path="src/b.py", take="real", seen=self.seen("src/b.py"))
        self.assertTrue(got["isError"])
        self.assertIn("link", got["content"][0]["text"])
        self.assertEqual(sorted(everything(self.outside)), ["stolen.txt"], "nothing was written through it")


class Numbers(unittest.TestCase):
    """How a run's output reads, from what the tools print. The runs themselves are in `InABox`."""

    PYTEST = ".F\n=== short test summary info ===\nPASSED tests/test_a.py::test_ok\nFAILED tests/test_a.py::test_no - assert 3 == 1\n1 failed, 1 passed in 0.12s\n"

    def test_pytest_is_read_test_by_test(self):
        got = testrun.parse("pytest", 1, self.PYTEST)
        self.assertEqual((got.passed, got.failed), ({"tests/test_a.py::test_ok"}, {"tests/test_a.py::test_no": "assert 3 == 1"}))

    def test_crlf_changes_nothing(self):
        self.assertEqual(testrun.parse("pytest", 1, self.PYTEST.replace("\n", "\r\n")).passed, {"tests/test_a.py::test_ok"})

    def test_a_run_that_ended_some_other_way_is_read_by_its_exit_code(self):
        self.assertFalse(testrun.parse("pytest", 4, "ERROR: file or directory not found: x\n\nno tests ran in 0.01s\n").read)
        self.assertFalse(testrun.parse("pytest", -9, "").read)

    def test_no_tests_is_not_a_pass(self):
        got = testrun.parse("pytest", 5, "\nno tests ran in 0.01s\n")
        self.assertTrue(got.none)
        self.assertEqual(testrun.compare(got, got).verdict, "nothing")
        self.assertNotIn("nothing", testrun.PASSES)

    def test_unittest_too(self):
        text = ("F.\n======================================================================\nFAIL: test_no (test_m.T.test_no)\n----------------------------------------------------------------------\n"
                "Traceback (most recent call last):\n  File \"x\", line 1\nAssertionError: 1 != 2\n\n----------------------------------------------------------------------\nRan 2 tests in 0.001s\n\nFAILED (failures=1)\n")
        got = testrun.parse("unittest", 1, text)
        self.assertEqual((got.count, got.failed), (1, {"test_m.T.test_no": "AssertionError: 1 != 2"}))

    def outcomes(self, passed=(), failed=()):
        return testrun.Outcome(0 if not failed else 1, "pytest", passed=set(passed), failed=dict.fromkeys(failed, "why"))

    def test_broken_by_this_change(self):
        got = testrun.compare(self.outcomes(["a"], ["b"]), self.outcomes(["a", "b"]))
        self.assertEqual(got.verdict, "broke")
        self.assertEqual(got.line, "tests: 1 ผ่าน, 1 ตก (ก่อนแก้ 2 ผ่าน) ← พังเพราะการแก้นี้")

    def test_failing_before(self):                                            # design case 22
        got = testrun.compare(self.outcomes(["a"], ["b"]), self.outcomes(["a"], ["b"]))
        self.assertEqual(got.verdict, "same")
        self.assertEqual(got.line, "tests: 1 ผ่าน, 1 ตก (ก่อนแก้ 1 ผ่าน, 1 ตก) ← ตกอยู่ก่อนแล้ว ไม่ใช่เพราะการแก้นี้")

    def test_a_new_test_that_fails(self):
        got = testrun.compare(self.outcomes(["a"], ["t.py::new"]), self.outcomes(["a"]))
        self.assertEqual(got.verdict, "new")
        self.assertIn("test ใหม่", got.line)

    def test_a_file_that_stopped_collecting_is_no_new_test(self):
        got = testrun.compare(self.outcomes([], ["t.py"]), self.outcomes(["t.py::a", "t.py::b"]))
        self.assertEqual(got.verdict, "broke")

    def test_fixed(self):
        got = testrun.compare(self.outcomes(["a", "b"]), self.outcomes(["a"], ["b"]))
        self.assertEqual(got.verdict, "fixed")
        self.assertIn("แก้ได้ 1 test", got.line)

    def test_flaky(self):                                                     # design case 23
        got = testrun.compare(self.outcomes(["a"], ["b"]), self.outcomes(["a", "b"]), {"b"})
        self.assertEqual(got.verdict, "flaky")
        self.assertIn("ไม่แน่นอน", got.line)

    def test_stopped_is_no_verdict_on_the_code(self):
        got = testrun.compare(testrun.parse("pytest", None, "", "เกินเวลา"), self.outcomes(["a"]))
        self.assertEqual(got.verdict, "stopped")
        self.assertNotIn("พัง", got.line)

    def test_a_base_that_could_not_run_is_said(self):
        got = testrun.compare(self.outcomes(["a"], ["b"]), None)
        self.assertEqual(got.verdict, "broke")
        self.assertIn("ก่อนแก้รันไม่ได้", got.line)
        self.assertIn("เทียบไม่ได้", got.line)

    def test_by_exit_code_alone(self):
        ok, bad, none = testrun.Outcome(0, "other", read=False), testrun.Outcome(1, "other", read=False), None
        self.assertEqual(testrun.compare(bad, ok).verdict, "broke")
        self.assertEqual(testrun.compare(bad, ok).line, "exit 1 (ก่อนแก้ exit 0) ← พังเพราะการแก้นี้")
        self.assertEqual(testrun.compare(bad, bad).verdict, "same")
        self.assertEqual(testrun.compare(ok, bad).verdict, "fixed")
        self.assertEqual(testrun.compare(ok, ok).verdict, "passed")
        self.assertEqual(testrun.compare(bad, none).verdict, "broke")

    def test_more_failures_than_fit_are_counted(self):
        got = testrun.compare(self.outcomes([], [f"t{i}" for i in range(20)]), self.outcomes([f"t{i}" for i in range(20)]))
        self.assertEqual(len(got.rows), testrun.MAX_LISTED + 1)
        self.assertIn("ตกอีก 12 test", got.rows[-1])

    def test_a_command_splits_as_a_shell_would_and_has_no_shell(self):
        self.assertEqual(testrun.split_command('python -c "print(1)" tests\\a.py'), ["python", "-c", "print(1)", "tests\\a.py"])
        for bad in ("a | b", "a && b", "a ; b", "a > b", "", "   "):
            with self.assertRaises(ValueError):
                testrun.split_command(bad)
        with self.assertRaises(ValueError):
            testrun.split_command('python "unclosed')

    def test_a_command_is_prepared_for_how_it_reads(self):
        self.assertEqual(testrun.prepared(["pytest", "-q"])[0][:3], ["python", "-m", "pytest"])
        self.assertIn("-rfEp", testrun.prepared(["python", "-m", "pytest"])[0])
        self.assertEqual(testrun.prepared(["python", "-m", "unittest"]), (["python", "-m", "unittest"], "unittest"))
        self.assertEqual(testrun.prepared(["python", "check.py"]), (["python", "check.py"], "other"))

    def test_what_the_box_lacks_is_named(self):
        self.assertEqual(testrun.blocked_by("ModuleNotFoundError: No module named 'torch'"), "ไม่มี library torch ใน box")
        self.assertEqual(testrun.blocked_by("socket.gaierror: [Errno 11001] getaddrinfo failed"), "ไม่มี network ใน box")
        self.assertIsNone(testrun.blocked_by("assert 1 == 2"))

    def test_test_files_are_told_from_the_rest(self):
        self.assertEqual(testrun.touches_tests(["tests/a.py", "src/test_b.py", "src/b_test.go", "ui/x.test.ts", "conftest.py", "src/latest.py"]), ["tests/a.py", "src/test_b.py", "src/b_test.go", "ui/x.test.ts", "conftest.py"])


def box_ready() -> bool:
    try:
        return Isolation.report()["appcontainer" if WINDOWS else "bubblewrap"][0]
    except Exception:  # noqa: BLE001 - no backend is a skip
        return False


@unittest.skipUnless(box_ready(), "no isolation backend on this machine")
class InABox(Case):
    """The real wall, with the test's own programs."""

    conf = ON
    start = staticmethod(Isolation.start)
    box_root = None

    def setUp(self):
        super().setUp()
        self.starts = []

        def counted(policy, root=None):
            self.starts.append(list(policy.readwrite))
            return Isolation.start(policy, root=root)
        self.server = SandboxServer(self.conf, None, self.logs, start=counted)
        self.addCleanup(self.server.close)
        self.put("calc.py", b"def add(a, b):\n    return a + b\n")
        self.put("tests/test_calc.py", b"from calc import add\n\ndef test_one():\n    assert add(1, 2) == 3\n\ndef test_two():\n    assert add(2, 2) == 4\n")
        self.commit()

    def test_python_runs_in_the_copy_with_the_project_importable(self):
        got = self.call("sandbox_run", code="import os, calc\nprint(calc.add(40, 2), os.path.basename(os.getcwd()))\n")
        self.assertTrue(got.startswith(f"{HEADER} exit 0,"), got)
        self.assertIn("42 copy", got)
        self.assertEqual(self.steps(), ["copying", "trying"])

    def test_a_command_and_its_stdin_and_exit_code(self):
        self.put("echo.py", b"import sys\nprint(sys.stdin.read().upper())\nsys.exit(4)\n")
        self.assertTrue(self.call("sandbox_run", command="python echo.py", stdin="hello").startswith(f"{HEADER} exit 4,"))
        self.assertIn("HELLO", self.call("sandbox_run", command="python echo.py", stdin="hello"))

    def test_a_program_with_no_input_is_told_how(self):                       # design case 4
        self.assertIn("ส่งผ่าน stdin", self.call("sandbox_run", code="input()\n"))

    def test_a_library_the_box_lacks(self):                                   # design case 3
        got = self.call("sandbox_run", code="import not_a_real_library\n")
        self.assertIn("box ไม่มี not_a_real_library", got)
        self.assertIn("sympy", got)

    def test_a_shell_line_is_refused_with_the_way_out(self):
        self.assertIn("no shell", self.call("sandbox_run", command="python a.py | grep x"))

    def test_what_a_run_writes_is_in_the_copy_and_in_the_diff(self):
        before = everything(self.proj)
        got = self.call("sandbox_run", code="open('made.txt', 'w').write('in the box')\nopen('calc.py', 'a').write('# touched\\n')\n")
        self.assertIn("แก้แล้ว 2 ไฟล์ในสำเนา", got)
        self.assertEqual(everything(self.proj), before)
        self.assertEqual(self.status_of(self.session().ws), {"calc.py": "M", "made.txt": "A"})
        self.assertIn("+# touched", self.call("sandbox_diff"))
        self.assertTrue(self.session().ws.close())                            # a file the box made can be removed

    def test_the_real_project_and_the_base_are_out_of_reach(self):            # I1, I10, and what protects the base
        before = everything(self.proj)
        self.call("sandbox_diff")
        ws = self.session().ws
        program = (f"import os\nfor action in (lambda: open({str(self.proj / 'a.txt')!r}).read(), lambda: open({str(self.proj / 'a.txt')!r}, 'w'), lambda: open({str(self.proj / 'new.txt')!r}, 'w'),\n"
                   f"               lambda: os.listdir({str(self.proj)!r}), lambda: open({os.path.join(ws.root, 'base.pack')!r}, 'rb').read()):\n"
                   "    try:\n        action()\n        print('open')\n    except PermissionError:\n        print('shut')\n")
        got = self.call("sandbox_run", code=program)
        self.assertEqual(got.count("shut"), 5, got)
        self.assertNotIn("open", got.replace("copy", ""))
        self.assertEqual(everything(self.proj), before)

    def test_a_network_call_fails_with_the_reason(self):                      # design case 26
        self.put("tests/test_net.py", b"import socket\n\ndef test_net():\n    socket.create_connection(('example.invalid', 80), 3)\n")
        self.commit()
        got = self.call("sandbox_test", command="python -m pytest -q tests/test_net.py")
        self.assertIn("ตกอยู่ก่อนแล้ว", got)
        self.assertIn("รันไม่ได้ใน box: ไม่มี network", got)

    def test_the_users_stop_ends_the_run_and_the_box_goes_on(self):           # design case 37
        ctx = self.context()
        threading.Timer(1.0, ctx.cancel.set).start()
        began = time.monotonic()
        got = self.call("sandbox_run", ctx=ctx, code="import time\ntime.sleep(60)\n")
        self.assertLess(time.monotonic() - began, 30)
        self.assertIn("ผู้ใช้กดหยุด", got)
        self.assertIn("5", self.call("sandbox_run", code="print(5)\n"))
        self.assertEqual(len(self.starts), 1, "the same box")

    def test_a_run_past_its_time_is_stopped_and_the_box_goes_on(self):        # design case 30
        server = SandboxServer({**self.conf, "run_seconds": 1.0}, None, start=Isolation.start)
        self.addCleanup(server.close)
        got = server.call("sandbox_run", {"code": "import time\ntime.sleep(60)\n"}, 0, None, self.ctx)["content"][0]["text"]
        self.assertIn("เกินเวลา", got)
        self.assertIn("5", server.call("sandbox_run", {"code": "print(5)\n"}, 0, None, self.ctx)["content"][0]["text"])

    def test_what_a_run_leaves_behind_ends_with_it(self):                     # design case 30
        leave = "import subprocess, sys\nsubprocess.Popen([sys.executable, '-c', \"import time\\nwhile True:\\n    open('still.txt', 'w').write('x')\\n    time.sleep(0.05)\"], creationflags=0x08000000 if sys.platform == 'win32' else 0)\nimport time\ntime.sleep(0.6)\n"
        self.call("sandbox_run", code=leave)
        ws = self.session().ws
        first = os.path.getmtime(Path(ws.copy) / "still.txt")
        time.sleep(0.4)
        self.assertEqual(os.path.getmtime(Path(ws.copy) / "still.txt"), first)

    def test_a_box_past_its_life_is_replaced(self):
        self.call("sandbox_run", code="print(1)\n")
        box = self.session()._box
        box._deadline = 0
        time.sleep(0.2)
        self.assertIn("5", self.call("sandbox_run", code="print(5)\n"))
        self.assertIsNot(self.session()._box, box)

    def test_a_box_that_cannot_start_says_why_and_the_rest_works(self):       # design case 48, I4
        def refuses(*_a, **_k):
            raise IsolationError("antivirus said no")
        server = SandboxServer(self.conf, None, start=refuses)
        self.addCleanup(server.close)
        got = server.call("sandbox_run", {"code": "print(1)\n"}, 0, None, self.ctx)
        self.assertTrue(got["isError"])
        self.assertIn("antivirus said no", got["content"][0]["text"])
        self.assertIn("nothing ran", got["content"][0]["text"])
        self.assertFalse(server.call("sandbox_read", {"path": "a.txt"}, 0, None, self.ctx)["isError"])

    def test_a_box_that_can_read_the_real_project_never_runs_anything(self):  # design case 49, I10
        class Open:
            isolated = True
            killed = False

            def run(self, argv, *_a, **_k):
                return RunResult(3, b"", b"", 1.0)

            def kill(self):
                self.killed = True
        made = Open()
        server = SandboxServer(self.conf, None, start=lambda *_a, **_k: made)
        self.addCleanup(server.close)
        got = server.call("sandbox_run", {"code": "print(1)\n"}, 0, None, self.ctx)
        self.assertTrue(got["isError"])
        self.assertIn("design I10", got["content"][0]["text"])
        self.assertTrue(made.killed)

    def test_the_policy_names_the_real_project(self):
        policies = []

        def seen(policy, root=None):
            policies.append(policy)
            return Isolation.start(policy, root=root)
        server = SandboxServer(self.conf, None, start=seen)
        self.addCleanup(server.close)
        server.call("sandbox_run", {"code": "print(1)\n"}, 0, None, self.ctx)
        self.assertEqual(list(policies[0].denied), [os.path.realpath(self.proj)])
        self.assertFalse(policies[0].network)

    # -------------------------------------------------------------------------------------------- the most important line
    def test_a_change_that_breaks_a_test_says_so(self):                       # design case 18
        self.call("sandbox_edit", path="calc.py", old="return a + b", new="return a - b")
        got = self.call("sandbox_test", command="python -m pytest -q")
        self.assertTrue(got.startswith(f"{HEADER} exit 1,"), got)
        self.assertIn("tests: 0 ผ่าน, 2 ตก (ก่อนแก้ 2 ผ่าน) ← พังเพราะการแก้นี้", got)
        self.assertIn("FAILED tests/test_calc.py::test_one", got)
        self.assertIn("แก้แล้ว 1 ไฟล์ในสำเนา ยังไม่แตะ project จริง", got)
        self.assertEqual(self.steps(), ["copying", "testing", "comparing", "retrying"])
        self.assertIn("not applied", self.call("sandbox_apply"))

    def test_a_test_failing_before_is_not_blamed_on_the_change(self):         # design case 22
        self.put("tests/test_old.py", b"def test_old():\n    assert False, 'broken long ago'\n")
        self.commit()
        self.call("sandbox_edit", path="calc.py", old="return a + b", new="return b + a")
        got = self.call("sandbox_test", command="python -m pytest -q")
        self.assertIn("tests: 2 ผ่าน, 1 ตก (ก่อนแก้ 2 ผ่าน, 1 ตก) ← ตกอยู่ก่อนแล้ว ไม่ใช่เพราะการแก้นี้", got)
        self.assertFalse(self.result("sandbox_apply")["isError"], "it may go on")

    def test_a_fix_says_what_it_fixed(self):
        self.put("calc.py", b"def add(a, b):\n    return a * b\n")
        self.commit()
        self.call("sandbox_edit", path="calc.py", old="a * b", new="a + b")
        self.assertIn("(ก่อนแก้ 1 ผ่าน, 1 ตก) ← แก้ได้ 1 test", self.call("sandbox_test", command="python -m pytest -q"))

    def test_the_whole_way_from_a_change_to_the_real_project(self):
        self.call("sandbox_edit", path="calc.py", old="return a + b", new="return b + a")
        self.assertIn("tests: 2 ผ่าน (ก่อนแก้ 2 ผ่าน)", self.call("sandbox_test", command="python -m pytest -q"))
        got = self.result("sandbox_apply")
        self.assertFalse(got["isError"], got)
        self.assertEqual(self.real("calc.py"), b"def add(a, b):\n    return b + a\n")
        self.assertEqual(self.steps()[-1], "applying")

    def test_the_base_runs_once_for_a_state_of_the_project(self):
        self.call("sandbox_edit", path="calc.py", old="return a + b", new="return b + a")
        self.call("sandbox_test", command="python -m pytest -q")
        self.assertEqual(len(self.starts), 2, "the copy's box and the base's")
        self.call("sandbox_edit", path="calc.py", old="return b + a", new="return (a + b)")
        self.call("sandbox_test", command="python -m pytest -q")
        self.assertEqual(len(self.starts), 2)
        self.put("other.py", b"x = 1\n")                                      # the project moved: the base is another one
        self.call("sandbox_test")
        self.assertEqual(len(self.starts), 3)
        self.assertNotIn("comparing", self.steps()[:1])

    def test_the_base_runs_where_the_copy_cannot_reach_it(self):
        self.call("sandbox_edit", path="calc.py", old="return a + b", new="return b + a")
        self.assertIn("tests: 2 ผ่าน (ก่อนแก้ 2 ผ่าน)", self.call("sandbox_test", command="python -m pytest -q"))
        ws = self.session().ws
        self.assertEqual(self.starts, [[ws.copy], [os.path.join(ws.root, "base")]], "each box is given its own tree and only that")
        self.assertFalse(os.path.exists(os.path.join(ws.root, "base")), "the base tree does not outlast its run")

    def test_nothing_changed_means_no_second_run(self):
        self.assertIn("tests: 2 ผ่าน (ก่อนแก้ 2 ผ่าน)", self.call("sandbox_test", command="python -m pytest -q"))
        self.assertEqual(len(self.starts), 1)

    def test_a_new_test_that_fails_is_told_from_a_broken_one(self):
        self.call("sandbox_edit", path="tests/test_new.py", old="", new="def test_new():\n    assert False\n")
        self.assertIn("ตกใน test ใหม่", self.call("sandbox_test", command="python -m pytest -q"))

    def test_a_change_that_stops_a_file_collecting_is_blamed_on_the_change(self):
        self.call("sandbox_edit", path="calc.py", old="def add(a, b):", new="def add(a, b:")
        got = self.call("sandbox_test", command="python -m pytest -q")
        self.assertIn("พังเพราะการแก้นี้", got)
        self.assertIn("ก่อนแก้ 2 ผ่าน", got)

    def test_a_flaky_test_is_called_that(self):                               # design case 23
        self.put("tests/test_flaky.py", b"import os\n\ndef test_flaky():\n    if os.path.exists('trigger.txt') and not os.path.exists('flag.log'):\n"
                                         b"        open('flag.log', 'w').close()\n        assert False, 'only the first time'\n")
        self.commit()
        self.call("sandbox_edit", path="trigger.txt", old="", new="x\n")       # passes on the base, fails once on the copy, then passes
        got = self.call("sandbox_test", command="python -m pytest -q tests/test_flaky.py")
        self.assertIn("ไม่แน่นอน", got)
        self.assertIn("retrying", self.steps())
        self.assertFalse(self.result("sandbox_apply")["isError"], "it is not blamed on the change")

    def test_unittest_is_compared_too(self):
        self.put("test_u.py", b"import unittest\nfrom calc import add\nclass T(unittest.TestCase):\n    def test_a(self):\n        self.assertEqual(add(1, 1), 2)\n")
        self.commit()
        self.call("sandbox_edit", path="calc.py", old="return a + b", new="return a - b")
        got = self.call("sandbox_test", command="python -m unittest test_u")
        self.assertIn("tests: 0 ผ่าน, 1 ตก (ก่อนแก้ 1 ผ่าน) ← พังเพราะการแก้นี้", got)
        self.assertIn("FAILED test_u.T.test_a", got)

    def test_any_other_command_by_its_exit_code(self):
        self.put("check.py", b"import calc, sys\nsys.exit(0 if calc.add(1, 1) == 2 else 1)\n")
        self.commit()
        self.call("sandbox_edit", path="calc.py", old="return a + b", new="return a - b")
        self.assertIn("exit 1 (ก่อนแก้ exit 0) ← พังเพราะการแก้นี้", self.call("sandbox_test", command="python check.py"))

    def test_the_last_command_is_used_again_and_pytest_is_found(self):
        self.assertIn("ผ่าน", self.call("sandbox_test"))
        self.assertEqual(self.session().command, "python -m pytest -q")

    def test_a_project_with_no_tests_needs_a_command(self):
        self.git("rm", "-qr", "tests")
        self.commit()
        self.assertIn("no test command is known", self.call("sandbox_test"))

    def test_no_tests_is_said(self):
        self.assertIn("ไม่มี test ให้รัน", self.call("sandbox_test", command="python -m pytest -q -k nothing_is_called_this"))

    def test_a_file_a_test_writes_is_no_change_when_it_is_ignored(self):      # design case 21
        self.put("tests/test_log.py", b"def test_log():\n    open('out.log', 'w').write('x')\n    open('kept.txt', 'w').write('x')\n")
        self.commit()
        self.call("sandbox_test", command="python -m pytest -q tests/test_log.py")
        self.assertEqual(self.status_of(self.session().ws), {"kept.txt": "A"})

    def test_a_change_to_the_tests_is_said_with_the_pass(self):               # design case 31
        self.call("sandbox_edit", path="tests/test_calc.py", old="== 3", new="!= 3")
        self.call("sandbox_edit", path="calc.py", old="return a + b", new="return a + b + 0")
        got = self.call("sandbox_test", command="python -m pytest -q tests/test_calc.py -k test_two")
        self.assertIn("[การแก้นี้แก้ไฟล์ test ด้วย: tests/test_calc.py]", got)

    def test_a_simulated_run_says_so(self):
        self.call("sandbox_edit", path="calc.py", old="return a + b", new="return b + a")
        self.assertIn("[จำลอง: no GPU]", self.call("sandbox_test", command="python -m pytest -q", simulated="no GPU"))

    def test_slow_tests_go_on_and_are_picked_up(self):                        # design case 24
        self.put("tests/test_slow.py", b"import time\n\ndef test_slow():\n    time.sleep(3)\n")
        self.commit()
        server = SandboxServer({**self.conf, "test_wait_seconds": 0.5}, None, start=Isolation.start)
        self.addCleanup(server.close)
        call = lambda tool, **a: server.call(tool, a, 0, None, self.ctx)["content"][0]["text"]      # noqa: E731
        first = call("sandbox_test", command="python -m pytest -q tests/test_slow.py")
        self.assertIn("ยังรันอยู่", first)
        self.assertIn("ยังรันอยู่", call("sandbox_edit", path="a.txt", old="one", new="1"))
        self.assertIn("ยังรันอยู่", call("sandbox_run", code="print(1)\n"))
        self.assertIn("ยังรันอยู่", call("sandbox_apply"))
        self.assertIn("one", call("sandbox_read", path="a.txt"), "reading goes on")
        got = first
        for _ in range(100):
            got = call("sandbox_test")
            if "ยังรันอยู่" not in got:
                break
        self.assertIn("tests: 1 ผ่าน", got)

    def test_a_stop_during_the_tests_is_no_verdict_on_the_code(self):         # design case 37
        self.put("tests/test_slow.py", b"import time\n\ndef test_slow():\n    time.sleep(60)\n")
        self.commit()
        self.call("sandbox_edit", path="calc.py", old="return a + b", new="return b + a")
        ctx = self.context()
        threading.Timer(2.0, ctx.cancel.set).start()
        got = self.call("sandbox_test", ctx=ctx, command="python -m pytest -q tests/test_slow.py")
        self.assertIn("ถูกหยุดก่อนจบ: ผู้ใช้กดหยุด", got)
        self.assertNotIn("พัง", got)
        self.assertIn("not applied", self.call("sandbox_apply"))

    def test_conflict_then_test_then_apply(self):
        self.call("sandbox_edit", path="calc.py", old="return a + b", new="return b + a")
        self.put("calc.py", b"# the user's line\ndef add(a, b):\n    return a + b\n")
        got = self.call("sandbox_test", command="python -m pytest -q")
        self.assertIn(CONFLICT.format(path="calc.py"), got)
        self.call("sandbox_resolve", path="calc.py", take="copy", seen=self.seen("calc.py"))
        self.assertIn("tests: 2 ผ่าน", self.call("sandbox_test"))
        self.assertFalse(self.result("sandbox_apply")["isError"])
        self.assertEqual(self.real("calc.py"), b"# the user's line\ndef add(a, b):\n    return b + a\n")


class ThroughTheServer(unittest.TestCase):
    """The route as a chat of the web app takes it: the server offers the tools, runs them in its tool loop, and sends the steps to the page."""

    def setUp(self):
        from serve.test_agent_chat import Fixture
        self.fixture = Fixture("run")
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        proj = self.fixture.proj
        for args in (("init", "-q"), ("add", "-A"), ("-c", "user.email=t@example.invalid", "-c", "user.name=t", "-c", "commit.gpgsign=false", "commit", "-qm", "x")):
            subprocess.run(["git", *args], cwd=proj, capture_output=True, check=True, creationflags=0x08000000 if WINDOWS else 0)
        self.rounds: list[int] = []

    def start(self, *scripts):
        from serve import agent as agent_mod
        from serve.frontend import ChatTemplate
        from serve.mcp import hub_from_config
        from serve.server import ByteTokenizer, Service, serve
        from serve.test_mcp import ScriptedEngine
        f = self.fixture
        f.agent = agent_mod.AgentServer()
        sandbox = SandboxServer(OFF, f.base / "sandbox-root", start=no_box)
        seen = sandbox.call

        def call(tool, arguments, timeout=0, cancel=None, ctx=None):
            self.rounds.append(ctx.round)
            return seen(tool, arguments, timeout, cancel, ctx)
        sandbox.call = call
        self.sandbox = sandbox
        f.hub = hub_from_config({}, None, builtins={"agent": f.agent, "sandbox": sandbox})
        f.hub.start(wait=True)
        tok = ByteTokenizer()
        f.engine = ScriptedEngine(tok, list(scripts), max_context=200_000)
        f.svc = Service(f.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        f.svc.mcp = f.hub
        f.svc.agent = f.agent
        f.httpd = serve(f.svc, port=0)
        f.base_url = f"http://127.0.0.1:{f.httpd.server_address[1]}"

    def test_a_chat_with_the_coding_tools_is_offered_them_and_the_steps_reach_the_page(self):
        from serve.test_harness import tool_call
        self.start(tool_call("sandbox_read", path="a.txt"), tool_call("sandbox_diff"), "</think>\n\nDone.")
        status, said, events = self.fixture.chat()
        self.assertEqual(status, 200, said)
        self.assertIn("sandbox_read", self.fixture.engine.prompt_text(0), "offered without the MCP switch")
        results = [e for e in events if e["event"] == "result"]
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r["text"].startswith(HEADER) for r in results), results)
        self.assertIn("hello", results[0]["text"])
        copying = [e for e in events if e["event"] == "sandbox"]
        self.assertEqual([e["step"] for e in copying], ["copying"])
        self.assertEqual(copying[0]["call_id"], results[0]["id"], "on the call it belongs to")
        self.assertEqual(self.rounds, [1, 2], "each round of tool calls is a new turn for the budget")

    def test_a_request_with_no_coding_tools_is_not_offered_them(self):
        self.start("</think>\n\nHi.")
        status, _, _ = self.fixture.chat({"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True, "strata_mcp": True})
        self.assertEqual(status, 200)
        self.assertNotIn("sandbox_read", self.fixture.engine.prompt_text(0))

    def test_deleting_the_chat_removes_its_copy(self):
        from serve.test_harness import tool_call
        self.start(tool_call("sandbox_diff"), "</think>\n\nDone.")
        self.fixture.chat()
        root = next(iter(self.sandbox._sessions.values())).ws.root
        with urllib.request.urlopen(self.fixture.post("/agent/checkpoints/forget", {"session": "chat-1"}), timeout=10) as r:
            self.assertEqual(r.status, 200)
        self.assertFalse(os.path.exists(root))
        self.assertEqual(len(self.sandbox._sessions), 0)


if __name__ == "__main__":
    unittest.main()
