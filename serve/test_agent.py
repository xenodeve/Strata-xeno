"""Tests for serve/agent.py: the chat's coding tools (Read, Write, Edit, Glob, Grep, TodoWrite; Bash is in test_shell.py) with Claude Code's names
and parameters, and the permission gate that every call goes through."""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from serve import agent  # noqa: E402
from serve.permissions import Policy  # noqa: E402


def text(res: dict) -> str:
    return res["content"][0]["text"]


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name).resolve()
        self.proj = self.base / "proj"
        (self.proj / "src").mkdir(parents=True)
        self.out = self.base / "elsewhere"
        self.out.mkdir()
        self.srv = agent.AgentServer()
        self.asked = []
        self.answer = "deny"

    def ctx(self, **kw):
        pol = Policy(cwd=str(self.proj), **{k: v for k, v in kw.items() if k in ("mode", "allow", "deny")})

        def ask(req):
            self.asked.append(req)
            return self.answer
        return agent.AgentContext(policy=pol, session=kw.get("session", "s1"), ask=ask, cancel=threading.Event())

    def call(self, tool, args, **kw):
        return self.srv.call(tool, args, 0, None, self.ctx(**kw))

    def write(self, rel, content, raw=False):
        p = self.proj / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if raw:
            p.write_bytes(content)
        else:
            p.write_text(content, encoding="utf-8", newline="")
        return p


class TheServer(Base):
    def test_it_offers_the_tools_by_claude_codes_names_and_parameters(self):
        names = {t["name"]: t for t in self.srv.tools}
        self.assertEqual(set(names), {"Read", "Write", "Edit", "Glob", "Grep", "TodoWrite", "ExitPlanMode", "NotebookEdit", "AskUserQuestion"})
        props = lambda n: set(names[n]["inputSchema"]["properties"])  # noqa: E731
        self.assertEqual(props("Read"), {"file_path", "offset", "limit", "pages"})
        self.assertEqual(props("Write"), {"file_path", "content"})
        self.assertEqual(props("Edit"), {"file_path", "old_string", "new_string", "replace_all"})
        self.assertEqual(props("Glob"), {"pattern", "path"})
        self.assertLessEqual({"pattern", "path", "glob", "type", "output_mode", "-i", "-n", "-A", "-B", "-C", "head_limit", "offset", "multiline"}, props("Grep"))
        self.assertEqual(names["Read"]["inputSchema"]["required"], ["file_path"])
        self.assertEqual(names["Edit"]["inputSchema"]["required"], ["file_path", "old_string", "new_string"])
        for t in names.values():
            self.assertTrue(t["description"].strip())

    def test_it_looks_like_a_server_to_the_hub(self):
        self.assertEqual((self.srv.name, self.srv.kind, self.srv.status), ("agent", "builtin", "ready"))
        self.assertTrue(self.srv.start())
        self.assertTrue(self.srv.wants_context)

    def test_a_call_without_a_context_is_refused(self):
        r = self.srv.call("Read", {"file_path": "x"}, 0, None)
        self.assertTrue(r["isError"])

    def test_an_unknown_tool_is_an_error(self):
        self.assertTrue(self.call("Teleport", {})["isError"])


class TheGate(Base):
    def test_a_call_that_asks_waits_for_the_answer_and_runs_only_on_allow(self):
        (self.out / "b.txt").write_text("hello\n", encoding="utf-8")
        r = self.call("Read", {"file_path": str(self.out / "b.txt")})
        self.assertTrue(r["isError"])
        self.assertIn("did not allow", text(r))
        self.assertEqual(len(self.asked), 1)
        self.assertEqual((self.asked[0]["tool"], self.asked[0]["arguments"]["file_path"]), ("Read", str(self.out / "b.txt")))
        self.assertIn("outside", self.asked[0]["why"])
        self.answer = "allow"
        r = self.call("Read", {"file_path": str(self.out / "b.txt")})
        self.assertFalse(r["isError"])
        self.assertIn("hello", text(r))

    def test_what_is_free_does_not_ask(self):
        self.write("src/a.py", "x = 1\n")
        self.call("Read", {"file_path": "src/a.py"})
        self.call("Glob", {"pattern": "**/*.py"})
        self.assertEqual(self.asked, [])

    def test_a_denied_call_is_refused_with_the_reason_and_nothing_is_asked(self):
        self.write("src/a.py", "x = 1\n")
        r = self.call("Read", {"file_path": "src/a.py"}, deny=["Read(src/**)"])
        self.assertTrue(r["isError"])
        self.assertIn("denied", text(r))
        self.assertEqual(self.asked, [])

    def test_plan_mode_refuses_a_write(self):
        r = self.call("Write", {"file_path": "new.txt", "content": "x"}, mode="plan")
        self.assertTrue(r["isError"])
        self.assertIn("plan", text(r))
        self.assertFalse((self.proj / "new.txt").exists())

    def test_the_card_carries_what_the_page_needs(self):
        self.call("Write", {"file_path": str(self.out / "c.txt"), "content": "x"})
        a = self.asked[0]
        self.assertEqual(set(a) >= {"tool", "arguments", "why", "danger", "judgeable", "rule"}, True)
        self.assertEqual(a["rule"], f"Write({self.out.as_posix()}/**)")
        self.assertFalse((self.out / "c.txt").exists())

    def test_the_answer_allow_chat_runs_it_too(self):
        self.answer = "allow_chat"
        self.assertFalse(self.call("Write", {"file_path": str(self.out / "c.txt"), "content": "x"})["isError"])

    def test_a_cancel_while_asking_is_a_refusal(self):
        ctx = self.ctx()
        ctx.cancel.set()
        ctx.ask = lambda req: "cancelled"
        r = self.srv.call("Write", {"file_path": str(self.out / "c.txt"), "content": "x"}, 0, None, ctx)
        self.assertTrue(r["isError"])
        self.assertFalse((self.out / "c.txt").exists())


class Reading(Base):
    def test_lines_are_numbered_like_cat_n(self):
        self.write("a.txt", "one\ntwo\nthree\n")
        self.assertEqual(text(self.call("Read", {"file_path": "a.txt"})), "     1\tone\n     2\ttwo\n     3\tthree")

    def test_offset_and_limit(self):
        self.write("a.txt", "".join(f"l{i}\n" for i in range(1, 11)))
        self.assertEqual(text(self.call("Read", {"file_path": "a.txt", "offset": 4, "limit": 2})), "     4\tl4\n     5\tl5")

    def test_a_long_file_is_cut_at_2000_lines_and_says_so(self):
        self.write("big.txt", "".join(f"l{i}\n" for i in range(1, 2501)))
        t = text(self.call("Read", {"file_path": "big.txt"}))
        self.assertIn("  2000\tl2000", t)
        self.assertNotIn("l2001", t)
        self.assertIn("offset", t)

    def test_a_very_long_line_is_cut(self):
        self.write("w.txt", "x" * 5000 + "\n")
        t = text(self.call("Read", {"file_path": "w.txt"}))
        self.assertLess(len(t), 2300)
        self.assertIn("truncated", t)

    def test_crlf_lines_are_read_as_lines(self):
        self.write("w.txt", "a\r\nb\r\n")
        self.assertEqual(text(self.call("Read", {"file_path": "w.txt"})), "     1\ta\n     2\tb")

    def test_an_empty_file_says_so(self):
        self.write("e.txt", "")
        self.assertIn("empty", text(self.call("Read", {"file_path": "e.txt"})))

    def test_missing_file_a_folder_and_a_binary_file_are_errors_with_a_reason(self):
        r = self.call("Read", {"file_path": "nope.txt"})
        self.assertTrue(r["isError"])
        self.assertIn("does not exist", text(r))
        self.assertIn("directory", text(self.call("Read", {"file_path": "src"})))
        self.write("b.bin", b"\x00\x01\x02abc", raw=True)
        r = self.call("Read", {"file_path": "b.bin"})
        self.assertTrue(r["isError"])
        self.assertIn("binary", text(r))

    def test_a_big_file_needs_a_window(self):
        self.write("huge.txt", "x" * 300_000)
        r = self.call("Read", {"file_path": "huge.txt"})
        self.assertTrue(r["isError"])
        self.assertIn("offset and limit", text(r))
        self.assertFalse(self.call("Read", {"file_path": "huge.txt", "limit": 1})["isError"])

    def test_bad_parameters_are_an_error_not_a_crash(self):
        self.write("a.txt", "x\n")
        for args in ({"file_path": "a.txt", "offset": "x"}, {"file_path": "a.txt", "limit": -3}, {"file_path": "a.txt", "offset": 0}):
            self.assertTrue(self.call("Read", args)["isError"], args)


class Writing(Base):
    def test_a_new_file_is_created_with_its_folders(self):
        r = self.call("Write", {"file_path": "new/deep/f.txt", "content": "hi\n"})
        self.assertFalse(r["isError"])
        self.assertIn("created", text(r))
        self.assertEqual((self.proj / "new/deep/f.txt").read_text(encoding="utf-8"), "hi\n")

    def test_an_existing_file_must_have_been_read_first(self):
        self.write("a.txt", "old\n")
        r = self.call("Write", {"file_path": "a.txt", "content": "new\n"})
        self.assertTrue(r["isError"])
        self.assertIn("not been read", text(r))
        self.assertEqual((self.proj / "a.txt").read_text(), "old\n")
        self.call("Read", {"file_path": "a.txt"})
        r = self.call("Write", {"file_path": "a.txt", "content": "new\n"})
        self.assertFalse(r["isError"])
        self.assertIn("updated", text(r))
        self.assertEqual((self.proj / "a.txt").read_text(), "new\n")

    def test_the_read_belongs_to_the_chat_that_made_it(self):
        self.write("a.txt", "old\n")
        self.call("Read", {"file_path": "a.txt"}, session="one")
        self.assertTrue(self.call("Write", {"file_path": "a.txt", "content": "x"}, session="two")["isError"])

    def test_a_file_changed_after_it_was_read_is_refused(self):
        p = self.write("a.txt", "old\n")
        self.call("Read", {"file_path": "a.txt"})
        time.sleep(0.02)
        p.write_text("someone else\n")
        os.utime(p, ns=(time.time_ns() + 5_000_000_000, time.time_ns() + 5_000_000_000))
        r = self.call("Write", {"file_path": "a.txt", "content": "mine\n"})
        self.assertTrue(r["isError"])
        self.assertIn("modified since", text(r))
        self.assertEqual(p.read_text(), "someone else\n")

    def test_after_a_write_the_file_counts_as_read(self):
        self.call("Write", {"file_path": "a.txt", "content": "1\n"})
        self.assertFalse(self.call("Write", {"file_path": "a.txt", "content": "2\n"})["isError"])

    def test_a_folder_is_not_a_file(self):
        self.assertTrue(self.call("Write", {"file_path": "src", "content": "x"})["isError"])

    def test_content_must_be_text(self):
        self.assertTrue(self.call("Write", {"file_path": "x.txt", "content": 5})["isError"])
        self.assertTrue(self.call("Write", {"file_path": "x.txt"})["isError"])


class Editing(Base):
    def read(self, rel="a.txt"):
        self.call("Read", {"file_path": rel})

    def test_a_unique_string_is_replaced(self):
        p = self.write("a.txt", "alpha\nbeta\ngamma\n")
        self.read()
        r = self.call("Edit", {"file_path": "a.txt", "old_string": "beta", "new_string": "BETA"})
        self.assertFalse(r["isError"])
        self.assertIn("updated successfully", text(r))
        self.assertEqual(p.read_text(), "alpha\nBETA\ngamma\n")
        self.assertIn("BETA", text(r))                                          # a few lines round the change

    def test_it_must_have_been_read(self):
        self.write("a.txt", "alpha\n")
        r = self.call("Edit", {"file_path": "a.txt", "old_string": "alpha", "new_string": "x"})
        self.assertTrue(r["isError"])
        self.assertIn("not been read", text(r))

    def test_a_string_that_is_not_there(self):
        self.write("a.txt", "alpha\n")
        self.read()
        r = self.call("Edit", {"file_path": "a.txt", "old_string": "zzz", "new_string": "x"})
        self.assertTrue(r["isError"])
        self.assertIn("not found", text(r))

    def test_several_matches_need_replace_all_or_more_context(self):
        p = self.write("a.txt", "x = 1\nx = 1\n")
        self.read()
        r = self.call("Edit", {"file_path": "a.txt", "old_string": "x = 1", "new_string": "x = 2"})
        self.assertTrue(r["isError"])
        self.assertIn("2 matches", text(r))
        self.assertIn("replace_all", text(r))
        self.assertEqual(p.read_text(), "x = 1\nx = 1\n")
        r = self.call("Edit", {"file_path": "a.txt", "old_string": "x = 1", "new_string": "x = 2", "replace_all": True})
        self.assertFalse(r["isError"])
        self.assertEqual(p.read_text(), "x = 2\nx = 2\n")

    def test_no_change_is_an_error(self):
        self.write("a.txt", "alpha\n")
        self.read()
        r = self.call("Edit", {"file_path": "a.txt", "old_string": "alpha", "new_string": "alpha"})
        self.assertTrue(r["isError"])
        self.assertIn("same", text(r))

    def test_an_empty_old_string_creates_a_file_that_is_not_there(self):
        r = self.call("Edit", {"file_path": "made.txt", "old_string": "", "new_string": "hello\n"})
        self.assertFalse(r["isError"])
        self.assertEqual((self.proj / "made.txt").read_text(), "hello\n")
        self.write("there.txt", "x\n")
        self.read("there.txt")
        self.assertTrue(self.call("Edit", {"file_path": "there.txt", "old_string": "", "new_string": "y"})["isError"])

    def test_a_missing_file_is_an_error(self):
        r = self.call("Edit", {"file_path": "none.txt", "old_string": "a", "new_string": "b"})
        self.assertTrue(r["isError"])
        self.assertIn("does not exist", text(r))

    def test_crlf_files_keep_their_line_ends(self):
        p = self.write("w.txt", "one\r\ntwo\r\nthree\r\n")
        self.read("w.txt")
        r = self.call("Edit", {"file_path": "w.txt", "old_string": "one\ntwo", "new_string": "ONE\nTWO"})
        self.assertFalse(r["isError"], text(r))
        self.assertEqual(p.read_bytes(), b"ONE\r\nTWO\r\nthree\r\n")

    def test_the_edit_counts_as_a_read_for_the_next_one(self):
        self.write("a.txt", "a b c\n")
        self.read()
        self.call("Edit", {"file_path": "a.txt", "old_string": "a", "new_string": "A"})
        self.assertFalse(self.call("Edit", {"file_path": "a.txt", "old_string": "b", "new_string": "B"})["isError"])

    def test_a_file_changed_after_it_was_read_is_refused(self):
        p = self.write("a.txt", "a b c\n")
        self.read()
        p.write_text("a b c d\n")
        os.utime(p, ns=(time.time_ns() + 5_000_000_000, time.time_ns() + 5_000_000_000))
        self.assertIn("modified since", text(self.call("Edit", {"file_path": "a.txt", "old_string": "a", "new_string": "A"})))


class Globbing(Base):
    def test_matches_newest_first(self):
        a = self.write("src/a.py", "1")
        b = self.write("src/b.py", "2")
        self.write("src/c.txt", "3")
        now = time.time()
        os.utime(a, (now - 100, now - 100))
        os.utime(b, (now, now))
        lines = text(self.call("Glob", {"pattern": "**/*.py"})).splitlines()
        self.assertEqual([Path(x).name for x in lines], ["b.py", "a.py"])

    def test_a_path_narrows_it_and_no_match_says_so(self):
        self.write("src/a.py", "1")
        self.write("top.py", "1")
        self.assertEqual([Path(x).name for x in text(self.call("Glob", {"pattern": "*.py", "path": "src"})).splitlines()], ["a.py"])
        self.assertIn("No files found", text(self.call("Glob", {"pattern": "*.rs"})))

    def test_it_stops_at_100_and_says_so(self):
        for i in range(120):
            self.write(f"m/f{i:03}.txt", "x")
        t = text(self.call("Glob", {"pattern": "m/*.txt"}))
        self.assertEqual(len([x for x in t.splitlines() if x.endswith(".txt")]), 100)
        self.assertIn("truncated", t)

    def test_the_git_folder_is_not_listed(self):
        self.write(".git/HEAD", "ref")
        self.write("a.txt", "x")
        self.assertNotIn(".git", text(self.call("Glob", {"pattern": "**/*"})))

    def test_not_a_folder_is_an_error(self):
        self.assertTrue(self.call("Glob", {"pattern": "*", "path": "nope"})["isError"])
        self.assertTrue(self.call("Glob", {})["isError"])


class Searching(Base):
    def setUp(self):
        super().setUp()
        self.write("src/a.py", "import os\nfoo = 1\nBar = 2\n")
        self.write("src/b.py", "foo()\nfoo()\n")
        self.write("notes.md", "# foo notes\n")
        self.write("node_modules/x/index.js", "foo\n")
        self.write("bin.dat", b"\x00foo\x00", raw=True)

    def names(self, t):
        return sorted(Path(x).name for x in t.splitlines() if x.strip() and not x.startswith("Found"))

    def test_files_with_matches_is_the_default(self):
        self.assertEqual(self.names(text(self.call("Grep", {"pattern": "foo"}))), ["a.py", "b.py", "notes.md"])      # not node_modules, not the binary one

    def test_a_glob_and_a_type_narrow_it(self):
        self.assertEqual(self.names(text(self.call("Grep", {"pattern": "foo", "glob": "*.md"}))), ["notes.md"])
        self.assertEqual(self.names(text(self.call("Grep", {"pattern": "foo", "type": "py"}))), ["a.py", "b.py"])

    def test_content_mode_has_numbers_and_can_ignore_case(self):
        t = text(self.call("Grep", {"pattern": "bar", "output_mode": "content", "-i": True, "path": "src"}))
        self.assertIn(":3:Bar = 2", t)
        self.assertEqual(text(self.call("Grep", {"pattern": "bar", "output_mode": "content", "path": "src"})).strip(), "No matches found")

    def test_context_lines(self):
        t = text(self.call("Grep", {"pattern": "foo = 1", "output_mode": "content", "-C": 1, "path": "src/a.py"}))
        self.assertIn("import os", t)
        self.assertIn("Bar = 2", t)
        self.assertEqual(text(self.call("Grep", {"pattern": "foo = 1", "output_mode": "content", "-A": 1, "path": "src/a.py"})).count("\n"), 1)

    def test_count_mode(self):
        t = text(self.call("Grep", {"pattern": "foo", "output_mode": "count", "path": "src"}))
        self.assertIn("a.py:1", t)
        self.assertIn("b.py:2", t)

    def test_head_limit_and_offset(self):
        t = text(self.call("Grep", {"pattern": "foo", "output_mode": "content", "head_limit": 1, "path": "src"}))
        self.assertEqual(len([x for x in t.splitlines() if "foo" in x]), 1)
        t2 = text(self.call("Grep", {"pattern": "foo", "output_mode": "content", "head_limit": 1, "offset": 1, "path": "src"}))
        self.assertNotEqual(t, t2)

    def test_multiline(self):
        t = text(self.call("Grep", {"pattern": "os\\nfoo", "output_mode": "files_with_matches", "multiline": True}))
        self.assertEqual(self.names(t), ["a.py"])
        self.assertIn("No files found", text(self.call("Grep", {"pattern": "os\\nfoo"})))

    def test_a_bad_pattern_is_an_error(self):
        r = self.call("Grep", {"pattern": "("})
        self.assertTrue(r["isError"])
        self.assertIn("regular expression", text(r))

    def test_a_single_file(self):
        self.assertEqual(self.names(text(self.call("Grep", {"pattern": "foo", "path": "src/b.py"}))), ["b.py"])

    def test_a_pattern_that_takes_forever_is_cut(self):
        self.write("slow.txt", "a" * 30_000 + "!\n")
        t0 = time.time()
        self.call("Grep", {"pattern": "(a+)+$", "path": "slow.txt"})
        self.assertLess(time.time() - t0, 15)


class PlanMode(Base):
    def test_the_user_approving_the_plan_leaves_plan_mode(self):
        ctx = self.ctx(mode="plan")
        events = []
        ctx.emit = events.append
        self.answer = "allow"
        r = self.srv.call("ExitPlanMode", {"plan": "1. do x"}, 0, None, ctx)
        self.assertFalse(r["isError"], text(r))
        self.assertIsNone(ctx.policy.mode)
        self.assertEqual(self.asked[0]["arguments"]["plan"], "1. do x")
        self.assertEqual(events, [{"event": "mode", "mode": "ask"}])
        self.assertFalse(self.srv.call("Write", {"file_path": "n.txt", "content": "x"}, 0, None, ctx)["isError"])      # writes work again

    def test_a_plan_the_user_does_not_approve_keeps_plan_mode(self):
        ctx = self.ctx(mode="plan")
        r = self.srv.call("ExitPlanMode", {"plan": "1. do x"}, 0, None, ctx)
        self.assertTrue(r["isError"])
        self.assertEqual(ctx.policy.mode, "plan")

    def test_outside_plan_mode_there_is_nothing_to_leave(self):
        self.assertTrue(self.call("ExitPlanMode", {"plan": "x"})["isError"])
        self.assertEqual(self.asked, [])


class Notebooks(Base):
    NB = {
        "nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"name": "python3"}},
        "cells": [
            {"id": "a1", "cell_type": "markdown", "metadata": {}, "source": ["# Title\n", "text"]},
            {"id": "b2", "cell_type": "code", "metadata": {}, "execution_count": 1, "source": "print('hi')\nx = 1",
             "outputs": [{"output_type": "stream", "name": "stdout", "text": ["hi\n"]}, {"output_type": "display_data", "data": {"image/png": "AAAA", "text/plain": ["<Figure>"]}, "metadata": {}}]},
            {"id": "c3", "cell_type": "code", "metadata": {}, "execution_count": None, "source": "", "outputs": []},
        ],
    }

    def nb(self, name="n.ipynb"):
        import json
        p = self.proj / name
        p.write_text(json.dumps(self.NB, indent=1) + "\n", encoding="utf-8")
        return p

    def load(self, name="n.ipynb"):
        import json
        return json.loads((self.proj / name).read_text(encoding="utf-8"))

    def test_reading_a_notebook_gives_its_cells_with_their_outputs(self):
        self.nb()
        t = text(self.call("Read", {"file_path": "n.ipynb"}))
        self.assertIn('<cell id="a1" type="markdown">', t)
        self.assertIn("# Title\ntext", t)
        self.assertIn('<cell id="b2" type="code">', t)
        self.assertIn("print('hi')", t)
        self.assertIn("hi", t.split('<cell id="b2"')[1])                      # the stream output
        self.assertIn("<Figure>", t)                                          # the text of a figure
        self.assertNotIn("AAAA", t)                                           # not the picture's bytes
        self.assertIn("image", t.lower())                                     # but it says there was one
        self.assertIn('<cell id="c3" type="code">', t)

    def test_a_notebook_that_is_not_json_is_an_error(self):
        self.write("bad.ipynb", "{ nope")
        r = self.call("Read", {"file_path": "bad.ipynb"})
        self.assertTrue(r["isError"])
        self.assertIn("notebook", text(r).lower())

    def test_the_tool_and_its_parameters(self):
        names = {t["name"]: t for t in self.srv.tools}
        self.assertEqual(set(names["NotebookEdit"]["inputSchema"]["properties"]), {"notebook_path", "new_source", "cell_id", "cell_type", "edit_mode"})
        self.assertEqual(names["NotebookEdit"]["inputSchema"]["required"], ["notebook_path", "new_source"])

    def test_replace_a_cell(self):
        self.nb()
        self.call("Read", {"file_path": "n.ipynb"})
        r = self.call("NotebookEdit", {"notebook_path": "n.ipynb", "cell_id": "b2", "new_source": "print('bye')"})
        self.assertFalse(r["isError"], text(r))
        cell = self.load()["cells"][1]
        self.assertEqual(cell["source"], "print('bye')")
        self.assertEqual(cell["outputs"], [])                                # the old output belongs to the old code
        self.assertIsNone(cell["execution_count"])
        self.assertEqual(self.load()["cells"][0]["source"], ["# Title\n", "text"])      # the other cells are as they were

    def test_replace_can_change_the_type_of_a_cell(self):
        self.nb()
        self.call("Read", {"file_path": "n.ipynb"})
        self.call("NotebookEdit", {"notebook_path": "n.ipynb", "cell_id": "b2", "new_source": "now text", "cell_type": "markdown"})
        cell = self.load()["cells"][1]
        self.assertEqual(cell["cell_type"], "markdown")
        self.assertNotIn("outputs", cell)

    def test_a_cell_can_be_named_by_its_number(self):
        self.nb()
        self.call("Read", {"file_path": "n.ipynb"})
        self.assertFalse(self.call("NotebookEdit", {"notebook_path": "n.ipynb", "cell_id": "cell-0", "new_source": "# New"})["isError"])
        self.assertEqual(self.load()["cells"][0]["source"], "# New")

    def test_insert_goes_after_the_named_cell_or_at_the_start(self):
        self.nb()
        self.call("Read", {"file_path": "n.ipynb"})
        r = self.call("NotebookEdit", {"notebook_path": "n.ipynb", "cell_id": "a1", "new_source": "y = 2", "cell_type": "code", "edit_mode": "insert"})
        self.assertFalse(r["isError"], text(r))
        ids = [c["id"] for c in self.load()["cells"]]
        self.assertEqual(ids[0], "a1")
        self.assertEqual(self.load()["cells"][1]["source"], "y = 2")
        self.assertTrue(ids[1] not in ("a1", "b2", "c3"))                     # it has an id of its own
        self.assertIn(ids[1], text(r))
        self.assertFalse(self.call("NotebookEdit", {"notebook_path": "n.ipynb", "new_source": "first", "cell_type": "markdown", "edit_mode": "insert"})["isError"])
        self.assertEqual(self.load()["cells"][0]["source"], "first")

    def test_insert_needs_a_cell_type(self):
        self.nb()
        self.call("Read", {"file_path": "n.ipynb"})
        r = self.call("NotebookEdit", {"notebook_path": "n.ipynb", "new_source": "x", "edit_mode": "insert"})
        self.assertTrue(r["isError"])
        self.assertIn("cell_type", text(r))

    def test_delete_a_cell(self):
        self.nb()
        self.call("Read", {"file_path": "n.ipynb"})
        r = self.call("NotebookEdit", {"notebook_path": "n.ipynb", "cell_id": "b2", "new_source": "", "edit_mode": "delete"})
        self.assertFalse(r["isError"], text(r))
        self.assertEqual([c["id"] for c in self.load()["cells"]], ["a1", "c3"])

    def test_a_cell_that_is_not_there_is_an_error_that_names_it(self):
        self.nb()
        self.call("Read", {"file_path": "n.ipynb"})
        r = self.call("NotebookEdit", {"notebook_path": "n.ipynb", "cell_id": "zzz", "new_source": "x"})
        self.assertTrue(r["isError"])
        self.assertIn("zzz", text(r))
        self.assertTrue(self.call("NotebookEdit", {"notebook_path": "n.ipynb", "cell_id": "cell-9", "new_source": "x"})["isError"])

    def test_it_must_have_been_read_and_not_changed_since(self):
        self.nb()
        r = self.call("NotebookEdit", {"notebook_path": "n.ipynb", "cell_id": "a1", "new_source": "x"})
        self.assertTrue(r["isError"])
        self.assertIn("not been read", text(r))

    def test_the_file_keeps_the_notebook_format_and_the_rest_of_it(self):
        p = self.nb()
        self.call("Read", {"file_path": "n.ipynb"})
        self.call("NotebookEdit", {"notebook_path": "n.ipynb", "cell_id": "a1", "new_source": "# Ünï"})
        raw = p.read_text(encoding="utf-8")
        self.assertTrue(raw.endswith("\n"))
        self.assertIn("Ünï", raw)                                             # not escaped
        self.assertEqual(self.load()["metadata"], {"kernelspec": {"name": "python3"}})
        self.assertEqual(self.load()["nbformat"], 4)

    def test_it_is_only_for_notebooks_and_the_options_are_checked(self):
        self.write("a.txt", "x")
        self.call("Read", {"file_path": "a.txt"})
        self.assertTrue(self.call("NotebookEdit", {"notebook_path": "a.txt", "cell_id": "a", "new_source": "x"})["isError"])
        self.nb()
        self.call("Read", {"file_path": "n.ipynb"})
        for bad in ({"edit_mode": "explode"}, {"cell_type": "raw-ish"}, {"new_source": 5}):
            args = {"notebook_path": "n.ipynb", "cell_id": "a1", "new_source": "x", **bad}
            self.assertTrue(self.call("NotebookEdit", args)["isError"], bad)

    def test_the_gate_treats_it_like_an_edit_of_that_file(self):
        self.nb()
        self.out.joinpath("o.ipynb").write_text("{}", encoding="utf-8")
        self.call("NotebookEdit", {"notebook_path": str(self.out / "o.ipynb"), "cell_id": "a", "new_source": "x"})
        self.assertEqual(len(self.asked), 1)
        self.assertEqual(self.asked[0]["tool"], "NotebookEdit")
        self.assertEqual(self.asked[0]["rule"], f"NotebookEdit({self.out.as_posix()}/**)")
        r = self.call("NotebookEdit", {"notebook_path": "n.ipynb", "cell_id": "a1", "new_source": "x"}, mode="plan")
        self.assertTrue(r["isError"])
        self.assertIn("plan", text(r))


class Todos(Base):
    def test_the_list_is_kept_for_the_chat_and_replaced_whole(self):
        todos = [{"content": "a", "status": "in_progress", "activeForm": "Doing a"}, {"content": "b", "status": "pending", "activeForm": "Doing b"}]
        r = self.call("TodoWrite", {"todos": todos})
        self.assertFalse(r["isError"])
        self.assertEqual(self.srv.todos("s1"), todos)
        self.call("TodoWrite", {"todos": todos[:1]})
        self.assertEqual(len(self.srv.todos("s1")), 1)
        self.assertEqual(self.srv.todos("other"), [])

    def test_a_bad_list_is_an_error(self):
        self.assertTrue(self.call("TodoWrite", {"todos": "x"})["isError"])
        self.assertTrue(self.call("TodoWrite", {"todos": [{"content": "a", "status": "nope", "activeForm": "A"}]})["isError"])
        self.assertTrue(self.call("TodoWrite", {"todos": [{"content": "", "status": "pending", "activeForm": "A"}]})["isError"])


if __name__ == "__main__":
    unittest.main()
