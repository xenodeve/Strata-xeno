"""The way back for the files the chat's tools change (serve/checkpoints.py): what is kept, what a rewind puts back, what it leaves alone, and the routes."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from serve import checkpoints  # noqa: E402
from serve.test_agent_chat import DONE, Fixture  # noqa: E402
from serve.test_harness import tool_call  # noqa: E402


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name).resolve()
        self.proj = self.base / "proj"
        self.proj.mkdir()
        self.store = checkpoints.Checkpoints(self.base / "cp")

    def put(self, name: str, text: str) -> str:
        p = self.proj / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(text.encode())
        return str(p)

    def read(self, path: str) -> str | None:
        return Path(path).read_bytes().decode() if os.path.exists(path) else None

    def change(self, session: str, cp: str, path: str, text: str):
        """What the tools do: keep what is there, write, note it."""
        self.store.before(session, cp, path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(text.encode())
        self.store.after(session, cp, path)


class Rewinding(Base):
    def test_a_changed_file_is_restored_and_a_made_file_is_deleted(self):
        a = self.put("a.txt", "original")
        new = str(self.proj / "new.txt")
        self.change("s1", "p1", a, "changed")
        self.change("s1", "p1", new, "made")
        pre = self.store.preview("s1", "p1")
        self.assertEqual({(f["action"], os.path.basename(f["path"]), f["changed"]) for f in pre["files"]}, {("restore", "a.txt", False), ("delete", "new.txt", False)})
        out = self.store.apply("s1", "p1")
        self.assertEqual((out["restored"], out["deleted"], out["kept"], out["failed"]), (1, 1, [], []))
        self.assertEqual(self.read(a), "original")
        self.assertIsNone(self.read(new))

    def test_only_the_first_state_in_a_prompt_is_kept(self):
        a = self.put("a.txt", "v0")
        self.change("s1", "p1", a, "v1")
        self.change("s1", "p1", a, "v2")                                    # changed twice in one prompt
        self.store.apply("s1", "p1")
        self.assertEqual(self.read(a), "v0")

    def test_rewinding_to_an_earlier_prompt_undoes_the_later_ones_too(self):
        a = self.put("a.txt", "A")
        b = str(self.proj / "b.txt")
        self.change("s1", "p1", a, "B")                                     # prompt 1
        self.change("s1", "p2", a, "C")                                     # prompt 2 changes it again and makes a file
        self.change("s1", "p2", b, "made in 2")
        self.change("s1", "p3", a, "D")                                     # prompt 3
        out = self.store.apply("s1", "p2")                                  # back to before prompt 2
        self.assertEqual((out["restored"], out["deleted"]), (1, 1))
        self.assertEqual(self.read(a), "B")
        self.assertIsNone(self.read(b))
        self.assertEqual([c["id"] for c in self.store.listing("s1")], ["p1"])      # prompts 2 and 3 are done with
        self.store.apply("s1", "p1")
        self.assertEqual(self.read(a), "A")

    def test_a_file_someone_else_changed_since_is_left_unless_asked_for(self):
        a = self.put("a.txt", "original")
        self.change("s1", "p1", a, "by the tools")
        Path(a).write_bytes(b"by the user afterwards")                      # not the tools
        pre = self.store.preview("s1", "p1")
        self.assertTrue(pre["files"][0]["changed"])
        out = self.store.apply("s1", "p1")
        self.assertEqual((out["restored"], len(out["kept"])), (0, 1))
        self.assertEqual(self.read(a), "by the user afterwards")
        self.change("s1", "p2", a, "again")                                  # (the prompts were done with; a new one starts)
        Path(a).write_bytes(b"edited again by the user")
        out = self.store.apply("s1", "p2", include_changed=True)
        self.assertEqual(out["restored"], 1)

    def test_a_file_that_was_taken_away_is_back_only_when_asked_for(self):
        a = self.put("a.txt", "original")
        self.change("s1", "p1", a, "changed")
        os.remove(a)
        self.assertEqual(self.store.apply("s1", "p1")["kept"], [a])
        self.assertIsNone(self.read(a))
        self.assertEqual(self.store.apply("s1", "p1", include_changed=True)["restored"], 0)       # p1 was done with by the first apply

    def test_a_file_in_a_new_folder_is_deleted_and_the_rewind_does_not_fail_when_it_is_already_gone(self):
        n = str(self.proj / "deep" / "er" / "new.txt")
        self.change("s1", "p1", n, "x")
        os.remove(n)
        out = self.store.apply("s1", "p1", include_changed=True)
        self.assertEqual((out["deleted"], out["failed"]), (0, []))

    def test_nothing_known_is_nothing_to_do(self):
        self.assertEqual(self.store.preview("nope", "p1"), {"ok": True, "files": [], "known": False})
        self.assertEqual(self.store.apply("nope", "p1")["restored"], 0)
        self.change("s1", "p1", self.put("a.txt", "x"), "y")
        self.assertEqual(self.store.preview("s1", "other"), {"ok": True, "files": [], "known": False})


class WhatIsKept(Base):
    def test_a_file_that_is_too_big_is_listed_as_one_that_cannot_be_restored(self):
        big = self.put("big.bin", "x" * 10)
        old = checkpoints.MAX_FILE
        checkpoints.MAX_FILE = 5
        self.addCleanup(setattr, checkpoints, "MAX_FILE", old)
        self.change("s1", "p1", big, "y")
        f = self.store.preview("s1", "p1")["files"][0]
        self.assertEqual((f["action"], f["why"]), ("skip", "it is too big to keep"))
        self.assertEqual(self.store.apply("s1", "p1")["skipped"], [big])
        self.assertEqual(self.read(big), "y")

    def test_no_more_files_than_the_limit_are_kept_for_one_prompt(self):
        old = checkpoints.MAX_FILES
        checkpoints.MAX_FILES = 2
        self.addCleanup(setattr, checkpoints, "MAX_FILES", old)
        paths = [self.put(f"f{i}.txt", "o") for i in range(4)]
        for p in paths:
            self.change("s1", "p1", p, "n")
        actions = [f["action"] for f in self.store.preview("s1", "p1")["files"]]
        self.assertEqual(sorted(actions), ["restore", "restore", "skip", "skip"])

    def test_ids_that_are_not_plain_are_not_kept(self):
        a = self.put("a.txt", "o")
        for session, cp in (("../x", "p1"), ("s1", "../p"), ("", "p1"), ("s1", "x" * 100), ("s/1", "p1")):
            self.store.before(session, cp, a)
            self.assertIsNone(self.store.scope(session, cp))
        self.assertFalse((self.base / "x").exists())
        self.assertEqual(self.store.listing("../x"), [])

    def test_a_chat_that_is_gone_takes_its_checkpoints(self):
        self.change("s1", "p1", self.put("a.txt", "o"), "n")
        self.assertEqual(len(self.store.listing("s1")), 1)
        self.store.forget("s1")
        self.assertEqual(self.store.listing("s1"), [])
        self.store.forget("never-was")

    def test_the_oldest_prompts_and_chats_are_trimmed(self):
        for i in range(checkpoints.KEEP_PROMPTS + 5):
            self.store.before("s1", f"p{i:03}", self.put(f"f{i}.txt", "o"))
        self.store.trim()
        ids = [c["id"] for c in self.store.listing("s1")]
        self.assertEqual((len(ids), ids[0]), (checkpoints.KEEP_PROMPTS, "p005"))
        for i in range(checkpoints.KEEP_CHATS + 3):
            self.store.before(f"chat{i:02}", "p1", self.put(f"c{i}.txt", "o"))
        self.store.trim()
        self.assertLessEqual(len([p for p in (self.base / "cp").iterdir()]), checkpoints.KEEP_CHATS)

    def test_what_is_kept_survives_a_restart(self):
        a = self.put("a.txt", "original")
        self.change("s1", "p1", a, "changed")
        again = checkpoints.Checkpoints(self.base / "cp")
        self.assertEqual(again.apply("s1", "p1")["restored"], 1)
        self.assertEqual(self.read(a), "original")


class WithTheChat(Fixture):
    def setUp(self):
        super().setUp()
        self.keep = checkpoints.Checkpoints(self.base / "cps")

    def go(self, *scripts):
        self.start(*scripts)
        self.svc.checkpoints = self.keep

    def call(self, path, body):
        try:
            with urllib.request.urlopen(self.post(path, body), timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_what_the_tools_write_in_a_prompt_can_be_put_back(self):
        target = self.proj / "a.txt"                                                   # exists: "hello\nworld\n"
        before = target.read_bytes()
        made = self.proj / "made.txt"
        self.go(tool_call("Read", file_path=str(target)), tool_call("Edit", file_path=str(target), old_string="world", new_string="there"),
                tool_call("Write", file_path=str(made), content="new file"), DONE)
        status, _, _ = self.chat(self.body(agent={"checkpoint": "prompt-1"}))
        self.assertEqual(status, 200)
        self.assertIn(b"there", target.read_bytes())
        self.assertTrue(made.exists())
        code, pre = self.call("/agent/rewind", {"session": "chat-1", "checkpoint": "prompt-1"})
        self.assertEqual(code, 200)
        self.assertEqual({(f["action"], os.path.basename(f["path"])) for f in pre["files"]}, {("restore", "a.txt"), ("delete", "made.txt")})
        code, out = self.call("/agent/rewind", {"session": "chat-1", "checkpoint": "prompt-1", "apply": True})
        self.assertEqual((code, out["restored"], out["deleted"]), (200, 1, 1))
        self.assertEqual(target.read_bytes(), before)
        self.assertFalse(made.exists())

    def test_a_request_without_a_checkpoint_keeps_nothing(self):
        target = self.proj / "a.txt"
        self.go(tool_call("Read", file_path=str(target)), tool_call("Edit", file_path=str(target), old_string="world", new_string="there"), DONE)
        self.chat()
        self.assertEqual(self.keep.listing("chat-1"), [])

    def test_a_change_that_is_refused_is_not_kept(self):
        outside = self.outside / "b.txt"
        self.go(tool_call("Write", file_path=str(outside), content="x"), DONE)
        seen = []
        self.chat(self.body(agent={"checkpoint": "p1"}), on_event=lambda e: e["event"] == "permission" and (seen.append(e), self.answer(e["id"], "deny")))
        self.assertEqual(len(seen), 1)
        self.assertEqual(self.keep.preview("chat-1", "p1")["files"], [])
        self.assertEqual(outside.read_text(encoding="utf-8"), "secret-ish\n")

    def test_forgetting_a_chat_and_odd_bodies(self):
        self.go(DONE)
        self.keep.before("chat-1", "p1", str(self.proj / "a.txt"))
        self.assertEqual(self.call("/agent/checkpoints/forget", {"session": "chat-1"})[0], 200)
        self.assertEqual(self.keep.listing("chat-1"), [])
        for body in ({}, {"session": 5}, [], {"session": "s"}):
            self.assertEqual(self.call("/agent/rewind", body)[0], 400, body)

    def test_a_foreign_origin_is_refused(self):
        self.go(DONE)
        req = self.post("/agent/rewind", {"session": "s", "checkpoint": "p"}, {"Origin": "http://evil.example"})
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req, timeout=10)
        self.assertEqual(cm.exception.code, 403)


if __name__ == "__main__":
    unittest.main()
