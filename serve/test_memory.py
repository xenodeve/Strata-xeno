"""The memory and instruction files the chat reads (serve/memory.py): the project's own, other apps' (off until switched on), what is imported, what is cut, the sub-folders', and what
reaches the model and the page."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from serve import memory  # noqa: E402
from serve.test_agent_chat import DONE, Fixture  # noqa: E402
from serve.test_harness import tool_call  # noqa: E402


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name).resolve()
        self.home = self.base / "home"
        self.proj = self.base / "proj"
        self.home.mkdir()
        self.proj.mkdir()

    def write(self, path: Path, text: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path


class Switches(unittest.TestCase):
    def test_nothing_of_another_app_is_on_until_it_is_listed(self):
        self.assertEqual(memory.settings({}), {"on": []})
        self.assertEqual(memory.settings({"import": {"memory": {"on": ["claude:memory", 5, "", "x" * 200]}}}), {"on": ["claude:memory"]})
        self.assertEqual(memory.settings({"import": "x"}), {"on": []})
        self.assertEqual(memory.settings(None), {"on": []})

    def test_what_a_page_sends_is_checked_strictly(self):
        ok, errors = memory.check_settings({"on": ["claude:memory", "codex:instructions"]})
        self.assertEqual((ok, errors), ({"on": ["claude:memory", "codex:instructions"]}, []))
        for bad in ("x", {"on": "claude:memory"}, {"on": ["../etc"]}, {"on": ["claude:other"]}, {"on": [1]}, {"other": 1}, {"on": ["claude:memory"] * 101}):
            self.assertIsNone(memory.check_settings(bad)[0], bad)


class Discover(Base):
    def test_the_project_files_are_always_on_and_in_order(self):
        self.write(self.proj / "CLAUDE.md", "main rules")
        self.write(self.proj / "CLAUDE.local.md", "my rules")
        self.write(self.proj / ".claude" / "rules" / "b.md", "rule b")
        self.write(self.proj / ".claude" / "rules" / "a.md", "rule a")
        self.write(self.proj / ".claude" / "rules" / "skip.txt", "not a rule")
        src = [s for s in memory.discover(str(self.home), [str(self.proj)], []) if s["kind"] == "project"][0]
        self.assertTrue(src["on"])
        self.assertEqual(src["shown"], ["CLAUDE.md", "CLAUDE.local.md", ".claude/rules/a.md", ".claude/rules/b.md"])

    def test_agents_md_is_read_when_there_is_no_claude_md(self):
        self.write(self.proj / "AGENTS.md", "agents")
        self.assertEqual([s["shown"] for s in memory.discover(str(self.home), [str(self.proj)], [])], [["AGENTS.md"]])
        self.write(self.proj / "CLAUDE.md", "claude")
        self.assertEqual([s["shown"] for s in memory.discover(str(self.home), [str(self.proj)], [])], [["CLAUDE.md"]])

    def test_every_folder_of_the_project_has_its_own(self):
        wt = self.base / "proj-wt"
        self.write(self.proj / "CLAUDE.md", "one")
        self.write(wt / "CLAUDE.md", "two")
        got = [(s["id"], s["label"]) for s in memory.discover(str(self.home), [str(self.proj), str(wt)], [])]
        self.assertEqual(got, [("project:0", "proj"), ("project:1", "proj-wt")])

    def test_other_apps_are_listed_and_off(self):
        self.write(self.home / ".claude" / "CLAUDE.md", "global claude")
        self.write(self.home / ".codex" / "AGENTS.md", "codex")
        found = {s["id"]: s for s in memory.discover(str(self.home), [], [])}
        self.assertEqual(set(found), {"claude:instructions", "codex:instructions"})
        self.assertFalse(found["claude:instructions"]["on"] or found["codex:instructions"]["on"])
        self.assertEqual(found["claude:instructions"]["shown"], ["~/.claude/CLAUDE.md"])
        on = {s["id"]: s["on"] for s in memory.discover(str(self.home), [], ["codex:instructions"])}
        self.assertEqual(on, {"claude:instructions": False, "codex:instructions": True})

    def test_claude_codes_memory_of_a_project_is_found_by_its_folder_name(self):
        d = self.home / ".claude" / "projects" / memory.slug(str(self.proj)) / "memory"
        self.write(d / "zeta.md", "z")
        self.write(d / "MEMORY.md", "- [zeta](zeta.md)")
        self.write(d / "alpha.md", "a")
        src = [s for s in memory.discover(str(self.home), [str(self.proj)], []) if s["kind"] == "memory"][0]
        self.assertEqual([os.path.basename(f) for f in src["files"]], ["MEMORY.md", "alpha.md", "zeta.md"])          # the index first
        self.assertFalse(src["on"])

    def test_slug_is_the_path_with_everything_but_letters_and_digits_as_a_dash(self):
        self.assertEqual(memory.slug("C:/"), memory.slug("C:\\"))
        self.assertTrue(set(memory.slug(str(self.proj))) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-"))


class Collect(Base):
    def test_the_project_notes_come_with_what_they_import_inside_the_folder_only(self):
        self.write(self.proj / "CLAUDE.md", "Use tabs.\n@docs/style.md\n@../outside.md\n")
        self.write(self.proj / "docs" / "style.md", "Style: short lines.")
        self.write(self.base / "outside.md", "SECRET OUTSIDE")
        text = memory.collect(str(self.home), [str(self.proj)], [])[0]["text"]
        self.assertIn("Use tabs.", text)
        self.assertIn("Style: short lines.", text)
        self.assertNotIn("SECRET OUTSIDE", text)

    def test_imports_do_not_loop_and_stop_after_a_few_levels(self):
        self.write(self.proj / "CLAUDE.md", "root @a.md")
        self.write(self.proj / "a.md", "A @b.md")
        self.write(self.proj / "b.md", "B @a.md @c.md")
        self.write(self.proj / "c.md", "C @d.md")
        self.write(self.proj / "d.md", "D @e.md")
        self.write(self.proj / "e.md", "E")
        text = memory.collect(str(self.home), [str(self.proj)], [])[0]["text"]
        for want in ("root", "A", "B", "C"):
            self.assertIn(want, text)
        self.assertEqual(text.count("[imported from a.md]"), 1)
        self.assertNotIn("\nE", text)                                                          # five levels down is not followed

    def test_other_apps_notes_come_only_when_switched_on(self):
        self.write(self.home / ".claude" / "CLAUDE.md", "I like haiku.")
        self.assertEqual(memory.collect(str(self.home), [str(self.proj)], []), [])
        got = memory.collect(str(self.home), [str(self.proj)], ["claude:instructions"])
        self.assertEqual([(b["kind"], b["label"], b["name"]) for b in got], [("instructions", "Claude Code", "CLAUDE.md")])
        self.assertEqual(got[0]["text"], "I like haiku.")

    def test_a_memory_that_is_on_brings_its_index_and_topics(self):
        d = self.home / ".claude" / "projects" / memory.slug(str(self.proj)) / "memory"
        self.write(d / "MEMORY.md", "index")
        self.write(d / "topic.md", "topic text")
        got = memory.collect(str(self.home), [str(self.proj)], ["claude:memory"])
        self.assertEqual([b["name"] for b in got], ["MEMORY.md", "topic.md"])
        self.assertEqual(memory.collect(str(self.home), [str(self.proj)], []), [])

    def test_a_long_file_is_cut_and_says_so_and_a_binary_one_is_left_out(self):
        self.write(self.proj / "CLAUDE.md", "x" * (memory.MAX_FILE + 500))
        self.write(self.proj / "CLAUDE.local.md", "ok")
        (self.proj / ".claude").mkdir()
        (self.proj / ".claude" / "CLAUDE.md").write_bytes(b"\0\1\2 binary")
        blocks = memory.collect(str(self.home), [str(self.proj)], [])
        self.assertEqual([b["name"] for b in blocks], ["CLAUDE.md", "CLAUDE.local.md"])
        self.assertTrue(blocks[0]["cut"])
        self.assertEqual(len(blocks[0]["text"]), memory.MAX_FILE)

    def test_all_of_it_together_is_cut_to_a_total(self):
        for name in ("CLAUDE.md", "CLAUDE.local.md"):
            self.write(self.proj / name, "y" * memory.MAX_FILE)
        self.write(self.proj / ".claude" / "CLAUDE.md", "z" * memory.MAX_FILE)
        for i in range(3):
            self.write(self.proj / ".claude" / "rules" / f"r{i}.md", "w" * memory.MAX_FILE)
        total = sum(len(b["text"]) for b in memory.collect(str(self.home), [str(self.proj)], []))
        self.assertLessEqual(total, memory.MAX_TOTAL)


class Nested(Base):
    def test_a_file_in_a_subfolder_brings_that_subfolders_rules_once(self):
        self.write(self.proj / "CLAUDE.md", "root rules")
        self.write(self.proj / "api" / "CLAUDE.md", "api rules")
        self.write(self.proj / "api" / "v1" / "AGENTS.md", "v1 rules")
        loaded: set = set()
        got = memory.nested(str(self.proj / "api" / "v1" / "x.py"), [str(self.proj)], loaded)
        self.assertIn("api rules", got)
        self.assertIn("v1 rules", got)
        self.assertNotIn("root rules", got)                                                  # the folder's own were read at the start
        self.assertLess(got.index("api rules"), got.index("v1 rules"))                       # the outer folder first
        self.assertIsNone(memory.nested(str(self.proj / "api" / "v1" / "y.py"), [str(self.proj)], loaded))      # once

    def test_a_file_at_the_top_or_outside_the_project_brings_nothing(self):
        self.write(self.proj / "CLAUDE.md", "root rules")
        self.write(self.base / "elsewhere" / "CLAUDE.md", "other")
        self.assertIsNone(memory.nested(str(self.proj / "top.py"), [str(self.proj)], set()))
        self.assertIsNone(memory.nested(str(self.base / "elsewhere" / "f.py"), [str(self.proj)], set()))
        self.assertIsNone(memory.nested(str(self.proj / "x.py"), [], set()))


class WithTheChat(Fixture):
    def setUp(self):
        super().setUp()
        self.home = self.base / "home"
        self.home.mkdir()
        self.cfg = self.base / "run.json"
        self.cfg.write_text("{}", encoding="utf-8")

    def settings(self, on):
        self.cfg.write_text(json.dumps({"import": {"memory": {"on": on}}}), encoding="utf-8")

    def go(self, *scripts):
        self.start(*scripts)
        self.svc.importer = SimpleNamespace(home=self.home)
        self.svc.config_path = str(self.cfg)

    def test_the_prompt_has_the_projects_notes_and_only_the_other_apps_notes_that_are_on(self):
        (self.proj / "CLAUDE.md").write_text("Project: use tabs.", encoding="utf-8")
        (self.home / ".claude").mkdir()
        (self.home / ".claude" / "CLAUDE.md").write_text("Global: I like haiku.", encoding="utf-8")
        self.go(DONE)
        self.chat()
        p = self.engine.prompt_text(0)
        self.assertIn("Project: use tabs.", p)
        self.assertNotIn("Global: I like haiku.", p)
        self.settings(["claude:instructions"])
        self.chat()
        p2 = self.engine.prompt_text(1)
        self.assertIn("Global: I like haiku.", p2)
        self.assertIn("Notes from files on this PC", p2)

    def test_the_other_folders_of_the_project_bring_theirs(self):
        wt = self.base / "proj-wt"
        wt.mkdir()
        (wt / "CLAUDE.md").write_text("Worktree rule: be brief.", encoding="utf-8")
        self.go(DONE)
        self.chat(self.body(agent={"dirs": [str(wt)]}))
        self.assertIn("Worktree rule: be brief.", self.engine.prompt_text(0))

    def test_reading_a_file_in_a_subfolder_hands_over_its_rules_once(self):
        sub = self.proj / "api"
        sub.mkdir()
        (sub / "CLAUDE.md").write_text("API rule: no globals.", encoding="utf-8")
        (sub / "a.py").write_text("x = 1\n", encoding="utf-8")
        (sub / "b.py").write_text("y = 2\n", encoding="utf-8")
        self.go(tool_call("Read", file_path=str(sub / "a.py")), tool_call("Read", file_path=str(sub / "b.py")), DONE)
        status, _, events = self.chat()
        self.assertEqual(status, 200)
        results = [e["text"] for e in events if e["event"] == "result"]
        self.assertEqual(len(results), 2)
        self.assertIn("API rule: no globals.", results[0])
        self.assertNotIn("API rule: no globals.", results[1])

    def get(self, path):
        try:
            with urllib.request.urlopen(urllib.request.Request(self.base_url + path), timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_the_page_can_list_the_sources_and_read_one(self):
        (self.proj / "CLAUDE.md").write_text("Project: use tabs.", encoding="utf-8")
        (self.home / ".claude").mkdir()
        (self.home / ".claude" / "CLAUDE.md").write_text("Global: I like haiku.", encoding="utf-8")
        self.go(DONE)
        status, body = self.get("/agent/memory?path=" + urllib.parse.quote(str(self.proj)))
        self.assertEqual(status, 200)
        got = {s["id"]: s for s in body["sources"]}
        self.assertEqual(set(got), {"project:0", "claude:instructions"})
        self.assertTrue(got["project:0"]["on"])
        self.assertFalse(got["claude:instructions"]["on"])
        self.assertNotIn("files", got["project:0"])                                          # no absolute paths of files outside are handed over by the list
        status, f = self.get("/agent/memory/file?path=" + urllib.parse.quote(str(self.proj)) + "&id=claude:instructions&n=0")
        self.assertEqual((status, f["ok"], f["text"]), (200, True, "Global: I like haiku."))

    def test_only_a_listed_file_can_be_read(self):
        self.go(DONE)
        for q in ("&id=nope&n=0", "&id=project:0&n=0", "&id=claude:instructions&n=5", "&id=claude:instructions&n=x", "&id=../../etc&n=0"):
            status, body = self.get("/agent/memory/file?path=" + urllib.parse.quote(str(self.proj)) + q)
            self.assertEqual(status, 404, q)

    def test_it_is_for_this_pc_only(self):
        self.go(DONE)
        req = urllib.request.Request(self.base_url + "/agent/memory?path=" + urllib.parse.quote(str(self.proj)), headers={"Host": "strata.example"})
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req, timeout=10)
        self.assertIn(cm.exception.code, (403, 421))


if __name__ == "__main__":
    unittest.main()
