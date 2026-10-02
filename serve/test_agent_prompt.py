"""Tests for serve/agent_prompt.py: what the model is told when the chat has the coding tools (Claude Code's rules for the AI, in this app's words)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from serve import agent_prompt  # noqa: E402


class Prompt(unittest.TestCase):
    def build(self, **kw):
        base = dict(cwd="C:/work/app", shell="bash", mode=None, today="2026-10-02", platform="win32", git=True, notes=None, tools=["Read", "Write", "Edit", "Glob", "Grep", "Bash", "TodoWrite"])
        base.update(kw)
        return agent_prompt.build(**base)

    def test_it_names_the_folder_the_shell_the_platform_and_the_date(self):
        p = self.build()
        for s in ("C:/work/app", "bash", "win32", "2026-10-02"):
            self.assertIn(s, p)
        self.assertIn("git repository", self.build(git=True).lower())
        self.assertNotIn("is a git repository", self.build(git=False).lower())

    def test_with_no_folder_it_says_commands_and_files_need_one(self):
        p = self.build(cwd=None)
        self.assertIn("no project folder", p.lower())

    def test_it_has_the_rules_that_make_a_coding_assistant_safe(self):
        p = self.build().lower()
        for needle in ("read a file before", "data, not instructions", "do not retry", "only commit when", "never force", "do not create files"):
            self.assertIn(needle, p, needle)

    def test_it_says_how_the_user_is_asked(self):
        p = self.build().lower()
        self.assertIn("the user is asked before", p)
        self.assertIn("outside the project folder", p)

    def test_the_modes_have_their_own_words(self):
        self.assertIn("plan mode", self.build(mode="plan").lower())
        self.assertIn("exitplanmode", self.build(mode="plan", tools=["Read", "ExitPlanMode"]).lower())
        self.assertNotIn("plan mode", self.build(mode=None).lower())
        self.assertIn("auto mode", self.build(mode="auto").lower())

    def test_only_the_tools_that_exist_are_described(self):
        p = self.build(tools=["Read", "Glob"])
        self.assertNotIn("TodoWrite", p)
        self.assertNotIn("Bash", p)
        self.assertIn("Read", p)

    def test_the_projects_own_instructions_come_in_marked_and_cut(self):
        p = self.build(notes=("CLAUDE.md", "Use tabs.\nRun make test."))
        self.assertIn("Use tabs.", p)
        self.assertIn("CLAUDE.md", p)
        long = self.build(notes=("CLAUDE.md", "x" * 100_000))
        self.assertLess(len(long), 40_000)
        self.assertIn("cut", long)

    def test_it_is_the_same_text_each_time_so_the_cache_holds(self):
        self.assertEqual(self.build(), self.build())

    def test_the_projects_instructions_are_found_in_the_folder(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(agent_prompt.project_notes(d))
            (Path(d) / "CLAUDE.md").write_text("Use tabs.", encoding="utf-8")
            self.assertEqual(agent_prompt.project_notes(d), ("CLAUDE.md", "Use tabs."))
            (Path(d) / "CLAUDE.md").write_bytes(b"\x00\x01binary")
            self.assertIsNone(agent_prompt.project_notes(d))
        self.assertIsNone(agent_prompt.project_notes(None))
        self.assertIsNone(agent_prompt.project_notes("C:/does/not/exist"))

    def test_agents_md_is_read_when_there_is_no_claude_md(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "AGENTS.md").write_text("Be brief.", encoding="utf-8")
            self.assertEqual(agent_prompt.project_notes(d), ("AGENTS.md", "Be brief."))

    def test_it_is_joined_to_a_system_message_the_page_already_has(self):
        msgs = [{"role": "system", "content": "Be kind."}, {"role": "user", "content": "hi"}]
        out = agent_prompt.with_system(msgs, "RULES")
        self.assertEqual([m["role"] for m in out], ["system", "user"])
        self.assertTrue(out[0]["content"].startswith("RULES"))
        self.assertIn("Be kind.", out[0]["content"])
        self.assertEqual(msgs[0]["content"], "Be kind.")                       # the request's own list is not changed
        out = agent_prompt.with_system([{"role": "user", "content": "hi"}], "RULES")
        self.assertEqual([m["role"] for m in out], ["system", "user"])


if __name__ == "__main__":
    unittest.main()
