"""The commit-time blueprint check (tools/blueprint_check.py): a commit that changes what docs/BLUEPRINT.md describes
must update it, or say in a `Blueprint: n/a - <why>` trailer why it does not (the developer, 2026-09-30)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
import blueprint_check as bc  # noqa: E402


def diff(path, *lines, new=False):
    head = f"diff --git a/{path} b/{path}\n" + ("new file mode 100644\n--- /dev/null\n" if new else f"--- a/{path}\n")
    return head + f"+++ b/{path}\n@@ -1,0 +1,{len(lines)} @@\n" + "\n".join(lines) + "\n"


FLAG = diff("src/program/generate.cpp", '+        else if (a == "--ban-ids") o.ban_ids = next("--ban-ids");')


class Triggers(unittest.TestCase):
    def test_a_new_engine_flag(self):
        self.assertTrue(any("--ban-ids" in r for r in bc.reasons(FLAG)))

    def test_a_removed_engine_flag(self):
        d = diff("src/program/generate.cpp", '-        else if (a == "--old-flag") o.old = true;')
        self.assertTrue(any("--old-flag" in r for r in bc.reasons(d)))

    def test_a_gen_key(self):
        d = diff("src/program/generate.cpp", '+                    else if (key == "ban") req_ban = 1;')
        self.assertTrue(any("ban" in r for r in bc.reasons(d)))

    def test_an_environment_variable_in_the_engine_and_the_server(self):
        self.assertTrue(bc.reasons(diff("src/prefill/prefill.cpp", '+    const char* v = std::getenv("STRATA_NEW");')))
        self.assertTrue(bc.reasons(diff("serve/server.py", '+    x = os.environ.get("STRATA_SIDE_BUDGET", "1")')))

    def test_a_run_config_key(self):
        d = diff("serve/server.py", '+        if cfg.get("cjk_guard"):')
        self.assertTrue(any("cjk_guard" in r for r in bc.reasons(d)))

    def test_a_new_source_file(self):
        self.assertTrue(bc.reasons(diff("serve/think_budget.py", "+import os", new=True)))

    def test_an_edited_line_that_keeps_its_key_is_not_a_change(self):
        # seen on 1620329: moving cfg["args"] into a local listed args/exe/cwd as removed AND added
        d = diff("serve/server.py", '-        engine = StrataEngine(cfg["exe"], cfg["args"])',
                 '+        engine = StrataEngine(cfg["exe"], args)', '+        args = list(cfg["args"])')
        self.assertEqual(bc.reasons(d), [])

    def test_nothing_structural(self):
        d = diff("serve/server.py", "+        # a comment about cfg and --flags", "+        x = 1") + \
            diff("tests/xeno/test_x.py", '+    self.assertEqual(a == "--flag", True)', new=True) + \
            diff("docs/reports/r.md", '+ a == "--flag"')
        self.assertEqual(bc.reasons(d), [])


class Verdict(unittest.TestCase):
    def test_a_trigger_without_the_blueprint_fails(self):
        self.assertTrue(bc.check(FLAG, ["src/program/generate.cpp"], "feat: a flag\n"))

    def test_the_blueprint_in_the_commit_passes(self):
        self.assertEqual(bc.check(FLAG, ["src/program/generate.cpp", "docs/BLUEPRINT.md"], "feat: a flag\n"), [])

    def test_a_reasoned_trailer_passes_and_a_bare_one_does_not(self):
        self.assertEqual(bc.check(FLAG, ["src/program/generate.cpp"],
                                  "feat: a flag\n\nBlueprint: n/a - an A/B flag removed next commit\n"), [])
        self.assertTrue(bc.check(FLAG, ["src/program/generate.cpp"], "feat: a flag\n\nBlueprint: n/a\n"))

    def test_no_trigger_needs_nothing(self):
        self.assertEqual(bc.check(diff("serve/server.py", "+        x = 1"), ["serve/server.py"], "fix\n"), [])


if __name__ == "__main__":
    unittest.main()
