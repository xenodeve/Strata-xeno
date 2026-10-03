"""Tests for serve/shell.py: the Bash tool (and BashOutput, KillShell) of the chat's coding tools, with Claude Code's parameters.

They run real commands through Git Bash (or bash) when there is one; without a shell the tests are skipped."""
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

from serve import agent, shell  # noqa: E402
from serve.permissions import Policy  # noqa: E402

SHELL = shell.find_shell(prefer="bash")
PY = Path(sys.executable).as_posix()


def text(res: dict) -> str:
    return res["content"][0]["text"]


@unittest.skipUnless(SHELL and SHELL.kind == "bash", "needs bash (Git Bash on Windows)")
class Bash(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.proj = Path(self._tmp.name).resolve() / "proj"
        self.proj.mkdir()
        self.srv = agent.AgentServer()
        shell.install(self.srv, SHELL)
        self.addCleanup(self.srv.close)
        self.asked = []
        self.answer = "allow"

    def ctx(self, cwd="__proj__", **kw):
        cancel = kw.pop("cancel", None) or threading.Event()
        pol = Policy(cwd=str(self.proj) if cwd == "__proj__" else cwd, allow=kw.pop("allow", []), **kw)

        def ask(req):
            self.asked.append(req)
            return self.answer
        return agent.AgentContext(policy=pol, session="s1", ask=ask, cancel=cancel)

    def run_(self, command, ctx=None, **args):
        return self.srv.call("Bash", {"command": command, **args}, 0, None, ctx or self.ctx())

    def test_the_tools_and_their_parameters(self):
        names = {t["name"]: t for t in self.srv.tools}
        self.assertLessEqual({"Bash", "BashOutput", "KillShell"}, set(names))
        self.assertEqual(set(names["Bash"]["inputSchema"]["properties"]), {"command", "timeout", "description", "run_in_background"})
        self.assertEqual(names["Bash"]["inputSchema"]["required"], ["command"])
        self.assertEqual(set(names["BashOutput"]["inputSchema"]["properties"]), {"bash_id", "filter"})
        self.assertEqual(set(names["KillShell"]["inputSchema"]["properties"]), {"shell_id"})
        self.assertIn("bash", names["Bash"]["description"].lower())            # which shell it is, so the model writes for it

    def test_a_plain_read_only_command_runs_without_asking(self):
        r = self.run_("echo hello")
        self.assertFalse(r["isError"])
        self.assertEqual(text(r).strip(), "hello")
        self.assertEqual(self.asked, [])

    def test_anything_else_asks_first_and_does_not_run_when_refused(self):
        self.answer = "deny"
        r = self.run_("touch made.txt")
        self.assertTrue(r["isError"])
        self.assertFalse((self.proj / "made.txt").exists())
        self.assertEqual(len(self.asked), 1)
        self.assertEqual(self.asked[0]["rule"], "Bash(touch:*)")
        self.answer = "allow"
        self.assertFalse(self.run_("touch made.txt")["isError"])
        self.assertTrue((self.proj / "made.txt").exists())

    def test_it_runs_in_the_project_folder(self):
        got = text(self.run_(f'"{PY}" -c "import os; print(os.getcwd())"')).strip()
        self.assertEqual(os.path.normcase(os.path.realpath(got)), os.path.normcase(str(self.proj)))

    def test_the_output_has_both_streams(self):
        r = self.run_(f'"{PY}" -c "import sys; print(\'out\'); print(\'err\', file=sys.stderr)"')
        self.assertIn("out", text(r))
        self.assertIn("err", text(r))

    def test_a_failing_command_is_an_error_with_its_exit_code(self):
        r = self.run_("exit 3")
        self.assertTrue(r["isError"])
        self.assertIn("Exit code 3", text(r))

    def test_a_command_that_takes_too_long_is_stopped_and_says_so(self):
        t0 = time.time()
        r = self.run_(f'"{PY}" -c "import time; print(\'started\', flush=True); time.sleep(30)"', timeout=1500)
        self.assertLess(time.time() - t0, 10)
        self.assertTrue(r["isError"])
        self.assertIn("timed out", text(r))
        self.assertIn("started", text(r))                                      # what it had written

    def test_a_cancel_stops_it(self):
        cancel = threading.Event()
        threading.Timer(0.7, cancel.set).start()
        t0 = time.time()
        r = self.run_(f'"{PY}" -c "import time; time.sleep(30)"', ctx=self.ctx(cancel=cancel))
        self.assertLess(time.time() - t0, 10)
        self.assertTrue(r["isError"])
        self.assertIn("cancel", text(r).lower())

    def test_output_is_capped_in_the_middle_and_says_so(self):
        r = self.run_(f'"{PY}" -c "print(\'a\' * 20000); print(\'MIDDLE\'); print(\'z\' * 40000)"')
        t = text(r)
        self.assertLess(len(t), shell.MAX_OUTPUT + 400)
        self.assertIn("truncated", t)
        self.assertTrue(t.startswith("a"))
        self.assertTrue(t.rstrip().endswith("z"))

    def test_the_timeout_is_checked(self):
        for bad in (0, -5, "x", 10_000_000):
            self.assertTrue(self.run_("echo hi", timeout=bad)["isError"], bad)

    def test_no_folder_no_command(self):
        r = self.run_("echo hi", ctx=self.ctx(cwd=None))
        self.assertTrue(r["isError"])
        self.assertIn("project folder", text(r))

    def test_the_servers_own_settings_are_not_in_the_commands_environment(self):
        os.environ["STRATA_SECRET_TEST"] = "hunter2"
        self.addCleanup(os.environ.pop, "STRATA_SECRET_TEST", None)
        self.assertNotIn("hunter2", text(self.run_("env | grep -i strata_secret_test || echo none")))
        self.assertIn("none", text(self.run_("env | grep -i strata_secret_test || echo none")))

    def test_a_dangerous_command_is_marked_when_it_asks(self):
        self.answer = "deny"
        self.run_("rm -rf ~")
        self.assertTrue(self.asked[0]["danger"])

    def test_a_background_command_is_read_later_and_can_be_killed(self):
        self.answer = "allow"
        r = self.run_(f'"{PY}" -c "import time; print(\'one\', flush=True); time.sleep(0.5); print(\'two\', flush=True); time.sleep(30)"', run_in_background=True)
        self.assertFalse(r["isError"])
        bid = text(r).split("ID:")[1].split()[0].strip(".")
        deadline = time.time() + 8
        got = ""
        while time.time() < deadline and "two" not in got:
            time.sleep(0.2)
            got += text(self.srv.call("BashOutput", {"bash_id": bid}, 0, None, self.ctx()))
        self.assertIn("one", got)
        self.assertIn("two", got)
        self.assertEqual(text(self.srv.call("BashOutput", {"bash_id": bid}, 0, None, self.ctx())).count("one"), 0)       # only what is new
        self.assertIn("running", text(self.srv.call("BashOutput", {"bash_id": bid}, 0, None, self.ctx())).lower())
        k = self.srv.call("KillShell", {"shell_id": bid}, 0, None, self.ctx())
        self.assertFalse(k["isError"])
        time.sleep(0.3)
        self.assertNotIn("running", text(self.srv.call("BashOutput", {"bash_id": bid}, 0, None, self.ctx())).lower())

    def test_another_chat_cannot_read_a_shell_it_did_not_start(self):
        r = self.run_("sleep 5", run_in_background=True)
        bid = text(r).split("ID:")[1].split()[0].strip(".")
        other = self.ctx()
        other.session = "other"
        self.assertTrue(self.srv.call("BashOutput", {"bash_id": bid}, 0, None, other)["isError"])
        self.assertTrue(self.srv.call("KillShell", {"shell_id": bid}, 0, None, other)["isError"])

    def test_closing_the_server_ends_the_background_commands(self):
        r = self.run_(f'"{PY}" -c "import time; time.sleep(60)"', run_in_background=True)
        bid = text(r).split("ID:")[1].split()[0].strip(".")
        proc = self.srv.session("s1").shells[bid].proc
        self.srv.close()
        proc.wait(timeout=5)
        self.assertIsNotNone(proc.returncode)


class Finding(unittest.TestCase):
    def test_the_shell_is_found_or_none(self):
        s = shell.find_shell()
        self.assertTrue(s is None or s.kind in ("bash", "powershell", "cmd"))

    def test_the_description_names_the_shell(self):
        if SHELL:
            self.assertIn(SHELL.kind, shell.describe(SHELL).lower())


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(SHELL and SHELL.kind == "bash", "needs bash (Git Bash on Windows)")
class WhereTheNextCommandRuns(unittest.TestCase):
    """A `cd` carries over to the next command of the chat, inside the project's folders; environment variables and shell state do not."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name).resolve()
        self.proj = self.base / "proj"
        (self.proj / "sub" / "deep").mkdir(parents=True)
        self.other = self.base / "other"
        self.other.mkdir()
        self.srv = agent.AgentServer()
        shell.install(self.srv, SHELL)
        self.addCleanup(self.srv.close)

    def ctx(self, session="s1", dirs=None):
        return agent.AgentContext(policy=Policy(cwd=str(self.proj), dirs=dirs or []), session=session, ask=lambda r: "allow")

    def run_(self, command, ctx=None, **args):
        return text(self.srv.call("Bash", {"command": command, **args}, 0, None, ctx or self.ctx()))

    def where(self, ctx=None):
        out = self.run_("pwd -W 2>/dev/null || pwd -P", ctx)                                  # the directory in this system's own form
        return os.path.normcase(os.path.realpath(out.split("\n")[0].strip()))

    def test_a_cd_carries_over_to_the_next_command_and_the_model_is_told(self):
        out = self.run_("cd sub")
        self.assertIn("Shell cwd is now", out)
        self.assertTrue(out.rstrip().endswith("sub]"))
        self.assertEqual(self.where(), os.path.normcase(str(self.proj / "sub")))
        self.run_("cd deep")
        self.assertEqual(self.where(), os.path.normcase(str(self.proj / "sub" / "deep")))
        self.run_("cd ../..")
        self.assertEqual(self.where(), os.path.normcase(str(self.proj)))

    def test_a_command_that_does_not_move_says_nothing_about_it(self):
        self.assertNotIn("Shell cwd", self.run_("echo hi"))
        self.run_("cd sub")
        self.assertNotIn("Shell cwd", self.run_("ls"))

    def test_environment_variables_and_shell_state_do_not_carry_over(self):
        self.run_("export STRATA_TEST_VAR=1; cd sub")
        self.assertEqual(self.run_('echo "[${STRATA_TEST_VAR}]"').split("\n")[0], "[]")
        self.run_("alias zz=echo")
        self.assertIn("not found", self.run_("zz hi").lower())

    def test_a_failing_command_and_an_exit_inside_it_still_say_where_it_ended(self):
        out = self.run_("cd sub && false")
        self.assertIn("Exit code 1", out)
        self.assertEqual(self.where(), os.path.normcase(str(self.proj / "sub")))
        self.run_("cd deep; exit 3")
        self.assertEqual(self.where(), os.path.normcase(str(self.proj / "sub" / "deep")))

    def test_a_directory_outside_the_project_is_forgotten_with_a_note(self):
        out = self.run_(f'cd "{self.other.as_posix()}"')
        self.assertIn("Shell cwd was reset", out)
        self.assertIn("outside the project's folders", out)
        self.assertEqual(self.where(), os.path.normcase(str(self.proj)))

    def test_the_other_folders_of_the_project_count(self):
        ctx = self.ctx(dirs=[str(self.other)])
        out = self.run_(f'cd "{self.other.as_posix()}"', ctx)
        self.assertIn("Shell cwd is now", out)
        self.assertEqual(self.where(ctx), os.path.normcase(str(self.other)))

    def test_a_folder_that_is_gone_means_the_project_folder_again(self):
        self.run_("cd sub/deep")
        os.rmdir(self.proj / "sub" / "deep")
        self.assertEqual(self.where(), os.path.normcase(str(self.proj)))

    def test_each_chat_has_its_own_directory(self):
        self.run_("cd sub", self.ctx("chat-a"))
        self.assertEqual(self.where(self.ctx("chat-b")), os.path.normcase(str(self.proj)))
        self.assertEqual(self.where(self.ctx("chat-a")), os.path.normcase(str(self.proj / "sub")))

    def test_a_background_command_does_not_move_the_chat(self):
        self.run_("cd sub")
        self.srv.call("Bash", {"command": "cd deep; sleep 0.2", "run_in_background": True}, 0, None, self.ctx())
        time.sleep(0.5)
        self.assertEqual(self.where(), os.path.normcase(str(self.proj / "sub")))

    def test_no_file_of_the_tool_is_left_behind(self):
        before = {p for p in Path(tempfile.gettempdir()).glob("strata-cwd-*")}
        self.run_("cd sub")
        self.run_("echo x")
        self.assertEqual({p for p in Path(tempfile.gettempdir()).glob("strata-cwd-*")}, before)

    def test_the_description_tells_the_model(self):
        desc = {t["name"]: t["description"] for t in self.srv.tools}["Bash"]
        self.assertIn("cd carries over", desc)
        self.assertNotIn("fresh shell", desc)
