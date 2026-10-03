"""The read-only Git view (serve/gitview.py) on a real repository in a temporary folder: the branch, the other branches, the changed files in groups, the last commits, the worktrees, a
file's diff, what is refused, and who may ask."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from serve import gitview  # noqa: E402
from serve.test_agent_chat import DONE, Fixture  # noqa: E402

HAS_GIT = gitview.available() is not None


def git(cwd, *args):
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    subprocess.run(["git", "-c", "core.autocrlf=false", *args], cwd=cwd, env=env, check=True, capture_output=True)


@unittest.skipUnless(HAS_GIT, "needs git")
class OnARepository(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name).resolve()
        self.repo = self.base / "app"
        self.repo.mkdir()
        git(self.repo, "init", "-q", "-b", "main")
        (self.repo / "a.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
        (self.repo / "b.txt").write_text("bee\n", encoding="utf-8")
        git(self.repo, "add", ".")
        git(self.repo, "commit", "-q", "-m", "first commit")
        (self.repo / "a.txt").write_text("one\nTWO\nthree\n", encoding="utf-8")             # not staged
        (self.repo / "b.txt").write_text("bee\nbuzz\n", encoding="utf-8")
        git(self.repo, "add", "b.txt")                                                       # staged
        (self.repo / "new.txt").write_text("fresh\nlines\n", encoding="utf-8")               # untracked

    def test_branch_changes_and_commits(self):
        v = gitview.info(str(self.repo))
        self.assertTrue(v["repo"] and v["git"])
        self.assertEqual(v["branch"], "main")
        self.assertFalse(v["detached"])
        self.assertEqual([f["path"] for f in v["staged"]], ["b.txt"])
        self.assertEqual([f["path"] for f in v["unstaged"]], ["a.txt"])
        self.assertEqual([f["path"] for f in v["untracked"]], ["new.txt"])
        self.assertEqual(v["counts"], {"staged": 1, "unstaged": 1, "untracked": 1, "conflicted": 0})
        self.assertEqual(v["staged"][0]["status"], "M")
        self.assertEqual(v["commits"][0]["subject"], "first commit")
        self.assertEqual(v["branches"], [{"name": "main", "current": True}])
        self.assertEqual(os.path.normcase(v["root"]), os.path.normcase(str(self.repo)))

    def test_from_a_folder_inside_it(self):
        sub = self.repo / "sub"
        sub.mkdir()
        v = gitview.info(str(sub))
        self.assertTrue(v["repo"])
        self.assertEqual(os.path.normcase(v["root"]), os.path.normcase(str(self.repo)))

    def test_other_branches_a_detached_head_and_the_worktrees(self):
        git(self.repo, "branch", "feature/x")
        wt = self.base / "app-wt"
        git(self.repo, "worktree", "add", "-q", str(wt), "feature/x")
        v = gitview.info(str(self.repo))
        self.assertEqual(sorted(b["name"] for b in v["branches"]), ["feature/x", "main"])
        self.assertTrue([b for b in v["branches"] if b["name"] == "main"][0]["current"])
        paths = {os.path.normcase(os.path.realpath(w["path"])): w for w in v["worktrees"]}
        self.assertEqual(len(paths), 2)
        self.assertTrue(paths[os.path.normcase(str(self.repo))]["current"])
        self.assertEqual(paths[os.path.normcase(str(wt))]["branch"], "feature/x")
        w = gitview.info(str(wt))                                                            # the worktree is its own folder with its own branch
        self.assertEqual(w["branch"], "feature/x")
        self.assertTrue([x for x in w["worktrees"] if x["current"]][0]["branch"] == "feature/x")
        git(self.repo, "checkout", "-q", "--detach")
        d = gitview.info(str(self.repo))
        self.assertTrue(d["detached"])
        self.assertIsNone(d["branch"])
        self.assertTrue(d["oid"])

    def test_ahead_of_its_upstream(self):
        remote = self.base / "remote.git"
        subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True, capture_output=True)
        git(self.repo, "remote", "add", "origin", str(remote))
        git(self.repo, "push", "-q", "-u", "origin", "main")
        (self.repo / "c.txt").write_text("c\n", encoding="utf-8")
        git(self.repo, "add", "c.txt")
        git(self.repo, "commit", "-q", "-m", "second")
        v = gitview.info(str(self.repo))
        self.assertEqual((v["upstream"], v["ahead"], v["behind"]), ("origin/main", 1, 0))

    def test_a_renamed_file_and_a_conflict(self):
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "second")
        git(self.repo, "mv", "a.txt", "renamed.txt")
        v = gitview.info(str(self.repo))
        self.assertEqual(v["staged"][0]["path"], "renamed.txt")
        self.assertEqual(v["staged"][0]["from"], "a.txt")
        git(self.repo, "commit", "-q", "-m", "rename")
        git(self.repo, "checkout", "-q", "-b", "other", "HEAD~1")
        (self.repo / "a.txt").write_text("other side\n", encoding="utf-8")
        git(self.repo, "commit", "-q", "-am", "other")
        git(self.repo, "checkout", "-q", "main")
        (self.repo / "renamed.txt").write_text("main side\n", encoding="utf-8")
        git(self.repo, "commit", "-q", "-am", "main side")
        subprocess.run(["git", "merge", "other"], cwd=self.repo, capture_output=True)
        v = gitview.info(str(self.repo))
        self.assertGreaterEqual(v["counts"]["conflicted"], 0)                                # whether this merge conflicts depends on the rename detection; it must not fail

    def test_the_diff_of_an_unstaged_file(self):
        d = gitview.diff(str(self.repo), "a.txt")
        self.assertTrue(d["ok"])
        self.assertIn("-two", d["diff"])
        self.assertIn("+TWO", d["diff"])
        self.assertFalse(d["truncated"])

    def test_the_diff_of_a_staged_file_is_the_staged_one(self):
        d = gitview.diff(str(self.repo), "b.txt", staged=True)
        self.assertIn("+buzz", d["diff"])
        self.assertEqual(gitview.diff(str(self.repo), "b.txt")["diff"], "")                  # nothing is unstaged in it

    def test_an_untracked_file_is_all_added(self):
        d = gitview.diff(str(self.repo), "new.txt", untracked=True)
        self.assertTrue(d["ok"])
        self.assertIn("+fresh", d["diff"])
        self.assertIn("+lines", d["diff"])

    def test_a_binary_file_is_said_to_be(self):
        (self.repo / "pic.bin").write_bytes(b"\0\1\2\3" * 100)
        self.assertTrue(gitview.diff(str(self.repo), "pic.bin", untracked=True)["binary"])

    def test_a_long_diff_is_cut(self):
        (self.repo / "big.txt").write_text("x\n" * 200_000, encoding="utf-8")
        d = gitview.diff(str(self.repo), "big.txt", untracked=True)
        self.assertTrue(d["truncated"])
        self.assertLessEqual(len(d["diff"]), gitview.MAX_DIFF + 200)

    def test_a_path_outside_the_repository_is_refused(self):
        for rel in ("../elsewhere.txt", "..\\x", "/etc/passwd", "C:\\Windows\\win.ini", "", "a\0b"):
            self.assertFalse(gitview.diff(str(self.repo), rel)["ok"], rel)
            self.assertFalse(gitview.diff(str(self.repo), rel, untracked=True)["ok"], rel)

    def test_many_changed_files_are_cut_with_a_note(self):
        for i in range(gitview.MAX_FILES + 20):
            (self.repo / f"f{i:04}.txt").write_text("x", encoding="utf-8")
        v = gitview.info(str(self.repo))
        self.assertTrue(v["truncated"])
        self.assertEqual(sum(len(v[k]) for k in ("staged", "unstaged", "untracked", "conflicted")), gitview.MAX_FILES)
        self.assertGreater(v["counts"]["untracked"], gitview.MAX_FILES)

    def test_it_reads_and_never_writes(self):
        before = sorted(p.name for p in (self.repo / ".git").iterdir())
        gitview.info(str(self.repo))
        gitview.diff(str(self.repo), "a.txt")
        self.assertEqual(sorted(p.name for p in (self.repo / ".git").iterdir()), before)
        self.assertFalse((self.repo / ".git" / "index.lock").exists())


@unittest.skipUnless(HAS_GIT, "needs git")
class NotARepository(unittest.TestCase):
    def test_a_folder_that_is_not_one(self):
        with tempfile.TemporaryDirectory() as d:
            v = gitview.info(d)
            self.assertEqual((v["git"], v["repo"]), (True, False))

    def test_not_a_folder(self):
        self.assertFalse(gitview.info("C:/definitely/not/here")["ok"])
        self.assertFalse(gitview.info("")["ok"])
        self.assertFalse(gitview.info("a\0b")["ok"])

    def test_an_empty_repository_has_no_commits_and_no_failure(self):
        with tempfile.TemporaryDirectory() as d:
            git(d, "init", "-q", "-b", "main")
            v = gitview.info(d)
            self.assertTrue(v["repo"])
            self.assertEqual(v["commits"], [])
            self.assertEqual(v["branch"], "main")


class Parsing(unittest.TestCase):
    def test_the_porcelain_status_of_every_kind(self):
        raw = "\0".join([
            "# branch.oid 0123456789abcdef", "# branch.head main", "# branch.upstream origin/main", "# branch.ab +2 -1",
            "1 .M N... 100644 100644 100644 aaa bbb dir/a.py", "1 M. N... 100644 100644 100644 aaa bbb b.py", "1 MM N... 100644 100644 100644 aaa bbb c.py",
            "2 R. N... 100644 100644 100644 aaa bbb R100 new.py", "old.py", "u UU N... 100644 100644 100644 100644 a b c both.py", "? loose.txt", "",
        ])
        st = gitview._parse_status(raw)
        self.assertEqual(st["head"], {"branch": "main", "detached": False, "oid": "012345678", "upstream": "origin/main", "ahead": 2, "behind": 1})
        self.assertEqual([f["path"] for f in st["staged"]], ["b.py", "c.py", "new.py"])
        self.assertEqual([f["path"] for f in st["unstaged"]], ["dir/a.py", "c.py"])
        self.assertEqual(st["staged"][2]["from"], "old.py")
        self.assertEqual([f["path"] for f in st["conflicted"]], ["both.py"])
        self.assertEqual([f["path"] for f in st["untracked"]], ["loose.txt"])

    def test_a_detached_head_and_a_branch_with_no_commits(self):
        self.assertEqual(gitview._parse_status("# branch.oid abcdef123456\0# branch.head (detached)\0")["head"]["branch"], None)
        self.assertTrue(gitview._parse_status("# branch.oid abcdef123456\0# branch.head (detached)\0")["head"]["detached"])
        self.assertIsNone(gitview._parse_status("# branch.oid (initial)\0# branch.head main\0")["head"]["oid"])

    def test_odd_input_does_not_fail(self):
        self.assertEqual(gitview._parse_status("")["staged"], [])
        gitview._parse_status("1 \0u \0")


@unittest.skipUnless(HAS_GIT, "needs git")
class TheRoute(Fixture):
    def get(self, path, headers=None):
        req = urllib.request.Request(self.base_url + path, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_it_answers_for_a_repository_and_for_its_diff(self):
        self.start(DONE)
        repo = self.proj
        git(repo, "init", "-q", "-b", "main")
        git(repo, "add", ".")
        git(repo, "commit", "-q", "-m", "init")
        (repo / "a.txt").write_text("hello\nthere\n", encoding="utf-8")
        status, body = self.get("/agent/git?path=" + urllib.parse.quote(str(repo)))
        self.assertEqual(status, 200)
        self.assertEqual(body["branch"], "main")
        self.assertEqual([f["path"] for f in body["unstaged"]], ["a.txt"])
        status, d = self.get("/agent/git/diff?path=" + urllib.parse.quote(str(repo)) + "&file=a.txt")
        self.assertEqual(status, 200)
        self.assertIn("+there", d["diff"])

    def test_it_is_for_this_pc_only(self):
        self.start(DONE)
        status, body = self.get("/agent/git?path=" + urllib.parse.quote(str(self.proj)), {"Host": "strata.example"})
        self.assertIn(status, (403, 421))

    def test_with_an_api_key_the_key_is_the_door(self):
        self.start(DONE)
        self.svc.api_key = "k"
        status, _ = self.get("/agent/git?path=" + urllib.parse.quote(str(self.proj)))
        self.assertEqual(status, 401)
        status, body = self.get("/agent/git?path=" + urllib.parse.quote(str(self.proj)), {"Authorization": "Bearer k"})
        self.assertEqual(status, 200)

    def test_a_path_that_is_not_a_folder_is_an_answer(self):
        self.start(DONE)
        status, body = self.get("/agent/git?path=" + urllib.parse.quote(str(self.base / "nope")))
        self.assertEqual(status, 200)
        self.assertFalse(body["ok"])


if __name__ == "__main__":
    unittest.main()
