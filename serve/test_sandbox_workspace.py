"""The sandbox's copy of a project (serve/sandbox): what goes into it, what a change looks like, how it follows the real project, and what it offers back. Every repository here is made by the test."""
from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from serve.sandbox import output, search, symbols, workspace  # noqa: E402
from serve.sandbox.workspace import ApplyRefused, QuotaExceeded, Workspace, WorkspaceError  # noqa: E402

NO_WINDOW = 0x08000000 if os.name == "nt" else 0
WINDOWS = os.name == "nt"


def long(path) -> str:
    p = os.path.abspath(str(path))
    return "\\\\?\\" + p if WINDOWS and not p.startswith("\\\\?\\") else p


def wipe(path) -> None:
    def again(func, p, _exc):
        os.chmod(p, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
        func(p)
    shutil.rmtree(long(path), onerror=again)


def junction(target, link) -> None:
    if not WINDOWS:
        raise unittest.SkipTest("a junction is a Windows thing")
    import _winapi
    _winapi.CreateJunction(str(target), str(link))


def everything(root) -> dict[str, bytes]:
    """Every file under `root`, .git too, as bytes: what "nothing was written" is checked against."""
    out = {}
    for folder, _dirs, names in os.walk(root):
        for name in names:
            p = os.path.join(folder, name)
            out[os.path.relpath(p, root).replace("\\", "/")] = Path(p).read_bytes()
    return out


class Base(unittest.TestCase):
    def setUp(self):
        self.base = Path(os.path.realpath(tempfile.mkdtemp(prefix="sbx-")))
        self.addCleanup(wipe, self.base)
        self.proj = self.base / "proj"
        self.proj.mkdir()
        self.outside = self.base / "outside"
        self.outside.mkdir()
        (self.outside / "stolen.txt").write_text("the content of another folder\n")
        self.scratch = self.base / "scratch"
        self.git("init", "-q")
        self.git("config", "core.autocrlf", "false")
        self.git("config", "core.safecrlf", "false")
        self.put(".gitignore", b"*.log\nbuilt/\n")
        self.put("a.txt", b"one\ntwo\nthree\n")
        self.put("src/b.py", b"def f():\n    return 1\n")
        self.commit()

    def git(self, *args, cwd=None, stdin=None) -> str:
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        p = subprocess.run(["git", "-c", "user.email=t@example.invalid", "-c", "user.name=t", "-c", "commit.gpgsign=false", "-c", "core.longpaths=true", "-c", "protocol.file.allow=always", *args],
                           cwd=str(cwd or self.proj), env=env, input=stdin, capture_output=True, creationflags=NO_WINDOW)
        if p.returncode:
            raise AssertionError(f"git {args}: {p.stderr.decode('utf-8', 'replace')}")
        return p.stdout.decode("utf-8", "replace")

    def commit(self, message: str = "x") -> None:
        self.git("add", "-A")
        self.git("commit", "-qm", message)

    def put(self, rel: str, data: bytes, root=None) -> None:
        p = long(Path(root or self.proj) / rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(data)

    def real(self, rel: str) -> bytes:
        return Path(self.proj / rel).read_bytes()

    def open_ws(self, **kw) -> Workspace:
        ws = Workspace(self.proj, self.scratch, **kw)
        self.addCleanup(ws.close)
        return ws

    def box(self, ws: Workspace, rel: str, data: bytes) -> None:
        """What code in the box does: it writes the copy directly, past everything in workspace.py."""
        self.put(rel, data, root=ws.copy)

    def in_copy(self, ws: Workspace, rel: str) -> bytes | None:
        p = long(Path(ws.copy) / rel)
        if not os.path.exists(p):
            return None
        with open(p, "rb") as f:
            return f.read()

    def status_of(self, ws: Workspace) -> dict[str, str]:
        return {r["path"]: r["status"] for r in ws.changed()}

    def status_of_full(self, ws: Workspace) -> dict[str, str]:
        return {r["path"]: r["status"] for r in ws.changed(full=True)}


class Snapshot(Base):
    def test_working_tree_not_head(self):
        self.put("a.txt", b"edited, not committed\n")
        self.put("new.txt", b"untracked\n")
        self.put("noise.log", b"ignored\n")
        self.put("built/out.bin", b"ignored folder\n")
        ws = self.open_ws()
        self.assertEqual(self.in_copy(ws, "a.txt"), b"edited, not committed\n")
        self.assertEqual(self.in_copy(ws, "new.txt"), b"untracked\n")
        self.assertIsNone(self.in_copy(ws, "noise.log"))
        self.assertIsNone(self.in_copy(ws, "built/out.bin"))
        self.assertEqual(ws.changed(), [], "a snapshot is its own base")

    def test_source_code_stays_in_whatever_its_name(self):
        for rel in ("src/secrets.py", "pkg/credentials.py", "app/secret.ts"):
            self.put(rel, b"x = 1\n")
        self.commit()
        ws = self.open_ws()
        for rel in ("src/secrets.py", "pkg/credentials.py", "app/secret.ts"):
            self.assertIsNotNone(self.in_copy(ws, rel), rel)

    def test_secrets_stay_out_also_when_tracked(self):
        secrets = [".env", ".env.local", "id_rsa", "deploy/key.pem", ".ssh/config", "credentials.json", ".aws/credentials", "certs/site.pfx", ".npmrc"]
        for rel in secrets:
            self.put(rel, b"TOKEN=abc\n")
        self.put(".env.example", b"TOKEN=\n")
        self.commit()
        self.put("untracked.key", b"TOKEN=abc\n")
        ws = self.open_ws()
        for rel in secrets + ["untracked.key"]:
            self.assertIsNone(self.in_copy(ws, rel), rel)
            self.assertIn("secret", ws.excluded[rel], rel)
        self.assertEqual(self.in_copy(ws, ".env.example"), b"TOKEN=\n", "a template is not a secret")

    def test_git_is_never_copied(self):
        ws = self.open_ws()
        self.assertFalse(os.path.exists(os.path.join(ws.copy, ".git")))

    def test_size_limit_at_the_edge(self):
        self.put("fits.bin", b"x" * 1000)
        self.put("over.bin", b"x" * 1001)
        ws = self.open_ws(max_file=1000)
        self.assertEqual(len(self.in_copy(ws, "fits.bin")), 1000)
        self.assertIsNone(self.in_copy(ws, "over.bin"))
        self.assertIn("size limit", ws.excluded["over.bin"])

    def test_not_ignored_output_folders_stay_out_but_tracked_ones_stay_in(self):
        self.put("dist/tracked.js", b"keep\n")
        self.commit()
        self.put("node_modules/pkg/index.js", b"drop\n")
        self.put("build-rel/obj.o", b"drop\n")
        self.put("dist/new_output.js", b"drop\n")
        ws = self.open_ws()
        self.assertEqual(self.in_copy(ws, "dist/tracked.js"), b"keep\n")
        for rel in ("node_modules/pkg/index.js", "build-rel/obj.o", "dist/new_output.js"):
            self.assertIsNone(self.in_copy(ws, rel), rel)

    def test_junction_is_dropped_even_though_git_lists_through_it(self):
        junction(self.outside, self.proj / "escape")
        listed = self.git("ls-files", "-o", "--exclude-standard")
        self.assertIn("escape/stolen.txt", listed, "if git stops listing through junctions this test no longer covers the case")
        ws = self.open_ws()
        self.assertIsNone(self.in_copy(ws, "escape/stolen.txt"))
        self.assertIn("link", ws.excluded["escape/stolen.txt"])

    def test_tracked_folder_replaced_by_a_junction(self):
        self.put("keys/t.txt", b"tracked\n")
        self.commit()
        wipe(self.proj / "keys")
        junction(self.outside, self.proj / "keys")
        ws = self.open_ws()
        self.assertIsNone(self.in_copy(ws, "keys/stolen.txt"))
        self.assertIsNone(self.in_copy(ws, "keys/t.txt"))
        self.assertFalse(any("the content of another folder" in str(self.in_copy(ws, r["path"])) for r in ws.changed()))

    def test_junction_that_stays_inside_the_project_is_dropped_too(self):
        self.put("real/file.txt", b"here\n")
        junction(self.proj / "real", self.proj / "alias")
        ws = self.open_ws()
        self.assertEqual(self.in_copy(ws, "real/file.txt"), b"here\n")
        self.assertIsNone(self.in_copy(ws, "alias/file.txt"))

    def test_symlink_out_is_dropped(self):
        try:
            os.symlink(self.outside / "stolen.txt", self.proj / "link.txt")
        except OSError:
            self.skipTest("this account cannot make symlinks (WinError 1314)")
        ws = self.open_ws()
        self.assertIsNone(self.in_copy(ws, "link.txt"))
        self.assertIn("link", ws.excluded["link.txt"])

    def test_bytes_are_kept_exactly(self):
        samples = {"crlf.txt": b"one\r\ntwo\r\n", "mixed.txt": b"a\r\nb\nc\r\n", "no_end.txt": b"last line", "cp874.txt": "ภาษาไทย\r\n".encode("cp874"),
                   "utf16.txt": "ไทย\r\n".encode("utf-16"), "bom.py": b"\xef\xbb\xbfx = 1\r\n", "all_bytes.bin": bytes(range(256)) * 4, "empty.txt": b""}
        for rel, data in samples.items():
            self.put(rel, data)
        self.commit()
        for autocrlf in ("false", "true"):                                    # true is this PC's global setting
            with self.subTest(autocrlf=autocrlf):
                self.git("config", "core.autocrlf", autocrlf)
                ws = self.open_ws()
                for rel, data in samples.items():
                    self.assertEqual(self.in_copy(ws, rel), data, rel)
                self.assertEqual(ws.changed(), [])

    def test_copy_is_never_a_hardlink(self):
        ws = self.open_ws()
        self.assertEqual(os.stat(os.path.join(ws.copy, "a.txt")).st_nlink, 1)
        before = self.real("a.txt")
        with open(os.path.join(ws.copy, "a.txt"), "ab") as f:                 # in place, the write that would go through a hardlink
            f.write(b"more\n")
        self.assertEqual(self.real("a.txt"), before)

    def test_unicode_names(self):
        names = ["ไฟล์ไทย.txt", "โฟลเดอร์/ข้างใน.py", "日本語/ファイル.txt", "emoji-😀.txt", "with space/and'quote.txt", "accent-é.txt"]
        for rel in names:
            self.put(rel, rel.encode("utf-8"))
        self.commit()
        self.put("ใหม่-ยังไม่ commit.txt", b"new\n")
        ws = self.open_ws()
        for rel in names + ["ใหม่-ยังไม่ commit.txt"]:
            self.assertIsNotNone(self.in_copy(ws, rel), rel)
        self.box(ws, "ไฟล์ไทย.txt", b"changed\n")
        self.assertEqual(self.status_of(ws), {"ไฟล์ไทย.txt": "M"})
        self.assertIn("ไฟล์ไทย.txt", ws.diff())

    def test_long_paths_past_the_old_windows_limit(self):
        rel = "/".join(["d" * 40] * 7) + "/deep-ไทย.txt"
        self.assertGreater(len(str(self.proj / rel)), 300)
        self.put(rel, b"deep\r\n")
        self.commit()
        ws = self.open_ws()
        self.assertEqual(self.in_copy(ws, rel), b"deep\r\n")
        self.box(ws, rel, b"deeper\r\n")
        self.assertEqual(self.status_of(ws), {rel: "M"})
        self.assertIn("+deeper", ws.diff())
        plan = ws.apply_plan()
        self.assertEqual([(w.path, w.data) for w in plan.writes], [(rel, b"deeper\r\n")])
        self.assertTrue(ws.close())
        self.assertFalse(os.path.exists(long(ws.root)))

    def test_project_in_a_sub_folder_of_the_repository(self):
        ws = Workspace(self.proj / "src", self.scratch)
        self.addCleanup(ws.close)
        self.assertEqual(self.in_copy(ws, "b.py"), b"def f():\n    return 1\n")
        self.assertIsNone(self.in_copy(ws, "a.txt"))
        self.put("src/b.py", b"def f():\n    return 2\n")
        self.put("a.txt", b"outside the project, changed\n")
        report = ws.sync()
        self.assertEqual(report["refreshed"], ["b.py"], "git status names paths from the top of the repository")
        self.assertEqual(self.in_copy(ws, "b.py"), b"def f():\n    return 2\n")

    def test_submodule_goes_in_as_it_is_now(self):
        lib = self.base / "lib"
        lib.mkdir()
        self.git("init", "-q", cwd=lib)
        self.put("code.py", b"x = 1\n", root=lib)
        self.git("add", "-A", cwd=lib)
        self.git("commit", "-qm", "lib", cwd=lib)
        self.git("submodule", "add", "-q", str(lib), "vendor/lib")
        self.commit()
        self.put("vendor/lib/code.py", b"x = 2\n")
        ws = self.open_ws()
        self.assertEqual(self.in_copy(ws, "vendor/lib/code.py"), b"x = 2\n")
        self.assertFalse(os.path.exists(os.path.join(ws.copy, "vendor", "lib", ".git")))
        self.assertEqual(ws.changed(), [])
        self.put("vendor/lib/code.py", b"x = 3\n")
        self.assertEqual(ws.sync()["refreshed"], ["vendor/lib/code.py"])
        self.assertEqual(self.in_copy(ws, "vendor/lib/code.py"), b"x = 3\n")

    def test_refuses_what_it_cannot_work_on(self):
        plain = self.base / "plain"
        plain.mkdir()
        with mock.patch.dict(os.environ, {"GIT_CEILING_DIRECTORIES": str(self.base)}):
            with self.assertRaisesRegex(WorkspaceError, "git work tree"):
                Workspace(plain, self.scratch)
        with self.assertRaisesRegex(WorkspaceError, "does not exist"):
            Workspace(self.base / "missing", self.scratch)
        with self.assertRaisesRegex(WorkspaceError, "must be apart"):
            Workspace(self.proj, self.proj / "scratch")

    @unittest.skipUnless(WINDOWS, "these are Windows paths")
    def test_a_network_path_is_refused_without_touching_the_network(self):
        with mock.patch.object(workspace.os.path, "realpath", side_effect=AssertionError("touched")):
            for bad in (r"\\server\share\proj", r"\\wsl$\Ubuntu\home\me\proj", "//server/share/proj"):
                with self.subTest(bad), self.assertRaisesRegex(WorkspaceError, "network path"):
                    Workspace(bad, self.scratch)

    @unittest.skipUnless(WINDOWS, "a mapped drive is a Windows thing")
    def test_a_mapped_network_drive_is_refused(self):
        with mock.patch.object(workspace.os.path, "realpath", return_value=r"\\server\share\proj"):
            with self.assertRaisesRegex(WorkspaceError, "network path"):
                Workspace(self.proj, self.scratch)


class Delta(Base):
    def test_clean_edit_new_delete(self):
        ws = self.open_ws()
        self.box(ws, "a.txt", b"one\nTWO\nthree\n")
        self.box(ws, "new.py", b"x = 1\n")
        os.remove(os.path.join(ws.copy, "src", "b.py"))
        self.assertEqual(self.status_of(ws), {"a.txt": "M", "new.py": "A", "src/b.py": "D"})

    def test_an_edit_that_restores_size_and_mtime_cannot_reach_the_project_as_tested(self):
        ws = self.open_ws()
        p = os.path.join(ws.copy, "a.txt")
        old = time.time_ns() - 3600 * 10**9
        os.utime(p, ns=(old, old))
        self.box(ws, "src/b.py", b"the tested change\n")
        ws.tested_record("pytest", 0, "passed")                               # hashes a.txt once more, now old enough for that hash to be reused
        self.box(ws, "a.txt", b"ONE\ntwo\nthree\n")
        os.utime(p, ns=(old, old))
        self.assertEqual(self.status_of(ws), {"src/b.py": "M"}, "the price of not opening every file after every command")
        plan = ws.apply_plan()                                                # first: a full read also refreshes what the quick one sees
        self.assertEqual(sorted(w.path for w in plan.writes), ["a.txt", "src/b.py"], "apply reads every file")
        self.assertIsNone(plan.tested, "what the test saw is not what is offered, and the plan says so")
        self.assertEqual(self.status_of_full(ws), {"a.txt": "M", "src/b.py": "M"})

    def test_a_second_write_in_the_same_clock_tick_is_seen(self):
        ws = self.open_ws()
        p = os.path.join(ws.copy, "a.txt")
        self.box(ws, "a.txt", b"AAA\nAAA\nAAAAA\n")
        st = os.stat(p)
        first = {r["path"]: r["hash"] for r in ws.changed()}
        self.box(ws, "a.txt", b"BBB\nBBB\nBBBBB\n")
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))                      # what two writes inside one tick of the file clock look like
        self.assertNotEqual({r["path"]: r["hash"] for r in ws.changed()}, first)

    def test_a_write_right_after_the_snapshot_in_the_same_clock_tick_is_seen(self):
        ws = self.open_ws()
        p = os.path.join(ws.copy, "a.txt")
        st = os.stat(p)
        self.box(ws, "a.txt", b"ONE\nTWO\nTHREE\n")
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))                      # the same size and, as within one tick, the same mtime the snapshot left
        self.assertEqual(self.status_of(ws), {"a.txt": "M"})

    def test_settled_files_are_not_opened_again(self):
        ws = self.open_ws()
        time.sleep(workspace.MTIME_TICK_NS / 1e9 * 2)
        ws.changed()
        with mock.patch.object(workspace, "_hash_file", side_effect=AssertionError("opened")):
            self.assertEqual(ws.changed(), [])
            self.assertEqual(ws.changed(), [])

    def test_rename_with_the_same_bytes(self):
        ws = self.open_ws()
        os.rename(os.path.join(ws.copy, "a.txt"), os.path.join(ws.copy, "renamed.txt"))
        rows = ws.changed()
        self.assertEqual([(r["path"], r["status"], r.get("from")) for r in rows], [("renamed.txt", "R", "a.txt")])
        diff = ws.diff()
        self.assertIn("rename from a.txt", diff)
        self.assertIn("rename to renamed.txt", diff)

    def test_rename_plus_edit_is_a_delete_and_a_new_file(self):
        ws = self.open_ws()
        os.remove(os.path.join(ws.copy, "a.txt"))
        self.box(ws, "renamed.txt", b"one\ntwo\nthree and more\n")
        self.assertEqual(self.status_of(ws), {"a.txt": "D", "renamed.txt": "A"})

    def test_two_empty_files_are_not_a_rename(self):
        self.put("empty_a", b"")
        self.commit()
        ws = self.open_ws()
        os.remove(os.path.join(ws.copy, "empty_a"))
        self.box(ws, "empty_b", b"")
        self.assertEqual(self.status_of(ws), {"empty_a": "D", "empty_b": "A"})

    def test_diff_shows_a_delete_after_the_real_file_is_gone(self):
        ws = self.open_ws()
        os.remove(os.path.join(ws.copy, "src", "b.py"))
        os.remove(self.proj / "src" / "b.py")
        diff = ws.diff()
        self.assertIn("deleted file mode", diff)
        self.assertIn("-    return 1", diff)

    def test_diff_goes_back_through_git_apply_byte_for_byte(self):
        self.put("crlf.txt", b"one\r\ntwo\r\nthree\r\n")
        self.put("no_end.txt", b"first\nsecond")
        self.put("gone.txt", b"bye\n")
        self.put("ff.txt", b"page one\x0cstill the same line\nend\n")
        self.commit()
        ws = self.open_ws()
        self.box(ws, "crlf.txt", b"one\r\nTWO\r\nthree\r\n")
        self.box(ws, "no_end.txt", b"first\nsecond, longer")
        self.box(ws, "new_end.txt", "ไทย\n".encode("utf-8"))
        self.box(ws, "ff.txt", b"page one\x0cstill the same line\nEND\n")
        self.box(ws, "a.txt", b"one\nTWO\nthree")                             # gains "no newline at end"
        os.remove(os.path.join(ws.copy, "gone.txt"))
        os.rename(os.path.join(ws.copy, "src", "b.py"), os.path.join(ws.copy, "src", "c.py"))
        diff = ws.diff()
        self.assertIn("\\ No newline at end of file", diff)
        self.git("apply", "--whitespace=nowarn", "-", stdin=diff.encode("utf-8"))
        for rel in ("crlf.txt", "no_end.txt", "new_end.txt", "ff.txt", "a.txt", "src/c.py"):
            self.assertEqual(self.real(rel), self.in_copy(ws, rel), rel)
        self.assertFalse((self.proj / "gone.txt").exists())
        self.assertFalse((self.proj / "src" / "b.py").exists())

    def test_line_ending_change_alone_shows(self):
        self.put("crlf.txt", b"one\r\ntwo\r\n")
        self.commit()
        ws = self.open_ws()
        self.box(ws, "crlf.txt", b"one\ntwo\n")
        self.git("config", "core.autocrlf", "true")                           # the setting that hides this from `git diff --no-index`
        self.assertEqual(ws.diff_stat(), {"files": 1, "added": 2, "removed": 2})

    def test_a_change_too_spread_out_to_diff_is_summed_up(self):
        lines = [f"line {i} {i * 2654435761 % 4294967296:x}\n" for i in range(workspace.MAX_DIFF_SPAN * 3)]
        self.put("big.txt", "".join(lines).encode())
        self.commit()
        ws = self.open_ws()
        self.box(ws, "big.txt", "".join(x if i % 2 else f"reformatted {i}\n" for i, x in enumerate(lines)).encode())
        diff = ws.diff()
        self.assertIn("too much changed to show", diff)
        self.assertNotIn("@@", diff, "difflib takes minutes on this shape, which is why it is not asked")
        half = workspace.MAX_DIFF_SPAN * 3 // 2
        self.assertEqual(ws.diff_stat(), {"files": 1, "added": half, "removed": half})

    def test_a_small_edit_in_a_big_file_is_still_a_diff(self):
        lines = [f"line {i}\n" for i in range(workspace.MAX_DIFF_SPAN * 3)]
        self.put("big.txt", "".join(lines).encode())
        self.commit()
        ws = self.open_ws()
        lines[1000] = "edited\n"
        self.box(ws, "big.txt", "".join(lines).encode())
        diff = ws.diff()
        self.assertIn("-line 1000\n+edited\n", diff)
        self.assertEqual(ws.diff_stat(), {"files": 1, "added": 1, "removed": 1})

    def test_changed_span(self):
        a = ["1", "2", "3", "4", "5"]
        self.assertEqual(workspace._changed_span(a, a), 0)
        self.assertEqual(workspace._changed_span(a, ["1", "2", "x", "4", "5"]), 1)
        self.assertEqual(workspace._changed_span(a, ["x", "2", "3", "4", "y"]), 5)
        self.assertEqual(workspace._changed_span(a, ["1", "5"]), 3)
        self.assertEqual(workspace._changed_span(["1", "1"], ["1", "1", "1"]), 1)
        self.assertEqual(workspace._changed_span([], a), 5)

    def test_binary_is_named_not_printed(self):
        self.put("blob.bin", b"\0\1\2" * 50)
        self.commit()
        ws = self.open_ws()
        self.box(ws, "blob.bin", b"\0\3\4" * 50)
        self.assertIn("Binary files a/blob.bin and b/blob.bin differ", ws.diff())

    def test_ignored_and_output_files_are_not_a_change(self):
        ws = self.open_ws()
        self.box(ws, "test.log", b"a test wrote this\n")
        self.box(ws, "built/obj.o", b"a build wrote this\n")
        self.box(ws, "src/__pycache__/b.cpython-311.pyc", b"\0compiled")
        self.box(ws, ".pytest_cache/v/cache", b"x")
        self.box(ws, "kept.txt", b"this one is a change\n")
        self.assertEqual(self.status_of(ws), {"kept.txt": "A"})

    def test_what_was_left_out_is_not_a_deletion(self):
        self.put(".env", b"TOKEN=abc\n")
        self.put("big.bin", b"x" * 5000)
        self.commit()
        ws = self.open_ws(max_file=1000)
        self.assertEqual(ws.changed(), [], "if these counted as deleted, apply would delete them from the project")
        self.assertEqual(ws.apply_plan().writes, [])

    def test_sandbox_folder_is_not_a_change(self):
        ws = self.open_ws()
        self.box(ws, f"{workspace.RESERVED}/output/1.txt", b"a long result\n")
        self.assertEqual(ws.changed(), [])
        self.assertEqual(ws.apply_plan().writes, [])

    @unittest.skipUnless(WINDOWS, "a name that differs by case is another folder elsewhere")
    def test_sandbox_folder_in_another_case_is_the_same_folder(self):
        ws = self.open_ws()
        self.box(ws, f"{workspace.RESERVED.upper()}/output/1.txt", b"a long result\n")
        self.assertEqual(ws.changed(), [])
        self.assertEqual(ws.apply_plan().writes, [])

    def test_a_copy_that_went_away_is_not_a_delete_of_everything(self):
        ws = self.open_ws()
        wipe(ws.copy)
        for call in (ws.changed, ws.diff, ws.apply_plan, ws.sync):
            with self.assertRaisesRegex(WorkspaceError, "gone"):
                call()

    def test_closed_workspace_refuses(self):
        ws = self.open_ws()
        ws.close()
        for call in (ws.changed, ws.diff, ws.apply_plan, ws.sync):
            with self.assertRaisesRegex(WorkspaceError, "closed"):
                call()


class PlantedInTheCopy(Base):
    """Code in the box can make links. The server reads and writes the copy with its own rights, so a link there must not lend it those rights."""

    def test_junction_to_a_secret_is_not_read(self):
        ws = self.open_ws()
        junction(self.outside, Path(ws.copy) / "peek")
        self.assertNotIn("peek/stolen.txt", ws.listing())
        self.assertEqual(self.status_of(ws), {"peek": "L"})
        self.assertNotIn("another folder", ws.diff())
        with self.assertRaisesRegex(WorkspaceError, "link"):
            ws.read_bytes("peek/stolen.txt")

    def test_a_link_stops_the_whole_apply(self):
        ws = self.open_ws()
        self.box(ws, "a.txt", b"changed\n")
        junction(self.outside, Path(ws.copy) / "peek")
        with self.assertRaises(ApplyRefused) as e:
            ws.apply_plan()
        self.assertEqual([r["path"] for r in e.exception.reasons], ["peek"])

    def test_write_cannot_go_through_a_planted_link(self):
        ws = self.open_ws()
        junction(self.outside, Path(ws.copy) / "peek")
        with self.assertRaisesRegex(WorkspaceError, "link"):
            ws.write_bytes("peek/stolen.txt", b"overwritten")
        self.assertEqual((self.outside / "stolen.txt").read_text(), "the content of another folder\n")

    def test_sync_does_not_write_through_a_planted_link(self):
        ws = self.open_ws()
        wipe(Path(ws.copy) / "src")
        target = self.outside / "b.py"
        target.write_bytes(self.real("src/b.py"))                             # the same bytes as the base, so that the link looks like a copy nobody touched
        junction(self.outside, Path(ws.copy) / "src")
        self.put("src/b.py", b"def f():\n    return 2\n")
        report = ws.sync()
        self.assertEqual(report["conflicts"], ["src/b.py"])
        self.assertEqual(target.read_bytes(), b"def f():\n    return 1\n", "written with the server's rights, through a link code in the box made")

    def test_a_folder_swapped_for_a_link_is_not_offered_as_deletes(self):
        ws = self.open_ws()
        wipe(Path(ws.copy) / "src")
        junction(self.outside, Path(ws.copy) / "src")
        with self.assertRaises(ApplyRefused):
            ws.apply_plan()

    def test_git_folder_made_in_the_copy_is_not_offered(self):
        ws = self.open_ws()
        self.box(ws, ".git/hooks/pre-commit", b"#!/bin/sh\nevil\n")
        self.box(ws, "sub/.git/config", b"[core]\n")
        self.box(ws, "other/.git", b"gitdir: somewhere\n")                    # a file, as a submodule has
        if WINDOWS:
            self.box(ws, "more/.GIT/hooks/pre-commit", b"#!/bin/sh\nevil\n")  # the same folder as .git to Windows
        self.assertEqual(ws.changed(), [])
        self.assertEqual(ws.apply_plan().writes, [])

    def test_names_that_lead_to_git_are_refused(self):
        ws = self.open_ws()
        self.box(ws, "GIT~1/hooks/pre-commit", b"#!/bin/sh\nevil\n")
        with self.assertRaises(ApplyRefused) as e:
            ws.apply_plan()
        self.assertIn(".git", e.exception.reasons[0]["why"])

    def test_bad_path(self):
        for rel in ("", "/abs", "../up", "a/../../up", "a//b", ".git", ".git/config", "x/.GIT/config", "a\\b", "git~1/config", "x\0y"):
            self.assertIsNotNone(workspace.bad_path(rel), repr(rel))
        for rel in ("a.txt", "src/b.py", ".gitignore", ".github/workflows/ci.yml", ".gitattributes", "ไทย/ไฟล์.txt"):
            self.assertIsNone(workspace.bad_path(rel), repr(rel))
        if WINDOWS:
            for rel in ("C:/x", "a.txt:stream", "a.txt::$DATA", "con", "NUL.txt", "dir/aux.c", "trailing.", "trailing ", ".git.", ".git "):
                self.assertIsNotNone(workspace.bad_path(rel), repr(rel))


class Sync(Base):
    def test_a_file_the_copy_did_not_touch_follows_the_project(self):
        ws = self.open_ws()
        self.put("a.txt", b"the user edited this\r\n")
        report = ws.sync()
        self.assertEqual(report["refreshed"], ["a.txt"])
        self.assertEqual(self.in_copy(ws, "a.txt"), b"the user edited this\r\n")
        self.assertEqual(ws.changed(), [], "the base moved with it")

    def test_nothing_to_do_is_quiet(self):
        ws = self.open_ws()
        report = ws.sync()
        self.assertEqual((report["refreshed"], report["added"], report["removed"], report["conflicts"], report["head"]), ([], [], [], [], None))

    def test_changed_on_both_sides_overwrites_neither(self):
        ws = self.open_ws()
        self.box(ws, "a.txt", b"the model's version\n")
        self.put("a.txt", b"the user's version\n")
        report = ws.sync()
        self.assertEqual(report["conflicts"], ["a.txt"])
        self.assertEqual(self.in_copy(ws, "a.txt"), b"the model's version\n")
        self.assertEqual(self.real("a.txt"), b"the user's version\n")
        self.assertEqual(ws.sync()["conflicts"], ["a.txt"], "it stays a conflict until it is settled")
        with self.assertRaises(ApplyRefused):
            ws.apply_plan()

    def test_same_change_on_both_sides_is_no_conflict(self):
        ws = self.open_ws()
        self.box(ws, "a.txt", b"the same\n")
        self.put("a.txt", b"the same\n")
        report = ws.sync()
        self.assertEqual(report["conflicts"], [])
        self.assertEqual(ws.changed(), [])

    def test_deleted_on_one_side_changed_on_the_other_is_a_conflict(self):
        ws = self.open_ws()
        self.box(ws, "a.txt", b"the model changed this\n")
        os.remove(self.proj / "a.txt")
        os.remove(os.path.join(ws.copy, "src", "b.py"))
        self.put("src/b.py", b"def f():\n    return 2\n")
        report = ws.sync()
        self.assertEqual(sorted(report["conflicts"]), ["a.txt", "src/b.py"])
        self.assertEqual(self.in_copy(ws, "a.txt"), b"the model changed this\n")
        self.assertEqual(self.real("src/b.py"), b"def f():\n    return 2\n")

    def test_new_on_both_sides_is_a_conflict(self):
        ws = self.open_ws()
        self.box(ws, "new.txt", b"the model's\n")
        self.put("new.txt", b"the user's\n")
        self.assertEqual(ws.sync()["conflicts"], ["new.txt"])
        self.assertEqual(self.in_copy(ws, "new.txt"), b"the model's\n")

    def test_a_dirty_file_put_back_is_noticed_though_status_is_clean(self):
        self.put("a.txt", b"dirty at the snapshot\n")
        ws = self.open_ws()
        self.git("checkout", "--", "a.txt")
        self.assertEqual(self.git("status", "--porcelain"), "", "the case is a file git no longer lists")
        self.assertEqual(ws.sync()["refreshed"], ["a.txt"])
        self.assertEqual(self.in_copy(ws, "a.txt"), b"one\ntwo\nthree\n")

    def test_an_untracked_file_deleted_is_noticed(self):
        self.put("scratchpad.txt", b"untracked at the snapshot\n")
        ws = self.open_ws()
        os.remove(self.proj / "scratchpad.txt")
        self.assertEqual(ws.sync()["removed"], ["scratchpad.txt"])
        self.assertIsNone(self.in_copy(ws, "scratchpad.txt"))

    def test_new_files_follow_but_only_what_the_snapshot_would_take(self):
        ws = self.open_ws(max_file=1000)
        self.put("fresh.txt", b"new in the project\n")
        self.put(".env", b"TOKEN=abc\n")
        self.put("too_big.bin", b"x" * 1001)
        self.put("noise.log", b"ignored\n")
        junction(self.outside, self.proj / "escape")
        report = ws.sync()
        self.assertEqual(report["added"], ["fresh.txt"])
        for rel in (".env", "too_big.bin", "noise.log", "escape/stolen.txt"):
            self.assertIsNone(self.in_copy(ws, rel), rel)

    def test_a_file_that_grew_past_the_limit_leaves_the_copy(self):
        self.put("grows.txt", b"small\n")
        self.commit()
        ws = self.open_ws(max_file=1000)
        self.put("grows.txt", b"x" * 1001)
        self.assertEqual(ws.sync()["removed"], ["grows.txt"])
        self.assertIsNone(self.in_copy(ws, "grows.txt"))
        self.assertIn("size limit", ws.excluded["grows.txt"])
        self.assertEqual(ws.changed(), [], "if this counted as a deletion, apply would delete the real file")

    def test_a_commit_is_seen_after_status_goes_clean(self):
        ws = self.open_ws()
        self.put("a.txt", b"committed\n")
        self.commit()
        self.assertEqual(self.git("status", "--porcelain"), "")
        report = ws.sync()
        self.assertEqual(report["refreshed"], ["a.txt"])
        self.assertIsNotNone(report["head"])

    def test_switching_branch_keeps_the_delta(self):
        self.git("checkout", "-q", "-b", "other")
        self.put("only_other.txt", b"on the other branch\n")
        self.put("a.txt", b"other branch's a\n")
        self.commit()
        self.git("checkout", "-q", "-")
        ws = self.open_ws()
        self.box(ws, "src/b.py", b"def f():\n    return 'the model'\n")
        self.git("checkout", "-q", "other")
        report = ws.sync()
        self.assertEqual(report["head"]["from"] != report["head"]["to"], True)
        self.assertEqual(sorted(report["refreshed"]), ["a.txt"])
        self.assertEqual(report["added"], ["only_other.txt"])
        self.assertEqual(self.status_of(ws), {"src/b.py": "M"})

    def test_it_writes_nothing_to_the_project(self):
        self.put("a.txt", b"dirty\n")
        (self.proj / "src" / "b.py").touch()                                  # a stale stat: a plain `git status` would refresh the index here
        before = everything(self.proj)
        ws = self.open_ws()
        self.box(ws, "a.txt", b"model\n")
        ws.sync()
        ws.changed()
        ws.diff()
        ws.tested_record("t", 0, "ok")
        with self.assertRaises(ApplyRefused):
            self.put("a.txt", b"moved\n")
            ws.apply_plan()
        self.put("a.txt", b"dirty\n")
        ws.sync()
        self.assertEqual(everything(self.proj), before)


class Apply(Base):
    def test_plan_carries_exact_bytes(self):
        self.put("crlf.txt", b"one\r\ntwo\r\n")
        self.commit()
        ws = self.open_ws()
        self.box(ws, "crlf.txt", b"one\r\nTWO\r\n")
        self.box(ws, "new.bin", b"\0\1\xff")
        os.remove(os.path.join(ws.copy, "a.txt"))
        plan = ws.apply_plan()
        self.assertEqual([(w.path, w.action, w.data) for w in plan.writes], [("crlf.txt", "write", b"one\r\nTWO\r\n"), ("new.bin", "write", b"\0\1\xff"), ("a.txt", "delete", None)])
        self.assertTrue(all(Path(w.full).is_relative_to(self.proj) for w in plan.writes))

    def test_rename_is_a_write_then_a_delete(self):
        ws = self.open_ws()
        os.rename(os.path.join(ws.copy, "a.txt"), os.path.join(ws.copy, "b.txt"))
        self.assertEqual([(w.path, w.action) for w in ws.apply_plan().writes], [("b.txt", "write"), ("a.txt", "delete")])

    def test_it_never_writes(self):
        ws = self.open_ws()
        self.box(ws, "a.txt", b"changed\n")
        os.remove(os.path.join(ws.copy, "src", "b.py"))
        before = everything(self.proj)
        ws.apply_plan()
        self.assertEqual(everything(self.proj), before)

    def test_a_moved_base_refuses_everything(self):
        ws = self.open_ws()
        self.box(ws, "a.txt", b"the model\n")
        self.box(ws, "ok.txt", b"would be fine alone\n")
        self.put("a.txt", b"the user moved it\n")
        with self.assertRaises(ApplyRefused) as e:
            ws.apply_plan()
        self.assertEqual([r["path"] for r in e.exception.reasons], ["a.txt"])

    def test_a_file_deleted_in_the_project_refuses(self):
        ws = self.open_ws()
        self.box(ws, "a.txt", b"the model\n")
        os.remove(self.proj / "a.txt")
        with self.assertRaisesRegex(ApplyRefused, "changed in the real project"):
            ws.apply_plan()

    def test_a_delete_of_a_changed_file_refuses(self):
        ws = self.open_ws()
        os.remove(os.path.join(ws.copy, "a.txt"))
        self.put("a.txt", b"the user moved it\n")
        with self.assertRaises(ApplyRefused):
            ws.apply_plan()

    def test_a_new_file_does_not_overwrite_one_that_appeared(self):
        ws = self.open_ws()
        self.box(ws, "new.txt", b"the model\n")
        self.put("new.txt", b"the user\n")
        with self.assertRaisesRegex(ApplyRefused, "already exists"):
            ws.apply_plan()

    def test_a_file_left_out_cannot_be_overwritten_by_making_it_again(self):
        self.put("big.bin", b"x" * 5000)
        self.commit()
        ws = self.open_ws(max_file=1000)
        self.box(ws, "big.bin", b"small now")
        with self.assertRaisesRegex(ApplyRefused, "already exists"):
            ws.apply_plan()
        self.assertEqual(len(self.real("big.bin")), 5000)

    def test_a_secret_made_in_the_copy_is_refused(self):
        ws = self.open_ws()
        self.box(ws, ".env", b"TOKEN=made up by a test\n")
        with self.assertRaisesRegex(ApplyRefused, "secret"):
            ws.apply_plan()
        self.assertNotIn("TOKEN", ws.diff())

    def test_a_file_over_the_limit_made_in_the_copy_is_refused(self):
        ws = self.open_ws(max_file=1000)
        self.box(ws, "huge.bin", b"x" * 1001)
        with self.assertRaisesRegex(ApplyRefused, "size limit"):
            ws.apply_plan()

    def test_a_junction_made_in_the_project_after_the_snapshot_is_not_written_through(self):
        ws = self.open_ws()
        self.box(ws, "src/b.py", b"the model\n")
        self.box(ws, "src/new.py", b"the model\n")
        wipe(self.proj / "src")
        junction(self.outside, self.proj / "src")
        with self.assertRaisesRegex(ApplyRefused, "link"):
            ws.apply_plan()

    @unittest.skipUnless(WINDOWS, "a name that differs by case is another file elsewhere")
    def test_renaming_only_the_case_does_not_delete_the_file(self):
        ws = self.open_ws()
        os.rename(os.path.join(ws.copy, "a.txt"), os.path.join(ws.copy, "A.TXT"))
        plan = ws.apply_plan()
        self.assertEqual([(w.path, w.action) for w in plan.writes], [("A.TXT", "write")], "a delete of a.txt would take the file with it")

    @unittest.skipUnless(WINDOWS, "a name that differs by case is another file elsewhere")
    def test_renaming_the_case_and_editing_does_not_delete_the_file(self):
        ws = self.open_ws()
        os.remove(os.path.join(ws.copy, "a.txt"))
        self.box(ws, "A.TXT", b"edited too\n")
        plan = ws.apply_plan()
        self.assertEqual([(w.path, w.action, w.base is not None) for w in plan.writes], [("A.TXT", "write", True)])

    def test_recheck_after_the_user_answers(self):
        ws = self.open_ws()
        self.box(ws, "a.txt", b"the model\n")
        plan = ws.apply_plan()
        ws.recheck(plan)
        self.put("a.txt", b"edited while the question was open\n")
        with self.assertRaises(ApplyRefused):
            ws.recheck(plan)

    def test_the_whole_way_round(self):
        ws = self.open_ws()
        self.box(ws, "a.txt", b"the model\n")
        self.box(ws, "new.txt", b"new\n")
        os.remove(os.path.join(ws.copy, "src", "b.py"))
        for w in ws.apply_plan().writes:                                      # what the caller does through permissions and checkpoints
            if w.action == "write":
                Path(w.full).write_bytes(w.data)
            else:
                os.remove(w.full)
        self.assertEqual(ws.sync()["conflicts"], [], "the project now holds what the copy holds")
        self.assertEqual(ws.changed(), [])


class Tested(Base):
    def test_a_record_binds_the_delta(self):
        ws = self.open_ws()
        self.box(ws, "a.txt", b"one\n")
        record = ws.tested_record("pytest", 0, "3 passed")
        self.assertEqual((record["command"], record["exit_code"], record["summary"], record["files"]), ("pytest", 0, "3 passed", ["a.txt"]))
        self.assertIn("a.txt", record["base"])
        self.assertEqual(ws.apply_plan().tested, record)

    def test_any_change_after_the_test_loses_it(self):
        ws = self.open_ws()
        self.box(ws, "a.txt", b"one\n")
        ws.tested_record("pytest", 0, "3 passed")
        self.box(ws, "a.txt", b"two\n")
        self.assertIsNone(ws.apply_plan().tested)
        self.box(ws, "a.txt", b"one\n")
        self.assertIsNotNone(ws.apply_plan().tested, "the same bytes are the same delta")

    def test_a_failed_run_is_named_as_failed(self):
        ws = self.open_ws()
        self.box(ws, "a.txt", b"one\n")
        ws.tested_record("pytest", 1, "1 failed")
        self.assertEqual(ws.apply_plan().tested["exit_code"], 1)

    def test_no_record_for_the_empty_delta_is_none(self):
        ws = self.open_ws()
        self.assertIsNone(ws.apply_plan().tested)

    def test_the_simulation_label_reaches_the_plan(self):
        ws = self.open_ws()
        self.box(ws, "a.txt", b"one\n")
        ws.tested_record("pytest", 0, "3 passed", simulated="no GPU, fake model")
        self.assertEqual(ws.apply_plan().tested["simulated"], "no GPU, fake model")


class Close(Base):
    def test_read_only_files_and_folders(self):
        ws = self.open_ws()
        path = os.path.join(ws.copy, "a.txt")
        os.chmod(path, stat.S_IREAD)
        folder = os.path.join(ws.copy, "src")
        os.chmod(os.path.join(folder, "b.py"), stat.S_IREAD)
        os.chmod(folder, stat.S_IREAD | stat.S_IEXEC)
        self.assertTrue(ws.close())
        self.assertFalse(os.path.exists(ws.root))

    def test_twice_is_fine(self):
        ws = self.open_ws()
        self.assertTrue(ws.close())
        self.assertTrue(ws.close())

    def test_it_removes_only_its_own_folder(self):
        ws = self.open_ws()
        (self.scratch / "someone_elses").write_text("keep\n")
        ws.close()
        self.assertEqual(os.listdir(self.scratch), ["someone_elses"])

    def test_a_junction_to_the_real_project_is_not_followed(self):
        before = everything(self.proj)
        ws = self.open_ws()
        junction(self.proj, Path(ws.copy) / "into_the_real_project")
        self.assertTrue(ws.close())
        self.assertEqual(everything(self.proj), before)
        self.assertFalse(os.path.exists(ws.root))


class Quota(Base):
    def test_snapshot_over_quota_leaves_nothing(self):
        with self.assertRaises(QuotaExceeded):
            Workspace(self.proj, self.scratch, quota=10)
        self.assertEqual(os.listdir(self.scratch), [])

    def test_a_project_too_big_is_refused_before_anything_is_copied(self):
        with mock.patch.object(Workspace, "_copy_one", side_effect=AssertionError("copied")):
            with self.assertRaises(QuotaExceeded):
                Workspace(self.proj, self.scratch, quota=10)

    def test_kept_base_counts_too(self):
        self.put("noise.bin", os.urandom(4000))
        self.commit()
        with self.assertRaises(QuotaExceeded):
            Workspace(self.proj, self.scratch, quota=6000)                    # the copy alone would fit; the kept bytes of random data do not
        self.assertEqual(os.listdir(self.scratch), [])

    def test_usage_counts_what_the_box_wrote(self):
        ws = self.open_ws()
        before = ws.usage()
        self.box(ws, "out.bin", b"x" * 5000)
        self.assertEqual(ws.usage(), before + 5000)

    def test_a_write_past_the_quota_is_refused(self):
        probe = self.open_ws()
        size = probe.usage()
        probe.close()
        ws = self.open_ws(quota=size + 100)
        ws.write_bytes("fits.txt", b"x" * 50)
        with self.assertRaises(QuotaExceeded):
            ws.write_bytes("does_not.txt", b"x" * 200)
        self.assertIsNone(self.in_copy(ws, "does_not.txt"))

    def test_replacing_a_file_counts_only_the_difference(self):
        probe = self.open_ws()
        size = probe.usage()
        probe.close()
        ws = self.open_ws(quota=size + 10)
        ws.write_bytes("a.txt", b"y" * 20)                                    # a.txt is 14 bytes now, so this adds 6

    def test_sync_past_the_quota_keeps_the_base_so_apply_refuses(self):
        probe = self.open_ws()
        size = probe.usage()
        probe.close()
        ws = self.open_ws(quota=size + 10)
        self.box(ws, "a.txt", b"one\ntwo\nthree\n")                           # unchanged bytes: only the real file moves
        self.put("src/b.py", os.urandom(500))
        self.box(ws, "note.txt", b"x")
        report = ws.sync()
        self.assertEqual(report["skipped"], ["src/b.py"])
        self.assertEqual(self.in_copy(ws, "src/b.py"), b"def f():\n    return 1\n")
        self.box(ws, "src/b.py", b"def f():\n    return 3\n")
        with self.assertRaises(ApplyRefused):
            ws.apply_plan()


class Finding(Base):
    def setUp(self):
        super().setUp()
        self.put("pkg/mod.py", b"def alpha():\n    return 'needle'\n")
        self.put("pkg/deep/er/file.py", b"# needle here too\n")
        self.put("docs/note.md", "ข้อความไทย needle\r\n".encode("utf-8"))
        self.put("data.bin", b"\0needle\0")
        self.put("wide.txt", "needle in UTF-16\r\n".encode("utf-16"))
        self.put("app.min.js", b"needle();\n")
        self.put(".env", b"needle=secret\n")
        self.commit()
        self.ws = self.open_ws()
        self.find = search.Search(self.ws)

    def test_a_pattern_in_a_child_finds_what_the_server_finds(self):
        plain = self.find.search("needle")
        with mock.patch.object(subprocess, "run", wraps=subprocess.run) as ran:
            child = self.find.search("needl+e")
        self.assertTrue(ran.called, "a repeat goes to the child")
        self.assertEqual(sorted((h["path"], h["line"], h["text"]) for h in child["hits"]), sorted((h["path"], h["line"], h["text"]) for h in plain["hits"]))
        self.assertEqual(child["skipped"], plain["skipped"])

    def test_a_pattern_with_no_repeat_stays_in_the_server(self):
        with mock.patch.object(subprocess, "run", side_effect=AssertionError("no child")):
            self.assertTrue(self.find.search("needle")["hits"])

    def test_a_pattern_that_runs_on_is_killed(self):
        (Path(self.ws.copy) / "long.txt").write_bytes(b"a" * 60 + b"!\n")
        t = time.monotonic()
        with mock.patch.object(search, "SEARCH_SECONDS", 1.0):
            out = self.find.search(r"(a|aa)+$")
        self.assertFalse(out["ok"])
        self.assertIn("still running", out["error"])
        self.assertLess(time.monotonic() - t, 5.0)

    def test_search_finds_lines_with_line_numbers(self):
        out = self.find.search("needle")
        hits = {(h["path"], h["line"]) for h in out["hits"]}
        self.assertIn(("pkg/mod.py", 2), hits)
        self.assertIn(("pkg/deep/er/file.py", 1), hits)
        self.assertIn(("docs/note.md", 1), hits)
        self.assertIn(("wide.txt", 1), hits)

    def test_a_hit_has_no_line_ending_in_it(self):
        hit = next(h for h in self.find.search("needle")["hits"] if h["path"] == "docs/note.md")
        self.assertEqual(hit["text"], "ข้อความไทย needle")

    def test_search_keeps_to_the_exclusions_and_says_what_it_skipped(self):
        out = self.find.search("needle")
        paths = {h["path"] for h in out["hits"]}
        self.assertNotIn(".env", paths)
        self.assertNotIn("data.bin", paths)
        self.assertNotIn("app.min.js", paths)
        self.assertEqual(out["skipped"], {"generated": 1, "binary": 1})

    def test_search_never_reads_git(self):
        self.assertEqual(self.find.search("ref: refs/heads", max_hits=5)["hits"], [])

    def test_search_sees_an_edit_at_once(self):
        self.find.search("needle")
        self.box(self.ws, "pkg/mod.py", b"def alpha():\n    return 'thread'\n")
        self.assertEqual(self.find.search("thread")["hits"][0]["path"], "pkg/mod.py")
        self.assertNotIn("pkg/mod.py", {h["path"] for h in self.find.search("needle")["hits"]})

    def test_a_same_size_edit_in_one_clock_tick_is_seen(self):
        p = os.path.join(self.ws.copy, "pkg", "mod.py")
        st = os.stat(p)
        self.box(self.ws, "pkg/mod.py", b"AAAA\n")
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
        self.assertTrue(self.find.search("AAAA")["hits"])
        self.box(self.ws, "pkg/mod.py", b"BBBB\n")
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
        self.assertTrue(self.find.search("BBBB")["hits"])

    def test_a_deleted_file_is_not_found_and_leaves_memory(self):
        time.sleep(workspace.MTIME_TICK_NS / 1e9 * 2)                         # a file is kept in memory once it has settled
        self.find.search("needle")
        self.assertIn("pkg/mod.py", self.find._texts)
        os.remove(os.path.join(self.ws.copy, "pkg", "mod.py"))
        self.assertNotIn("pkg/mod.py", {h["path"] for h in self.find.search("needle")["hits"]})
        self.assertNotIn("pkg/mod.py", self.find._texts)

    def test_a_first_search_does_not_open_every_file_of_the_copy(self):
        with mock.patch.object(workspace.time, "time_ns", return_value=time.time_ns() + 10**10):
            ws = self.open_ws()                                               # as if the copy took longer than the file clock's tick, as a real one does: its files count as settled
        opened = []
        real_read = ws.read_bytes
        with mock.patch.object(ws, "read_bytes", side_effect=lambda *a: opened.append(a) or real_read(*a)):
            self.assertTrue(search.Search(ws).search("needle")["hits"])
        self.assertEqual(opened, [], "the kept store holds these bytes in one file; opening each one costs about 4 ms here")

    def test_new_output_of_a_test_is_not_searched(self):
        self.box(self.ws, "run.log", b"needle in ignored output\n")
        self.box(self.ws, "src/__pycache__/x.pyc", b"needle\n")
        self.box(self.ws, f"{workspace.RESERVED}/output/1.txt", b"needle in saved output\n")
        paths = {h["path"] for h in self.find.search("needle")["hits"]}
        self.assertFalse(paths & {"run.log", "src/__pycache__/x.pyc", f"{workspace.RESERVED}/output/1.txt"})

    def test_a_planted_link_is_not_followed(self):
        (self.outside / "stolen.txt").write_text("needle from outside\n")
        junction(self.outside, Path(self.ws.copy) / "peek")
        self.assertNotIn("peek/stolen.txt", {h["path"] for h in self.find.search("needle")["hits"]})

    def test_a_secret_made_in_the_copy_is_not_searched(self):
        self.box(self.ws, "fresh.pem", b"needle\n")
        self.assertNotIn("fresh.pem", {h["path"] for h in self.find.search("needle")["hits"]})

    def test_max_hits(self):
        self.box(self.ws, "many.txt", b"needle\n" * 500)
        out = self.find.search("needle", "many.txt", max_hits=10)
        self.assertEqual((len(out["hits"]), out["truncated"]), (10, True))

    def test_one_hit_for_a_line(self):
        self.box(self.ws, "twice.txt", b"needle needle needle\nother\nneedle\n")
        self.assertEqual([h["line"] for h in self.find.search("needle", "twice.txt")["hits"]], [1, 3])

    def test_globs(self):
        self.assertEqual({h["path"] for h in self.find.search("needle", "*.py")["hits"]}, {"pkg/mod.py", "pkg/deep/er/file.py"})
        self.assertEqual({h["path"] for h in self.find.search("needle", "pkg/*.py")["hits"]}, {"pkg/mod.py"})
        self.assertEqual({h["path"] for h in self.find.search("needle", "pkg/**/*.py")["hits"]}, {"pkg/mod.py", "pkg/deep/er/file.py"})

    def test_a_pattern_with_anchors_works_by_line(self):
        self.assertEqual([h["line"] for h in self.find.search(r"^\s+return", "mod.py")["hits"]], [2])

    def test_a_bad_pattern_is_an_answer(self):
        out = self.find.search("(unclosed")
        self.assertFalse(out["ok"])
        self.assertIn("pattern", out["error"])

    def test_a_pattern_that_would_never_end_is_refused(self):
        self.box(self.ws, "a.txt", b"a" * 200 + b"!\n")
        for pattern in (r"(a+)+$", r"(a*)*$", r"(\s*\w+)+$", r"(?:x|(a+))+$"):
            with self.subTest(pattern):
                self.assertFalse(self.find.search(pattern)["ok"])
        self.assertTrue(self.find.search(r"(ab)+$")["ok"], "a repeat with nothing repeating inside it is fine")

    def test_find_file(self):
        self.assertEqual(self.find.find_file("*.py")["paths"], ["pkg/deep/er/file.py", "pkg/mod.py", "src/b.py"])
        self.assertEqual(self.find.find_file("docs/*")["paths"], ["docs/note.md"])
        self.assertEqual(self.find.find_file("nothing*")["paths"], [])
        self.assertEqual(self.find.find_file(".env")["paths"], [])

    def test_tree(self):
        out = self.find.tree(".", 2)
        self.assertTrue(out["ok"])
        self.assertIn("pkg/ (2 files)", out["text"])
        self.assertIn("  mod.py", out["text"])
        self.assertIn("  deep/ (1 files)", out["text"])
        self.assertNotIn("file.py", out["text"], "past the depth")
        self.assertNotIn(".env", out["text"])

    def test_tree_of_a_folder(self):
        self.assertEqual(self.find.tree("pkg/deep", 2)["text"], "er/ (1 files)\n  file.py")

    def test_a_made_up_folder_names_what_is_there(self):
        out = self.find.tree("pkgs")
        self.assertFalse(out["ok"])
        self.assertIn("no such folder", out["error"])

    def test_read_gives_pieces_that_join_up(self):
        text = "ไทย-" * 2000
        self.box(self.ws, "long.txt", text.encode("utf-8"))
        got, offset = "", 0
        while offset is not None:
            part = self.find.read("long.txt", offset, 997)                    # a unit of this text is 10 bytes, so a round number would always land between characters
            self.assertTrue(part["ok"])
            self.assertLessEqual(len(part["text"].encode("utf-8")), 997)
            got += part["text"]
            offset = part["next"]
        self.assertEqual(got, text, "no character is cut in two")

    def test_read_of_a_binary_or_big_file_gives_its_size(self):
        out = self.find.read("data.bin")
        self.assertEqual((out["size"], out["kind"]), (8, "binary"))
        self.assertNotIn("text", out)
        self.box(self.ws, "big.txt", b"x" * (self.ws.max_file + 1))
        out = self.find.read("big.txt")
        self.assertEqual(out["kind"], "too big to show")
        self.assertNotIn("text", out)

    def test_read_of_utf16_is_text(self):
        self.assertIn("needle in UTF-16", self.find.read("wide.txt")["text"])

    def test_read_of_a_made_up_path_names_the_near_ones(self):
        out = self.find.read("pkg/mode.py")
        self.assertFalse(out["ok"])
        self.assertIn("pkg/mod.py", out["near"])
        self.assertIn("pkg/mod.py", self.find.read("wrong/folder/mod.py")["near"], "the same name in another folder")

    def test_read_refuses_what_is_left_out(self):
        for path in (".git/config", ".env", "../../outside/stolen.txt", str(self.proj / "a.txt")):
            with self.subTest(path):
                self.assertFalse(self.find.read(path)["ok"])


PY = b'''"""A module."""
import os

LIMIT = 3


@decorated
def first(a):
    return a


class Thing:
    size: int = 1

    def method(self):
        return 1

    @property
    def value(self):
        return 2


class Other:
    def method(self):
        return 3


async def later():
    def inner():
        local = 1
        return local
    return inner
'''


class Structure(Base):
    def setUp(self):
        super().setUp()
        self.put("code.py", PY)
        self.commit()
        self.ws = self.open_ws()
        self.sym = symbols.Symbols(self.ws)

    def src(self, rel: str = "code.py") -> bytes:
        return self.in_copy(self.ws, rel)

    def names(self, path: str = "code.py") -> list[str]:
        return [s["name"] for s in self.sym.outline(path)["symbols"]]

    def test_outline(self):
        self.assertEqual(self.names(), ["LIMIT", "first", "Thing", "Thing/size", "Thing/method", "Thing/value", "Other", "Other/method", "later", "later/inner"])
        kinds = {s["name"]: s["kind"] for s in self.sym.outline("code.py")["symbols"]}
        self.assertEqual((kinds["first"], kinds["Thing"], kinds["Thing/method"], kinds["LIMIT"], kinds["later/inner"]), ("function", "class", "method", "variable", "function"))

    def test_outline_depth(self):
        self.assertEqual([s["name"] for s in self.sym.outline("code.py", depth=1)["symbols"]], ["LIMIT", "first", "Thing", "Other", "later"])

    def test_outline_of_a_made_up_path(self):
        out = self.sym.outline("cod.py")
        self.assertFalse(out["ok"])
        self.assertIn("code.py", out["near"])

    def test_find_symbol_with_body(self):
        out = self.sym.find_symbol("first", body=True)
        self.assertEqual(len(out["matches"]), 1)
        self.assertEqual(out["matches"][0]["body"], "@decorated\ndef first(a):\n    return a")
        self.assertEqual(out["matches"][0]["line"], 7, "a decorator is part of the symbol")

    def test_find_symbol_by_path(self):
        for name in ("Thing/method", "Thing.method"):
            with self.subTest(name):
                self.assertEqual([m["name"] for m in self.sym.find_symbol(name)["matches"]], ["Thing/method"])

    def test_a_missing_symbol(self):
        out = self.sym.find_symbol("nothing_like_this")
        self.assertFalse(out["ok"])
        self.assertIn("no symbol", out["error"])

    def test_duplicates_answer_all_and_change_nothing(self):
        out = self.sym.find_symbol("method")
        self.assertEqual({m["name"] for m in out["matches"]}, {"Thing/method", "Other/method"})
        before = self.src()
        for edit in (self.sym.replace_body, self.sym.insert_after, self.sym.insert_before):
            with self.subTest(edit.__name__):
                out = edit("method", "    def method(self):\n        return 9\n")
                self.assertFalse(out["ok"])
                self.assertEqual({m["name"] for m in out["matches"]}, {"Thing/method", "Other/method"})
                self.assertEqual(self.src(), before)

    def test_duplicates_across_files(self):
        self.box(self.ws, "second.py", b"def first():\n    return 2\n")
        out = self.sym.replace_body("first", "def first():\n    return 3\n")
        self.assertFalse(out["ok"])
        self.assertEqual(len(out["matches"]), 2)

    def test_a_ref_from_an_answer_picks_one(self):
        self.box(self.ws, "second.py", b"def first():\n    return 2\n")
        ref = next(m["ref"] for m in self.sym.find_symbol("first")["matches"] if m["file"] == "second.py")
        self.assertTrue(self.sym.replace_body(ref, "def first():\n    return 3\n")["ok"])
        self.assertEqual(self.src("second.py"), b"def first():\n    return 3\n")
        self.assertIn(b"@decorated", self.src())

    def test_a_path_in_the_name_picks_one(self):
        self.assertTrue(self.sym.replace_body("Other.method", "    def method(self):\n        return 9\n")["ok"])
        self.assertIn(b"return 9", self.src())
        self.assertIn(b"return 1\n", self.src(), "Thing.method is as it was")

    def test_the_same_name_twice_in_one_class_needs_the_line(self):
        self.box(self.ws, "prop.py", b"class C:\n    @property\n    def x(self):\n        return 1\n\n    @x.setter\n    def x(self, v):\n        pass\n")
        out = self.sym.replace_body("C.x", "    def x(self):\n        return 2\n")
        self.assertFalse(out["ok"])
        self.assertEqual(sorted(m["line"] for m in out["matches"]), [2, 6])
        self.assertTrue(self.sym.replace_body("prop.py:C/x@6", "    @x.setter\n    def x(self, v):\n        self.v = v\n")["ok"])
        self.assertIn(b"self.v = v", self.src("prop.py"))

    def test_what_was_read_can_be_given_back_unchanged(self):
        crlf = PY.replace(b"\n", b"\r\n")
        for name, data in (("lf.py", PY), ("crlf.py", crlf), ("bom.py", b"\xef\xbb\xbf" + crlf), ("no_end.py", b"def z():\n    return 0")):
            self.box(self.ws, name, data)
            for row in self.sym.outline(name)["symbols"]:
                if row["kind"] == "variable":
                    continue
                with self.subTest(file=name, symbol=row["name"]):
                    body = self.sym.find_symbol(row["ref"], body=True)["matches"][0]["body"]
                    self.assertTrue(self.sym.replace_body(row["ref"], body)["ok"])
                    self.assertEqual(self.src(name), data)

    def test_crlf_stays_crlf(self):
        self.box(self.ws, "win.py", PY.replace(b"\n", b"\r\n"))
        self.assertTrue(self.sym.replace_body("win.py:first", "def first(a):\n    b = a\n    return b\n")["ok"])
        self.assertTrue(self.sym.insert_after("win.py:first", "\n\ndef second():\n    pass\n")["ok"])
        self.assertTrue(self.sym.insert_before("win.py:Other", "# a note\n")["ok"])
        data = self.src("win.py")
        self.assertEqual(data.count(b"\n"), data.count(b"\r\n"), "no bare line ending got in")
        self.assertIn(b"def first(a):\r\n    b = a\r\n    return b\r\n", data)
        self.assertIn(b"# a note\r\nclass Other:", data)

    def test_a_line_ending_mix_outside_the_symbol_is_left_alone(self):
        data = b"def a():\r\n    return 1\r\n\r\ndef b():\n    return 2\n\ndef c():\r\n    return 3\r\n"
        self.box(self.ws, "mix.py", data)
        self.assertTrue(self.sym.replace_body("mix.py:b", "def b():\n    return 20\n")["ok"])
        self.assertEqual(self.src("mix.py"), data.replace(b"def b():\n    return 2\n", b"def b():\r\n    return 20\n"))

    def test_thai_before_the_symbol_does_not_move_the_offsets(self):
        data = "x = 'ไทย'  # ภาษาไทย\n\n\ndef f():  # ไทย\n    return 'ไทย'\n\n\ndef g():\n    return 2\n".encode("utf-8")
        self.box(self.ws, "thai.py", data)
        self.assertEqual(self.sym.find_symbol("thai.py:f", body=True)["matches"][0]["body"], "def f():  # ไทย\n    return 'ไทย'")
        self.assertTrue(self.sym.replace_body("thai.py:f", "def f():\n    return 'สวัสดี'\n")["ok"])
        self.assertEqual(self.src("thai.py"), data.replace("def f():  # ไทย\n    return 'ไทย'".encode(), "def f():\n    return 'สวัสดี'".encode()))

    def test_a_form_feed_does_not_shift_the_lines(self):
        data = b"a = 1\n\x0c\ndef f():\n    return 1\n\x0c\ndef g():\n    return 2\n"
        self.box(self.ws, "ff.py", data)
        self.assertEqual(self.sym.find_symbol("ff.py:g", body=True)["matches"][0]["body"], "def g():\n    return 2")

    def test_a_trailing_comment_stays(self):
        self.box(self.ws, "tail.py", b"LIMIT = 1  # why\n\n\ndef f():\n    return 1  # because\n\n\nx = 2\n")
        self.assertTrue(self.sym.replace_body("tail.py:f", "def f():\n    return 2\n")["ok"])
        self.assertIn(b"return 2  # because\n", self.src("tail.py"))

    def test_a_cp874_file_keeps_its_encoding(self):
        data = "# -*- coding: cp874 -*-\nx = 'ไทย'\n\n\ndef f():\n    return 1\n".encode("cp874")
        self.box(self.ws, "old.py", data)
        self.assertTrue(self.sym.replace_body("old.py:f", "def f():\n    return 'ภาษา'\n")["ok"])
        self.assertEqual(self.src("old.py"), data.replace(b"return 1", "return 'ภาษา'".encode("cp874")))

    def test_text_the_encoding_cannot_hold_is_refused(self):
        data = "# -*- coding: cp874 -*-\ndef f():\n    return 1\n".encode("cp874")
        self.box(self.ws, "old.py", data)
        out = self.sym.replace_body("old.py:f", "def f():\n    return '日本語'\n")
        self.assertFalse(out["ok"])
        self.assertEqual(self.src("old.py"), data)

    def test_insert_after_at_the_end_of_a_file_with_no_line_end(self):
        self.box(self.ws, "end.py", b"def f():\n    return 1")
        self.assertTrue(self.sym.insert_after("end.py:f", "\n\ndef g():\n    return 2\n")["ok"])
        self.assertEqual(self.src("end.py"), b"def f():\n    return 1\n\n\ndef g():\n    return 2")

    def test_insert_before_goes_above_the_decorator(self):
        self.assertTrue(self.sym.insert_before("first", "# above\n")["ok"])
        self.assertIn(b"# above\n@decorated\ndef first", self.src())

    def test_an_edit_that_breaks_the_file_is_refused(self):
        before = self.src()
        out = self.sym.replace_body("first", "def first(a:\n    return a\n")
        self.assertFalse(out["ok"])
        self.assertIn("would not parse", out["error"])
        self.assertEqual(self.src(), before)

    def test_a_symbol_sharing_a_line_is_refused(self):
        self.box(self.ws, "semi.py", b"a = 1; b = 2\nc = 3\n")
        out = self.sym.replace_body("semi.py:a", "a = 9\n")
        self.assertFalse(out["ok"])
        self.assertIn("shares a line", out["error"])
        self.assertEqual(self.src("semi.py"), b"a = 1; b = 2\nc = 3\n")

    def test_a_symbol_is_never_read_from_a_file_that_changed_within_one_clock_tick(self):
        path = os.path.join(self.ws.copy, "code.py")
        st = os.stat(path)
        self.sym.find_symbol("first", body=True)
        self.box(self.ws, "code.py", PY.replace(b"    return a\n", b"    return b\n"))
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))                   # the same size and mtime, as two writes inside one tick leave
        self.assertEqual(self.sym.find_symbol("first", body=True)["matches"][0]["body"], "@decorated\ndef first(a):\n    return b")

    def test_two_quick_edits_of_the_same_size(self):
        self.assertTrue(self.sym.replace_body("first", "@decorated\ndef first(a):\n    return 1\n")["ok"])
        self.assertTrue(self.sym.replace_body("first", "@decorated\ndef first(a):\n    return 2\n")["ok"])
        self.assertIn(b"return 2", self.src())
        self.assertEqual(self.src().count(b"def first"), 1)

    def test_a_symbol_edit_shows_in_the_delta(self):
        self.assertTrue(self.sym.replace_body("first", "def first(a):\n    return 2\n")["ok"])
        self.assertEqual(self.status_of(self.ws), {"code.py": "M"})
        self.assertEqual(self.real("code.py"), PY, "the real project is not touched")

    def test_an_edit_past_the_quota_is_an_answer(self):
        self.ws.quota = self.ws.usage()
        out = self.sym.insert_after("first", "x = " + repr("y" * 5000) + "\n")
        self.assertFalse(out["ok"])
        self.assertIn("quota", out["error"])


class NoStructure(Base):
    def setUp(self):
        super().setUp()
        self.put("broken.py", b"def good():\n    return 1\n\ndef bad(:\n    return findme\n")
        self.put("native.cpp", b"int findme() {\n  return 1;\n}\n")
        self.put("notes.md", b"# findme\n")
        self.commit()
        self.ws = self.open_ws()
        self.sym = symbols.Symbols(self.ws)

    def test_a_file_that_does_not_parse_says_so(self):
        out = self.sym.outline("broken.py")
        self.assertFalse(out["ok"])
        self.assertTrue(out["text_only"])
        self.assertIn("does not parse at line 4", out["error"])
        self.assertIn("text search", out["error"])

    def test_find_symbol_falls_back_to_text_and_labels_it(self):
        out = self.sym.find_symbol("findme")
        self.assertEqual(out["matches"], [], "text is not a symbol")
        self.assertEqual({h["path"] for h in out["text_hits"]}, {"broken.py", "native.cpp", "notes.md"})
        self.assertIn("does not parse", out["note"])

    def test_a_symbol_in_a_file_that_does_not_parse_is_not_edited(self):
        before = self.in_copy(self.ws, "broken.py")
        out = self.sym.replace_body("broken.py:good", "def good():\n    return 2\n")
        self.assertFalse(out["ok"])
        self.assertTrue(out["text_only"])
        self.assertEqual(self.in_copy(self.ws, "broken.py"), before)

    def test_search_still_reads_it(self):
        hits = search.Search(self.ws).search("findme", "*.py")["hits"]
        self.assertEqual([(h["path"], h["line"]) for h in hits], [("broken.py", 5)])

    def test_files_that_attack_the_parser_are_answers(self):
        cases = {"deep.py": b"x = " + b"(" * 5000 + b"1" + b")" * 5000 + b"\n", "nul.py": b"def f():\0\n", "enc.py": b"# -*- coding: no-such-codec -*-\nx = 1\n", "bad.py": b"x = '\xff\xfe'\n",
                 "ind.py": b"\n".join(b" " * i + b"if x:" for i in range(0, 400)) + b"\n",
                 "late_bytes.py": b"a = 1\nb = 2\nc = '\xff\xfe'\n",                             # past the two lines the encoding is looked for in: an error of its own kind
                 "chain.py": b"x = " + b"+".join([b"1"] * 100000) + b"\n"}                         # shallow to the parser, deep once it is a tree
        for name, data in cases.items():
            with self.subTest(name):
                self.box(self.ws, name, data)
                out = self.sym.outline(name)
                self.assertFalse(out["ok"])
                self.assertTrue(out["text_only"])

    def test_a_file_that_gets_fixed_has_an_outline(self):
        self.box(self.ws, "broken.py", b"def good():\n    return 1\n")
        self.assertTrue(self.sym.outline("broken.py")["ok"])

    def test_other_languages_say_only_text_search_when_a_grammar_is_missing(self):
        with mock.patch.object(symbols, "_parser", return_value=None):
            out = self.sym.outline("native.cpp")
        self.assertFalse(out["ok"])
        self.assertTrue(out["text_only"])
        self.assertIn("Only text search is available", out["error"])
        self.assertIn("tree_sitter", out["error"])

    def test_a_grammar_that_cannot_be_imported_is_noticed_for_real(self):
        symbols._parser.cache_clear()
        self.addCleanup(symbols._parser.cache_clear)
        grammar = symbols.Grammar("tree_sitter_no_such_grammar", "language", ())
        self.assertIsNone(symbols._parser(grammar))

    def test_an_unknown_kind_of_file_says_so(self):
        out = self.sym.outline("notes.md")
        self.assertFalse(out["ok"])
        self.assertIn("no grammar is known for .md", out["error"])

    def test_binary(self):
        self.box(self.ws, "blob.py", b"\0\1\2")
        self.assertFalse(self.sym.outline("blob.py")["ok"])


@unittest.skipUnless(symbols._parser(symbols.GRAMMARS[".cpp"]), "tree_sitter_cpp is not installed here")
class TreeSitter(Base):
    def test_cpp(self):
        data = b"namespace ns {\r\ntemplate <typename T>\r\nT twice(T x) {\r\n  return x * 2;\r\n}\r\n\r\nclass Box {\r\n public:\r\n  int size() const { return 1; }\r\n};\r\n}  // namespace ns\r\n"
        self.put("box.hpp", data)
        self.commit()
        ws = self.open_ws()
        sym = symbols.Symbols(ws)
        names = [s["name"] for s in sym.outline("box.hpp")["symbols"]]
        self.assertEqual(names, ["ns", "ns/twice", "ns/Box", "ns/Box/size"])
        self.assertTrue(sym.find_symbol("ns::twice", body=True)["matches"][0]["body"].startswith("template <typename T>"))
        self.assertTrue(sym.replace_body("twice", "template <typename T>\nT twice(T x) {\n  return x + x;\n}\n")["ok"])
        self.assertEqual(self.in_copy(ws, "box.hpp"), data.replace(b"x * 2", b"x + x"))
        self.assertFalse(sym.replace_body("twice", "T twice(T x) {{{\n")["ok"], "an edit that breaks a clean file")


class Results(Base):
    def setUp(self):
        super().setUp()
        self.ws = self.open_ws()
        self.out = output.Output(self.ws)

    def test_header_is_the_designs_line(self):
        self.assertEqual(output.header(1, 2.3), "[sandbox: สำเนา ไม่ใช่ของจริง] exit 1, 2.3 s")
        self.assertEqual(output.header(), "[sandbox: สำเนา ไม่ใช่ของจริง]")

    def test_a_short_result(self):
        self.assertEqual(self.out.render("ok\n", exit_code=0, seconds=0.5), "[sandbox: สำเนา ไม่ใช่ของจริง] exit 0, 0.5 s\nok\n")

    def test_the_changed_line_closes_the_result(self):
        self.assertTrue(self.out.render("41 passed", changed=2).endswith("41 passed\nแก้แล้ว 2 ไฟล์ในสำเนา ยังไม่แตะ project จริง\n"))

    def test_a_long_result_fits_saves_and_says_how_to_read_on(self):
        body = "".join(f"line {i}\n" for i in range(2000))
        text = self.out.render(body, exit_code=1, seconds=2.0)
        self.assertLessEqual(len(text.encode("utf-8")), 1024)
        self.assertTrue(text.startswith("[sandbox: สำเนา ไม่ใช่ของจริง] exit 1, 2.0 s\n"))
        self.assertIn("line 0\n", text)
        self.assertIn("line 1999\n", text, "the end of a run is where the failure is")
        match = re.search(r'sandbox_read\("([^"]+)", offset=(\d+)\)', text)
        self.assertIsNotNone(match)
        rel, offset = match[1], int(match[2])
        self.assertEqual(self.in_copy(self.ws, rel), body.encode("utf-8"), "the whole of it")
        part = search.Search(self.ws).read(rel, offset, 200)
        self.assertTrue(part["ok"], "the way the text gives must work")
        self.assertEqual(part["text"], body.encode()[offset:offset + 200].decode())

    def test_budget_counts_bytes_not_characters(self):
        for n in (300, 341, 342, 343, 344, 345, 1000, 5000):
            with self.subTest(n=n):
                self.out.new_turn()
                text = self.out.render("ก" * n, exit_code=0, seconds=1.0, changed=3)
                self.assertLessEqual(len(text.encode("utf-8")), 1024)
                self.assertNotIn("\ufffd", text, "no character is cut in two")

    def test_every_size_around_the_edge_fits(self):
        for n in range(900, 1100):
            self.out.new_turn()
            self.assertLessEqual(len(self.out.render("x" * n, exit_code=0, seconds=1.0, changed=2).encode("utf-8")), 1024, n)

    def test_output_that_is_not_utf8(self):
        self.assertIn("\ufffd", self.out.render(b"caf\xe9\n", exit_code=0))

    def test_saved_output_is_not_a_change(self):
        self.out.render("x" * 5000)
        self.assertEqual(self.ws.changed(), [])
        self.assertEqual(self.ws.apply_plan().writes, [])
        self.assertEqual(search.Search(self.ws).find_file("*.txt")["paths"], ["a.txt"])

    def test_each_long_result_has_its_own_file(self):
        one = self.out.render("a" * 5000)
        two = self.out.render("b" * 5000)
        self.assertNotEqual(re.search(r'read\("([^"]+)"', one)[1], re.search(r'read\("([^"]+)"', two)[1])

    def test_the_turn_budget_runs_out_and_the_header_stays(self):
        out = output.Output(self.ws, per_turn=2500)
        texts = [out.render("x" * 5000) for _ in range(5)]
        self.assertTrue(all(t.startswith(output.HEADER) for t in texts), "I5 holds when nothing else fits")
        self.assertIn(output.OUT_OF_BUDGET, texts[-1])
        self.assertLessEqual(sum(len(t.encode()) for t in texts if output.OUT_OF_BUDGET not in t), 2500)

    def test_short_results_spend_the_turn_too(self):
        out = output.Output(self.ws, per_turn=600)
        texts = [out.render("x" * 100) for _ in range(6)]
        self.assertIn(output.OUT_OF_BUDGET, texts[-1])
        self.assertLessEqual(sum(len(t.encode()) for t in texts if output.OUT_OF_BUDGET not in t), 600)

    def test_the_last_of_a_turn_fits_what_is_left(self):
        out = output.Output(self.ws, per_turn=1500)
        first = out.render("x" * 5000)
        second = out.render("y" * 5000)
        self.assertLessEqual(len(first.encode()) + len(second.encode()), 1500)
        self.assertNotIn(output.OUT_OF_BUDGET, second)

    def test_a_new_turn_has_a_new_budget(self):
        out = output.Output(self.ws, per_turn=2000)
        for _ in range(4):
            out.render("x" * 5000)
        out.new_turn()
        self.assertNotIn(output.OUT_OF_BUDGET, out.render("short"))

    def test_past_the_quota_it_says_nothing_was_saved(self):
        self.ws.quota = self.ws.usage()
        text = self.out.render("x" * 5000)
        self.assertLessEqual(len(text.encode()), 1024)
        self.assertIn("not saved", text)
        self.assertNotIn("sandbox_read", text)

    def test_conflicts_in_the_designs_words(self):
        self.assertEqual(output.conflict_lines(["a.txt", "b.py"]), "[เปลี่ยนในของจริงด้วย: a.txt]\n[เปลี่ยนในของจริงด้วย: b.py]")


class GitCalls(Base):
    """What every git run is given. `git status` only sometimes rewrites the index, so a before-and-after comparison can miss a lost flag; this cannot."""

    def setUp(self):
        super().setUp()
        self.calls: list[dict] = []
        real_run = subprocess.run

        def spy(cmd, **kw):
            self.calls.append(kw)
            return real_run(cmd, **kw)
        patcher = mock.patch.object(workspace.subprocess, "run", side_effect=spy)
        patcher.start()
        self.addCleanup(patcher.stop)

    def exercise(self) -> None:
        ws = Workspace(self.proj, self.scratch)
        self.addCleanup(ws.close)
        self.box(ws, "new.txt", b"makes check-ignore run\n")
        ws.changed()
        ws.sync()
        ws.close()

    def test_no_run_opens_a_window_or_takes_a_lock(self):
        self.exercise()
        self.assertGreaterEqual(len(self.calls), 4, "ls-files, status, rev-parse, check-ignore")
        for kw in self.calls:
            self.assertEqual(kw["creationflags"], 0x08000000 if WINDOWS else 0)
            self.assertEqual(kw["env"]["GIT_OPTIONAL_LOCKS"], "0", "without it `git status` writes the real project's index")

    def test_a_git_hook_environment_does_not_point_git_elsewhere(self):
        with mock.patch.dict(os.environ, {"GIT_DIR": str(self.base / "another" / ".git"), "GIT_WORK_TREE": str(self.base), "GIT_INDEX_FILE": "x"}):
            self.exercise()
        for kw in self.calls:
            for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
                self.assertNotIn(key, kw["env"])


class NoWindow(unittest.TestCase):
    def test_every_subprocess_is_told_to_open_none(self):
        """By reading the source: a window on the developer's display is the one thing a passing test here would not show."""
        for name in ("workspace", "search", "symbols", "output"):
            source = (ROOT / "serve" / "sandbox" / f"{name}.py").read_text(encoding="utf-8")
            for line in source.splitlines():
                if re.search(r"subprocess\.(run|Popen|call|check_output|check_call)\(", line):
                    self.assertIn("creationflags=NO_WINDOW", line, line)
        self.assertEqual(workspace.NO_WINDOW, 0x08000000 if WINDOWS else 0)


if __name__ == "__main__":
    unittest.main()
