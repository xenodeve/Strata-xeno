"""`@file` in the prompt (serve/files.py): which files go with typed letters, the text of a mentioned one, and what is never offered or read."""
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

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from serve import files  # noqa: E402
from serve.test_agent_chat import DONE, Fixture  # noqa: E402


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name).resolve()
        self.proj = self.base / "proj"
        self.proj.mkdir()

    def write(self, rel: str, text: str = "x", root: Path | None = None):
        p = (root or self.proj) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(text.encode("utf-8"))                                                  # as it is: no line ends of this system
        return p


class Find(Base):
    def test_the_name_that_starts_with_it_first_then_the_name_that_has_it_then_the_path(self):
        for rel in ("src/app.py", "src/util/application.py", "docs/my-app.md", "src/other.py", "app/readme.txt"):
            self.write(rel)
        got = [f["path"] for f in files.find([str(self.proj)], "app")]
        self.assertEqual(got[:2], ["src/app.py", "src/util/application.py"])            # starts with it; shallow first
        self.assertIn("docs/my-app.md", got)
        self.assertIn("app/readme.txt", got)                                            # only the path has it
        self.assertNotIn("src/other.py", got)

    def test_any_case_and_a_typed_path_with_either_slash(self):
        self.write("src/Components/Button.tsx")
        self.assertEqual([f["path"] for f in files.find([str(self.proj)], "BUTTON")], ["src/Components/Button.tsx"])
        self.assertEqual([f["path"] for f in files.find([str(self.proj)], "src\\comp")], ["src/Components/Button.tsx"])

    def test_nothing_typed_lists_the_shallow_files(self):
        self.write("a.txt"); self.write("deep/er/est/z.txt"); self.write("b.txt")
        self.assertEqual([f["path"] for f in files.find([str(self.proj)], "")][:2], ["a.txt", "b.txt"])

    def test_secrets_and_the_usual_junk_are_not_offered(self):
        for rel in (".env", "config/id_rsa", "keys/server.pem", ".git/config", "node_modules/pkg/index.js", "__pycache__/m.pyc", ".hidden/x.txt", "dist/bundle.js", "ok.txt"):
            self.write(rel)
        self.assertEqual([f["path"] for f in files.find([str(self.proj)], "")], ["ok.txt"])
        self.assertEqual(files.find([str(self.proj)], "env"), [])

    def test_every_folder_of_the_project_is_searched_and_says_which(self):
        wt = self.base / "proj-wt"
        self.write("same.py"); self.write("only_here.py", root=wt)
        got = files.find([str(self.proj), str(wt)], "py")
        self.assertEqual({(f["path"], f["folder"]) for f in got}, {("same.py", 0), ("only_here.py", 1)})

    def test_at_most_a_few_are_offered(self):
        for i in range(files.LIMIT + 15):
            self.write(f"f{i:03}.txt")
        self.assertEqual(len(files.find([str(self.proj)], "f")), files.LIMIT)

    def test_odd_input_is_nothing(self):
        self.write("a.txt")
        self.assertEqual(files.find([str(self.proj)], "x" * 300), [])
        self.assertEqual(files.find([str(self.proj)], "a\0b"), [])
        self.assertEqual(files.find([], "a"), [])
        self.assertEqual(files.find(["C:/not/there", "", None], "a"), [])


class Read(Base):
    def test_the_text_of_a_mentioned_file(self):
        self.write("src/a.py", "print(1)\n")
        got = files.read([str(self.proj)], "src/a.py")
        self.assertEqual((got["ok"], got["name"], got["text"], got["cut"], got["folder"]), (True, "src/a.py", "print(1)\n", False, 0))

    def test_the_first_folder_that_has_it(self):
        wt = self.base / "proj-wt"
        self.write("only.txt", "in the worktree", root=wt)
        self.assertEqual(files.read([str(self.proj), str(wt)], "only.txt")["folder"], 1)

    def test_a_long_file_is_cut_and_says_so(self):
        self.write("big.txt", "y" * (files.MAX_TEXT + 100))
        got = files.read([str(self.proj)], "big.txt")
        self.assertTrue(got["cut"])
        self.assertEqual(len(got["text"]), files.MAX_TEXT)

    def test_a_way_out_of_the_project_is_refused(self):
        self.write("../outside.txt", "secret outside", root=self.proj)
        for rel in ("../outside.txt", "..\\outside.txt", "sub/../../outside.txt", str(self.base / "outside.txt"), "/etc/passwd", "C:\\Windows\\win.ini", "", "a\0b", "x" * 600):
            self.assertFalse(files.read([str(self.proj)], rel)["ok"], rel)

    def test_a_link_that_leads_out_is_refused(self):
        outside = self.base / "outside.txt"
        outside.write_text("secret", encoding="utf-8")
        link = self.proj / "link.txt"
        try:
            os.symlink(outside, link)
        except (OSError, NotImplementedError):
            self.skipTest("no symbolic links here")
        self.assertFalse(files.read([str(self.proj)], "link.txt")["ok"])

    def test_a_secret_and_a_binary_file_are_not_read(self):
        self.write(".env", "TOKEN=abc")
        (self.proj / "pic.bin").write_bytes(b"\0\1\2" * 100)
        self.assertIn("secret", files.read([str(self.proj)], ".env")["error"])
        self.assertIn("binary", files.read([str(self.proj)], "pic.bin")["error"])

    def test_a_file_that_is_not_there(self):
        self.assertIn("no such file", files.read([str(self.proj)], "nope.txt")["error"])
        self.assertFalse(files.read([str(self.proj)], "sub")["ok"])


class TheRoutes(Fixture):
    def get(self, path, headers=None):
        try:
            with urllib.request.urlopen(urllib.request.Request(self.base_url + path, headers=headers or {}), timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_the_page_can_search_and_read(self):
        (self.proj / "src").mkdir()
        (self.proj / "src" / "main.py").write_bytes(b"hi\n")
        self.start(DONE)
        status, body = self.get("/agent/files?path=" + urllib.parse.quote(str(self.proj)) + "&q=main")
        self.assertEqual((status, [f["path"] for f in body["files"]]), (200, ["src/main.py"]))
        status, body = self.get("/agent/mention?path=" + urllib.parse.quote(str(self.proj)) + "&rel=src/main.py")
        self.assertEqual((status, body["ok"], body["text"]), (200, True, "hi\n"))
        status, body = self.get("/agent/mention?path=" + urllib.parse.quote(str(self.proj)) + "&rel=../elsewhere/b.txt")
        self.assertFalse(body["ok"])

    def test_it_is_for_this_pc_only(self):
        self.start(DONE)
        req = urllib.request.Request(self.base_url + "/agent/files?path=" + urllib.parse.quote(str(self.proj)) + "&q=a", headers={"Host": "strata.example"})
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req, timeout=10)
        self.assertIn(cm.exception.code, (403, 421))

    def test_with_an_api_key_the_key_is_the_door(self):
        self.start(DONE)
        self.svc.api_key = "k"
        self.assertEqual(self.get("/agent/files?path=" + urllib.parse.quote(str(self.proj)) + "&q=a")[0], 401)
        self.assertEqual(self.get("/agent/files?path=" + urllib.parse.quote(str(self.proj)) + "&q=a", {"Authorization": "Bearer k"})[0], 200)


if __name__ == "__main__":
    unittest.main()
