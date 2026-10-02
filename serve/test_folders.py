"""The folders of this PC, to choose a project's folder from: what is listed, what is not, and who may ask."""
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

from serve import folders  # noqa: E402
from serve.test_agent_chat import DONE, Fixture  # noqa: E402


class Look(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name).resolve()
        for name in ("beta", "Alpha", "gamma", ".hidden"):
            (self.base / name).mkdir()
        (self.base / "a-file.txt").write_text("x", encoding="utf-8")

    def test_the_folders_in_it_by_name_and_no_files(self):
        seen = folders.look(str(self.base))
        self.assertEqual(seen["path"], str(self.base))
        self.assertEqual([d["name"] for d in seen["dirs"]], ["Alpha", "beta", "gamma"])             # not a-file.txt, not .hidden; any case
        self.assertEqual(seen["dirs"][0]["path"], str(self.base / "Alpha"))
        self.assertEqual(seen["parent"], str(self.base.parent))
        self.assertFalse(seen["truncated"])

    def test_what_is_not_a_folder_is_nothing(self):
        for raw in (str(self.base / "a-file.txt"), str(self.base / "nope"), "a\0b", "x" * 2000):
            self.assertIsNone(folders.look(raw), raw[:20])

    def test_blank_is_the_home_folder(self):
        self.assertEqual(folders.look("")["path"], os.path.realpath(os.path.expanduser("~")))
        self.assertEqual(folders.look(None)["path"], os.path.realpath(os.path.expanduser("~")))

    def test_the_top_has_no_parent_but_the_drives_on_windows(self):
        top = Path(self.base.anchor)
        seen = folders.look(str(top))
        self.assertEqual(seen["parent"], "@drives" if os.name == "nt" else None)

    @unittest.skipUnless(os.name == "nt", "drives are a Windows thing")
    def test_the_drives(self):
        seen = folders.look("@drives")
        self.assertEqual(seen["path"], "")
        self.assertIsNone(seen["parent"])
        self.assertTrue(all(len(d["name"]) == 3 for d in seen["dirs"]) and seen["dirs"])

    def test_a_very_full_folder_is_cut_and_says_so(self):
        for i in range(folders.LIMIT + 3):
            (self.base / f"d{i:04}").mkdir()
        seen = folders.look(str(self.base))
        self.assertEqual(len(seen["dirs"]), folders.LIMIT)
        self.assertTrue(seen["truncated"])


class TheRoute(Fixture):
    def get(self, path, headers=None):
        req = urllib.request.Request(self.base_url + path, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_it_lists_a_folder(self):
        self.start(DONE)
        status, body = self.get("/agent/folders?path=" + urllib.parse.quote(str(self.base)))
        self.assertEqual(status, 200)
        self.assertEqual(sorted(d["name"] for d in body["dirs"]), ["elsewhere", "proj"])

    def test_it_says_when_it_is_not_a_folder(self):
        self.start(DONE)
        status, body = self.get("/agent/folders?path=" + urllib.parse.quote(str(self.proj / "a.txt")))
        self.assertEqual(status, 200)
        self.assertFalse(body["ok"])
        self.assertIn("not a folder", body["error"])

    def test_it_is_for_this_pc_only(self):
        self.start(DONE)
        status, body = self.get("/agent/folders", {"Host": "strata.example"})              # a page that names another site
        self.assertIn(status, (403, 421))                                                  # the server's own Host check answers first
        self.assertIn("error", body)

    def test_with_an_api_key_the_key_is_the_door(self):
        self.start(DONE)
        self.svc.api_key = "k"
        status, _ = self.get("/agent/folders")
        self.assertEqual(status, 401)
        status, _ = self.get("/agent/folders", {"Authorization": "Bearer k"})
        self.assertEqual(status, 200)


if __name__ == "__main__":
    unittest.main()
