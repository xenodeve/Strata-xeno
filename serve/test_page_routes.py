"""serve/test_page_routes.py - the nine web-app routes that act on a project's machine (#310 part 2, task 5), as one
pure function the server calls for the workstation and the runner calls for My PC: the same answers as the modules
give directly, the same 400 / 404 texts, and the check of a runner's answer before it reaches the page.

    python -m pytest serve/test_page_routes.py -q
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from serve import checkpoints, files, folders, gitview, memory, page_routes  # noqa: E402
from serve.test_gitview import HAS_GIT, git  # noqa: E402


class Answers(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name).resolve()
        self.repo = self.base / "app"
        self.repo.mkdir()
        (self.repo / "a.txt").write_text("one\ntwo\n", encoding="utf-8")
        (self.repo / "CLAUDE.md").write_bytes(b"# rules\nbe kind\n")
        (self.repo / "sub").mkdir()
        self.home = self.base / "home"
        (self.home / ".claude").mkdir(parents=True)
        (self.home / ".claude" / "CLAUDE.md").write_text("user rules\n", encoding="utf-8")
        self.cp = checkpoints.Checkpoints(self.base / "cp")

    def answer(self, path, query=None, body=None, on=()):
        return page_routes.answer(path, query or {}, body, home=str(self.home), memory_on=list(on), checkpoints=self.cp)

    @unittest.skipUnless(HAS_GIT, "needs git")
    def test_answers_equal_direct_calls(self):
        d = str(self.repo)
        git(d, "init", "-q", "-b", "main")
        git(d, "add", ".")
        git(d, "commit", "-q", "-m", "first")
        (self.repo / "a.txt").write_text("one\nTWO\n", encoding="utf-8")
        self.assertEqual(self.answer("/agent/git", {"path": [d]}), (200, gitview.info(d)))
        self.assertEqual(self.answer("/agent/git/diff", {"path": [d], "file": ["a.txt"]}), (200, gitview.diff(d, "a.txt", False, False)))
        self.assertEqual(self.answer("/agent/git/diff", {"path": [d], "file": ["a.txt"], "staged": ["1"], "untracked": ["1"]}),
                         (200, gitview.diff(d, "a.txt", True, True)))
        self.assertEqual(self.answer("/agent/folders", {"path": [d]}), (200, folders.look(d)))
        self.assertEqual(self.answer("/agent/folders", {"path": [str(self.repo / "a.txt")]}),
                         (200, {"ok": False, "error": "That is not a folder on this PC"}))
        self.assertEqual(self.answer("/agent/files", {"path": [d], "q": ["a"]}), (200, {"files": files.find([d], "a")}))
        self.assertEqual(self.answer("/agent/mention", {"path": [d], "rel": ["a.txt"]}), (200, files.read([d], "a.txt")))
        on = ["claude:instructions"]
        sources = memory.discover(str(self.home), [d], on)
        listing = {"max": memory.MAX_TOTAL, "sources": [{k: s[k] for k in ("id", "app", "label", "kind", "shown", "bytes", "on")} for s in sources]}
        self.assertEqual(self.answer("/agent/memory", {"path": [d]}, on=on), (200, listing))
        self.assertEqual(len(listing["sources"]), 2)
        self.assertEqual(self.answer("/agent/memory/file", {"path": [d], "id": ["project:0"]}, on=on),
                         (200, {"ok": True, "name": "CLAUDE.md", "text": "# rules\nbe kind\n", "cut": False}))
        self.assertEqual(self.answer("/agent/memory/file", {"path": [d], "id": ["project:0"], "n": ["1"]}),
                         (404, {"ok": False, "error": "no such file"}))
        self.assertEqual(self.answer("/agent/memory/file", {"path": [d], "id": ["project:0"], "n": ["x"]}),
                         (404, {"ok": False, "error": "no such file"}))
        # the way back: what checkpoints.preview / apply / forget give
        f = str(self.repo / "a.txt")
        self.cp.before("s1", "c1", f)
        Path(f).write_text("changed\n", encoding="utf-8")
        self.cp.after("s1", "c1", f)
        self.assertEqual(self.answer("/agent/rewind", body={"session": "s1", "checkpoint": "c1"}), (200, self.cp.preview("s1", "c1")))
        self.assertEqual(self.answer("/agent/rewind", body={"session": "s1", "checkpoint": "c1"})[1]["files"][0]["action"], "restore")
        self.assertEqual(self.answer("/agent/checkpoints/forget", body={"session": "s1"}), (200, {"ok": True}))
        self.assertEqual(self.cp.preview("s1", "c1"), {"ok": True, "files": [], "known": False})

    def test_apply_puts_files_back(self):
        f = str(self.repo / "a.txt")
        self.cp.before("s1", "c1", f)
        Path(f).write_text("changed\n", encoding="utf-8")
        self.cp.after("s1", "c1", f)
        code, out = self.answer("/agent/rewind", body={"session": "s1", "checkpoint": "c1", "apply": True})
        self.assertEqual((code, out["restored"]), (200, 1))
        self.assertEqual(Path(f).read_text(encoding="utf-8"), "one\ntwo\n")

    def test_folders_and_files_from_several_folders(self):
        other = self.base / "other"
        other.mkdir()
        (other / "b.txt").write_text("bee\n", encoding="utf-8")
        q = {"path": [str(self.repo)], "dirs": [str(other), " ", "x\0y"], "q": ["b"]}
        self.assertEqual(self.answer("/agent/files", q), (200, {"files": files.find([str(self.repo), str(other)], "b")}))
        self.assertEqual(self.answer("/agent/folders", {}), (200, folders.look("")))

    def test_rewind_validation_texts(self):
        need = (400, {"error": {"message": "send {\"session\": the chat's id, \"checkpoint\": the prompt's id}"}})
        for path in ("/agent/rewind", "/agent/checkpoints/forget"):
            for body in (None, [], {}, {"session": 5}, {"checkpoint": "c1"}):
                self.assertEqual(self.answer(path, body=body), need, (path, body))
        self.assertEqual(self.answer("/agent/rewind", body={"session": "s1"}),
                         (400, {"error": {"message": "send the checkpoint, the prompt's id"}}))
        self.assertEqual(self.answer("/agent/rewind", body={"session": "s1", "checkpoint": 3}),
                         (400, {"error": {"message": "send the checkpoint, the prompt's id"}}))

    def test_other_paths_are_not_answered(self):
        for path in ("/agent", "/agent/run", "/mcp", "/agent/folders/x", ""):
            self.assertEqual(self.answer(path), (404, {"error": {"message": "not found"}}), path)
        self.assertEqual(set(page_routes.PATHS), set(page_routes.GET_PATHS) | set(page_routes.POST_PATHS))
        self.assertEqual(len(page_routes.PATHS), 9)


class Relay(unittest.TestCase):
    """What a runner's answer becomes on the page: Runner.request's errors, and an answer the server checks first."""

    def test_good_answers_pass(self):
        for status in (200, 400, 404):
            self.assertEqual(page_routes.relay({"status": status, "body": {"ok": True}}, "ann-pc"), (status, {"ok": True}))

    def test_offline_and_timeout(self):
        code, body = page_routes.relay({"error": "offline", "message": "your PC (ann-pc) is offline"}, "ann-pc")
        self.assertEqual((code, body), (409, {"error": {"type": "runner_offline", "message": "your PC (ann-pc) is offline"}}))
        code, body = page_routes.relay({"error": "timeout", "message": "your PC (ann-pc) did not answer within 30 s"}, "ann-pc")
        self.assertEqual((code, body), (504, {"error": {"type": "runner_timeout", "message": "your PC (ann-pc) did not answer within 30 s"}}))
        code, body = page_routes.relay({"error": "offline", "message": "x" * 1000}, "ann-pc")
        self.assertEqual(len(body["error"]["message"]), 300)

    def test_unreadable_answers_are_502(self):
        unreadable = "your PC's runner (ann-pc) sent an answer the server could not read"
        for res in (None, [], "x", {}, {"status": 200}, {"status": 200, "body": []}, {"status": 200, "body": "x"},
                    {"status": "200", "body": {}}, {"status": True, "body": {}}, {"status": 200.0, "body": {}},
                    {"status": 401, "body": {}}, {"status": 403, "body": {}}, {"status": 500, "body": {}},
                    {"status": 302, "body": {}}, {"error": "other", "message": "m"}):
            self.assertEqual(page_routes.relay(res, "ann-pc"),
                             (502, {"error": {"type": "runner_answer", "message": unreadable}}), res)

    def test_answer_size_is_capped(self):
        at = page_routes.MAX_ANSWER - len(json.dumps({"text": ""}, ensure_ascii=False))
        self.assertEqual(page_routes.relay({"status": 200, "body": {"text": "x" * at}}, "m")[0], 200)
        code, body = page_routes.relay({"status": 200, "body": {"text": "x" * (at + 1)}}, "m")
        self.assertEqual(code, 502)
        self.assertEqual(body["error"]["type"], "runner_answer")
        self.assertIn("too large", body["error"]["message"])
        # the cap is above every answer the routes give on their own: the largest text (a note's file) at JSON's
        # worst escaping (6 characters for one), and a diff or a mentioned file the same way
        self.assertGreaterEqual(page_routes.MAX_ANSWER, 6 * max(page_routes.MEMORY_FILE_MAX, gitview.MAX_DIFF, files.MAX_TEXT))


if __name__ == "__main__":
    unittest.main()
