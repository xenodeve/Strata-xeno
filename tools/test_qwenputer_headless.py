"""The launcher must pass a batch-file path with spaces intact to cmd.exe."""
import importlib.util
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import time
import unittest

SPEC = importlib.util.spec_from_file_location(
    "qwenputer_headless", Path(__file__).parent / "qwenputer" / "headless.py")
headless = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(headless)


@unittest.skipUnless(os.name == "nt", "Windows batch environment")
class HeadlessTest(unittest.TestCase):
    def test_timeout_finishes_owned_grandchild_before_done(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            marker, ready, log = root / "orphan.txt", root / "ready.txt", root / "run.log"
            grandchild = root / "grandchild.py"
            grandchild.write_text("import time,pathlib,sys\ntime.sleep(2)\npathlib.Path(sys.argv[1]).write_text('survived')\n")
            parent = root / "parent.py"
            parent.write_text("import subprocess,sys,time,pathlib\n"
                              "subprocess.Popen([sys.executable,sys.argv[1],sys.argv[2]],creationflags=0x08000000)\n"
                              "pathlib.Path(sys.argv[3]).write_text('started')\ntime.sleep(5)\n")
            proc = subprocess.run([sys.executable, str(Path(headless.__file__)), "--log", str(log),
                                   "--timeout", "1", "--", sys.executable, str(parent),
                                   str(grandchild), str(marker), str(ready)],
                                  capture_output=True, text=True, timeout=15, creationflags=0x08000000)
            self.assertTrue(ready.is_file(), "fixture must actually start the grandchild")
            self.assertEqual(proc.returncode, 124, proc.stderr)
            self.assertIn("DONE exit=124", log.read_text())
            time.sleep(2)
            self.assertFalse(marker.exists(), "DONE was emitted while a task-owned grandchild survived")

    def test_vcvars_path_with_spaces_sets_child_environment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "tool chain"
            source.mkdir()
            batch = source / "vcvars test.bat"
            batch.write_text("@echo off\nset QWENPUTER_TEST_ENV=ready\n", encoding="ascii")
            env = headless.msvc_environment(batch, root / "setup.cmd")
            self.assertEqual(env.get("QWENPUTER_TEST_ENV"), "ready")
            self.assertNotIn("QWENPUTER_TEST_ENV", os.environ)


if __name__ == "__main__":
    unittest.main()
