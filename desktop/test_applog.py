"""desktop/test_applog.py - Strata Desktop's log (#310 part 3 task 4): every token the app holds is blanked as `***`
in messages, arguments and tracebacks, unhandled exceptions of any thread land there scrubbed, and the runner's lines
go there too.

    python -m pytest desktop/test_applog.py -q
"""
from __future__ import annotations

import io
import json
import logging
import logging.handlers
import sys
import tempfile
import threading
import unittest
from pathlib import Path

from desktop import applog

TOKEN = "sk-strata-abc"


class LogBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "sub" / "desktop.log"
        self.log = applog.setup(self.path)
        self._hooks = (sys.excepthook, threading.excepthook, sys.stdout, sys.stderr)

    def tearDown(self):
        sys.excepthook, threading.excepthook, sys.stdout, sys.stderr = self._hooks
        applog.close()
        applog.forget_secrets()
        self._tmp.cleanup()

    def text(self) -> str:
        for h in logging.getLogger().handlers:
            h.flush()
        return self.path.read_text(encoding="utf-8")


class Scrub(LogBase):
    def test_log_scrubs_token(self):
        applog.add_secret(TOKEN)
        self.log.info("the key is " + TOKEN)
        self.log.info("the key is %s, once more %r", TOKEN, TOKEN)
        try:
            raise RuntimeError(f"refused {TOKEN}")
        except RuntimeError:
            self.log.exception("a failure")
        logging.getLogger("pywebview").error("from a library: %s", TOKEN)
        text = self.text()
        self.assertNotIn(TOKEN, text)
        self.assertEqual(text.count("***"), 5)
        self.assertIn("RuntimeError: refused ***", text)
        self.assertIn("from a library: ***", text)

    def test_escaped_forms_are_scrubbed(self):
        token = 'sk-"q\\x'
        applog.add_secret(token)
        self.log.info("raw %s json %s twice %s repr %r", token, json.dumps(token), json.dumps(json.dumps(token)), token)
        text = self.text()
        self.assertNotIn(token, text)
        self.assertNotIn(json.dumps(token)[1:-1], text)
        self.assertNotIn(json.dumps(json.dumps(token))[1:-1], text)
        self.assertNotIn(repr(token)[1:-1], text)

    def test_the_inject_script_form_is_scrubbed(self):
        from desktop import inject
        token = 'sk-<b>&"q\\x'
        applog.add_secret(token)
        values, remove = inject.storage_values(token, None, {"id": "AbCdEfGhIjKlMnOpQrStUv"}, "Strata Desktop on x")
        script = inject.inject_script(values, remove)
        self.assertIn(inject.js_literal(json.dumps(token))[1:-1], script)      # the form the script carries
        self.log.info("script %s literal %s", script, inject.js_literal(token))
        text = self.text()
        for form in (token, inject.js_literal(token)[1:-1], inject.js_literal(json.dumps(token))[1:-1], "<b>", "u003cb"):
            self.assertNotIn(form, text)
        self.assertIn("strata.device", text)                                    # the rest of the line is kept

    def test_empty_and_odd_secrets_are_ignored(self):
        applog.add_secret("")
        applog.add_secret(None)
        self.log.info("plain words")
        self.assertIn("plain words", self.text())
        self.assertNotIn("*", self.text())

    def test_longest_secret_first(self):
        applog.add_secret("abc")
        applog.add_secret("abcdef")
        self.log.info("x abcdef y")
        text = self.text()
        self.assertIn("x *** y", text)
        self.assertNotIn("def", text)

    def test_excepthook_scrubs(self):
        applog.add_secret(TOKEN)
        applog.install_hooks()

        def boom():
            local = "-".join(["LOCAL", "VALUE"])      # a local's value: never in the log (no locals are printed)
            assert local
            raise ValueError("the token " + TOKEN + " did not work")

        t = threading.Thread(target=boom, name="t-boom")
        t.start()
        t.join()
        try:
            raise KeyError(TOKEN)
        except KeyError:
            sys.excepthook(*sys.exc_info())
        text = self.text()
        self.assertNotIn(TOKEN, text)
        self.assertIn("ValueError: the token *** did not work", text)
        self.assertIn("t-boom", text)
        self.assertIn("KeyError: '***'", text)
        self.assertNotIn("LOCAL-VALUE", text)

    def test_system_exit_in_a_thread_is_quiet(self):
        applog.install_hooks()
        t = threading.Thread(target=lambda: sys.exit(3))
        t.start()
        t.join()
        self.assertNotIn("SystemExit", self.text() if self.path.exists() else "")


class Lines(LogBase):
    def test_runner_say_goes_to_log(self):
        import serve.runner as runner
        applog.add_secret(TOKEN)
        old = runner._say
        runner._say = applog.say
        try:
            runner._say("[runner] connected with " + TOKEN)
        finally:
            runner._say = old
        text = self.text()
        self.assertIn("[runner] connected with ***", text)
        self.assertNotIn(TOKEN, text)

    def test_stdout_and_stderr_go_to_log(self):
        applog.add_secret(TOKEN)
        applog.capture_stdio()
        print("[strata] MCP server x started with", TOKEN)
        sys.stderr.write("half a line ")
        sys.stderr.write("and the rest\n")
        print("no newline yet", end="")
        sys.stdout.flush()
        text = self.text()
        self.assertIn("[strata] MCP server x started with ***", text)
        self.assertIn("half a line and the rest", text)
        self.assertIn("no newline yet", text)
        self.assertNotIn(TOKEN, text)

    def test_rotation_is_one_mb_three_files(self):
        handlers = [h for h in logging.getLogger().handlers if isinstance(h, logging.handlers.RotatingFileHandler)]
        self.assertEqual(len(handlers), 1)
        self.assertEqual(handlers[0].maxBytes, 1 << 20)
        self.assertEqual(handlers[0].backupCount, 3)

    def test_setup_twice_keeps_one_handler(self):
        applog.setup(self.path)
        ours = [h for h in logging.getLogger().handlers if isinstance(h, logging.handlers.RotatingFileHandler)]
        self.assertEqual(len(ours), 1)
        self.log.info("once")
        self.assertEqual(self.text().count("once"), 1)


def test_scrub_without_setup():
    applog.add_secret(TOKEN)
    try:
        assert applog.scrub("a " + TOKEN + " b") == "a *** b"
        stream = io.StringIO()
        h = logging.StreamHandler(stream)
        h.setFormatter(applog.ScrubFormatter("%(message)s"))
        lg = logging.getLogger("strata.desktop.test-scrub")
        lg.propagate = False
        lg.addHandler(h)
        lg.warning("x %s", TOKEN)
        lg.removeHandler(h)
        assert stream.getvalue() == "x ***\n"
    finally:
        applog.forget_secrets()
