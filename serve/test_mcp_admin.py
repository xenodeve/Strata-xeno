"""serve/test_mcp_admin.py - setting up MCP servers from the web app (issue #79): who may change them, what is checked, what is
written to the run config, and that no secret leaves the server.

    python -m unittest serve.test_mcp_admin -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import mcp_admin  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FAKE = str(ROOT / "serve" / "mcp_fake_server.py")
SECRET = "s3cret-token-value"


class Validation(unittest.TestCase):
    def test_a_program_and_an_address_are_both_fine(self):
        a, err = mcp_admin.validate_entry("files", {"command": "npx", "args": ["-y", "pkg"], "env": {"A": "1"}, "cwd": "C:/x"})
        self.assertEqual(err, [])
        self.assertEqual(a["command"], "npx")
        b, err = mcp_admin.validate_entry("web", {"url": "http://127.0.0.1:3000/mcp", "headers": {"Authorization": "Bearer x"}})
        self.assertEqual(err, [])
        self.assertEqual(b["url"], "http://127.0.0.1:3000/mcp")

    def test_a_bad_entry_is_refused_with_the_name_of_the_field(self):
        def fields(name, entry):
            return [e["field"] for e in mcp_admin.validate_entry(name, entry)[1]]
        self.assertIn("name", fields("two words", {"command": "x"}))
        self.assertIn("name", fields("", {"command": "x"}))
        self.assertEqual(sorted(fields("a", {"command": "x", "url": "http://h/mcp"})), ["command", "url"])      # one or the other
        self.assertIn("command", fields("a", {}))
        self.assertIn("command", fields("a", {"command": "  "}))
        self.assertIn("url", fields("a", {"url": "ftp://h/mcp"}))
        self.assertIn("url", fields("a", {"url": "http://h/mcp", "type": "sse"}))
        self.assertIn("args", fields("a", {"command": "x", "args": "not a list"}))
        self.assertIn("args", fields("a", {"command": "x", "args": ["a"] * 65}))
        self.assertIn("env", fields("a", {"command": "x", "env": {"A": 1}}))
        self.assertIn("headers", fields("a", {"url": "http://h/mcp", "headers": {"A": ["x"]}}))
        self.assertIn("command", fields("a", {"command": "x\ny"}))

    def test_the_limits_have_ranges(self):
        ok, err = mcp_admin.validate_settings({"timeout_s": 30, "max_result_chars": 5000, "max_rounds": 4})
        self.assertEqual((err, ok), ([], {"timeout_s": 30.0, "max_result_chars": 5000, "max_rounds": 4}))
        for bad in ({"timeout_s": 0}, {"timeout_s": 9999}, {"max_rounds": 0}, {"max_rounds": 99}, {"max_result_chars": 10}, {"max_rounds": 1.5},
                    {"timeout_s": "x"}, {"unknown": 1}):
            with self.subTest(bad=bad):
                self.assertTrue(mcp_admin.validate_settings(bad)[1], bad)


class Masking(unittest.TestCase):
    def test_secrets_show_as_a_mask_and_the_mask_keeps_the_stored_value(self):
        stored = {"command": "x", "env": {"TOKEN": SECRET, "OTHER": "o"}, "headers": None}
        shown = mcp_admin.public_entry(stored)
        self.assertEqual(shown["env"], {"TOKEN": mcp_admin.MASK, "OTHER": mcp_admin.MASK})
        self.assertNotIn(SECRET, json.dumps(shown))
        merged = mcp_admin.merge_masked({"command": "x", "env": {"TOKEN": mcp_admin.MASK, "OTHER": "changed", "NEW": "n"}}, stored)
        self.assertEqual(merged["env"], {"TOKEN": SECRET, "OTHER": "changed", "NEW": "n"})
        h = mcp_admin.merge_masked({"url": "http://h/mcp", "headers": {"Authorization": mcp_admin.MASK}}, {"url": "http://h/mcp", "headers": {"Authorization": "Bearer q"}})
        self.assertEqual(h["headers"], {"Authorization": "Bearer q"})
        gone = mcp_admin.merge_masked({"command": "x", "env": {"A": mcp_admin.MASK}}, {"command": "x"})       # a mask with nothing behind it
        self.assertEqual(gone["env"], {})


class Who(unittest.TestCase):
    def test_a_key_or_this_pc_itself(self):
        may = mcp_admin.may_edit
        self.assertEqual(may(True, "10.0.0.7", "pc.example.com")[0], True)            # with a key the key is the gate
        self.assertEqual(may(False, "127.0.0.1", "127.0.0.1:8091")[0], True)
        self.assertEqual(may(False, "::1", "localhost:8091")[0], True)
        self.assertEqual(may(False, "127.0.0.1", "[::1]:8091")[0], True)
        for client, host in (("10.0.0.7", "10.0.0.2:8091"), ("10.0.0.7", "localhost"), ("127.0.0.1", "10.0.0.2:8091"), ("127.0.0.1", "my-pc"),
                             ("127.0.0.1", "attacker.com")):
            with self.subTest(client=client, host=host):
                ok, why = may(False, client, host)
                self.assertFalse(ok)
                self.assertIn("API key", why)


class WriteBack(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "strata-test.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_other_keys_stay_a_backup_is_kept_once_and_the_key_in_use_is_kept(self):
        original = {"model": "m", "args": ["--x"], "mcpServers": {"old": {"command": "a"}}, "mcp": {"timeout_s": 10}}
        self.path.write_text(json.dumps(original), encoding="utf-8")
        mcp_admin.write_config(self.path, {"new": {"command": "b"}}, {"max_rounds": 3})
        now = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(now["model"], "m")
        self.assertEqual(now["args"], ["--x"])
        self.assertEqual(now["mcpServers"], {"new": {"command": "b"}})          # the file used mcpServers: that key is the one written
        self.assertNotIn("mcp_servers", now)
        self.assertEqual(now["mcp"], {"timeout_s": 10, "max_rounds": 3})        # a limit not sent is kept
        bak = Path(str(self.path) + ".bak-mcp")
        self.assertEqual(json.loads(bak.read_text(encoding="utf-8")), original)
        mcp_admin.write_config(self.path, {}, {})
        self.assertEqual(json.loads(bak.read_text(encoding="utf-8")), original)   # the first backup is not overwritten

    def test_mcp_servers_is_the_key_when_the_file_has_neither(self):
        self.path.write_text(json.dumps({"model": "m"}), encoding="utf-8")
        mcp_admin.write_config(self.path, {"a": {"command": "b"}}, {})
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8"))["mcp_servers"], {"a": {"command": "b"}})

    def test_a_failed_write_leaves_the_file_as_it_was(self):
        self.path.write_text(json.dumps({"model": "m"}), encoding="utf-8")
        before = self.path.read_text(encoding="utf-8")
        with self.assertRaises(TypeError):
            mcp_admin.write_config(self.path, {"a": {"command": object()}}, {})          # not JSON: the dump fails
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)
        self.assertEqual([p.name for p in Path(self.tmp.name).iterdir() if p.suffix == ".tmp"], [])


class Endpoints(unittest.TestCase):
    """The HTTP seam: a served mock with a run config file; the fake MCP server is the one that starts."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = Path(self.tmp.name) / "strata-test.json"
        self.cfg.write_text(json.dumps({"model": "m", "mcp_servers": {
            "fake": {"command": sys.executable, "args": [FAKE], "env": {"TOKEN": SECRET}},
            "off": {"command": "nothing", "disabled": True}}}), encoding="utf-8")
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "ok", max_context=8192), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.config_path = str(self.cfg)
        self.svc.mcp_config_path = None
        mcp_admin.reload_hub(self.svc)
        self.svc.mcp.wait(30)
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        if self.svc.mcp:
            self.svc.mcp.close()
        self.tmp.cleanup()

    def call(self, path, method="GET", body=None, headers=None):
        data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
        h = {"Content-Type": "application/json"} if data is not None else {}
        h.update(headers or {})
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=h)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def js(self, path, method="GET", body=None, headers=None):
        code, raw = self.call(path, method, body, headers)
        return code, json.loads(raw)

    def test_the_servers_are_listed_with_their_state_and_no_secret(self):
        code, raw = self.call("/mcp/config")
        self.assertEqual(code, 200)
        self.assertNotIn(SECRET.encode(), raw)
        d = json.loads(raw)
        self.assertTrue(d["editable"])
        by = {s["name"]: s for s in d["servers"]}
        self.assertEqual(set(by), {"fake", "off"})
        self.assertEqual(by["fake"]["kind"], "program")
        self.assertEqual(by["fake"]["env"], {"TOKEN": mcp_admin.MASK})
        self.assertEqual(by["fake"]["status"], "ready")
        self.assertGreater(len(by["fake"]["tools"]), 0)
        self.assertTrue(by["off"]["disabled"])
        self.assertEqual(by["fake"]["source"], "config")
        self.assertIn("timeout_s", d["settings"])

    def test_an_edit_is_written_and_takes_effect_without_a_restart(self):
        servers = {"fake": {"command": sys.executable, "args": [FAKE], "env": {"TOKEN": mcp_admin.MASK}},
                   "second": {"command": sys.executable, "args": [FAKE]}}
        code, d = self.js("/mcp/config", "POST", {"servers": servers, "settings": {"max_rounds": 3}}, {"Origin": self.base})
        self.assertEqual(code, 200, d)
        self.assertEqual({s["name"] for s in d["servers"]}, {"fake", "second"})                  # "off" was not sent: it is deleted
        self.svc.mcp.wait(30)
        code, status = self.js("/mcp")
        self.assertEqual({s["name"] for s in status["servers"]}, {"fake", "second"})
        written = json.loads(self.cfg.read_text(encoding="utf-8"))
        self.assertEqual(written["mcp_servers"]["fake"]["env"], {"TOKEN": SECRET})               # the mask kept the stored value
        self.assertEqual(written["mcp"]["max_rounds"], 3)
        self.assertEqual(written["model"], "m")
        self.assertEqual(self.svc.mcp.settings["max_rounds"], 3)

    def test_a_changed_secret_replaces_the_stored_one(self):
        servers = {"fake": {"command": sys.executable, "args": [FAKE], "env": {"TOKEN": "new-value"}}}
        code, d = self.js("/mcp/config", "POST", {"servers": servers}, {"Origin": self.base})
        self.assertEqual(code, 200, d)
        self.assertEqual(json.loads(self.cfg.read_text(encoding="utf-8"))["mcp_servers"]["fake"]["env"], {"TOKEN": "new-value"})

    def test_bad_input_is_refused_with_the_field_and_nothing_is_written(self):
        before = self.cfg.read_text(encoding="utf-8")
        code, d = self.js("/mcp/config", "POST", {"servers": {"a b": {"command": "x"}}}, {"Origin": self.base})
        self.assertEqual(code, 400)
        self.assertEqual(d["error"]["fields"][0]["field"], "name")
        code, d = self.js("/mcp/config", "POST", {"servers": {}, "settings": {"max_rounds": 0}}, {"Origin": self.base})
        self.assertEqual(code, 400)
        self.assertEqual(self.cfg.read_text(encoding="utf-8"), before)

    def test_an_edit_needs_json_and_this_pc_s_own_page(self):
        body = {"servers": {}}
        self.assertEqual(self.call("/mcp/config", "POST", b"{}", {"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.call("/mcp/config", "POST", body, {"Origin": "http://evil.example"})[0], 403)

    def test_from_another_address_it_is_read_only_unless_there_is_a_key(self):
        host = {"Host": "10.0.0.5:8091"}                     # an IP address passes the Host check; it is not this PC
        code, d = self.js("/mcp/config", "GET", None, host)
        self.assertEqual(code, 200)
        self.assertFalse(d["editable"])
        self.assertIn("API key", d["reason"])
        self.assertEqual(self.call("/mcp/config", "POST", {"servers": {}}, host)[0], 403)
        self.svc.api_key = "k"
        ok = {**host, "Authorization": "Bearer k"}
        self.assertTrue(self.js("/mcp/config", "GET", None, ok)[1]["editable"])
        self.assertEqual(self.call("/mcp/config", "POST", {"servers": {"fake": {"command": sys.executable, "args": [FAKE]}}}, ok)[0], 200)
        self.assertEqual(self.call("/mcp/config", "POST", {"servers": {}}, host)[0], 401)       # a key is set: none sent

    def test_without_a_config_file_it_can_be_read_and_not_changed(self):
        self.svc.config_path = None
        code, d = self.js("/mcp/config")
        self.assertEqual(code, 200)
        self.assertFalse(d["editable"])
        self.assertIn("run config", d["reason"])
        self.assertEqual(self.call("/mcp/config", "POST", {"servers": {}}, {"Origin": self.base})[0], 409)


class FromTheMcpConfigFile(unittest.TestCase):
    def test_servers_that_come_from_the_mcp_config_file_are_shown_and_not_edited(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, other = Path(d) / "run.json", Path(d) / "desktop.json"
            cfg.write_text(json.dumps({"mcp_servers": {"mine": {"command": "a"}}}), encoding="utf-8")
            other.write_text(json.dumps({"mcpServers": {"theirs": {"command": "b"}, "mine": {"command": "c"}}}), encoding="utf-8")
            view = mcp_admin.read_view(str(cfg), str(other))
            by = {s["name"]: s for s in view["servers"]}
            self.assertEqual(by["theirs"]["source"], "file")
            self.assertFalse(by["theirs"]["editable"])
            self.assertEqual(by["mine"]["source"], "file")                    # the file wins a name, as the hub does
            self.assertTrue(by["mine"]["shadowed"])
            mine = [s for s in view["servers"] if s["name"] == "mine"]
            self.assertEqual(sorted(s["source"] for s in mine), ["config", "file"])   # the run config's entry stays listed: a save keeps it
            self.assertTrue(next(s for s in mine if s["source"] == "config")["overridden"])


if __name__ == "__main__":
    unittest.main()
