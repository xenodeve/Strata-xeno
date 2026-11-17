"""Strata's own tools on the user's own PC (issue #344): StrataSettings shows each PC apart (central_pc, my_pc), and StrataSettingsChange
with pc "my_pc" changes the MCP servers and the skill switches of the chat's own PC through its runner's jobs - the whole list sent back as
the web app sends it, the secrets kept there, a new server still waiting for the user's approval in Strata Desktop.

    python -m pytest serve/test_strata_my_pc.py -q
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from serve import agent, mcp_admin, permissions, runner as runner_mod, strata_tools  # noqa: E402
from serve.test_strata_tools import CONFIG, Svc, text_of  # noqa: E402

SECRET = "tok-on-my-pc-5"
OFF = {"command": "never-started", "args": ["-x"], "env": {"TOKEN": SECRET}, "disabled": True}


class FakeRunner:
    """serve/runner_hub.py Runner as Strata's tools use it: its machine, online(), and request() answered by a real Executor."""

    def __init__(self, ex, machine="TOP_PC"):
        self.ex, self.info, self.up, self.device = ex, {"machine": machine}, True, "dev-1"
        self.groups, self.skills, self.jobs = [], [], []
        self.unknown = set()

    def online(self):
        return self.up

    def request(self, kind, payload, timeout=None):
        self.jobs.append(kind)
        if kind in self.unknown:                        # an older Strata Desktop
            return {"error": "unknown", "message": f"this runner does not know the job {kind!r}: update Strata Desktop"}
        return self.ex.handle({"kind": kind, **payload}, None)


class TheUsersPc(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        (self.base / "central").mkdir()
        self.config = self.base / "central" / "run.json"
        self.config.write_text(json.dumps(CONFIG), encoding="utf-8")
        self.svc = Svc(str(self.config))
        self.server = strata_tools.StrataServer(self.svc)
        for target, value in ((runner_mod, "_say"),):
            p = mock.patch.object(target, value, lambda text: None)
            p.start()
            self.addCleanup(p.stop)
        home = self.base / "home"
        home.mkdir()
        p = mock.patch.dict("os.environ", {"APPDATA": str(home / "AppData")})
        p.start()
        self.addCleanup(p.stop)
        self.runner_json = self.base / "runner.json"
        self.runner_json.write_text(json.dumps({"mcp_servers": {"old": OFF}}), encoding="utf-8")
        self.ex = runner_mod.Executor(home, self.runner_json, self.base)
        self.addCleanup(self.ex.close)
        self.mine = FakeRunner(self.ex)
        self.cards, self.answer, self.during = [], "allow", None

    def ctx(self, mine=True, why=""):
        def ask(card):
            self.cards.append(card)
            if self.during:
                self.during()
            return self.answer
        c = agent.AgentContext(policy=permissions.Policy(cwd=None), session="s", ask=ask)
        c.my_pc, c.my_pc_why = (self.mine if mine else None), why
        return c

    def change(self, setting, value, pc="my_pc", **kw):
        return self.server.call("StrataSettingsChange", {"pc": pc, "setting": setting, "value": value}, ctx=self.ctx(**kw))

    def runner_cfg(self):
        return json.loads(self.runner_json.read_text(encoding="utf-8"))

    def settings(self, **kw):
        r = self.server.call("StrataSettings", {}, ctx=self.ctx(**kw))
        self.assertFalse(r["isError"], text_of(r))
        return json.loads(text_of(r))

    # ---------------------------------------------------------------------------------------- reading
    def test_the_settings_are_apart_by_pc_and_name_where_each_runs(self):
        got = self.settings()
        self.assertIn("central PC", got["central_pc"]["about"])
        self.assertEqual([s["name"] for s in got["central_pc"]["mcp_servers"]], ["files", "remote"])
        mine = got["my_pc"]
        self.assertEqual((mine["connected"], mine["machine"]), (True, "TOP_PC"))
        self.assertIn("my_<server>__<tool>", mine["about"])
        self.assertIn("approves it in Strata Desktop", mine["about"])
        [old] = mine["mcp_servers"]
        self.assertEqual((old["name"], old["disabled"], old["env"]), ("old", True, {"TOKEN": mcp_admin.MASK}))
        self.assertIn("skills", mine)
        self.assertNotIn(SECRET, json.dumps(got))
        self.assertNotIn("files", json.dumps(mine))                       # the central PC's servers are never listed as the user's

    def test_no_pc_connected_says_why(self):
        got = self.settings(mine=False)
        self.assertEqual(got["my_pc"]["connected"], False)
        self.assertIn("Strata Desktop", got["my_pc"]["note"])
        why = "this message was sent from another browser"
        self.assertEqual(self.settings(mine=False, why=why)["my_pc"]["note"], why)

    def test_an_offline_pc_is_shown_as_it_last_connected(self):
        self.mine.up = False
        self.mine.groups = [{"name": "old", "kind": "mcp"}, {"name": "agent", "kind": "coding"}]
        got = self.settings()["my_pc"]
        self.assertEqual((got["online"], got["mcp_servers"]), (False, ["old"]))
        self.assertEqual(self.mine.jobs, [])
        r = self.change("mcp.delete", {"name": "old"})
        self.assertTrue(r["isError"])
        self.assertIn("offline", text_of(r))
        self.assertEqual(self.cards, [])

    # ---------------------------------------------------------------------------------------- changing MCP servers
    def test_an_added_server_waits_for_approval_and_the_others_and_the_waiting_ones_survive(self):
        code = self.ex.handle({"kind": "mcp.config.set", "body": {"servers": {"old": OFF, "wait": {"command": "w"}}}}, None)["status"]
        self.assertEqual(code, 200)
        r = self.change("mcp.add", {"name": "playwright", "server": {"command": "cmd", "args": ["/c", "npx", "-y", "@playwright/mcp@latest"]}})
        self.assertFalse(r["isError"], text_of(r))
        self.assertIn("approve it in Strata Desktop", text_of(r))
        self.assertIn("TOP_PC", text_of(r))
        [card] = self.cards
        self.assertEqual((card["change"]["machine"], card["change"]["machine_name"], card["change"]["target"]), ("my", "TOP_PC", "playwright"))
        self.assertEqual(card["change"]["after"]["args"], ["/c", "npx", "-y", "@playwright/mcp@latest"])
        cfg = self.runner_cfg()
        self.assertEqual(cfg["mcp_servers"], {"old": OFF})                         # the approved one, its secret kept
        self.assertEqual(set(cfg["mcp_pending"]), {"wait", "playwright"})          # nothing new runs from the chat's word alone
        self.assertEqual(json.loads(self.config.read_text(encoding="utf-8")), CONFIG)     # the central PC's config is untouched

    def test_an_env_value_the_model_sends_is_shown_in_full_and_never_unchanged(self):
        self.change("mcp.edit", {"name": "old", "server": {"command": "never-started", "args": ["-x"], "disabled": True,
                                                            "env": {"TOKEN": "--require C:/x/evil.js", "NODE_OPTIONS": "--require y.js"}}})
        keys = {k["key"]: k for k in self.cards[0]["change"]["keys"]}
        self.assertEqual((keys["TOKEN"]["state"], keys["TOKEN"]["value"]), ("changed", "--require C:/x/evil.js"))
        self.assertEqual((keys["NODE_OPTIONS"]["state"], keys["NODE_OPTIONS"]["value"]), ("new", "--require y.js"))
        self.assertNotIn(SECRET, json.dumps(self.cards))

    def test_a_mask_keeps_the_secret_stored_on_that_pc(self):
        r = self.change("mcp.edit", {"name": "old", "server": {"command": "never-started", "args": ["-y"], "env": {"TOKEN": mcp_admin.MASK}, "disabled": True}})
        self.assertFalse(r["isError"], text_of(r))
        self.assertEqual(self.cards[0]["change"]["after"]["env"], {"TOKEN": strata_tools.STORED})
        self.assertEqual(self.runner_cfg()["mcp_servers"]["old"], {**OFF, "args": ["-y"]})

    def test_switching_off_and_deleting_are_done_at_once(self):
        self.runner_json.write_text(json.dumps({"mcp_servers": {"old": OFF, "b": {"command": "b", "disabled": True}}}), encoding="utf-8")
        self.ex.cfg = self.ex._read_config()
        r = self.change("mcp.delete", {"name": "b"})
        self.assertFalse(r["isError"], text_of(r))
        self.assertIn("gone from the user's next message", text_of(r))
        self.assertEqual(self.runner_cfg()["mcp_servers"], {"old": OFF})

    def test_a_no_changes_nothing_on_that_pc(self):
        before = self.runner_json.read_bytes()
        self.answer = "deny"
        r = self.change("mcp.add", {"name": "x", "server": {"command": "x"}})
        self.assertTrue(r["isError"])
        self.assertEqual(self.runner_json.read_bytes(), before)

    def test_a_change_made_on_that_pc_while_the_user_decides_stops_this_one(self):
        def meanwhile():
            self.ex.handle({"kind": "mcp.config.set", "body": {"servers": {"old": {**OFF, "args": ["-z"]}}}}, None)
        self.during = meanwhile
        r = self.change("mcp.switch", {"name": "old", "on": True})
        self.assertTrue(r["isError"])
        self.assertIn("changed while", text_of(r))

    def test_reserved_and_existing_names_are_refused_without_a_card(self):
        for name in ("agent", "skills", "strata", "my_x", "old"):
            self.assertTrue(self.change("mcp.add", {"name": name, "server": {"command": "x"}})["isError"], name)
        self.assertIn("TOP_PC", text_of(self.change("mcp.delete", {"name": "files"})))      # the central PC's server is not the user's
        self.assertEqual(self.cards, [])

    # ---------------------------------------------------------------------------------------- the PC is said every time
    def test_the_pc_is_said_every_time_and_server_settings_stay_central(self):
        r = self.server.call("StrataSettingsChange", {"setting": "web.on", "value": True}, ctx=self.ctx())
        self.assertIn('pc: "central"', text_of(r))
        for setting, value in (("web.on", True), ("helpers.on", True), ("chat.temperature", 0.5)):
            r = self.change(setting, value)
            self.assertTrue(r["isError"])
            self.assertIn("central PC only", text_of(r))
        r = self.change("mcp.add", {"name": "x", "server": {"command": "x"}}, mine=False)
        self.assertIn("No PC of the user's is connected", text_of(r))
        self.assertEqual(self.cards, [])

    def test_an_older_strata_desktop_says_to_install_the_new_one(self):
        self.mine.unknown = {"skills.config", "skills.config.set"}
        got = self.settings()["my_pc"]
        self.assertIn("installs the new Strata Desktop", got["skills_error"])
        r = self.change("skills.enabled", False)
        self.assertIn("installs the new Strata Desktop", text_of(r))

    # ---------------------------------------------------------------------------------------- skills
    def test_a_skill_switch_is_written_on_that_pc(self):
        r = self.change("skills.enabled", False)
        self.assertFalse(r["isError"], text_of(r))
        self.assertEqual(self.cards[0]["change"]["machine"], "my")
        self.assertEqual(self.runner_cfg()["import"]["skills"], {"enabled": False, "harness_off": [], "off": {}})
        self.assertNotIn("import", json.loads(self.config.read_text(encoding="utf-8")))
        self.assertEqual(self.ex.handle({"kind": "skills.config"}, None)["body"]["settings"]["enabled"], False)

    def test_the_skills_job_refuses_a_bad_body(self):
        for body in (None, {}, {"skills": {"nope": 1}}, {"skills": {"enabled": False}, "x": 1}):
            self.assertEqual(self.ex.handle({"kind": "skills.config.set", "body": body}, None)["status"], 400, body)


if __name__ == "__main__":
    unittest.main()
