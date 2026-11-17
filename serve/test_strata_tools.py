"""Strata's own tools for the chat (serve/strata_tools.py, issue #339): the docs are searched and read from a fixed list of pages, the settings are
read without a secret, and every change asks the user - in every mode, YOLO included, never settled by auto mode's judge - is refused for a team
guest, changes nothing when the user says no, and writes exactly what the Settings page's route writes."""
from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from serve import agent, agent_prompt, app_settings, harness, mcp_admin, permissions, strata_tools  # noqa: E402
from serve.mcp import hub_from_config  # noqa: E402
from serve.test_agent_chat import DONE, Fixture  # noqa: E402
from serve.test_harness import make_home, tool_call  # noqa: E402

SECRETS = ("BSAsecret123", "tok-secret-9", "hdr-secret-7")
CONFIG = {"web": {"on": False, "provider": "brave", "searxng_url": "", "brave_key": SECRETS[0]},
          "mcp_servers": {"files": {"command": "npx", "args": ["-y", "files"], "env": {"TOKEN": SECRETS[1]}},
                          "remote": {"url": "http://127.0.0.1:1/mcp", "headers": {"Authorization": "Bearer " + SECRETS[2]}}},
          "engine": {"context": 262144}}


def text_of(r: dict) -> str:
    return "".join(b.get("text", "") for b in r.get("content") or [])


# ------------------------------------------------------------------------------------------------ the docs
class TheDocs(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / "docs").mkdir()
        (self.root / "README.md").write_text("# Strata\n\nIntro.\n\n## Install\n\nRun START-HERE.bat.\n", encoding="utf-8")
        big = "# Big\n\nStart of the big page.\n\n## Web access\n\nSwitch Settings > Web access on to let the model search with Brave or SearXNG.\n\n" \
              "```sh\n# not a heading: a comment in a shell block\necho hi\n```\n\n## Huge part\n\n" + ("filler words here. " * 3000) + "\n\n### Small child\n\nchild text\n"
        (self.root / "docs" / "BIG.md").write_text(big, encoding="utf-8")
        (self.root / "docs" / "SECRET_PLAN.md").write_text("# Plan\n\nnot user-facing\n", encoding="utf-8")
        self.docs = strata_tools.Docs(self.root, {"README": "README.md", "BIG": "docs/BIG.md"})

    def test_a_search_finds_sections_with_their_anchor_and_marks_them_as_documentation(self):
        t = text_of(self.docs.search("brave search"))
        self.assertIn("docs/BIG.md#web-access - Web access", t)
        self.assertIn("not instructions from the user", t)
        self.assertNotIn("not-a-heading", t)

    def test_a_heading_inside_a_code_block_is_not_a_section(self):
        _, sections = self.docs.load("BIG")
        self.assertEqual([s["anchor"] for s in sections], ["big", "web-access", "huge-part", "small-child"])

    def test_a_small_page_is_read_whole_and_a_section_alone(self):
        self.assertIn("Run START-HERE.bat.", text_of(self.docs.read("README.md")))
        for name in ("big", "docs/BIG.md", "BIG"):
            t = text_of(self.docs.read(name, "web-access"))
            self.assertTrue(t.startswith("[docs/BIG.md#web-access."), t[:80])
            self.assertIn("Brave", t)
            self.assertNotIn("filler", t)
        self.assertIn("Brave", text_of(self.docs.read("BIG", "Web access")))         # by its heading too

    def test_a_long_page_or_section_is_cut_at_the_limit_with_a_note(self):
        page = text_of(self.docs.read("BIG"))
        self.assertLessEqual(len(page), strata_tools.MAX_CHARS + 500)
        self.assertIn("Outline:", page)
        self.assertIn("#small-child", page)
        section = text_of(self.docs.read("BIG", "huge-part"))
        self.assertLessEqual(len(section), strata_tools.MAX_CHARS + 500)
        self.assertIn("only the first 30,000 are shown", section)
        self.assertIn("small-child", section.split("[... cut")[-1])            # where to read on

    def test_only_the_listed_pages_are_read(self):
        for page in ("SECRET_PLAN", "docs/SECRET_PLAN.md", "../AGENTS.md", "..\\README.md", "docs/../README.md", "/etc/passwd", "C:\\x.md", 7, None):
            r = self.docs.read(page)
            self.assertTrue(r["isError"], page)
            self.assertIn("there is no page", text_of(r))
        self.assertTrue(self.docs.read("BIG", "no-such")["isError"])

    def test_a_changed_file_is_read_again(self):
        self.assertIn("Run START", text_of(self.docs.read("README")))
        (self.root / "README.md").write_text("# Strata\n\nNew text, longer than before.\n", encoding="utf-8")
        self.assertIn("New text", text_of(self.docs.read("README")))

    def test_the_real_allow_list_is_user_facing_and_exists(self):
        for name, rel in strata_tools.DOCS.items():
            self.assertTrue((ROOT / rel).is_file(), rel)
            self.assertNotRegex(rel, r"reports/|plans/|adr/|superpowers/")
        self.assertIn("BLUEPRINT", strata_tools.DOCS)
        self.assertIn("TEAM_WORKSTATION", strata_tools.DOCS)


# ------------------------------------------------------------------------------------------------ the gate
class TheGate(unittest.TestCase):
    def test_a_change_asks_in_every_mode_and_no_judge_settles_it(self):
        for mode in (None, "auto", "yolo", "nonsense"):
            d = permissions.decide("StrataSettingsChange", {"setting": "web.on", "value": True}, permissions.Policy(cwd=None, mode=mode, allow=["StrataSettingsChange", "*"]))
            self.assertEqual((d.kind, d.judgeable, d.danger), ("ask", False, False), mode)
        self.assertEqual(permissions.decide("StrataSettingsChange", {}, permissions.Policy(cwd=None, mode="plan")).kind, "deny")
        self.assertEqual(permissions.decide("StrataSettingsChange", {}, permissions.Policy(cwd=None, mode="yolo", deny=["StrataSettingsChange"])).kind, "deny")
        self.assertIsNone(permissions.rule_for("StrataSettingsChange", {"file_path": "/x/y"}))

    def test_reading_asks_nothing(self):
        for mode in (None, "auto", "yolo", "plan"):
            for tool in ("StrataDocs", "StrataSettings"):
                self.assertEqual(permissions.decide(tool, {"query": "x"}, permissions.Policy(cwd=None, mode=mode)).kind, "allow", (tool, mode))


# ------------------------------------------------------------------------------------------------ the tools, with a stand-in for the server
class Svc:
    """What the tools use of serve/server.py's Service."""

    def __init__(self, config_path):
        self.config_path, self.mcp_config_path, self.shared, self.agent, self.mcp, self.api_key = config_path, None, {}, None, None, None
        paths = make_home(Path(config_path).parent)
        self.importer = harness.Importer(paths["home"], paths["appdata"], environ={})
        self.importer.rescan({})
        self.strata = None

    def set_shared(self, defaults):
        self.shared = app_settings.clean_shared_defaults(defaults)
        return self.shared


class ToolsBase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        self.config = self.base / "run.json"
        self.config.write_text(json.dumps(CONFIG), encoding="utf-8")
        self.svc = Svc(str(self.config))
        self.server = strata_tools.StrataServer(self.svc)
        self.addCleanup(lambda: self.svc.mcp and self.svc.mcp.close())
        self.cards = []
        self.answer = "allow"

    def ctx(self, mode=None, guest=False):
        def ask(card):
            self.cards.append(card)
            return self.answer
        c = agent.AgentContext(policy=permissions.Policy(cwd=None, mode=mode), session="s", ask=ask)
        c.guest = guest
        return c

    def change(self, setting, value, pc="central", **kw):
        return self.server.call("StrataSettingsChange", {"pc": pc, "setting": setting, "value": value}, ctx=self.ctx(**kw))


class TheTools(ToolsBase):
    CASES = [("web.on", True), ("web.provider", "searxng"), ("web.searxng_url", "http://localhost:8080"), ("helpers.on", True),
             ("skills.enabled", False), ("skills.app", {"app": "codex", "on": False}), ("chat.temperature", 0.6), ("chat.reasoning_effort", "low"),
             ("mcp.add", {"name": "new", "server": {"url": "http://127.0.0.1:1/x", "disabled": True}}),
             ("mcp.edit", {"name": "files", "server": {"command": "npx", "args": ["-y", "files2"], "env": {"TOKEN": mcp_admin.MASK}}}),
             ("mcp.switch", {"name": "files", "on": False}), ("mcp.delete", {"name": "remote"}),
             ("skills.skill", {"app": "codex", "name": "codex-only", "on": False}), ("mcp.import", {"app": "codex", "name": "off-here"})]

    def test_every_change_asks_in_ask_auto_and_yolo_with_a_before_and_after(self):
        for setting, value in self.CASES:
            for mode in (None, "auto", "yolo"):
                self.config.write_text(json.dumps(CONFIG), encoding="utf-8")
                self.svc.shared = {}
                self.cards.clear()
                r = self.change(setting, value, mode=mode)
                self.assertFalse(r["isError"], (setting, mode, text_of(r)))
                self.assertEqual(len(self.cards), 1, (setting, mode))
                card = self.cards[0]
                self.assertEqual((card["tool"], card["judgeable"], card["change"]["setting"]), ("StrataSettingsChange", False, setting))
                self.assertIn("before", card["change"])
                self.assertNotEqual(card["change"]["before"], card["change"]["after"], setting)
                for s in SECRETS:
                    self.assertNotIn(s, json.dumps(card), setting)

    def test_a_no_changes_nothing(self):
        before = self.config.read_bytes()
        self.answer = "deny"
        for setting, value in self.CASES:
            r = self.change(setting, value, mode="yolo")
            self.assertTrue(r["isError"], setting)
            self.assertIn("did not allow", text_of(r))
        self.assertEqual(self.config.read_bytes(), before)
        self.assertEqual(self.svc.shared, {})
        self.assertFalse((self.base / "run.json.bak-mcp").exists())

    def test_a_guest_can_read_the_docs_but_neither_read_nor_change_the_settings(self):
        before = self.config.read_bytes()
        for setting, value in self.CASES:
            r = self.change(setting, value, guest=True)
            self.assertTrue(r["isError"])
            self.assertIn("team guest", text_of(r))
        r = self.server.call("StrataSettings", {}, ctx=self.ctx(guest=True))
        self.assertTrue(r["isError"])
        self.assertEqual(self.cards, [])
        self.assertEqual(self.config.read_bytes(), before)
        self.assertFalse(self.server.call("StrataDocs", {"query": "install"}, ctx=self.ctx(guest=True))["isError"])

    def test_without_a_chat_only_the_docs_answer(self):
        self.assertTrue(self.server.call("StrataSettings", {})["isError"])
        self.assertTrue(self.server.call("StrataSettingsChange", {"setting": "web.on", "value": True})["isError"])
        self.assertFalse(self.server.call("StrataDocs", {"page": "README"})["isError"])

    def test_the_settings_never_hold_a_secret(self):
        t = text_of(self.server.call("StrataSettings", {}, ctx=self.ctx()))
        for s in SECRETS:
            self.assertNotIn(s, t)
        got = json.loads(t)
        self.assertEqual(set(got), {"central_pc", "my_pc"})
        self.assertEqual(got["my_pc"]["connected"], False)
        got = got["central_pc"]
        self.assertEqual(got["web"]["brave_key"], "set")
        self.assertEqual(got["mcp_servers"][0]["env"], {"TOKEN": mcp_admin.MASK})
        self.assertNotIn("engine", got)

    def test_an_unknown_setting_is_refused_with_the_list_and_nothing_is_asked(self):
        for setting in ("engine.context", "api_key", "hooks", "web.brave_key", "guest_mcp", "allowed_hosts", None):
            r = self.change(setting, 1)
            self.assertTrue(r["isError"])
            self.assertIn("helpers.on", text_of(r))
            self.assertIn("mcp.import", text_of(r))
        self.assertIn("secret", text_of(self.change("web.brave_key", "x")))
        for setting, value in (("web.on", "yes"), ("chat.temperature", 9), ("mcp.add", {"name": "files", "server": {"command": "x"}}),
                               ("mcp.delete", {"name": "nope"}), ("skills.app", {"app": "nope", "on": True}), ("mcp.add", {"name": "bad name", "server": {"command": "x"}})):
            self.assertTrue(self.change(setting, value)["isError"], setting)
        self.assertEqual(self.cards, [])

    def test_a_change_that_changes_nothing_is_refused_without_a_card(self):
        before = self.config.read_bytes()
        for setting, value in (("web.on", False), ("web.provider", "brave"), ("mcp.switch", {"name": "files", "on": True}), ("helpers.on", False),
                               ("mcp.edit", {"name": "files", "server": mcp_admin.public_entry(CONFIG["mcp_servers"]["files"])})):
            r = self.change(setting, value)
            self.assertTrue(r["isError"], setting)
            self.assertIn("nothing to change", text_of(r))
        self.assertEqual(self.cards, [])
        self.assertEqual(self.config.read_bytes(), before)

    def test_a_built_in_groups_name_is_never_given_to_a_server(self):
        self.assertIn("built-in", text_of(self.change("mcp.add", {"name": "strata", "server": {"command": "x"}})))
        found = {"claude": {"servers": {"strata": {"command": "python", "args": ["tools/strata_mcp.py"]}}}}
        own, name = harness.import_server({}, "claude", "strata", found, {})
        self.assertEqual(name, "strata-claude")                                  # Strata's own MCP server, imported from Claude Code
        code, out = mcp_admin.apply(self.svc, {"servers": {**CONFIG["mcp_servers"], "skills": {"command": "x"}}})
        self.assertEqual(code, 400)
        self.assertEqual(self.cards, [])

    def test_an_mcp_change_says_its_tools_come_with_the_next_message(self):
        r = self.change("mcp.switch", {"name": "files", "on": False})
        self.assertIn("next message", text_of(r))

    def test_a_secret_kept_by_its_mask_and_the_card_shows_the_mask(self):
        self.change("mcp.edit", self.CASES[9][1])
        stored = json.loads(self.config.read_text(encoding="utf-8"))["mcp_servers"]["files"]
        self.assertEqual(stored["env"], {"TOKEN": SECRETS[1]})
        self.assertEqual(stored["args"], ["-y", "files2"])
        self.assertEqual(self.cards[0]["change"]["after"]["env"], {"TOKEN": strata_tools.STORED})

    def test_a_change_made_while_the_user_decides_stops_this_one(self):
        def ask(card):
            mcp_admin.update_config(str(self.config), lambda cfg: cfg.__setitem__("web", {**cfg["web"], "provider": "searxng"}))
            return "allow"
        c = agent.AgentContext(policy=permissions.Policy(cwd=None), session="s", ask=ask)
        r = self.server.call("StrataSettingsChange", {"pc": "central", "setting": "web.provider", "value": "searxng"}, ctx=c)
        self.assertTrue(r["isError"])
        self.assertIn("changed while", text_of(r))

    def test_plan_mode_and_a_helper_change_nothing(self):
        self.assertIn("plan mode", text_of(self.change("web.on", True, mode="plan")))
        c = self.ctx()
        c.in_helper = True
        self.assertTrue(self.server.call("StrataSettingsChange", {"pc": "central", "setting": "web.on", "value": True}, ctx=c)["isError"])
        self.assertEqual(self.cards, [])


class TheCard(ToolsBase):
    """The review of 38920f6f: the card shows what the model sends, in full, and names what it changes."""

    def entries(self):
        return {(k["field"], k["key"]): k for k in self.cards[0]["change"]["keys"]}

    def test_an_env_value_the_model_changes_is_shown_in_full_and_marked_changed(self):
        self.change("mcp.edit", {"name": "files", "server": {"command": "npx", "args": ["-y", "files"],
                                                              "env": {"TOKEN": "--require C:/x/evil.js"}}})
        card = self.cards[0]
        keys = self.entries()
        self.assertEqual(keys[("env", "TOKEN")]["state"], "changed")
        self.assertEqual(keys[("env", "TOKEN")]["value"], "--require C:/x/evil.js")
        self.assertIn("--require C:/x/evil.js", json.dumps(card["change"]["after"]))
        self.assertIn("--require C:/x/evil.js", json.dumps(card["arguments"]))          # the model's own words are not masked
        self.assertNotIn(SECRETS[1], json.dumps(card))                                     # the stored one never shows
        self.assertEqual(card["change"]["target"], "files")

    def test_a_value_kept_from_storage_is_masked_and_said_to_be_unchanged_and_a_new_key_is_new(self):
        self.change("mcp.edit", {"name": "files", "server": {"command": "npx", "args": ["-y", "files"],
                                                              "env": {"TOKEN": mcp_admin.MASK, "NODE_OPTIONS": "--require x.js"}}})
        keys = self.entries()
        self.assertEqual((keys[("env", "TOKEN")]["state"], keys[("env", "TOKEN")]["value"]), ("unchanged", "******** (unchanged, stored)"))
        self.assertEqual((keys[("env", "NODE_OPTIONS")]["state"], keys[("env", "NODE_OPTIONS")]["value"]), ("new", "--require x.js"))
        self.assertNotIn(SECRETS[1], json.dumps(self.cards[0]))

    def test_a_key_dropped_is_marked_removed(self):
        self.change("mcp.edit", {"name": "remote", "server": {"url": "http://127.0.0.1:1/mcp"}})
        self.assertEqual(self.entries()[("headers", "Authorization")]["state"], "removed")
        self.assertNotIn(SECRETS[2], json.dumps(self.cards[0]))

    def test_an_added_server_shows_its_values(self):
        self.change("mcp.add", {"name": "pw", "server": {"command": "npx", "env": {"NODE_OPTIONS": "--require evil.js"}, "disabled": True}})
        self.assertEqual(self.entries()[("env", "NODE_OPTIONS")], {"field": "env", "key": "NODE_OPTIONS", "state": "new", "value": "--require evil.js"})
        self.assertEqual(self.cards[0]["change"]["target"], "pw")

    def test_the_card_names_its_target(self):
        for setting, value, target in (("mcp.switch", {"name": "files", "on": False}, "files"), ("mcp.delete", {"name": "remote"}, "remote"),
                                       ("skills.app", {"app": "codex", "on": False}, "codex"),
                                       ("skills.skill", {"app": "codex", "name": "codex-only", "on": False}, "codex / codex-only"),
                                       ("mcp.import", {"app": "codex", "name": "off-here"}, "off-here (from codex)")):
            self.config.write_text(json.dumps(CONFIG), encoding="utf-8")
            self.cards.clear()
            self.change(setting, value)
            self.assertEqual(self.cards[0]["change"]["target"], target, setting)

    def test_my_is_kept_for_the_users_pc(self):
        for name in ("my_files", "MY_x", "My_y"):
            r = self.change("mcp.add", {"name": name, "server": {"command": "x"}})
            self.assertTrue(r["isError"])
            self.assertIn("my_", text_of(r))
            code, out = mcp_admin.apply(self.svc, {"servers": {**CONFIG["mcp_servers"], name: {"command": "x", "disabled": True}}})
            self.assertEqual(code, 400, name)
            self.assertIn("my_", out["error"]["message"])
        found = {"claude": {"servers": {"my_tools": {"command": "python", "args": ["t.py"]}}}}
        _, final = harness.import_server({}, "claude", "my_tools", found, {})
        self.assertFalse(final.lower().startswith("my_"), final)
        self.assertEqual(self.cards, [])

    def test_the_settings_mask_secrets_in_args_and_addresses(self):
        cfg = {"mcp_servers": {"a": {"command": "npx", "args": ["srv", "--api-key", "plain-secret-1", "--token=plain-secret-2", "key=plain-secret-3",
                                                                "ghp_" + "A" * 30, "--port", "8080"]},
                               "b": {"url": "https://mcp.example.com/mcp?apikey=plain-secret-4&x=1"}}}
        self.config.write_text(json.dumps(cfg), encoding="utf-8")
        t = text_of(self.server.call("StrataSettings", {}, ctx=self.ctx()))
        for secret in ("plain-secret-1", "plain-secret-2", "plain-secret-3", "plain-secret-4", "ghp_" + "A" * 30):
            self.assertNotIn(secret, t)
        self.assertIn("8080", t)                                                       # a plain value stays
        self.assertIn("https://mcp.example.com/mcp?", t)

    def test_the_check_after_the_answer_sees_a_stored_secret_that_changed(self):
        def ask(card):
            mcp_admin.update_config(str(self.config), lambda cfg: cfg["mcp_servers"]["files"].__setitem__("env", {"TOKEN": "other-secret"}))
            return "allow"
        c = agent.AgentContext(policy=permissions.Policy(cwd=None), session="s", ask=ask)
        r = self.server.call("StrataSettingsChange", {"pc": "central", "setting": "mcp.edit", "value": {"name": "files", "server": {
            "command": "npx", "args": ["-y", "files2"], "env": {"TOKEN": mcp_admin.MASK}}}}, ctx=c)
        self.assertIn("changed while", text_of(r))

    def test_a_helper_keeps_the_guest_mark(self):
        import inspect
        from serve import subagent
        self.assertIn("helper.ctx.guest", inspect.getsource(subagent.execute))


# ------------------------------------------------------------------------------------------------ the same as the routes
class SameAsTheRoute(Fixture):
    def start(self, *scripts, **kw):
        super().start(*scripts, **kw)
        self.config = self.base / "run.json"
        self.svc.config_path = str(self.config)
        paths = make_home(self.base)
        self.svc.importer = harness.Importer(paths["home"], paths["appdata"], environ={})
        self.svc.importer.rescan({})
        self.server = strata_tools.StrataServer(self.svc)
        self.svc.strata = self.server

    def reset(self, shared=None):
        self.config.write_text(json.dumps(CONFIG, indent=2), encoding="utf-8")
        bak = Path(str(self.config) + ".bak-mcp")
        if bak.exists():
            bak.unlink()
        self.svc.shared = dict(shared or {})

    def route(self, path, body):
        with urllib.request.urlopen(self.post(path, body), timeout=10) as r:
            return r.status

    def tool(self, setting, value):
        c = agent.AgentContext(policy=permissions.Policy(cwd=None, mode="yolo"), session="s", ask=lambda card: "allow")
        r = self.server.call("StrataSettingsChange", {"pc": "central", "setting": setting, "value": value}, ctx=c)
        self.assertFalse(r["isError"], text_of(r))

    def test_each_change_writes_what_the_route_writes(self):
        self.start(DONE)
        files = mcp_admin.public_entry(CONFIG["mcp_servers"]["files"])
        remote = mcp_admin.public_entry(CONFIG["mcp_servers"]["remote"])
        cases = [
            ("/agent/web", {"on": True}, "web.on", True),
            ("/agent/web", {"provider": "searxng"}, "web.provider", "searxng"),
            ("/agent/helpers", {"on": True}, "helpers.on", True),
            ("/import", {"skills": {**harness.skill_settings(CONFIG), "harness_off": ["codex"]}}, "skills.app", {"app": "codex", "on": False}),
            ("/import", {"skills": {**harness.skill_settings(CONFIG), "enabled": False}}, "skills.enabled", False),
            ("/mcp/config", {"servers": {"files": {**files, "disabled": True}, "remote": remote}}, "mcp.switch", {"name": "files", "on": False}),
            ("/mcp/config", {"servers": {"files": files}}, "mcp.delete", {"name": "remote"}),
            ("/mcp/config", {"servers": {"files": files, "remote": remote, "new": {"url": "http://127.0.0.1:1/x", "disabled": True}}},
             "mcp.add", {"name": "new", "server": {"url": "http://127.0.0.1:1/x", "disabled": True}}),
            ("/mcp/config", {"servers": {"files": {**files, "args": ["-y", "files2"]}, "remote": remote}},
             "mcp.edit", {"name": "files", "server": {**files, "args": ["-y", "files2"]}}),
            ("/import", {"skills": {**harness.skill_settings(CONFIG), "off": {"codex": ["codex-only"]}}}, "skills.skill", {"app": "codex", "name": "codex-only", "on": False}),
            ("/import", {"import_mcp": {"harness": "codex", "name": "off-here"}}, "mcp.import", {"app": "codex", "name": "off-here"}),
        ]
        for path, body, setting, value in cases:
            self.reset()
            self.assertEqual(self.route(path, body), 200, path)
            by_route = self.config.read_bytes()
            self.reset()
            self.tool(setting, value)
            self.assertEqual(self.config.read_bytes(), by_route, setting)
            self.assertNotEqual(by_route, json.dumps(CONFIG, indent=2).encode(), setting)
            if setting.startswith("mcp."):
                self.assertIs(self.svc.mcp.servers.get("strata"), self.server, setting)     # the restarted hub keeps Strata's own group

    def test_a_shared_default_is_the_routes(self):
        self.start(DONE)
        self.reset({"temperature": 0.6})
        self.assertEqual(self.route("/settings", {"defaults": {"temperature": 0.6, "top_k": 20}}), 200)
        by_route = dict(self.svc.shared)
        self.reset({"temperature": 0.6})
        self.tool("chat.top_k", 20)
        self.assertEqual(self.svc.shared, by_route)
        self.tool("chat.top_k", None)                                            # null removes it
        self.assertEqual(self.svc.shared, {"temperature": 0.6})


# ------------------------------------------------------------------------------------------------ in the chat
class InTheChat(Fixture):
    def start(self, *scripts, **kw):
        super().start(*scripts, **kw)
        self.config = self.base / "run.json"
        self.config.write_text(json.dumps({"agents": {"on": False}}), encoding="utf-8")
        self.svc.config_path = str(self.config)
        self.strata = strata_tools.StrataServer(self.svc)
        self.svc.strata = self.strata
        self.hub.close()
        self.hub = hub_from_config({}, None, builtins={"agent": self.agent, "strata": self.strata})
        self.hub.start(wait=True)
        self.svc.mcp = self.hub

    def test_the_tools_are_offered_with_the_coding_tools_and_described_in_the_rules(self):
        self.start(DONE)
        self.chat()
        p = self.engine.prompt_text(0)
        for name in ("StrataDocs", "StrataSettings", "StrataSettingsChange", "# Strata itself", "# Where things run"):
            self.assertIn(name, p)

    def test_a_change_in_yolo_still_asks_on_its_call_and_a_no_keeps_the_config(self):
        self.start(tool_call("StrataSettingsChange", pc="central", setting="helpers.on", value="true"), DONE)
        before = self.config.read_bytes()
        cards = []
        status, _, events = self.chat(self.body(agent={"mode": "yolo"}), on_event=lambda e: e["event"] == "permission" and (cards.append(e), self.answer(e["id"], "deny")))
        self.assertEqual(status, 200)
        self.assertEqual(len(cards), 1)
        call = [e for e in events if e["event"] == "call"][0]
        self.assertEqual(cards[0]["call_id"], call["id"])                               # the card lands on its call
        self.assertEqual(cards[0]["change"], {"setting": "helpers.on", "target": None, "description": strata_tools.SETTINGS["helpers.on"], "before": False, "after": True,
                                              "machine": "server", "machine_name": None})
        self.assertEqual(self.config.read_bytes(), before)

    def test_allowed_it_is_written(self):
        self.start(tool_call("StrataSettingsChange", pc="central", setting="helpers.on", value="true"), DONE)
        status, _, events = self.chat(on_event=lambda e: e["event"] == "permission" and self.answer(e["id"], "allow"))
        self.assertEqual(status, 200)
        self.assertTrue([e for e in events if e["event"] == "result"][0]["ok"])
        self.assertEqual(json.loads(self.config.read_text(encoding="utf-8"))["agents"], {"on": True})

    def test_a_guest_run_hides_the_settings_tools(self):
        self.start(DONE)
        run = self.svc.start_agent_run({"cwd": str(self.proj), "session": "chat-1"}, [{"role": "user", "content": "hi"}], "bob", guest=True)
        self.assertTrue(run.ctx.guest)
        self.assertTrue({"StrataSettings", "StrataSettingsChange"} <= run.hidden)
        self.assertIn("StrataDocs", run.prompt)
        self.assertNotIn("StrataSettingsChange", run.prompt)
        own = self.svc.start_agent_run({"cwd": str(self.proj), "session": "chat-1"}, [{"role": "user", "content": "hi"}], "owner")
        self.assertFalse(own.ctx.guest)
        self.assertIn("StrataSettingsChange", own.prompt)

    def test_a_chat_without_the_coding_tools_is_offered_the_docs_only(self):
        self.start(DONE)
        body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True, "strata_mcp": True}
        self.chat(body)
        p = self.engine.prompt_text(0)
        self.assertIn("StrataDocs", p)
        self.assertNotIn("StrataSettingsChange", p)
        self.assertNotIn("StrataSettings", p.replace("StrataSettingsChange", ""))

    def test_a_guest_may_use_the_docs_group(self):
        self.start(DONE)
        self.assertIn("strata", mcp_admin.guest_allowed({}, self.hub))


# ------------------------------------------------------------------------------------------------ where things run
class WhereThingsRun(unittest.TestCase):
    def build(self, **kw):
        return agent_prompt.build("C:/p", "bash", None, "2026-11-17", "win32", False, None, ["Read", "Bash", "WebFetch", "StrataDocs"], **kw)

    def test_a_page_on_the_central_pc_itself_is_the_users_pc(self):
        p = self.build(same_pc=True)
        self.assertIn("working on the user's own computer, in a chat", p.split("\n")[0])
        self.assertIn("the central PC is the user's PC", p)
        self.assertIn('"localhost" in their calls is this machine', p)
        self.assertNotIn("not on the user's PC", p)

    def test_a_server_project_does_not_claim_the_users_pc(self):
        p = self.build()
        self.assertNotIn("the user's own computer", p.split("\n")[0])
        self.assertIn("run on the central PC (the server), not on the user's PC", p)
        self.assertIn("WebFetch, Strata's own tools always run on the central PC", p)
        self.assertIn("none is offered in this chat", p)
        self.assertNotIn("runs on the user's PC", p)

    def test_a_my_pc_project_names_the_machine_and_the_my_rule(self):
        p = self.build(machine="ann-pc", my_tools=True)
        self.assertIn("user's own computer (ann-pc)", p.split("\n")[0])
        self.assertIn("run on the user's PC (ann-pc)", p)
        self.assertIn("starts with my_", p)
        self.assertIn("Playwright", p)
        self.assertNotIn("starts with my_", self.build(machine="ann-pc", my_tools=False))


if __name__ == "__main__":
    unittest.main()
