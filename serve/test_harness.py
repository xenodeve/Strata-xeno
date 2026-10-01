"""serve/test_harness.py - skills and MCP servers of the other coding apps on this PC (issue #94): what discovery finds in a home folder,
how an entry is read, what the skill switches select, and how a server is imported. Every test reads a FAKE home folder built here,
never the real one.

    python -m unittest serve.test_harness -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import harness  # noqa: E402

SECRET = "tok-s3cret-planted"


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def skill(root: Path, name: str, description: str = "does a thing", extra: str = "", body: str = "# Body\nsteps") -> Path:
    d = root / name
    write(d / "SKILL.md", f"---\nname: {name}\ndescription: {description}\n{extra}---\n\n{body}\n")
    return d


def make_home(root: Path) -> dict:
    """A home folder shaped like the real ones. Returns the paths the tests need."""
    home, appdata = root / "home", root / "appdata"
    out = {"home": home, "appdata": appdata}
    # Claude Code: user skills, one plugin that is enabled and one that is not, MCP servers
    skill(home / ".claude/skills", "pdf-tools", "Read and write PDF files")
    skill(home / ".claude/skills", "shared-skill", "in two apps")
    plug_on, plug_off = home / ".claude/plugins/cache/on", home / ".claude/plugins/cache/off"
    skill(plug_on / "skills", "plugin-skill", "from an enabled plugin")
    skill(plug_off / "skills", "disabled-plugin-skill", "from a plugin that is not enabled")
    write(home / ".claude/plugins/installed_plugins.json", json.dumps({"version": 2, "plugins": {
        "on@market": [{"scope": "user", "installPath": str(plug_on)}], "off@market": [{"scope": "user", "installPath": str(plug_off)}]}}))
    write(home / ".claude/settings.json", json.dumps({"enabledPlugins": {"on@market": True, "off@market": False}}))
    write(home / ".claude.json", json.dumps({"mcpServers": {
        "files": {"type": "stdio", "command": "npx", "args": ["-y", "@x/server-files", "C:/work"], "env": {"TOKEN": SECRET}},
        "web": {"type": "http", "url": "https://mcp.example.com/mcp", "headers": {"Authorization": "Bearer " + SECRET}},
        "legacy": {"type": "sse", "url": "https://old.example.com/sse"},
        "bad name!": {"command": "node", "args": ["a.js"]},
    }, "projects": {"C:/repo": {"mcpServers": {"per-repo": {"command": "x"}}}}}))
    # Codex: skills, and TOML servers (one disabled there)
    skill(home / ".codex/skills", "codex-only", "only in Codex")
    skill(home / ".codex/skills", "shared-skill", "in two apps")
    write(home / ".codex/config.toml", f'''model = "x"

[mcp_servers.files]
command = "npx"
args = ["-y", "@x/server-files", "C:/work"]
enabled = true
tools = {{ a = 1 }}

[mcp_servers.files.env]
TOKEN = "{SECRET}"

[mcp_servers.off-here]
command = "node"
args = ["off.js"]
enabled = false

[mcp_servers.remote]
url = "https://remote.example.com/mcp"
bearer_token_env_var = "REMOTE_TOKEN"

[mcp_servers.nothing]
enabled = true
''')
    # the shared agents folder, Antigravity, Cursor, Gemini CLI, Claude Desktop
    skill(home / ".agents/skills", "agents-skill", "in the shared folder")
    skill(home / ".gemini/antigravity/skills", "ag-skill", "in Antigravity")
    skill(home / ".cursor/skills", "cursor-skill", "in Cursor")
    write(home / ".cursor/mcp.json", json.dumps({"mcpServers": {"cur": {"command": "python", "args": ["-m", "x"]}}}))
    write(home / ".gemini/settings.json", json.dumps({"mcpServers": {
        "g-stdio": {"command": "node", "args": ["g.js"]}, "g-http": {"httpUrl": "https://g.example.com/mcp"}, "g-sse": {"url": "https://g.example.com/sse"}}}))
    write(appdata / "Claude/claude_desktop_config.json", json.dumps({"mcpServers": {"desk": {"command": "uvx", "args": ["thing"]}}}))
    return out


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.paths = make_home(Path(self.tmp.name))
        self.home, self.appdata = self.paths["home"], self.paths["appdata"]

    def tearDown(self):
        self.tmp.cleanup()

    def scan(self):
        return harness.scan(self.home, self.appdata)


class Home(unittest.TestCase):
    def test_the_folder_is_the_one_in_the_environment_or_the_user_s(self):
        self.assertEqual(harness.home_dir({"STRATA_HOME": "C:/x/fake"}), Path("C:/x/fake"))
        self.assertEqual(harness.home_dir({}), Path.home())
        self.assertEqual(harness.appdata_dir(Path("C:/h"), {"STRATA_HOME": "C:/h"}), Path("C:/h/AppData/Roaming"))
        self.assertEqual(harness.appdata_dir(Path("C:/h"), {"APPDATA": "C:/real/roaming"}), Path("C:/real/roaming"))


class Discovery(Fixture):
    def test_finds_what_is_planted_and_nothing_else(self):
        found = self.scan()
        names = {h: sorted(s["name"] for s in v["skills"]) for h, v in found.items()}
        self.assertEqual(names["claude"], ["pdf-tools", "plugin-skill", "shared-skill"])
        self.assertEqual(names["codex"], ["codex-only", "shared-skill"])
        self.assertEqual(names["agents"], ["agents-skill"])
        self.assertEqual(names["antigravity"], ["ag-skill"])
        self.assertEqual(names["cursor"], ["cursor-skill"])
        self.assertEqual(sum(len(v) for v in names.values()), 8)
        servers = {h: sorted(v["servers"]) for h, v in found.items()}
        self.assertEqual(servers["claude"], ["bad name!", "files", "legacy", "web"])             # the user's own, not the per-repo one
        self.assertEqual(servers["claude-desktop"], ["desk"])
        self.assertEqual(servers["codex"], ["files", "nothing", "off-here", "remote"])
        self.assertEqual(servers["gemini"], ["g-http", "g-sse", "g-stdio"])
        self.assertEqual(servers["cursor"], ["cur"])

    def test_an_empty_home_finds_nothing_and_says_so(self):
        with tempfile.TemporaryDirectory() as d:
            found = harness.scan(Path(d) / "h", Path(d) / "a")
            for h, v in found.items():
                self.assertFalse(v["found"], h)
                self.assertEqual((v["skills"], v["servers"], v["errors"]), ([], {}, []), h)

    def test_found_names_the_files_that_were_read(self):
        found = self.scan()
        self.assertTrue(found["codex"]["found"])
        self.assertTrue(any(f.endswith("config.toml") for f in found["codex"]["files"]))
        self.assertTrue(any(f.endswith("skills") for f in found["codex"]["files"]))
        self.assertTrue(all(not os.path.isabs(f) or str(self.home) in f or str(self.appdata) in f for f in found["codex"]["files"]))

    def test_a_skill_carries_its_name_description_and_origin(self):
        by = {s["name"]: s for s in self.scan()["claude"]["skills"]}
        self.assertEqual(by["pdf-tools"]["description"], "Read and write PDF files")
        self.assertEqual(by["pdf-tools"]["id"], "claude:pdf-tools")
        self.assertEqual(by["pdf-tools"]["origin"], "user")
        self.assertEqual(by["plugin-skill"]["origin"], "plugin:on")

    def test_only_the_skills_of_enabled_plugins(self):
        self.assertNotIn("disabled-plugin-skill", {s["name"] for s in self.scan()["claude"]["skills"]})

    def test_a_plugin_with_no_skills_folder_is_not_an_error(self):
        bare = self.home / ".claude/plugins/cache/bare"
        bare.mkdir(parents=True)
        data = json.loads((self.home / ".claude/plugins/installed_plugins.json").read_text(encoding="utf-8"))
        data["plugins"]["bare@market"] = [{"scope": "user", "installPath": str(bare)}]
        write(self.home / ".claude/plugins/installed_plugins.json", json.dumps(data))
        write(self.home / ".claude/settings.json", json.dumps({"enabledPlugins": {"on@market": True, "bare@market": True}}))
        found = self.scan()
        self.assertEqual(found["claude"]["errors"], [])                       # most plugins are not about skills: that is not a problem to report
        self.assertIn("plugin-skill", {x["name"] for x in found["claude"]["skills"]})

    def test_a_broken_file_is_an_error_not_an_exception(self):
        write(self.home / ".codex/config.toml", "this is [not toml")
        write(self.home / ".cursor/mcp.json", "{ nope")
        found = self.scan()
        self.assertTrue(found["codex"]["errors"])
        self.assertTrue(found["cursor"]["errors"])
        self.assertEqual(found["codex"]["servers"], {})
        self.assertEqual(sorted(s["name"] for s in found["codex"]["skills"]), ["codex-only", "shared-skill"])      # its skills still show
        self.assertTrue(found["claude"]["servers"])                                                              # the others too

    def test_a_skill_folder_without_a_skill_file_is_left_out(self):
        (self.home / ".claude/skills/empty-dir").mkdir()
        write(self.home / ".claude/skills/stray.txt", "x")
        self.assertNotIn("empty-dir", {s["name"] for s in self.scan()["claude"]["skills"]})

    def test_a_skill_that_is_the_same_folder_twice_is_one(self):
        # a junction or symlink to a folder that is already listed (the real ~/.claude/skills has 17 of them)
        link = self.home / ".claude/skills/pdf-tools-link"
        try:
            os.symlink(self.home / ".claude/skills/pdf-tools", link, target_is_directory=True)
        except OSError:
            import subprocess
            r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(self.home / ".claude/skills/pdf-tools")], capture_output=True)
            if r.returncode:
                self.skipTest("cannot make a link here")
        names = [s["name"] for s in self.scan()["claude"]["skills"]]
        self.assertEqual(names.count("pdf-tools"), 1)
        self.assertEqual(len(names), 3)


class FrontMatter(unittest.TestCase):
    def parse(self, text):
        return harness.parse_front(text)

    def test_plain_quoted_and_folded_values(self):
        self.assertEqual(self.parse("---\nname: a\ndescription: plain words here\n---\nbody"), {"name": "a", "description": "plain words here"})
        self.assertEqual(self.parse('---\nname: "b"\ndescription: \'single: colon\'\n---\n')["description"], "single: colon")
        folded = self.parse("---\nname: c\ndescription: >\n  first line\n  second line\nlicense: MIT\n---\n")
        self.assertEqual(folded["description"], "first line second line")
        literal = self.parse("---\nname: d\ndescription: |-\n  keeps\n  lines\n---\n")
        self.assertEqual(literal["description"], "keeps\nlines")
        multi = self.parse("---\nname: e\ndescription: starts here\n  and goes on\nother: 1\n---\n")
        self.assertEqual(multi["description"], "starts here and goes on")

    def test_no_front_matter_or_a_broken_one(self):
        self.assertEqual(self.parse("# Just a title\ntext"), {})
        self.assertEqual(self.parse("---\nname: x\nno end"), {})
        self.assertEqual(self.parse("\ufeff---\nname: bom\n---\n"), {"name": "bom"})

    def test_extra_keys_do_not_break_it(self):
        got = self.parse("---\nname: f\ndescription: d\nmetadata:\n  version: 1\nallowed-tools: Bash\n---\n")
        self.assertEqual((got["name"], got["description"]), ("f", "d"))


class Switches(Fixture):
    def test_everything_is_on_with_no_settings(self):
        sel = harness.select_skills(self.scan(), harness.skill_settings({}))
        self.assertEqual(sorted(s["name"] for s in sel), ["ag-skill", "agents-skill", "codex-only", "cursor-skill", "pdf-tools", "plugin-skill", "shared-skill"])

    def test_the_same_skill_in_two_apps_is_one_and_the_first_app_wins(self):
        sel = harness.select_skills(self.scan(), harness.skill_settings({}))
        shared = [s for s in sel if s["name"] == "shared-skill"]
        self.assertEqual([s["harness"] for s in shared], ["claude"])
        view = {s["id"]: s for s in harness.skill_items(self.scan(), harness.skill_settings({}))}
        self.assertEqual(view["codex:shared-skill"]["same_as"], "claude:shared-skill")
        self.assertFalse(view["codex:shared-skill"]["used"])

    def test_the_three_levels_of_off(self):
        found = self.scan()
        names = lambda cfg: sorted(s["name"] for s in harness.select_skills(found, harness.skill_settings(cfg)))
        self.assertEqual(names({"import": {"skills": {"enabled": False}}}), [])
        self.assertNotIn("codex-only", names({"import": {"skills": {"harness_off": ["codex"]}}}))
        self.assertIn("pdf-tools", names({"import": {"skills": {"harness_off": ["codex"]}}}))
        self.assertNotIn("pdf-tools", names({"import": {"skills": {"off": {"claude": ["pdf-tools"]}}}}))
        self.assertIn("plugin-skill", names({"import": {"skills": {"off": {"claude": ["pdf-tools"]}}}}))

    def test_when_the_first_app_s_copy_is_off_the_next_one_is_used(self):
        got = harness.select_skills(self.scan(), harness.skill_settings({"import": {"skills": {"off": {"claude": ["shared-skill"]}}}}))
        self.assertEqual([s["harness"] for s in got if s["name"] == "shared-skill"], ["codex"])

    def test_a_skill_that_turns_up_later_is_on_and_a_switched_off_one_stays_off(self):
        cfg = {"import": {"skills": {"off": {"claude": ["pdf-tools"]}}}}
        skill(self.home / ".claude/skills", "brand-new")
        names = sorted(s["name"] for s in harness.select_skills(self.scan(), harness.skill_settings(cfg)))
        self.assertIn("brand-new", names)
        self.assertNotIn("pdf-tools", names)

    def test_settings_are_read_defensively(self):
        for bad in (None, 5, "x", [], {"import": 3}, {"import": {"skills": "no"}}, {"import": {"skills": {"harness_off": "codex", "off": [1], "enabled": "yes"}}}):
            s = harness.skill_settings(bad if isinstance(bad, dict) else {"import": bad})
            self.assertEqual(s, {"enabled": True, "harness_off": [], "off": {}}, repr(bad))

    def test_the_settings_come_out_as_they_are_saved(self):
        s = harness.skill_settings({"import": {"skills": {"enabled": False, "harness_off": ["codex", "nope"], "off": {"claude": ["a", "b"], "nope": ["x"]}}}})
        self.assertEqual(s, {"enabled": False, "harness_off": ["codex"], "off": {"claude": ["a", "b"]}})      # only the apps there are


# ------------------------------------------------------------------------------------------------ the built-in skills server
from serve import skills as skills_mod  # noqa: E402


def make_server(root: Path, n_extra: int = 0):
    """A SkillsServer over the skills planted in `root` (plus `n_extra` more that all mention 'filler')."""
    base = root / "skills"
    d = skill(base, "pdf-tools", "Read and write PDF files", body="# PDF\nUse the helper.")
    write(d / "references/forms.md", "# Forms\nfill them")
    write(d / "scripts/run.py", "print('x')")
    write(d / ".hidden/notes.txt", "hidden")
    write(d / ".env", "SECRET=1")
    write(d / "sub/.env.local", "SECRET=2")
    write(d / "id.pem", "-----BEGIN-----")
    (d / "bin.dat").write_bytes(b"ab\x00cd")
    (d / "big.txt").write_bytes(b"x" * (skills_mod.MAX_FILE + 10))
    skill(base, "git-helper", "Commit and branch with git")
    skill(base, "csv-tools", "Clean and convert CSV data")
    descriptions = {"pdf-tools": "Read and write PDF files", "git-helper": "Commit and branch with git", "csv-tools": "Clean and convert CSV data"}
    items = [{"id": f"claude:{n}", "harness": "claude", "name": n, "description": descriptions[n], "origin": "user", "dir": str(base / n),
              "real": os.path.realpath(base / n)} for n in sorted(descriptions)]
    for i in range(n_extra):
        p = skill(base, f"filler-{i:02d}", "filler skill about nothing")
        items.append({"id": f"claude:{p.name}", "harness": "claude", "name": p.name, "description": "filler skill about nothing", "origin": "user",
                      "dir": str(p), "real": os.path.realpath(p)})
    return skills_mod.SkillsServer(items), d


def text_of(result: dict) -> str:
    return "".join(c.get("text", "") for c in result["content"])


def link_dir(link: Path, target: Path) -> bool:
    """A symlink, or a junction where symlinks need rights; False when neither can be made here."""
    try:
        os.symlink(target, link, target_is_directory=True)
        return True
    except OSError:
        import subprocess
        return subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True).returncode == 0


class SkillsServer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.srv, self.pdf = make_server(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def call(self, tool, **args):
        return self.srv.call(tool, args, 5, None)

    def test_it_is_a_server_with_three_tools_and_is_ready(self):
        self.assertEqual((self.srv.name, self.srv.kind, self.srv.status), ("skills", "builtin", "ready"))
        self.assertEqual([t["name"] for t in self.srv.tools], ["find_skills", "use_skill", "read_skill_file"])
        for t in self.srv.tools:
            self.assertEqual(t["inputSchema"]["type"], "object")
            self.assertTrue(t["description"])

    def test_a_few_skills_are_listed_in_the_description_and_many_are_not(self):
        use = next(t for t in self.srv.tools if t["name"] == "use_skill")["description"]
        self.assertIn("pdf-tools", use)
        self.assertIn("Read and write PDF files", use)
        many, _ = make_server(self.root / "m", n_extra=40)
        use = next(t for t in many.tools if t["name"] == "use_skill")["description"]
        self.assertNotIn("filler-07", use)                           # the catalog is not carried in every prompt
        self.assertIn("find_skills", use)
        self.assertIn("43", use)                                     # how many there are
        self.assertLess(len(use), 1500)

    def test_find_ranks_by_the_words_and_says_when_nothing_matches(self):
        r = self.call("find_skills", query="pdf files")
        self.assertFalse(r["isError"])
        self.assertTrue(text_of(r).splitlines()[0].startswith("pdf-tools"))
        self.assertNotIn("git-helper", text_of(r))
        self.assertIn("No skill", text_of(self.call("find_skills", query="zzzqqq")))
        self.assertTrue(self.call("find_skills", query="")["isError"])

    def test_find_gives_at_most_twenty(self):
        many, _ = make_server(self.root / "m2", n_extra=40)
        r = many.call("find_skills", {"query": "filler skill"}, 5, None)
        self.assertEqual(len([ln for ln in text_of(r).splitlines() if ln.startswith("filler-")]), 20)

    def test_use_skill_gives_the_skill_file_and_the_files_beside_it(self):
        r = self.call("use_skill", name="pdf-tools")
        self.assertFalse(r["isError"])
        t = text_of(r)
        self.assertIn("Use the helper.", t)
        self.assertIn("references/forms.md", t)
        self.assertIn("scripts/run.py", t)
        listed = t.split("Files in this skill")[-1]
        for hidden in (".env", ".hidden", "id.pem", "bin.dat"):
            self.assertNotIn(hidden, listed, hidden)

    def test_use_skill_of_a_name_that_is_not_there_says_so_and_what_to_do(self):
        r = self.call("use_skill", name="nope")
        self.assertTrue(r["isError"])
        self.assertIn("find_skills", text_of(r))

    def test_read_a_file_of_the_skill(self):
        r = self.call("read_skill_file", name="pdf-tools", path="references/forms.md")
        self.assertFalse(r["isError"])
        self.assertIn("fill them", text_of(r))
        self.assertIn("print('x')", text_of(self.call("read_skill_file", name="pdf-tools", path="scripts\\run.py")))      # a Windows path

    def test_the_fence_refuses_what_is_not_a_plain_file_of_the_skill(self):
        cases = {
            "up": "../git-helper/SKILL.md", "up deep": "references/../../git-helper/SKILL.md", "up windows": "..\\git-helper\\SKILL.md",
            "absolute": str(self.root / "skills/git-helper/SKILL.md"), "slash": "/etc/passwd", "drive": "C:/Windows/win.ini",
            "dot file": ".env", "dot folder": ".hidden/notes.txt", "nested dot file": "sub/.env.local", "a key": "id.pem",
            "binary": "bin.dat", "too big": "big.txt", "a folder": "references", "missing": "nothing.md", "empty": "", "nul": "a\x00b",
        }
        for label, path in cases.items():
            with self.subTest(label):
                r = self.call("read_skill_file", name="pdf-tools", path=path)
                self.assertTrue(r["isError"], f"{label}: {text_of(r)[:80]}")
        self.assertTrue(self.call("read_skill_file", name="pdf-tools", path=123)["isError"])
        self.assertTrue(self.call("read_skill_file", name="nope", path="x")["isError"])

    def test_a_link_that_leads_out_of_the_skill_is_refused(self):
        outside = self.root / "outside"
        write(outside / "secret.txt", "TOP SECRET")
        if not link_dir(self.pdf / "linked", outside):
            self.skipTest("cannot make a link here")
        r = self.call("read_skill_file", name="pdf-tools", path="linked/secret.txt")
        self.assertTrue(r["isError"])
        self.assertNotIn("TOP SECRET", text_of(r))
        self.assertNotIn("linked", text_of(self.call("use_skill", name="pdf-tools")).split("Files in this skill")[-1])     # and it is not offered

    def test_a_skill_that_is_itself_a_link_may_read_its_own_real_folder(self):
        real = self.root / "shared" / "real-skill"
        skill(real.parent, "real-skill", "the real one")
        write(real / "refs/a.md", "inside")
        link = self.root / "skills" / "link-skill"
        if not link_dir(link, real):
            self.skipTest("cannot make a link here")
        srv = skills_mod.SkillsServer([{"id": "claude:link-skill", "harness": "claude", "name": "link-skill", "description": "d", "origin": "user",
                                        "dir": str(link), "real": os.path.realpath(link)}])
        self.assertIn("inside", text_of(srv.call("read_skill_file", {"name": "link-skill", "path": "refs/a.md"}, 5, None)))      # junctions into the shared folder work

    def test_an_unknown_tool_and_bad_arguments_are_errors_not_exceptions(self):
        self.assertTrue(self.srv.call("rm_rf", {}, 5, None)["isError"])
        self.assertTrue(self.srv.call("use_skill", "not a dict", 5, None)["isError"])
        self.assertTrue(self.srv.call("find_skills", {"query": 5}, 5, None)["isError"])

    def test_the_skills_can_be_changed_while_it_runs(self):
        self.srv.set_skills([])
        self.assertEqual(self.srv.tools, [])
        self.assertTrue(self.call("use_skill", name="pdf-tools")["isError"])
        self.srv.set_skills([{"id": "x:a", "harness": "x", "name": "a", "description": "d", "origin": "user", "dir": str(self.root), "real": str(self.root)}])
        self.assertEqual([t["name"] for t in self.srv.tools], ["find_skills", "use_skill", "read_skill_file"])


# ------------------------------------------------------------------------------------------------ MCP servers: read from the other apps, imported by a click
class Translate(unittest.TestCase):
    def tr(self, harness_id, raw, environ=None):
        return harness.translate(harness_id, "x", raw, environ if environ is not None else {})

    def test_a_program(self):
        r = self.tr("claude", {"type": "stdio", "command": "npx", "args": ["-y", "pkg"], "env": {"A": "1"}, "cwd": "C:/w"})
        self.assertEqual(r["entry"], {"command": "npx", "args": ["-y", "pkg"], "env": {"A": "1"}, "cwd": "C:/w"})
        self.assertEqual((r["reason"], r["disabled"]), (None, False))

    def test_an_address_with_headers(self):
        r = self.tr("claude", {"type": "http", "url": "https://h/mcp", "headers": {"Authorization": "Bearer t"}})
        self.assertEqual(r["entry"], {"url": "https://h/mcp", "headers": {"Authorization": "Bearer t"}})

    def test_sse_is_listed_with_its_reason_and_cannot_be_imported(self):
        for raw in ({"type": "sse", "url": "https://h/sse"}, {"url": "https://h/sse"}):
            r = self.tr("gemini", raw)
            self.assertIsNone(r["entry"])
            self.assertIn("SSE", r["reason"])
        self.assertEqual(self.tr("gemini", {"httpUrl": "https://h/mcp"})["entry"], {"url": "https://h/mcp"})       # Gemini's name for Streamable HTTP
        self.assertEqual(self.tr("cursor", {"url": "https://h/mcp"})["entry"], {"url": "https://h/mcp"})          # Cursor does not say: tried as Streamable HTTP

    def test_nothing_to_run_is_listed_with_its_reason(self):
        r = self.tr("codex", {"enabled": True})
        self.assertIsNone(r["entry"])
        self.assertIn("command", r["reason"])
        self.assertIsNone(self.tr("codex", "not a table")["entry"])

    def test_codex_enabled_false_and_claude_disabled_are_off_at_the_source(self):
        self.assertTrue(self.tr("codex", {"command": "node", "enabled": False})["disabled"])
        self.assertTrue(self.tr("claude", {"command": "node", "disabled": True})["disabled"])
        self.assertFalse(self.tr("codex", {"command": "node", "enabled": True})["disabled"])

    def test_codex_fields_that_strata_does_not_use_are_dropped(self):
        r = self.tr("codex", {"command": "node", "args": ["a"], "enabled": True, "tools": {"x": 1}, "startup_timeout_sec": 30, "env": {"K": "v"}})
        self.assertEqual(r["entry"], {"command": "node", "args": ["a"], "env": {"K": "v"}})

    def test_a_bearer_token_that_codex_keeps_in_an_environment_variable(self):
        raw = {"url": "https://h/mcp", "bearer_token_env_var": "REMOTE_TOKEN"}
        self.assertEqual(self.tr("codex", raw, {"REMOTE_TOKEN": "abc"})["entry"]["headers"], {"Authorization": "Bearer abc"})
        r = self.tr("codex", raw, {})
        self.assertNotIn("headers", r["entry"])
        self.assertIn("REMOTE_TOKEN", r["note"])

    def test_variables_in_a_value_are_filled_from_the_environment(self):
        r = self.tr("claude", {"command": "${TOOLS}/run", "args": ["--key=${KEY:-none}", "${MISSING}"]}, {"TOOLS": "C:/t", "KEY": "k"})
        self.assertEqual(r["entry"]["command"], "C:/t/run")
        self.assertEqual(r["entry"]["args"], ["--key=k", "${MISSING}"])
        self.assertEqual(self.tr("claude", {"command": "x", "args": ["${NOPE:-fallback}"]}, {})["entry"]["args"], ["fallback"])

    def test_an_entry_strata_would_refuse_is_listed_with_why(self):
        r = self.tr("claude", {"command": "x", "args": ["a"] * 100})
        self.assertIsNone(r["entry"])
        self.assertIn("args", r["reason"])
        self.assertIsNone(self.tr("claude", {"command": "a\nb"})["entry"])
        self.assertIsNone(self.tr("claude", {"url": "ftp://h"})["entry"])

    def test_the_same_server_looks_the_same_whatever_the_app(self):
        f = harness.fingerprint
        self.assertEqual(f({"command": "npx", "args": ["-y", "a"]}), f({"command": "C:/Program Files/nodejs/npx.cmd", "args": ["-y", "a"], "env": {"X": "1"}}))
        self.assertNotEqual(f({"command": "npx", "args": ["-y", "a"]}), f({"command": "npx", "args": ["-y", "b"]}))
        self.assertEqual(f({"url": "https://h/mcp"}), f({"url": "https://h/mcp/", "headers": {"A": "b"}}))
        self.assertNotEqual(f({"url": "https://h/mcp"}), f({"command": "https://h/mcp"}))


class Candidates(Fixture):
    def by(self, own=None):
        cands = harness.mcp_candidates(self.scan(), own or {}, {})
        return {h: {c["name"]: c for c in v} for h, v in cands.items()}

    def test_every_server_of_every_app_is_listed_with_a_state(self):
        c = self.by()
        self.assertEqual(c["claude"]["files"]["state"], "available")
        self.assertEqual(c["claude"]["legacy"]["state"], "unsupported")
        self.assertEqual(c["codex"]["nothing"]["state"], "unsupported")
        self.assertEqual(c["codex"]["off-here"]["state"], "available")
        self.assertTrue(c["codex"]["off-here"]["disabled_in_source"])
        self.assertEqual(c["gemini"]["g-sse"]["state"], "unsupported")
        self.assertEqual(c["claude"]["files"]["kind"], "program")
        self.assertEqual(c["claude"]["web"]["kind"], "address")

    def test_one_that_strata_already_has_says_so_whatever_it_is_called_there(self):
        own = {"my-files": {"command": "npx", "args": ["-y", "@x/server-files", "C:/work"]}}
        c = self.by(own)
        self.assertEqual(c["claude"]["files"]["state"], "already")
        self.assertEqual(c["claude"]["files"]["already_as"], "my-files")
        self.assertEqual(c["codex"]["files"]["state"], "already")             # the same server in another app
        self.assertEqual(c["cursor"]["cur"]["state"], "available")

    def test_a_page_never_gets_the_entry_and_only_who_may_change_the_settings_gets_the_command_line(self):
        for items in harness.mcp_candidates(self.scan(), {}, {}).values():
            for item in items:
                plain = harness.public_candidate(item, details=False)
                self.assertNotIn(SECRET, json.dumps(plain))
                for k in ("entry", "command", "args", "url", "env", "headers", "cwd"):
                    self.assertNotIn(k, plain, k)
                full = harness.public_candidate(item, details=True)
                self.assertNotIn(SECRET, json.dumps(full))                                  # the values of env and headers are masked
                self.assertNotIn("entry", full)
        files = next(i for i in harness.mcp_candidates(self.scan(), {}, {})["claude"] if i["name"] == "files")
        shown = harness.public_candidate(files, details=True)
        self.assertEqual((shown["command"], shown["args"][0]), ("npx", "-y"))
        self.assertEqual(shown["env"], {"TOKEN": mcp_admin_mask()})


def mcp_admin_mask():
    from serve import mcp_admin
    return mcp_admin.MASK


class ImportOne(Fixture):
    def do(self, hid, name, own=None):
        return harness.import_server(own if own is not None else {}, hid, name, self.scan(), {})

    def test_the_server_is_copied_with_its_secrets_into_strata_s_own_servers(self):
        own, name = self.do("claude", "files")
        self.assertEqual(name, "files")
        self.assertEqual(own["files"], {"command": "npx", "args": ["-y", "@x/server-files", "C:/work"], "env": {"TOKEN": SECRET}})
        own, _ = self.do("claude", "web")
        self.assertEqual(own["web"]["headers"], {"Authorization": "Bearer " + SECRET})

    def test_nothing_else_in_strata_s_servers_changes(self):
        before = {"mine": {"command": "python", "args": ["a"]}}
        own, _ = self.do("cursor", "cur", dict(before))
        self.assertEqual(own["mine"], before["mine"])
        self.assertEqual(set(own), {"mine", "cur"})

    def test_a_name_strata_already_uses_gets_the_app_added(self):
        own, name = self.do("codex", "files", {"files": {"command": "other"}})
        self.assertEqual(name, "files-codex")
        own, name = self.do("codex", "files", {"files": {"command": "other"}, "files-codex": {"command": "other2"}})
        self.assertEqual(name, "files-codex-2")

    def test_a_name_with_characters_strata_does_not_take_is_cleaned(self):
        own, name = self.do("claude", "bad name!")
        self.assertEqual(name, "bad-name")
        self.assertEqual(own[name]["command"], "node")

    def test_one_that_is_off_in_its_own_app_is_imported_off(self):
        own, name = self.do("codex", "off-here")
        self.assertTrue(own[name]["disabled"])

    def test_refused_cases_say_why(self):
        for hid, name, code in (("claude", "legacy", "unsupported"), ("codex", "nothing", "unsupported"), ("claude", "nope", "missing"), ("nope", "x", "missing")):
            with self.subTest(hid=hid, name=name):
                with self.assertRaises(harness.ImportRefused) as cm:
                    self.do(hid, name)
                self.assertEqual(cm.exception.code, code)
        with self.assertRaises(harness.ImportRefused) as cm:
            self.do("claude", "files", {"mine": {"command": "npx", "args": ["-y", "@x/server-files", "C:/work"]}})
        self.assertEqual(cm.exception.code, "already")

    def test_the_other_apps_files_are_not_touched(self):
        before = {p: p.read_bytes() for p in list(self.home.rglob("*.json")) + list(self.home.rglob("*.toml"))}
        self.do("claude", "files")
        self.assertEqual(before, {p: p.read_bytes() for p in before})


# ------------------------------------------------------------------------------------------------ in the hub, in the chat, and over HTTP
import urllib.error  # noqa: E402
import urllib.request  # noqa: E402

from serve import mcp_admin  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.mcp import hub_from_config  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402
from serve.test_mcp import ScriptedEngine  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FAKE = str(ROOT / "serve" / "mcp_fake_server.py")


class InTheHub(Fixture):
    def setUp(self):
        super().setUp()
        self.imp = harness.Importer(self.home, self.appdata, environ={})
        self.imp.rescan({})

    def test_the_skills_are_a_server_of_the_hub_with_namespaced_tools(self):
        hub = hub_from_config({}, None, builtins=self.imp.builtins())
        hub.start(wait=True)
        names = [t["name"] for t in hub.template_tools()]
        self.assertEqual(names, ["skills__find_skills", "skills__use_skill", "skills__read_skill_file"])
        r = hub.call("skills__use_skill", {"name": "pdf-tools"})
        self.assertTrue(r["ok"])
        self.assertIn("steps", r["text"])
        self.assertEqual((r["server"], r["tool"]), ("skills", "use_skill"))
        st = {s["name"]: s for s in hub.status()["servers"]}
        self.assertEqual((st["skills"]["transport"], st["skills"]["status"]), ("builtin", "ready"))
        hub.close()

    def test_a_hub_with_only_the_skills_is_a_hub_and_with_nothing_is_none(self):
        self.assertIsNotNone(hub_from_config({}, None, builtins=self.imp.builtins()))
        self.assertIsNone(hub_from_config({}, None))

    def test_with_no_skills_the_server_offers_no_tools_and_is_not_listed(self):
        self.imp.apply({"import": {"skills": {"enabled": False}}})
        hub = hub_from_config({}, None, builtins=self.imp.builtins())
        hub.start(wait=True)
        self.assertEqual(hub.template_tools(), [])
        self.assertNotIn("skills", {s["name"] for s in hub.status()["servers"]})
        hub.close()

    def test_a_switch_changes_the_tools_of_a_hub_that_is_running(self):
        hub = hub_from_config({}, None, builtins=self.imp.builtins())
        hub.start(wait=True)
        self.assertEqual(len(hub.template_tools()), 3)
        self.imp.apply({"import": {"skills": {"enabled": False}}})
        self.assertEqual(hub.template_tools(), [])
        self.imp.apply({})
        self.assertEqual(len(hub.template_tools()), 3)
        hub.close()


def tool_call(function: str, **params) -> str:
    """The model's text for one tool call (like test_mcp.call_script, whose first argument is called `name`, as a skill's parameter is)."""
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"</think>\n\nLet me check.\n\n<tool_call>\n<function={function}>\n{body}</function>\n</tool_call>"


class SkillsInTheChat(Fixture):
    """The model finds and loads a skill through the chat's own tool loop, as it does any MCP tool."""

    def setUp(self):
        super().setUp()
        self.imp = harness.Importer(self.home, self.appdata, environ={})
        self.imp.rescan({})
        self.hub = hub_from_config({}, None, builtins=self.imp.builtins())
        self.hub.start(wait=True)
        self.httpd = None

    def tearDown(self):
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
        self.hub.close()
        super().tearDown()

    def start(self, *scripts):
        tok = ByteTokenizer()
        self.engine = ScriptedEngine(tok, list(scripts))
        self.svc = Service(self.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.mcp = self.hub
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def post(self, body):
        body = {"model": "m", "messages": [{"role": "user", "content": "make a pdf"}], "stream": True, "strata_mcp": True, **body}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read().decode()

    def test_the_model_loads_a_skill_and_reads_it(self):
        self.start(tool_call("skills__use_skill", name="pdf-tools"), "</think>\n\nI used the skill.")
        text = self.post({})
        said = "".join((json.loads(ln[6:]).get("choices") or [{"delta": {}}])[0]["delta"].get("content") or "" for ln in text.splitlines() if ln.startswith("data: {"))
        self.assertIn("I used the skill.", said)
        self.assertEqual(len(self.engine.prompts), 2)
        first, second = self.engine.prompt_text(0), self.engine.prompt_text(1)
        self.assertIn('"name": "skills__use_skill"', first)                 # the three tools are offered
        self.assertIn("pdf-tools", first)                                   # few skills: they are named in the description
        self.assertNotIn("# Body", first)                                   # but their text is not in the prompt
        self.assertIn("# Body\nsteps", second)                              # it came back as the tool's result

    def test_a_skill_that_is_switched_off_is_not_offered(self):
        self.imp.apply({"import": {"skills": {"off": {"claude": ["pdf-tools"]}}}})
        self.start("</think>\n\nok")
        self.post({})
        self.assertNotIn("pdf-tools", self.engine.prompt_text(0))
        self.assertIn("plugin-skill", self.engine.prompt_text(0))


class ImportEndpoints(Fixture):
    def setUp(self):
        super().setUp()
        write(self.home / ".cursor/mcp.json", json.dumps({"mcpServers": {
            "fake": {"command": sys.executable, "args": [FAKE]},                                  # the one a test imports and sees start
            "quiet": {"command": "node", "args": ["q.js"], "disabled": True},                     # off at its source: imported off, so nothing starts
            "secret-off": {"command": "node", "args": ["s.js"], "env": {"TOKEN": SECRET}, "disabled": True},
            "secret-web": {"url": "https://x.example/mcp", "headers": {"Authorization": "Bearer " + SECRET}, "disabled": True}}}))
        self.cfg_path = self.home / "run.json"
        write(self.cfg_path, json.dumps({"model": "m", "mcp_servers": {"mine": {"command": "python", "args": ["-m", "mine"], "disabled": True}}}))
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "ok", max_context=8192), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.config_path = str(self.cfg_path)
        self.svc.mcp_config_path = None
        self.svc.importer = harness.Importer(self.home, self.appdata, environ={})
        self.svc.importer.rescan(json.loads(self.cfg_path.read_text(encoding="utf-8")))
        mcp_admin.reload_hub(self.svc)
        self.svc.mcp.wait(30)
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.seen = []                                           # every response body, to look for the secret in

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        if self.svc.mcp:
            self.svc.mcp.close()
        super().tearDown()

    def call(self, path, method="GET", body=None, headers=None):
        data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
        h = {"Content-Type": "application/json"} if data is not None else {}
        h.update(headers or {})
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=h)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                code, raw = r.status, r.read()
        except urllib.error.HTTPError as e:
            code, raw = e.code, e.read()
        self.seen.append(raw)
        return code, raw

    def js(self, *a, **k):
        code, raw = self.call(*a, **k)
        return code, json.loads(raw)

    def saved(self):
        return json.loads(self.cfg_path.read_text(encoding="utf-8"))

    def own(self, headers=None):
        return {"Origin": self.base, **(headers or {})}

    def test_the_view_lists_the_apps_with_their_skills_and_servers(self):
        code, d = self.js("/import")
        self.assertEqual(code, 200)
        self.assertTrue(d["available"] and d["editable"])
        self.assertEqual((d["skills"]["used"], d["skills"]["total"]), (7, 8))                  # the second copy of a shared skill is not in use
        by = {h["id"]: h for h in d["mcp"]["harnesses"]}
        self.assertEqual({s["name"]: s["state"] for s in by["claude"]["servers"]}["files"], "available")
        self.assertEqual({s["name"]: s["state"] for s in by["claude"]["servers"]}["legacy"], "unsupported")
        self.assertEqual(d["skills"]["settings"], {"enabled": True, "harness_off": [], "off": {}})
        files = next(s for s in by["claude"]["servers"] if s["name"] == "files")
        self.assertEqual(files["command"], "npx")                                               # who may change the settings sees the command line
        self.assertEqual(files["env"], {"TOKEN": mcp_admin.MASK})
        self.assertTrue(any(h["found"] and h["files"] for h in d["harnesses"]))

    def test_from_another_address_it_is_a_list_of_names_and_says_why(self):
        host = {"Host": "10.0.0.5:8091"}
        code, raw = self.call("/import", "GET", None, host)
        d = json.loads(raw)
        self.assertEqual(code, 200)
        self.assertFalse(d["editable"])
        self.assertIn("API key", d["reason"])
        for key in ('"command"', '"args"', '"url"', '"env"', '"headers"', '"cwd"'):
            self.assertNotIn(key, raw.decode(), key)
        self.assertIn("files", raw.decode())                                                    # the names are there
        for body in ({"rescan": True}, {"skills": {"enabled": False}}, {"import_mcp": {"harness": "claude", "name": "files"}}):
            self.assertEqual(self.call("/import", "POST", body, host)[0], 403, body)

    def test_a_skill_switch_is_saved_as_an_off_list_and_takes_effect_at_once(self):
        code, d = self.js("/import", "POST", {"skills": {"harness_off": ["codex"], "off": {"claude": ["pdf-tools"]}}}, self.own())
        self.assertEqual(code, 200, d)
        self.assertEqual(self.saved()["import"]["skills"], {"enabled": True, "harness_off": ["codex"], "off": {"claude": ["pdf-tools"]}})
        self.assertEqual(self.saved()["model"], "m")                                            # the rest of the config is kept
        self.assertEqual(self.saved()["mcp_servers"]["mine"]["args"], ["-m", "mine"])
        in_use = sorted(self.svc.mcp.servers["skills"]._skills)
        self.assertNotIn("pdf-tools", in_use)
        self.assertNotIn("codex-only", in_use)
        self.assertIn("plugin-skill", in_use)
        self.assertEqual(d["skills"]["used"], 5)
        self.assertTrue(Path(str(self.cfg_path) + ".bak-mcp").exists())

    def test_a_switch_that_is_off_stays_off_after_a_rescan_and_a_new_skill_is_on(self):
        self.js("/import", "POST", {"skills": {"off": {"claude": ["pdf-tools"]}}}, self.own())
        skill(self.home / ".claude/skills", "brand-new", "turned up later")
        code, d = self.js("/import", "POST", {"rescan": True}, self.own())
        self.assertEqual(code, 200)
        used = sorted(self.svc.mcp.servers["skills"]._skills)
        self.assertIn("brand-new", used)
        self.assertNotIn("pdf-tools", used)

    def test_a_rescan_is_what_finds_a_skill_added_since(self):
        skill(self.home / ".claude/skills", "later-one")
        code, d = self.js("/import")
        self.assertNotIn("later-one", [i["name"] for i in d["skills"]["items"]])
        code, d = self.js("/import?rescan=1")
        self.assertIn("later-one", [i["name"] for i in d["skills"]["items"]])

    def test_bad_skill_settings_are_refused_with_the_field_and_nothing_is_written(self):
        before = self.cfg_path.read_text(encoding="utf-8")
        for bad, field in (({"enabled": "yes"}, "enabled"), ({"harness_off": ["nope"]}, "harness_off"), ({"harness_off": "codex"}, "harness_off"),
                           ({"off": {"claude": "x"}}, "off"), ({"off": {"nope": ["x"]}}, "off"), ({"off": {"claude": [1]}}, "off"), ({"unknown": 1}, "unknown")):
            with self.subTest(bad=bad):
                code, d = self.js("/import", "POST", {"skills": bad}, self.own())
                self.assertEqual(code, 400)
                self.assertEqual(d["error"]["fields"][0]["field"], field)
        self.assertEqual(self.cfg_path.read_text(encoding="utf-8"), before)

    def test_importing_a_server_copies_it_with_its_secret_and_starts_it_like_any_other(self):
        code, d = self.js("/import", "POST", {"import_mcp": {"harness": "cursor", "name": "fake"}}, self.own())
        self.assertEqual(code, 200, d)
        self.assertEqual(self.saved()["mcp_servers"]["fake"], {"command": sys.executable, "args": [FAKE]})
        self.svc.mcp.wait(30)
        self.assertEqual(self.svc.mcp.servers["fake"].status, "ready")                          # it runs now: an ordinary server of Strata's
        cur = {s["name"]: s for h in d["mcp"]["harnesses"] if h["id"] == "cursor" for s in h["servers"]}
        self.assertEqual(cur["fake"]["state"], "already")
        self.assertEqual(self.saved()["mcp_servers"]["mine"]["args"], ["-m", "mine"])

    def test_the_secret_is_copied_on_the_server_and_is_in_no_response(self):
        code, d = self.js("/import", "POST", {"import_mcp": {"harness": "cursor", "name": "secret-off"}}, self.own())
        self.assertEqual(code, 200, d)
        self.assertEqual(self.saved()["mcp_servers"]["secret-off"]["env"], {"TOKEN": SECRET})   # in the run config, where Strata's own servers are
        self.assertTrue(self.saved()["mcp_servers"]["secret-off"]["disabled"])                   # off at its source: imported off
        self.js("/import", "POST", {"import_mcp": {"harness": "cursor", "name": "secret-web"}}, self.own())
        self.assertEqual(self.saved()["mcp_servers"]["secret-web"]["headers"], {"Authorization": "Bearer " + SECRET})
        self.js("/import")
        self.js("/mcp/config")
        self.js("/import", "GET", None, {"Host": "10.0.0.5:8091"})
        self.assertTrue(self.seen)
        for raw in self.seen:
            self.assertNotIn(SECRET.encode(), raw)

    def test_a_server_that_cannot_be_imported_says_why(self):
        for body, status in (({"harness": "claude", "name": "legacy"}, 422), ({"harness": "codex", "name": "nothing"}, 422), ({"harness": "claude", "name": "nope"}, 404),
                             ({"harness": "nope", "name": "x"}, 404), ({"harness": "claude"}, 400), ({"harness": 5, "name": "x"}, 400)):
            with self.subTest(body=body):
                code, d = self.js("/import", "POST", {"import_mcp": body}, self.own())
                self.assertEqual(code, status, d)
                self.assertTrue(d["error"]["message"])
        self.js("/import", "POST", {"import_mcp": {"harness": "cursor", "name": "quiet"}}, self.own())
        code, d = self.js("/import", "POST", {"import_mcp": {"harness": "cursor", "name": "quiet"}}, self.own())
        self.assertEqual(code, 409)                                                             # it is already there now
        self.assertEqual(list(self.saved()["mcp_servers"]).count("quiet"), 1)

    def test_discovery_alone_starts_nothing(self):
        before = set(self.svc.mcp.servers)
        self.js("/import?rescan=1")
        self.js("/import", "POST", {"rescan": True}, self.own())
        self.assertEqual(set(self.svc.mcp.servers), before)                                     # only "mine" and the skills

    def test_who_may_ask_and_how(self):
        self.assertEqual(self.call("/import", "POST", b"{}", {"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.call("/import", "POST", {"rescan": True}, {"Origin": "http://evil.example"})[0], 403)
        self.assertEqual(self.call("/import", "POST", b"not json", self.own())[0], 400)
        self.assertEqual(self.call("/import", "POST", {"nothing": 1}, self.own())[0], 400)
        host = {"Host": "10.0.0.5:8091"}
        self.svc.api_key = "k"
        ok = {**host, "Authorization": "Bearer k"}
        self.assertTrue(self.js("/import", "GET", None, ok)[1]["editable"])
        self.assertEqual(self.call("/import", "POST", {"rescan": True}, ok)[0], 200)
        self.assertEqual(self.call("/import", "POST", {"rescan": True}, host)[0], 401)
        self.assertEqual(self.call("/import", "GET", None, host)[0], 401)

    def test_without_a_config_file_it_is_a_read_only_list(self):
        self.svc.config_path = None
        code, d = self.js("/import")
        self.assertEqual(code, 200)
        self.assertFalse(d["editable"])
        self.assertIn("run config", d["reason"])
        self.assertEqual(self.call("/import", "POST", {"skills": {"enabled": False}}, self.own())[0], 409)

    def test_a_server_without_the_importer_says_it_is_not_there(self):
        self.svc.importer = None
        code, d = self.js("/import")
        self.assertEqual((code, d), (200, {"available": False}))
        self.assertEqual(self.call("/import", "POST", {"rescan": True}, self.own())[0], 404)

if __name__ == "__main__":
    unittest.main()
