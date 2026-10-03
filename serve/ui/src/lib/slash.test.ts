import { describe, expect, test } from "bun:test"
import { commandsOf, markOf, matchCommands, pickCommand, skillOfMessage, slashQuery, type Command } from "./slash"
import type { SkillItem } from "./importer"

// "/" at the start of a message opens the skills in use, to be picked by name (like Claude Code's slash commands); a message that
// starts with the name of a skill in use asks the server to load that skill.
const item = (name: string, extra: Partial<SkillItem> = {}): SkillItem =>
  ({ id: `claude:${name}`, harness: "claude", name, description: `does ${name}`, origin: "user", off: false, used: true, same_as: null, ...extra })
const cmd = (name: string, description = "", extra: Partial<Command> = {}): Command => ({ name, description, from: "Claude Code", plugin: null, ...extra })

describe("slashQuery: is the caret in a slash command at the start", () => {
  test("a / alone is an empty query", () => { expect(slashQuery("/", 1)).toBe("") })
  test("the letters after it are the query", () => { expect(slashQuery("/pdf", 4)).toBe("pdf") })
  test("only what is before the caret counts", () => { expect(slashQuery("/pdf-tools", 4)).toBe("pdf") })
  test("not when the / is not the first character", () => {
    expect(slashQuery("see /pdf", 8)).toBeNull()
    expect(slashQuery(" /pdf", 5)).toBeNull()
    expect(slashQuery("a\n/pdf", 6)).toBeNull()
  })
  test("not once there is a space (the arguments are being typed) or when the caret is before the /", () => {
    expect(slashQuery("/pdf make", 9)).toBeNull()
    expect(slashQuery("/pdf", 0)).toBeNull()
  })
  test("a path or a URL is not a command", () => {
    expect(slashQuery("/usr/bin", 8)).toBeNull()
    expect(slashQuery("//", 2)).toBeNull()
  })
  test("no text, no query", () => { expect(slashQuery("", 0)).toBeNull() })
})

describe("commandsOf: the skills that can be picked", () => {
  test("only the copy in use of each name, sorted by name", () => {
    const got = commandsOf([item("b"), item("a"), item("a", { id: "codex:a", harness: "codex", used: false, same_as: "claude:a" }), item("c", { used: false, off: true })])
    expect(got.map((c) => c.name)).toEqual(["a", "b"])
  })
  test("a name that two apps both have in use is listed once", () => {
    expect(commandsOf([item("a"), item("a", { id: "codex:a", harness: "codex" })]).map((c) => c.name)).toEqual(["a"])
  })
})

describe("commandsOf: where a skill comes from", () => {
  const labels = { claude: "Claude Code", codex: "Codex CLI" }
  test("the app's name and, for a plugin's skill, the plugin", () => {
    const got = commandsOf([item("a"), item("b", { origin: "plugin:superpowers", harness: "claude" }), item("c", { harness: "codex" })], labels)
    expect(got.map((c) => [c.name, c.from, c.plugin])).toEqual([["a", "Claude Code", null], ["b", "Claude Code", "superpowers"], ["c", "Codex CLI", null]])
  })
  test("an app with no label is called by its id", () => { expect(commandsOf([item("a", { harness: "zed" })], labels)[0].from).toBe("zed") })
})

describe("matchCommands: the list as it is typed", () => {
  const all = [cmd("pdf-tools", "make a pdf"), cmd("tdd", "test first"), cmd("spdf", "a name with pdf inside"), cmd("notes", "writes up a PDF of notes")]
  test("an empty query lists them all, by name", () => { expect(matchCommands(all, "").map((c) => c.name)).toEqual(["notes", "pdf-tools", "spdf", "tdd"]) })
  test("a name that starts with it comes first, then one that contains it, then a description", () => {
    expect(matchCommands(all, "pdf").map((c) => c.name)).toEqual(["pdf-tools", "spdf", "notes"])
  })
  test("capitals do not matter", () => { expect(matchCommands(all, "TDD").map((c) => c.name)).toEqual(["tdd"]) })
  test("an exact name is first", () => { expect(matchCommands([cmd("pdf-tools"), cmd("pdf")], "pdf").map((c) => c.name)).toEqual(["pdf", "pdf-tools"]) })
  test("nothing matches nothing", () => { expect(matchCommands(all, "zzz")).toEqual([]) })
})

describe("pickCommand: choosing one puts its name in the first word", () => {
  test("a bare slash becomes the name and a space", () => { expect(pickCommand("/", "pdf-tools")).toEqual({ text: "/pdf-tools ", caret: 11 }) })
  test("what was typed of the name is replaced", () => { expect(pickCommand("/pd", "pdf-tools")).toEqual({ text: "/pdf-tools ", caret: 11 }) })
  test("what follows is kept, after one space", () => { expect(pickCommand("/pd make it", "pdf-tools")).toEqual({ text: "/pdf-tools make it", caret: 11 }) })
})

describe("skillOfMessage: the skill a message asks for", () => {
  const names = ["pdf-tools", "tdd"]
  test("a message that starts with the name of a skill in use", () => {
    expect(skillOfMessage("/pdf-tools make a pdf", names)).toBe("pdf-tools")
    expect(skillOfMessage("/tdd", names)).toBe("tdd")
    expect(skillOfMessage("  /tdd now", names)).toBe("tdd")
  })
  test("not a name that is not a skill in use, or only the start of one", () => {
    expect(skillOfMessage("/nope go", names)).toBeNull()
    expect(skillOfMessage("/pdf go", names)).toBeNull()
    expect(skillOfMessage("/pdf-toolsx go", names)).toBeNull()
  })
  test("not when the slash is in the middle", () => { expect(skillOfMessage("use /tdd", names)).toBeNull() })
  test("no skills, no skill", () => { expect(skillOfMessage("/tdd", [])).toBeNull() })
})

describe("markOf: where the slash command is in a message", () => {
  const cmds = [cmd("pdf-tools", "make a pdf"), cmd("tdd")]
  test("the range of /name at the start, and the skill", () => {
    const m = markOf("/pdf-tools make a pdf", cmds)
    expect([m?.at, m?.end, m?.cmd.name]).toEqual([0, 10, "pdf-tools"])
  })
  test("after leading white space the range starts at the slash", () => {
    const m = markOf("  /tdd now", cmds)
    expect([m?.at, m?.end]).toEqual([2, 6])
  })
  test("a name alone is a command too", () => { expect(markOf("/tdd", cmds)?.end).toBe(4) })
  test("nothing for a name that is not a skill in use, a half name, or a slash further in", () => {
    expect(markOf("/nope go", cmds)).toBeNull()
    expect(markOf("/pdf go", cmds)).toBeNull()
    expect(markOf("use /tdd", cmds)).toBeNull()
    expect(markOf("", cmds)).toBeNull()
  })
})
