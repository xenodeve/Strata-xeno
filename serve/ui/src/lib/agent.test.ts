import { describe, expect, test } from "bun:test"
import { addRule, agentRequest, diffRows, forgetRules, NO_AGENT, rulesOf, toolSummary, type AgentInfo } from "./agent"
import type { Backing } from "./sessions"

// The page's side of the coding tools (issue #96): what a request says, the rules the user chose "allow for this chat" for, and how a call
// is summarised and shown. The server decides everything; this only carries the user's choices and shows what happened.
const ON: AgentInfo = { available: true, allowed: true, shell: "bash", tools: ["Read", "Write", "Bash"] }
const mem = (): Backing & { data: Map<string, unknown> } => {
  const data = new Map<string, unknown>()
  return { data, get: <T,>(k: string, d: T) => (data.has(k) ? (data.get(k) as T) : d), set: (k, v) => { data.set(k, JSON.parse(JSON.stringify(v))); return true }, remove: (k) => { data.delete(k) } }
}

describe("agentRequest: what a chat request says about the coding tools", () => {
  test("on: the folder, the mode, the chat and its rules", () => {
    expect(agentRequest({ agent: true, agentMode: "plan" }, ON, "C:/work/app", "chat1", ["Bash(git status:*)"])).toEqual({
      strata_agent: { cwd: "C:/work/app", mode: "plan", session: "chat1", allow: ["Bash(git status:*)"] },
    })
  })
  test("no folder: no cwd", () => {
    const r = agentRequest({ agent: true }, ON, null, "c", []).strata_agent!
    expect("cwd" in r).toBe(false)
    expect(r.mode).toBe("ask")
  })
  test("an empty folder text is no folder", () => { expect("cwd" in agentRequest({ agent: true }, ON, "  ", "c", []).strata_agent!).toBe(false) })
  test("switched off, or not available, or not allowed from here: nothing", () => {
    expect(agentRequest({ agent: false }, ON, "x", "c", [])).toEqual({})
    expect(agentRequest({ agent: true }, NO_AGENT, "x", "c", [])).toEqual({})
    expect(agentRequest({ agent: true }, { ...ON, allowed: false }, "x", "c", [])).toEqual({})
  })
  test("the default is on (a setting that was never made)", () => { expect(agentRequest({}, ON, "x", "c", []).strata_agent).toBeDefined() })
  test("a mode that is not known is the default one", () => { expect(agentRequest({ agent: true, agentMode: "bypass" }, ON, "x", "c", []).strata_agent!.mode).toBe("ask") })
  test("a chat that is not saved yet still has a session name", () => { expect(agentRequest({ agent: true }, ON, "x", null, []).strata_agent!.session).toBe("new") })
})

describe("the rules a user allowed for a chat", () => {
  test("they are kept per chat", () => {
    const b = mem()
    addRule(b, "a", "Bash(git status:*)")
    addRule(b, "b", "Read(/x/**)")
    expect(rulesOf(b, "a")).toEqual(["Bash(git status:*)"])
    expect(rulesOf(b, "b")).toEqual(["Read(/x/**)"])
    expect(rulesOf(b, "none")).toEqual([])
  })
  test("a rule is kept once", () => {
    const b = mem()
    addRule(b, "a", "Bash(ls:*)")
    addRule(b, "a", "Bash(ls:*)")
    expect(rulesOf(b, "a")).toEqual(["Bash(ls:*)"])
  })
  test("a chat can forget them", () => {
    const b = mem()
    addRule(b, "a", "Bash(ls:*)")
    forgetRules(b, "a")
    expect(rulesOf(b, "a")).toEqual([])
  })
  test("the chats that were used longest ago are dropped first, and a chat keeps at most 100 rules", () => {
    const b = mem()
    for (let i = 0; i < 60; i++) addRule(b, `c${i}`, "Bash(ls:*)")
    expect(rulesOf(b, "c0")).toEqual([])
    expect(rulesOf(b, "c59")).toEqual(["Bash(ls:*)"])
    for (let i = 0; i < 150; i++) addRule(b, "big", `Bash(cmd${i}:*)`)
    expect(rulesOf(b, "big").length).toBe(100)
    expect(rulesOf(b, "big")[99]).toBe("Bash(cmd149:*)")
  })
  test("what is stored is checked when it is read back", () => {
    const b = mem()
    b.set("agent.rules", { order: ["a"], rules: { a: ["ok", 5, null, "also ok"] } })
    expect(rulesOf(b, "a")).toEqual(["ok", "also ok"])
    b.set("agent.rules", "garbage")
    expect(rulesOf(b, "a")).toEqual([])
  })
})

describe("toolSummary: a call in a few words", () => {
  test("by tool", () => {
    expect(toolSummary("Read", { file_path: "src/a.py" })).toBe("src/a.py")
    expect(toolSummary("Write", { file_path: "x.txt", content: "y" })).toBe("x.txt")
    expect(toolSummary("Edit", { file_path: "src/a.py", old_string: "a", new_string: "b" })).toBe("src/a.py")
    expect(toolSummary("NotebookEdit", { notebook_path: "analysis.ipynb", new_source: "x" })).toBe("analysis.ipynb")
    expect(toolSummary("Glob", { pattern: "**/*.py" })).toBe("**/*.py")
    expect(toolSummary("Grep", { pattern: "foo", path: "src" })).toBe("foo in src")
    expect(toolSummary("Grep", { pattern: "foo" })).toBe("foo")
    expect(toolSummary("Bash", { command: "npm test\nnpm run build" })).toBe("npm test")
    expect(toolSummary("Bash", { command: "ls", description: "List the files" })).toBe("List the files")
    expect(toolSummary("TodoWrite", { todos: [1, 2, 3] })).toBe("3 steps")
    expect(toolSummary("BashOutput", { bash_id: "bash_1" })).toBe("bash_1")
    expect(toolSummary("KillShell", { shell_id: "bash_2" })).toBe("bash_2")
  })
  test("arguments that are not what they should be give nothing, not a crash", () => {
    expect(toolSummary("Read", null)).toBe("")
    expect(toolSummary("Read", { file_path: 5 })).toBe("")
    expect(toolSummary("Bash", "ls")).toBe("")
    expect(toolSummary("Mystery", { a: 1 })).toBe("")
  })
  test("a long one is cut", () => { expect(toolSummary("Bash", { command: "x".repeat(500) }).length).toBeLessThanOrEqual(121) })
})

describe("diffRows: an edit as removed and added lines", () => {
  test("the old lines, then the new ones", () => {
    expect(diffRows("a\nb", "A\nb\nc")).toEqual([{ kind: "del", text: "a" }, { kind: "del", text: "b" }, { kind: "add", text: "A" }, { kind: "add", text: "b" }, { kind: "add", text: "c" }])
  })
  test("a line that is the same at the start or the end is not shown as changed", () => {
    expect(diffRows("keep\nold\nend", "keep\nnew\nend")).toEqual([{ kind: "same", text: "keep" }, { kind: "del", text: "old" }, { kind: "add", text: "new" }, { kind: "same", text: "end" }])
  })
  test("nothing old is only additions", () => { expect(diffRows("", "x")).toEqual([{ kind: "add", text: "x" }]) })
  test("a very long edit is cut", () => {
    const rows = diffRows(Array.from({ length: 500 }, (_, i) => `o${i}`).join("\n"), "x")
    expect(rows.length).toBeLessThanOrEqual(201)
    expect(rows.at(-1)?.kind).toBe("more")
  })
})
