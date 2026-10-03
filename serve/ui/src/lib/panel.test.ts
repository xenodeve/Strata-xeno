import { describe, expect, test } from "bun:test"
import { planOf, skillCall, skillsUsed, panelState } from "./panel"
import type { Message, ToolCall } from "./chat"

const tool = (extra: Partial<ToolCall>): ToolCall => ({ id: Math.random().toString(36), name: "x", at: 0, rat: 0, state: "done", ...extra })
const a = (extra: Partial<Message> = {}): Message => ({ role: "assistant", text: "", time: 1, ...extra })
const u = (text: string): Message => ({ role: "user", text, time: 1 })

describe("panelState", () => {
  test("kept as it was; anything odd is the default", () => {
    expect(panelState({ open: true, tab: "plan" })).toEqual({ open: true, tab: "plan" })
    expect(panelState({ open: "yes", tab: "nope" })).toEqual({ open: false, tab: "git" })
    expect(panelState(null)).toEqual({ open: false, tab: "git" })
    expect(panelState("x")).toEqual({ open: false, tab: "git" })
  })
})

describe("planOf", () => {
  const todos = [{ content: "a", status: "completed" as const, activeForm: "A" }, { content: "b", status: "in_progress" as const, activeForm: "B" }, { content: "c", status: "pending" as const, activeForm: "C" }]
  test("the list as it was last sent, how far it is and the step in progress", () => {
    const p = planOf([u("go"), a({ todos: [todos[0]] }), a({ todos })])
    expect(p).toMatchObject({ done: 1, total: 3 })
    expect(p.current?.content).toBe("b")
    expect(p.todos).toHaveLength(3)
  })
  test("no list: nothing", () => { expect(planOf([u("hi"), a({ text: "ok" })])).toMatchObject({ done: 0, total: 0, current: null, approved: null }) })
  test("the plan sent for approval is kept", () => {
    const p = planOf([a({ tools: [tool({ name: "ExitPlanMode", arguments: { plan: "1. do it" } })] }), a({ tools: [tool({ name: "ExitPlanMode", arguments: { plan: "1. do it better" } })] })])
    expect(p.approved).toBe("1. do it better")
  })
  test("a plan that is no text is not a plan", () => { expect(planOf([a({ tools: [tool({ name: "ExitPlanMode", arguments: { plan: 5 } })] })]).approved).toBeNull() })
})

describe("skillCall", () => {
  test("loading a skill, reading its file, looking for one", () => {
    expect(skillCall({ server: "skills", tool: "use_skill", arguments: { name: "pdf-tools" } })).toEqual({ kind: "use", name: "pdf-tools" })
    expect(skillCall({ server: "skills", tool: "read_skill_file", arguments: { name: "pdf-tools", path: "a.md" } })).toEqual({ kind: "file", name: "pdf-tools" })
    expect(skillCall({ server: "skills", tool: "find_skills", arguments: { query: "pdf" } })).toEqual({ kind: "find" })
  })
  test("another server, another tool, odd arguments: nothing, or no name", () => {
    expect(skillCall({ server: "fs", tool: "use_skill", arguments: { name: "x" } })).toBeNull()
    expect(skillCall({ server: "skills", tool: "other" })).toBeNull()
    expect(skillCall({ server: "skills", tool: "use_skill", arguments: "x" })).toEqual({ kind: "use", name: undefined })
  })
})

describe("skillsUsed", () => {
  test("the skills the user asked for with /name and the ones the model loaded, in order of first use, with counts", () => {
    const list = skillsUsed([
      u("/pdf-tools read this"),
      a({ tools: [tool({ server: "skills", tool: "use_skill", arguments: { name: "git-flow" } }), tool({ server: "skills", tool: "read_skill_file", arguments: { name: "git-flow", path: "x.md" } }), tool({ server: "skills", tool: "find_skills", arguments: { query: "q" } })] }),
      u("/pdf-tools again"),
      a({ tools: [tool({ server: "skills", tool: "use_skill", arguments: { name: "git-flow" } })] }),
    ], ["pdf-tools"])
    expect(list).toEqual([{ name: "pdf-tools", by: "you", uses: 2, files: 0 }, { name: "git-flow", by: "model", uses: 2, files: 1 }])
  })
  test("a /name that is not a skill in use is not one; a call that did not run does not count; a summary is not a prompt", () => {
    const list = skillsUsed([
      u("/nothing here"), { role: "user", text: "/pdf-tools in a summary", time: 1, compact: { before: 1, after: 1, auto: false } },
      a({ tools: [tool({ server: "skills", tool: "use_skill", arguments: { name: "x" }, state: "skipped" })] }),
    ], ["pdf-tools"])
    expect(list).toEqual([])
  })
})
