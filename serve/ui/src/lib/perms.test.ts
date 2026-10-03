import { describe, expect, test } from "bun:test"
import { addPerm, countRules, forgetPerms, loadPerms, MAX_RULES, permsFor, removePerm, ruleSetOf, validRule } from "./perms"
import type { Backing } from "./sessions"

const memory = (): Backing & { data: Map<string, unknown> } => {
  const data = new Map<string, unknown>()
  return { data, get: <T,>(k: string, d: T) => (data.has(k) ? (JSON.parse(JSON.stringify(data.get(k))) as T) : d), set: (k, v) => { data.set(k, v); return true }, remove: (k) => { data.delete(k) } }
}

describe("validRule: a tool's name, and between brackets what it is limited to", () => {
  test("rules in Claude Code's syntax", () => {
    for (const r of ["Bash", "Read", "Bash(npm test:*)", "Read(src/**)", "Edit(C:/work/app/**)", "WebFetch(domain:example.com)", "Bash(git commit -m \"x (y)\":*)"]) expect(validRule(r)).toBe(true)
  })
  test("not rules", () => {
    for (const r of ["", "  ", "1Bash", "Bash(", "Bash()", "Bash (x)", "(x)", "Bash(x) extra", "x".repeat(501)]) expect(validRule(r)).toBe(false)
  })
})

describe("the rules for a project and for everywhere", () => {
  const P = { kind: "project", id: "p1" } as const
  const E = { kind: "everywhere" } as const
  test("added, once each, newest last, kept", () => {
    const b = memory()
    expect(addPerm(b, E, "allow", "Bash(git status:*)")).toBe(true)
    expect(addPerm(b, E, "allow", "Bash(git status:*)")).toBe(false)
    expect(addPerm(b, P, "deny", "Bash(rm:*)")).toBe(true)
    expect(addPerm(b, P, "allow", "Read(src/**)")).toBe(true)
    const p = loadPerms(b)
    expect(p.everywhere).toEqual({ allow: ["Bash(git status:*)"], deny: [] })
    expect(p.projects.p1).toEqual({ allow: ["Read(src/**)"], deny: ["Bash(rm:*)"] })
    expect(countRules(p)).toBe(3)
  })
  test("something that is not a rule is not added", () => {
    const b = memory()
    expect(addPerm(b, E, "allow", "not a rule at all")).toBe(false)
    expect(addPerm(b, E, "allow", "  Bash(ls:*)  ")).toBe(true)
    expect(loadPerms(b).everywhere.allow).toEqual(["Bash(ls:*)"])             // trimmed
  })
  test("a rule is in one list of a place: the other one loses it", () => {
    const b = memory()
    addPerm(b, P, "allow", "Bash(npm test:*)")
    addPerm(b, P, "deny", "Bash(npm test:*)")
    expect(loadPerms(b).projects.p1).toEqual({ allow: [], deny: ["Bash(npm test:*)"] })
  })
  test("removed; a project with none is not kept", () => {
    const b = memory()
    addPerm(b, P, "allow", "Read")
    addPerm(b, E, "deny", "Bash(rm:*)")
    removePerm(b, P, "allow", "Read")
    removePerm(b, E, "deny", "Bash(rm:*)")
    expect(loadPerms(b)).toEqual({ everywhere: { allow: [], deny: [] }, projects: {} })
    removePerm(b, P, "allow", "never there")
    expect(ruleSetOf(loadPerms(b), P)).toEqual({ allow: [], deny: [] })
  })
  test("a project that is gone takes its rules with it, and not the others'", () => {
    const b = memory()
    addPerm(b, P, "allow", "Read")
    addPerm(b, { kind: "project", id: "p2" }, "allow", "Glob")
    forgetPerms(b, "p1")
    expect(Object.keys(loadPerms(b).projects)).toEqual(["p2"])
    forgetPerms(b, "nope")
    expect(Object.keys(loadPerms(b).projects)).toEqual(["p2"])
  })
  test("a list is full at MAX_RULES", () => {
    const b = memory()
    for (let i = 0; i < MAX_RULES; i++) expect(addPerm(b, E, "allow", `Bash(cmd${i}:*)`)).toBe(true)
    expect(addPerm(b, E, "allow", "Bash(one-more:*)")).toBe(false)
  })
})

describe("permsFor: what a request carries", () => {
  test("the chat's allowed rules with the project's and the everywhere ones, and the refused ones of the project and everywhere, each once", () => {
    const b = memory()
    addPerm(b, { kind: "everywhere" }, "allow", "Read")
    addPerm(b, { kind: "everywhere" }, "deny", "Bash(rm:*)")
    addPerm(b, { kind: "project", id: "p1" }, "allow", "Edit(src/**)")
    addPerm(b, { kind: "project", id: "p1" }, "deny", "Bash(curl:*)")
    addPerm(b, { kind: "project", id: "p2" }, "allow", "Glob")
    expect(permsFor(b, ["Bash(npm test:*)", "Read"], "p1")).toEqual({ allow: ["Bash(npm test:*)", "Read", "Edit(src/**)"], deny: ["Bash(curl:*)", "Bash(rm:*)"] })
    expect(permsFor(b, [], undefined)).toEqual({ allow: ["Read"], deny: ["Bash(rm:*)"] })
    expect(permsFor(b, [], "p2").allow).toEqual(["Glob", "Read"])
  })
  test("odd storage is read as nothing", () => {
    const b = memory()
    b.set("agent.perms", { everywhere: "x", projects: [1, 2] })
    expect(loadPerms(b)).toEqual({ everywhere: { allow: [], deny: [] }, projects: {} })
    b.set("agent.perms", { everywhere: { allow: ["Read", 5, "bad rule", "Read"], deny: "Bash" }, projects: { p: { allow: [], deny: [] } } })
    expect(loadPerms(b)).toEqual({ everywhere: { allow: ["Read"], deny: [] }, projects: {} })
  })
})
