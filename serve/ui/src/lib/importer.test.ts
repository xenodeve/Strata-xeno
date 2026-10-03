import { describe, expect, test } from "bun:test"
import { filterItems, harnessOn, itemOn, setHarness, setItem, setMaster, skillSummary, type SkillItem, type SkillSettings } from "./importer"

// Skills of the other coding apps are on until switched off, at three levels, and the page keeps only what is OFF (an off-list), so a skill
// found later is on. These are the settings a switch sends, and the filter over a long list.
const ON: SkillSettings = { enabled: true, harness_off: [], off: {} }
const item = (harness: string, name: string, extra: Partial<SkillItem> = {}): SkillItem =>
  ({ id: `${harness}:${name}`, harness, name, description: `does ${name}`, origin: "user", off: false, used: true, same_as: null, ...extra })

describe("the switches", () => {
  test("everything is on with nothing in the off-lists", () => {
    expect(harnessOn(ON, "claude")).toBe(true)
    expect(itemOn(ON, "claude", "pdf-tools")).toBe(true)
  })
  test("the whole import: off and on again, the other choices kept", () => {
    const s = setItem(ON, "claude", "pdf-tools", false)
    const off = setMaster(s, false)
    expect(off.enabled).toBe(false)
    expect(off.off).toEqual({ claude: ["pdf-tools"] })
    expect(setMaster(off, true)).toEqual(s)
  })
  test("an app: its name goes in and out of the list, once", () => {
    const s = setHarness(setHarness(ON, "codex", false), "codex", false)
    expect(s.harness_off).toEqual(["codex"])
    expect(harnessOn(s, "codex")).toBe(false)
    expect(harnessOn(s, "claude")).toBe(true)
    expect(setHarness(s, "codex", true)).toEqual(ON)
  })
  test("a skill: switched off by app and name, and on again leaves no empty list behind", () => {
    const s = setItem(setItem(ON, "claude", "a", false), "claude", "b", false)
    expect(s.off).toEqual({ claude: ["a", "b"] })
    expect(itemOn(s, "claude", "a")).toBe(false)
    expect(itemOn(s, "codex", "a")).toBe(true)                       // the same name in another app is another switch
    const back = setItem(setItem(s, "claude", "a", true), "claude", "b", true)
    expect(back).toEqual(ON)
    expect(setItem(s, "claude", "a", false)).toEqual(s)               // already off: the same
  })
  test("the settings that come in are not changed", () => {
    const before = JSON.stringify(ON)
    setItem(ON, "claude", "a", false); setHarness(ON, "codex", false); setMaster(ON, false)
    expect(JSON.stringify(ON)).toBe(before)
  })
})

describe("the filter", () => {
  const items = [item("claude", "pdf-tools", { description: "Read and write PDF files" }), item("claude", "git-helper"), item("codex", "csv-tools", { description: "Clean CSV data" })]
  test("by a few letters of the name or of what it says, any case", () => {
    expect(filterItems(items, "pdf").map((i) => i.name)).toEqual(["pdf-tools"])
    expect(filterItems(items, "CSV").map((i) => i.name)).toEqual(["csv-tools"])
    expect(filterItems(items, "write files").map((i) => i.name)).toEqual(["pdf-tools"])          // every word, anywhere
    expect(filterItems(items, "  ")).toHaveLength(3)
    expect(filterItems(items, "zzz")).toEqual([])
  })
  test("by the app's name too", () => {
    expect(filterItems(items, "claude", { claude: "Claude Code" }).map((i) => i.name)).toEqual(["pdf-tools", "git-helper"])
  })
})

describe("the line that says how many", () => {
  test("in use of found, in how many apps", () => {
    const items = [item("claude", "a"), item("claude", "b"), item("codex", "a", { used: false, same_as: "claude:a" }), item("cursor", "c", { off: true, used: false })]
    expect(skillSummary(items)).toEqual({ used: 2, total: 4, apps: 3 })
    expect(skillSummary([])).toEqual({ used: 0, total: 0, apps: 0 })
  })
})
