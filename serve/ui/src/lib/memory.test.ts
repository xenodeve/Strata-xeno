import { describe, expect, test } from "bun:test"
import { INIT_PROMPT, isInitCommand, isMemoryCommand, sizeText, withSwitch } from "./memory"

describe("withSwitch", () => {
  test("switching on adds it once, switching off takes it away", () => {
    expect(withSwitch([], "claude:memory", true)).toEqual(["claude:memory"])
    expect(withSwitch(["claude:memory"], "claude:memory", true)).toEqual(["claude:memory"])
    expect(withSwitch(["claude:memory", "codex:instructions"], "claude:memory", false)).toEqual(["codex:instructions"])
    expect(withSwitch(["a"], "b", false)).toEqual(["a"])
  })
  test("it does not change what it was given", () => {
    const on = ["x"]
    withSwitch(on, "y", true)
    expect(on).toEqual(["x"])
  })
})

describe("sizeText", () => {
  test("bytes, then kilobytes, none for an unknown size", () => {
    expect(sizeText(512)).toBe("512 B")
    expect(sizeText(2048)).toBe("2.0 KB")
    expect(sizeText(20480)).toBe("20 KB")
    expect(sizeText(null)).toBe("")
  })
})

describe("the commands", () => {
  test("/init and /memory alone, in any case", () => {
    expect(isInitCommand("/init")).toBe(true)
    expect(isInitCommand("  /INIT ")).toBe(true)
    expect(isMemoryCommand("/memory")).toBe(true)
  })
  test("not with words after, not another name", () => {
    expect(isInitCommand("/init now")).toBe(false)
    expect(isInitCommand("/initialize")).toBe(false)
    expect(isMemoryCommand("/memory x")).toBe(false)
    expect(isMemoryCommand("see /memory")).toBe(false)
  })
  test("/init asks for a CLAUDE.md, to improve one that is there, and for no generic advice", () => {
    expect(INIT_PROMPT).toContain("CLAUDE.md")
    expect(INIT_PROMPT).toContain("improve it")
    expect(INIT_PROMPT).toContain("do not put in generic advice")
  })
})
