import { describe, expect, test } from "bun:test"
import { isRewindCommand, rewindSummary, shortPath } from "./rewind"

describe("shortPath", () => {
  test("the last two parts, either separator", () => {
    expect(shortPath("C:\\work\\app\\src\\a.py")).toBe("src/a.py")
    expect(shortPath("/home/me/app/b.txt")).toBe("app/b.txt")
    expect(shortPath("c.txt")).toBe("c.txt")
  })
})

describe("rewindSummary", () => {
  test("counts what was done and what was left", () => {
    expect(rewindSummary({ ok: true, restored: 2, deleted: 1, kept: ["a"], failed: [], skipped: ["b", "c"] })).toEqual({ restored: 2, deleted: 1, kept: 1, failed: 0 })
  })
})

describe("isRewindCommand", () => {
  test("/rewind alone, in any case", () => {
    expect(isRewindCommand("/rewind")).toBe(true)
    expect(isRewindCommand("  /Rewind ")).toBe(true)
  })
  test("not with words after, not another name", () => {
    expect(isRewindCommand("/rewind now")).toBe(false)
    expect(isRewindCommand("/rewinds")).toBe(false)
  })
})
