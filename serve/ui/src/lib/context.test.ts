import { describe, expect, test } from "bun:test"
import { contextView, isContextCommand, weigh } from "./context"
import type { Message } from "./chat"

const u = (text: string): Message => ({ role: "user", text, time: 1 })
const a = (text: string, extra: Partial<Message> = {}): Message => ({ role: "assistant", text, time: 2, ...extra })

describe("weigh: what the messages are made of", () => {
  test("what was said, the tools' calls and results, and the summary, apart", () => {
    const w = weigh([
      { role: "user", text: "S".repeat(260), time: 0, compact: { before: 1, after: 1, auto: false } },
      u("x".repeat(26)),
      a("y".repeat(26), { tools: [{ id: "c", name: "Read", at: 0, rat: 0, state: "done", arguments: { file_path: "a" }, result: "r".repeat(52) }] }),
    ])
    expect(w.summary).toBe(100)
    expect(w.conversation).toBe(20)
    expect(w.tools).toBeGreaterThan(20)
  })
  test("a tool that did not run, and an answer that failed, count for nothing", () => {
    const w = weigh([u("hi"), a("x".repeat(500), { error: "boom" }), a("ok", { tools: [{ id: "c", name: "Bash", at: 0, rat: 0, state: "skipped", result: "z".repeat(500) }] })])
    expect(w.tools).toBe(0)
    expect(w.conversation).toBeLessThan(10)
  })
  test("a file in a prompt counts", () => { expect(weigh([{ role: "user", text: "see", time: 0, files: [{ name: "f", text: "q".repeat(260) }] }]).conversation).toBeGreaterThan(90) })
})

describe("contextView", () => {
  const msgs = [u("x".repeat(260)), a("y".repeat(260))]
  test("what the server reported is the figure, and what it added is the rest", () => {
    const v = contextView(msgs, 1000, 4000, true)
    expect(v).toMatchObject({ known: true, max: 4000, used: 1000, exact: true, pct: 25, autoAt: 3200, level: "ok" })
    expect(v.parts.find((p) => p.key === "system")!.tokens).toBe(1000 - 200)
    expect(v.parts.find((p) => p.key === "free")!.tokens).toBe(3000)
    expect(v.parts.reduce((n, p) => n + p.tokens, 0)).toBe(4000)
  })
  test("with nothing reported it is a guess from the text, and says so", () => {
    const v = contextView(msgs, null, 4000, true)
    expect(v.exact).toBe(false)
    expect(v.used).toBe(200)
    expect(v.parts.find((p) => p.key === "system")!.tokens).toBe(0)
  })
  test("levels: under 60 %, from 60 %, and at the compacting point", () => {
    expect(contextView(msgs, 2300, 4000, true).level).toBe("ok")
    expect(contextView(msgs, 2400, 4000, true).level).toBe("warn")
    expect(contextView(msgs, 3200, 4000, true).level).toBe("full")
  })
  test("with automatic compacting off there is no point, and no \"full\"", () => {
    const v = contextView(msgs, 3900, 4000, false)
    expect(v.autoAt).toBeNull()
    expect(v.level).toBe("warn")
  })
  test("a window that is not known: no share, no point, nothing free", () => {
    const v = contextView(msgs, 500, 0, true)
    expect(v).toMatchObject({ known: false, pct: 0, autoAt: null, level: "ok" })
    expect(v.parts.find((p) => p.key === "free")!.tokens).toBe(0)
  })
  test("more than the window is shown as 100 %, with nothing free", () => {
    const v = contextView(msgs, 5000, 4000, true)
    expect(v.pct).toBe(100)
    expect(v.parts.find((p) => p.key === "free")!.tokens).toBe(0)
  })
})

describe("isContextCommand", () => {
  test("/context alone, in any case", () => {
    expect(isContextCommand("/context")).toBe(true)
    expect(isContextCommand("  /Context  ")).toBe(true)
  })
  test("not with words after it, not another name", () => {
    expect(isContextCommand("/context now")).toBe(false)
    expect(isContextCommand("/contexts")).toBe(false)
    expect(isContextCommand("see /context")).toBe(false)
  })
})
