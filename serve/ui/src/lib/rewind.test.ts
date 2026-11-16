import { afterEach, describe, expect, test } from "bun:test"
import { applyRewind, forgetCheckpoints, isRewindCommand, previewRewind, rewindSummary, shortPath } from "./rewind"

;(globalThis as Record<string, unknown>).location ??= { pathname: "/next/", search: "" }
const realFetch = globalThis.fetch
afterEach(() => { globalThis.fetch = realFetch })
/** fetch replaced: what each request sent, answered with `answer`. */
function record(answer: unknown = { ok: true, known: true, files: [] }) {
  const sent: { url: string; body: unknown }[] = []
  globalThis.fetch = (async (u: string, init?: RequestInit) => {
    sent.push({ url: String(u), body: JSON.parse(String(init?.body ?? "null")) })
    return new Response(JSON.stringify(answer), { status: 200, headers: { "Content-Type": "application/json" } })
  }) as typeof fetch
  return sent
}

describe("#310 part 2: a rewind on My PC names the runner in the body", () => {
  test("preview, apply and forget carry runner when given", async () => {
    const sent = record()
    await previewRewind("s1", "123", "r0123456789abcdef")
    await applyRewind("s1", "123", true, "r0123456789abcdef")
    forgetCheckpoints("s1", "r0123456789abcdef")
    await new Promise((r) => setTimeout(r, 0))
    expect(sent.map((x) => x.url)).toEqual(["/agent/rewind", "/agent/rewind", "/agent/checkpoints/forget"])
    expect(sent[0].body).toEqual({ session: "s1", checkpoint: "123", runner: "r0123456789abcdef" })
    expect(sent[1].body).toEqual({ session: "s1", checkpoint: "123", apply: true, include_changed: true, runner: "r0123456789abcdef" })
    expect(sent[2].body).toEqual({ session: "s1", runner: "r0123456789abcdef" })
  })
  test("without one the body is today's", async () => {
    const sent = record()
    await previewRewind("s1", "123")
    await applyRewind("s1", "123", false)
    forgetCheckpoints("s1")
    await new Promise((r) => setTimeout(r, 0))
    expect(sent.map((x) => x.body)).toEqual([{ session: "s1", checkpoint: "123" }, { session: "s1", checkpoint: "123", apply: true, include_changed: false }, { session: "s1" }])
  })
})

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
