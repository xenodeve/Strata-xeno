import { beforeAll, describe, expect, test } from "bun:test"
import { messageSig } from "./msgsig"
import { appendPage, mergeNewest } from "./requestlist"
import { feedTrails, trailOf } from "./trails"

// What lets the page leave things as they are (a message that did not change is not drawn again), and keep what it had (a list, a graph, a draft) when another page is opened and this one is opened again.
type Message = import("./chat").Message
type Metrics = import("./metrics").Metrics

const answer = (): Message => ({ role: "assistant", text: "Hello", reasoning: "thinking", time: 1, tools: [{ id: "c1", name: "Read", at: 5, rat: 3, state: "running" }] })

describe("what a message looks like (its signature)", () => {
  test("it is the same while nothing changes, and different at each thing the message draws", () => {
    const m = answer()
    const first = messageSig(m)
    expect(messageSig(m)).toBe(first)
    const changes: ((m: Message) => void)[] = [
      (m) => { m.text += " world" },
      (m) => { m.reasoning += "!" },
      (m) => { m.tools![0].state = "done" },
      (m) => { m.tools![0].open = true },
      (m) => { m.tools![0].result = "ok" },
      (m) => { m.tools![0].ms = 12 },
      (m) => { m.tools![0].doneAt = 99 },
      (m) => { m.tools![0].ask = { id: "a", tool: "Bash", why: "", danger: false, rule: null, answer: "allow" } },
      (m) => { m.tools![0].question = { id: "q", questions: [], answers: { a: ["x"] } } },
      (m) => { m.tools![0].hooks = [{ hook: "h", on: "after_tool", command: "c", ok: true, code: 0, blocked: false, timeout: false, text: "", ms: 1 }] },
      (m) => { m.tools![0].steps = [{ id: "s", name: "Read", state: "running" }] },
      (m) => { m.tools!.push({ id: "c2", name: "Edit", at: 5, rat: 3, state: "writing" }) },
      (m) => { m.error = "failed" },
      (m) => { m.stopped = true },
      (m) => { m.thinkSecs = 2.5 },
      (m) => { m.stats = { tokens: 10, tokS: 40 } },
      (m) => { m.todos = [{ content: "a", status: "pending", activeForm: "doing a" }] },
      (m) => { m.prefill = { state: "done", rate: null, mean: 100, read: 5, cached: 0 } },
      (m) => { m.hookRunning = "echo" },
    ]
    for (const change of changes) {
      const m2 = answer()
      change(m2)
      expect(messageSig(m2)).not.toBe(first)
    }
  })

  test("a long result is counted by its length, not read, so a signature is cheap for every message at every draw", () => {
    const m = answer()
    m.tools![0].result = "x".repeat(20000)
    const t0 = performance.now()
    for (let i = 0; i < 200; i++) messageSig(m)
    expect(performance.now() - t0).toBeLessThan(200)
  })
})

describe("the list of requests, kept while the page is left", () => {
  const rows = (...ids: number[]) => ids.map((n) => ({ id: `r${n}` }))
  test("the newest page is put in front, and what was kept below it stays", () => {
    const kept = rows(10, 9, 8, 7, 6, 5, 4, 3)
    const r = mergeNewest(kept, rows(12, 11, 10, 9))
    expect(r.rows.map((x) => x.id)).toEqual(["r12", "r11", "r10", "r9", "r8", "r7", "r6", "r5", "r4", "r3"])
    expect(r.reset).toBe(false)
  })
  test("when so many came that the newest page does not reach what was kept, the list starts again from it", () => {
    const r = mergeNewest(rows(10, 9, 8), rows(30, 29, 28))
    expect(r.rows.map((x) => x.id)).toEqual(["r30", "r29", "r28"])
    expect(r.reset).toBe(true)
  })
  test("nothing was kept: it is just the page, and it is not a reset", () => {
    expect(mergeNewest([], rows(3, 2, 1))).toEqual({ rows: rows(3, 2, 1), reset: false })
  })
  test("a page further down does not show a row twice", () => {
    expect(appendPage(rows(5, 4, 3), rows(3, 2, 1)).map((x) => x.id)).toEqual(["r5", "r4", "r3", "r2", "r1"])
  })
})

describe("the disks' graphs, fed by every reading", () => {
  const reading = (time: number, read: number): Metrics => ({ time, hardware: { disks: [{ index: 0, read_mb: read, write_mb: 1, read_ms_op: 0.5 }, { index: 1, read_mb: null, write_mb: null, read_ms_op: null }] } } as unknown as Metrics)
  test("they go on from one reading to the next, a reading counts once, and a disk with no figure keeps its place as a gap", () => {
    feedTrails(reading(1, 10)); feedTrails(reading(1, 10)); feedTrails(reading(2, 20))
    expect(trailOf(0).read).toEqual([10, 20])
    expect(trailOf(1).read).toEqual([null, null])
    expect(trailOf(7).read).toEqual([])
  })
  test("only the last two minutes (120 readings) are kept", () => {
    for (let i = 3; i < 200; i++) feedTrails(reading(i, i))
    expect(trailOf(0).read.length).toBe(120)
    expect(trailOf(0).read.at(-1)).toBe(199)
  })
})

describe("the controller keeps what the page cannot", () => {
  let ChatController: typeof import("./chat").ChatController
  beforeAll(async () => {
    const g = globalThis as Record<string, unknown>
    g.localStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} }
    g.requestAnimationFrame = (f: () => void) => setTimeout(f, 0)
    g.cancelAnimationFrame = clearTimeout
    g.location = { pathname: "/next/", search: "" }
    ;({ ChatController } = await import("./chat"))
  })
  test("the draft is the controller's: a new page reads what the last one left", () => {
    const c = new ChatController()
    expect(c.draft).toEqual({ text: "", files: [] })
    c.draft.text = "half a sentence"
    c.draft.files = [{ kind: "file", name: "a.txt", text: "x" }]
    expect(c.draft.text).toBe("half a sentence")
    expect(c.draft.files.length).toBe(1)
  })
  test("the list of conversations changes for the list, not for the words of an answer", () => {
    const c = new ChatController()
    const before = c.listSig()
    c.notify()
    c.notify()
    expect(c.listSig()).toBe(before)                           // a notice is not a change of the list
    c.newSession()
    c.index.items.push({ id: "z", title: "A new chat", time: 5 })
    expect(c.listSig()).not.toBe(before)
  })
})
