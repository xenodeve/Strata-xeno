import { describe, expect, test } from "bun:test"
import { LiveTokens } from "./livetokens"

const gen = (generated: number, tok_s = 40) => ({ state: "generating", generated, tok_s })

describe("the tokens written so far, counted live", () => {
  test("nothing is shown before the model has written anything, or when no answer is being written", () => {
    const c = new LiveTokens()
    expect(c.shown(0)).toBeNull()
    c.sample(null, gen(50), 100)
    expect(c.shown(100)).toBeNull()
    c.sample("m1", { state: "reading" }, 200)
    expect(c.shown(200)).toBeNull()
  })

  test("a round's count is shown, and it runs on by the speed until the next reading", () => {
    const c = new LiveTokens()
    c.sample("m1", gen(100, 50), 1000)
    expect(c.shown(1000)).toEqual({ tokens: 100, tokS: 50 })
    expect(c.shown(1400)?.tokens).toBe(120)                    // 0.4 s at 50 tokens a second
    c.sample("m1", gen(130, 50), 1500)
    expect(c.shown(1500)?.tokens).toBe(130)                    // the reading replaces the estimate
  })

  test("it does not run on for ever when readings stop", () => {
    const c = new LiveTokens()
    c.sample("m1", gen(100, 50), 0)
    expect(c.shown(60_000)?.tokens).toBe(175)                  // at most 1.5 s of it
  })

  test("the rounds of one answer add up, when the server goes through reading the prompt between them", () => {
    const c = new LiveTokens()
    c.sample("m1", gen(80), 0)
    c.sample("m1", { state: "reading", generated: null }, 1000)      // a tool ran, the next round reads its prompt
    c.sample("m1", gen(30), 3000)
    expect(c.shown(3000)?.tokens).toBe(110)
    c.sample("m1", { state: "idle" }, 4000)
    expect(c.shown(4000)?.tokens).toBe(110)
  })

  test("and when the count simply starts again between two readings", () => {
    const c = new LiveTokens()
    c.sample("m1", gen(200), 0)
    c.sample("m1", gen(20), 500)
    expect(c.shown(500)?.tokens).toBe(220)
  })

  test("a different answer starts from nothing", () => {
    const c = new LiveTokens()
    c.sample("m1", gen(200), 0)
    c.sample("m2", gen(5), 100)
    expect(c.shown(100)?.tokens).toBe(5)
    c.sample(null, gen(5), 200)
    expect(c.shown(200)).toBeNull()
  })

  test("readings that are not numbers are ignored, and a missing speed stops the running on", () => {
    const c = new LiveTokens()
    c.sample("m1", { state: "generating", generated: null, tok_s: 30 }, 0)
    expect(c.shown(0)).toBeNull()
    c.sample("m1", { state: "generating", generated: 40, tok_s: null }, 0)
    expect(c.shown(1000)).toEqual({ tokens: 40, tokS: null })
    c.sample("m1", { state: "generating", generated: -3 as number, tok_s: 10 }, 10)
    expect(c.shown(10)?.tokens).toBe(40)
  })
})
