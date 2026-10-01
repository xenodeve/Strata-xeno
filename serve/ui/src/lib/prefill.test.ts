import { describe, expect, test } from "bun:test"
import { PrefillMeter, prefillText } from "./prefill"

// The engine reports how far it has read the prompt, one number per chunk. The position counts a reused (cached) prefix as
// read, so the first number is a baseline, never a speed. The speed shown while it reads is the mean over the last second.
describe("prefill meter", () => {
  test("no speed from one sample, and none while nothing has moved", () => {
    const m = new PrefillMeter()
    m.push(0, 500)
    expect(m.rate()).toBeNull()
    m.push(500, 500)
    expect(m.rate()).toBeNull()
  })

  test("the mean over the last second, from the newest sample back", () => {
    const m = new PrefillMeter()
    m.push(0, 100)                                // 100 of the prompt were cached: the baseline
    m.push(500, 600)
    expect(m.rate()).toBe(1000)                   // 500 tokens in 0.5 s
    m.push(1000, 1100)
    expect(m.rate()).toBe(1000)
    m.push(1500, 2100)                            // speeds up: the window is [500, 1500] = 1500 tokens in 1 s
    expect(m.rate()).toBe(1500)
  })

  test("chunks arrive slower than the window: it reaches back to the last chunk, so the speed is not zero in between", () => {
    const m = new PrefillMeter()
    m.push(0, 0)
    m.push(500, 2000)                             // one 2000-token chunk
    m.push(1000, 2000)
    m.push(1500, 2000)
    expect(m.rate()).toBeCloseTo(2000 / 1.5, 6)   // 2000 tokens since the sample before the chunk
  })

  test("the mean of the whole read excludes the baseline", () => {
    const m = new PrefillMeter()
    m.push(0, 4000)
    m.push(1000, 5000)
    m.push(3000, 9000)
    expect(m.mean()).toBeCloseTo(5000 / 3, 6)
  })
})

describe("the line under the prompt", () => {
  test("while it reads: the last-second speed, or just that it reads", () => {
    expect(prefillText({ state: "reading", rate: null, mean: null, read: null, cached: null })).toBe("Reading the prompt…")
    expect(prefillText({ state: "reading", rate: 1240.4, mean: null, read: null, cached: null })).toMatch(/^Reading the prompt · 1,?240 tok\/s · last second$/)
  })
  test("when done: the mean, what was read and what the cache held, never added together", () => {
    const t = prefillText({ state: "done", rate: 900, mean: 1180.2, read: 2400, cached: 5000 })!
    expect(t).toMatch(/^Prefill 1,?180 tok\/s mean · 2,?400 tokens read · 5,?000 from the cache$/)
    expect(prefillText({ state: "done", rate: null, mean: 800, read: 300, cached: 0 })).toMatch(/^Prefill 800 tok\/s mean · 300 tokens read$/)
  })
  test("a prompt the cache held whole has nothing to read and no speed", () => {
    expect(prefillText({ state: "done", rate: null, mean: null, read: 0, cached: 5000 })).toMatch(/^Nothing to read: all 5,?000 tokens came from the cache$/)
  })
  test("nothing is shown when nothing was measured", () => {
    expect(prefillText({ state: "done", rate: null, mean: null, read: null, cached: null })).toBeNull()
    expect(prefillText({ state: "done", rate: null, mean: null, read: 300, cached: 0 })).toMatch(/^300 tokens read$/)
    expect(prefillText({ state: "done", rate: null, mean: 1000, read: null, cached: null })).toBe("Prefill 1,000 tok/s mean")
  })
})
