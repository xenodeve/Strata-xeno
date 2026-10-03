import { describe, expect, test } from "bun:test"
import { closeTrace, parseTrace, spansIn, topNames } from "./trace"

// The engine's STRATA_TIMELINE file: a Chrome trace array that is left OPEN (it appends after every request).
const OPEN = `[
{"ph":"M","pid":8,"tid":0,"name":"process_name","args":{"name":"strata 8"}},
{"ph":"M","pid":8,"tid":1,"name":"thread_name","args":{"name":"main"}},
{"ph":"M","pid":8,"tid":2,"name":"thread_name","args":{"name":"pool worker 0"}},
{"ph":"X","pid":8,"tid":1,"ts":1000.0,"dur":500.0,"name":"verify window"},
{"ph":"X","pid":8,"tid":2,"ts":1100.0,"dur":200.0,"name":"cpu experts","args":{"a":7,"b":2}},
{"ph":"X","pid":8,"tid":2,"ts":1350.0,"dur":100.0,"name":"cpu experts"},
{"ph":"X","pid":8,"tid":1,"ts":900.0,"dur":50.0,"name":"commit"},
{"ph":"X","pid":8,"tid":1,"ts":1600.0,"dur":0.0,"name":"instant-ish"},
`

describe("trace parsing", () => {
  test("an open array (no closing bracket, a trailing comma) parses", () => {
    const t = parseTrace(OPEN)
    expect(t.count).toBe(5)
    expect(t.lanes.map((l) => l.name)).toEqual(["main", "pool worker 0"])
  })
  test("a closed array parses too, and a half-written last line is dropped", () => {
    expect(parseTrace(OPEN + "]").count).toBe(5)
    expect(parseTrace(OPEN + '{"ph":"X","pid":8,"tid":1,"ts":17').count).toBe(5)       // the engine was killed mid-line
  })
  test("spans are sorted by start within a lane, and the time range covers all of them", () => {
    const t = parseTrace(OPEN)
    const main = t.lanes[0]
    expect(Array.from(main.ts)).toEqual([900, 1000, 1600])
    expect([t.t0, t.t1]).toEqual([900, 1600])
  })
  test("a lane without a name is called by its thread id, and not text at all is an empty trace", () => {
    const t = parseTrace('[{"ph":"X","pid":3,"tid":9,"ts":1,"dur":2,"name":"x"}')
    expect(t.lanes[0].name).toBe("thread 9")
    expect(parseTrace("").count).toBe(0)
    expect(() => parseTrace("not json at all")).toThrow()
  })
  test("closeTrace gives a valid JSON array of the same events (for ui.perfetto.dev)", () => {
    const closed = closeTrace(OPEN)
    expect(JSON.parse(closed)).toHaveLength(8)
    expect(closeTrace(OPEN + "]")).toBe(OPEN.trimEnd().replace(/,$/, "") + "\n]")        // already closed: unchanged in content
  })
})

describe("looking at a window", () => {
  const t = parseTrace(OPEN)
  test("spansIn finds what overlaps [a, b], including a long span that began before it", () => {
    const main = t.lanes[0]
    const [lo, hi] = spansIn(main, 1400, 1700)
    expect(Array.from(main.ts.slice(lo, hi))).toEqual([1000, 1600])                       // 1000..1500 overlaps 1400
    expect(spansIn(main, 2000, 3000)).toEqual([3, 3])                                      // nothing: an empty range
  })
  test("topNames adds up the time of each name inside the window, clipped to it", () => {
    const top = topNames(t, 1000, 1200, 5)
    const cpu = top.find((x) => x.name === "cpu experts")!
    expect(cpu).toMatchObject({ count: 1, total: 100 })                                    // 1100..1300 clipped to 1100..1200
    expect(top[0].name).toBe("verify window")                                              // 1000..1500 clipped to 200
    expect(top[0].total).toBe(200)
  })
})
