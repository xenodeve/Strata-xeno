import { describe, expect, test } from "bun:test"
import { makeOrbClock } from "./index.es.js"

// The orb's own time (xeno's change to the vendored library): a change of speed must not make the picture jump.
describe("the orb's clock", () => {
  test("time adds up at the speed it passes at", () => {
    const c = makeOrbClock()
    c.start(10)
    expect(c.advance(11, 1)).toBeCloseTo(1)
    expect(c.advance(12, 1)).toBeCloseTo(2)
  })
  test("a new speed does not move what has already passed: no jump, no restart", () => {
    const c = makeOrbClock()
    c.start(100)
    expect(c.advance(101, 1)).toBeCloseTo(1)
    expect(c.advance(101.0001, 1.5)).toBeCloseTo(1.00015)             // the old way: 101.0001 * 1.5 = 151.5, a jump of fifty seconds
    expect(c.advance(102, 1.5)).toBeCloseTo(2.5, 3)
  })
  test("the same speed again changes nothing", () => {
    const a = makeOrbClock(), b = makeOrbClock()
    a.start(0); b.start(0)
    a.advance(1, 1); b.advance(1, 1)
    a.start(1)                                                         // an effect that ran again (a prop that did not matter changed)
    expect(a.advance(2, 1)).toBeCloseTo(b.advance(2, 1))
  })
  test("time that passed while it was stopped does not count", () => {
    const c = makeOrbClock()
    c.start(0)
    c.advance(1, 1)
    c.start(60)                                                        // it was hidden for a minute
    expect(c.advance(61, 1)).toBeCloseTo(2)
  })
  test("time stands still at speed 0, and a clock never started begins at zero", () => {
    const c = makeOrbClock()
    expect(c.advance(5, 1)).toBe(0)
    expect(c.advance(6, 0)).toBe(0)
  })
})
