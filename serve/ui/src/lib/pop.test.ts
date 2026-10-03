import { describe, expect, test } from "bun:test"
import { pace, popRuns, reelMove, staggerOf } from "./pop"

// Numbers in a line of text pop in when they change (transitions.dev "Number pop-in"): the text is cut into plain runs and number
// runs, and the last two characters of a number come in a step behind the others.
describe("pop runs", () => {
  test("a line is cut into words and numbers, with their separators kept inside the number", () => {
    expect(popRuns("Reading the prompt · 1,240.5 tok/s · last 1 second")).toEqual([
      { num: false, text: "Reading the prompt · " }, { num: true, text: "1,240.5" }, { num: false, text: " tok/s · last " },
      { num: true, text: "1" }, { num: false, text: " second" },
    ])
  })
  test("a full stop or a comma after a number is not part of it", () => {
    expect(popRuns("5 tokens read, 12.")).toEqual([{ num: true, text: "5" }, { num: false, text: " tokens read, " }, { num: true, text: "12" }, { num: false, text: "." }])
  })
  test("Thai text with numbers, and text with none", () => {
    expect(popRuns("อ่านแล้ว 303 token")).toEqual([{ num: false, text: "อ่านแล้ว " }, { num: true, text: "303" }, { num: false, text: " token" }])
    expect(popRuns("idle")).toEqual([{ num: false, text: "idle" }])
    expect(popRuns("")).toEqual([])
  })
})

describe("stagger", () => {
  test("the last character is two steps behind, the one before it one step, the rest none", () => {
    expect([0, 1, 2, 3].map(staggerOf)).toEqual([2, 1, 0, 0])
  })
})

// A reel (transitions.dev "Spinning counter") turns the way the figure moved: up when the number grew, down when it fell.
describe("reel move", () => {
  test("a figure that grew turns the reel up, wrapping past 9", () => {
    expect(reelMove(3, 7, true)).toBe(4)
    expect(reelMove(7, 3, true)).toBe(6)
    expect(reelMove(9, 0, true)).toBe(1)
  })
  test("a figure that fell turns it down, wrapping past 0", () => {
    expect(reelMove(7, 3, false)).toBe(-4)
    expect(reelMove(3, 7, false)).toBe(-6)
    expect(reelMove(0, 9, false)).toBe(-1)
  })
  test("a digit that did not change does not move", () => {
    expect(reelMove(5, 5, true)).toBe(0)
    expect(reelMove(5, 5, false)).toBe(0)
  })
})

// A figure that changes often must stay readable: the faster it changes, the shorter its move, and past a point it does not move at all.
describe("pace", () => {
  test("a figure that was still for a good while moves as it was made to", () => {
    expect(pace(null, 320)).toEqual({ animate: true, dur: 320 })
    expect(pace(2000, 320)).toEqual({ animate: true, dur: 320 })
    expect(pace(2000, 450)).toEqual({ animate: true, dur: 450 })
  })
  test("one that changes every half second moves in a short part of that time", () => {
    expect(pace(500, 320)).toEqual({ animate: true, dur: 200 })
    expect(pace(500, 450)).toEqual({ animate: true, dur: 200 })
  })
  test("the move gets shorter the faster it changes, but not below what the eye can follow", () => {
    expect(pace(300, 320).dur).toBe(120)
    expect(pace(150, 320)).toEqual({ animate: true, dur: 90 })
  })
  test("one that changes faster than that does not move: the digits just show", () => {
    expect(pace(149, 320)).toEqual({ animate: false, dur: 0 })
    expect(pace(40, 450)).toEqual({ animate: false, dur: 0 })
  })
})
