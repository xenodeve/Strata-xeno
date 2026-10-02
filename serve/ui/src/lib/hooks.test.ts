import { describe, expect, test } from "bun:test"
import { eventText, noteFrom, noteText } from "./hooks"

const report = { hook: "h0123abcd", on: "before_tool", command: "./check.sh", ok: false, code: 2, blocked: true, timeout: false, error: null, text: "no commits to main", ms: 120 }

describe("what a hook did, from the server's stream", () => {
  test("a report becomes a note, with what the hook printed", () => {
    expect(noteFrom(report)).toEqual({ hook: "h0123abcd", on: "before_tool", command: "./check.sh", ok: false, code: 2, blocked: true, timeout: false, error: null, text: "no commits to main", ms: 120 })
  })

  test("something that is not a hook's report is left out, and odd fields are made safe", () => {
    expect(noteFrom({ on: "sometime" })).toBeNull()
    expect(noteFrom({})).toBeNull()
    expect(noteFrom({ on: "stop", command: 5, text: 7, code: "x", ok: "yes", ms: "slow" })).toEqual({ hook: "", on: "stop", command: "", ok: false, code: null, blocked: false, timeout: false, error: null, text: "", ms: 0 })
  })

  test("in a few words: stopped, out of time, could not run, ran, or the exit code", () => {
    const n = (more: Record<string, unknown>) => noteFrom({ ...report, blocked: false, ok: false, ...more })!
    expect(noteText(n({ blocked: true }))).toBe("A hook stopped this call")
    expect(noteText(n({ timeout: true, code: null }))).toBe("A hook ran out of time and was stopped")
    expect(noteText(n({ error: "it could not be started", code: null }))).toBe("A hook could not run")
    expect(noteText(n({ ok: true, code: 0 }))).toBe("A hook ran")
    expect(noteText(n({ code: 1 }))).toBe("A hook finished with exit code 1")
  })

  test("each event has a name", () => {
    expect(["before_tool", "after_tool", "prompt", "stop"].map((e) => eventText(e as never))).toEqual(["Before a tool call", "After a tool call", "When you send a prompt", "When the model has finished"])
  })
})
