import { describe, expect, test } from "bun:test"
import { agentStatus, readProgress, type StatusInput } from "./status"

// The design principle (AGENTS.md, "Web app: always show what the model is doing now"): at every moment of an answer there is one status in words, and a step that ends gives way to the next one's
// words, never to silence.
const base: StatusInput = { streaming: true, reasoning: "", text: "", tools: [] }
const tool = (name: string, state: string, more: Record<string, unknown> = {}) => ({ name, state, at: 0, rat: 0, ...more })
const kind = (i: Partial<StatusInput>) => agentStatus({ ...base, ...i })?.kind ?? null
const label = (i: Partial<StatusInput>) => agentStatus({ ...base, ...i })?.label ?? null

describe("the status of an answer being written", () => {
  test("there is none when nothing is being written, or nothing has happened yet (the page's own waiting line covers that)", () => {
    expect(kind({ streaming: false, text: "done", tools: [tool("Read", "done")] })).toBeNull()
    expect(kind({})).toBeNull()
  })

  test("without tools: thinking, then answering", () => {
    expect(kind({ reasoning: "hmm" })).toBe("thinking")
    expect(kind({ reasoning: "hmm", text: "The" })).toBe("composing")
    expect(label({ text: "The" })).toBe("Answering…")
  })

  test("a call being written, a tool running, a helper working: each says what", () => {
    expect(label({ tools: [tool("Bash", "writing")] })).toBe("Writing the call to Bash…")
    expect(label({ tools: [tool("Bash", "running")] })).toBe("Running Bash…")
    expect(label({ tools: [tool("Task", "running")] })).toBe("A helper is working…")
    expect(kind({ tools: [tool("Bash", "writing")] })).toBe("writing")
    expect(kind({ tools: [tool("Bash", "running")] })).toBe("running")
  })

  test("the last of several running calls is the one named", () => {
    expect(label({ tools: [tool("Read", "done"), tool("Bash", "running")] })).toBe("Running Bash…")
  })

  test("a question for the user, auto mode checking, a hook running: each has its own words, and they come before the tool's", () => {
    expect(kind({ tools: [tool("Bash", "asking")] })).toBe("asking")
    expect(label({ tools: [tool("Bash", "asking")] })).toBe("Waiting for you…")
    expect(kind({ tools: [tool("Bash", "running", { judging: true })] })).toBe("judging")
    expect(label({ tools: [tool("Bash", "running", { judging: true })] })).toBe("Auto mode is checking the call…")
    expect(kind({ tools: [tool("Bash", "running", { hookRunning: "./check.sh" })] })).toBe("hook")
    expect(label({ tools: [tool("Bash", "running", { hookRunning: "./check.sh" })] })).toBe("Running your hook…")
    expect(kind({ tools: [tool("Read", "done", { hookRunning: "npm run lint" })] })).toBe("hook")             // an after hook runs once the tool is done
  })

  test("a hook of the answer itself (a prompt or an end hook) is a status too", () => {
    expect(kind({ text: "all done", hookRunning: "npm test" })).toBe("hook")
    expect(kind({ tools: [tool("Read", "done")], text: "x", hookRunning: "npm test" })).toBe("hook")
  })

  test("once a tool has answered the page says it is reading the result, not that the tool is still running", () => {
    expect(kind({ tools: [tool("Bash", "done")], serverState: "reading" })).toBe("reading")
    expect(label({ tools: [tool("Bash", "done")], serverState: "reading" })).toBe("Reading the tool's result…")
    expect(kind({ tools: [tool("Bash", "error")], serverState: "idle" })).toBe("reading")           // the next request has not begun yet: still the result being taken in
    expect(kind({ tools: [tool("Bash", "done")], serverState: undefined })).toBe("reading")
  })

  test("then, while the model writes, it is planning the next step", () => {
    expect(kind({ tools: [tool("Bash", "done")], serverState: "generating" })).toBe("planning")
    expect(label({ tools: [tool("Bash", "done")], serverState: "generating" })).toBe("Planning the next step…")
  })

  test("what the model did after the tool decides: thinking again, or writing, even when it wrote something before the tool", () => {
    // text before the call (at = 12) does not make it "answering" while the result is being read
    const before = { text: "Let me check", tools: [tool("Read", "done", { at: 12, rat: 0 })], reasoning: "" }
    expect(kind({ ...before, serverState: "reading" })).toBe("reading")
    expect(kind({ ...before, reasoning: "now what", serverState: "generating" })).toBe("thinking")
    expect(kind({ ...before, reasoning: "now what", text: "Let me check\n\nThe file", serverState: "generating" })).toBe("composing")
    // thinking before the call (rat = 4) is not the thinking after it
    expect(kind({ reasoning: "plan", tools: [tool("Read", "done", { at: 0, rat: 4 })], serverState: "reading" })).toBe("reading")
  })

  test("it never goes quiet from the first tool to the end: every moment of a run has words", () => {
    const run: StatusInput[] = [
      { ...base, reasoning: "plan" },                                                                                           // thinking
      { ...base, reasoning: "plan", tools: [tool("Read", "writing", { rat: 4 })] },                                              // the call is written
      { ...base, reasoning: "plan", tools: [tool("Read", "running", { rat: 4 })] },                                              // it runs
      { ...base, reasoning: "plan", tools: [tool("Read", "done", { rat: 4 })], serverState: "idle" },                           // done; the next request is being prepared
      { ...base, reasoning: "plan", tools: [tool("Read", "done", { rat: 4 })], serverState: "reading" },                        // the result is read
      { ...base, reasoning: "plan", tools: [tool("Read", "done", { rat: 4 })], serverState: "generating" },                     // the model writes
      { ...base, reasoning: "plan next", tools: [tool("Read", "done", { rat: 4 })], serverState: "generating" },                // it thinks
      { ...base, reasoning: "plan next", text: "Here it is", tools: [tool("Read", "done", { rat: 4 })], serverState: "generating" },   // it answers
    ]
    expect(run.map((i) => agentStatus(i)?.kind)).toEqual(["thinking", "writing", "running", "reading", "reading", "planning", "thinking", "composing"])
    for (const i of run) expect(agentStatus(i)).not.toBeNull()
  })

  test("each status has an orb form and words", () => {
    for (const i of [{ tools: [tool("Bash", "running")] }, { tools: [tool("Bash", "asking")] }, { tools: [tool("Read", "done")], serverState: "reading" }, { text: "x" }]) {
      const s = agentStatus({ ...base, ...i })!
      expect(s.label.length).toBeGreaterThan(3)
      expect(typeof s.design).toBe("string")
    }
  })
})

describe("how far the reading is", () => {
  test("a share and the tokens read of the tokens to read", () => {
    expect(readProgress(4200, 10000)).toEqual({ read: 4200, total: 10000, percent: 42 })
    expect(readProgress(0, 500)).toEqual({ read: 0, total: 500, percent: 0 })
  })
  test("it never says 100 % while it is still reading, and cannot read more than there is", () => {
    expect(readProgress(9999, 10000)?.percent).toBe(99)
    expect(readProgress(10000, 10000)?.percent).toBe(99)
    expect(readProgress(12000, 10000)).toEqual({ read: 10000, total: 10000, percent: 99 })
  })
  test("nothing when the server does not say", () => {
    for (const [r, t] of [[null, 100], [5, null], [undefined, undefined], [5, 0], [-1, 100], [5, -3]] as const) expect(readProgress(r, t)).toBeNull()
  })
})

