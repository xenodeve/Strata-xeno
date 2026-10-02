import { describe, expect, test } from "bun:test"
import { COMPACT_AT, MAX_SUMMARY, compactCommand, compactPrompt, continuationText, estimateTokens, shouldCompact, summaryOf, summaryRoom } from "./compact"

describe("summaryOf: the summary in what the model wrote", () => {
  test("what is inside <summary>, without the analysis", () => {
    expect(summaryOf("<analysis>thinking</analysis>\n<summary>\n1. Primary request: x\n</summary>")).toBe("1. Primary request: x")
  })
  test("tags in any case, and only the first summary", () => { expect(summaryOf("<SUMMARY>a</SUMMARY> and <summary>b</summary>")).toBe("a") })
  test("a summary cut off by the limit keeps what there is", () => { expect(summaryOf("<analysis>a</analysis><summary>1. Request: do x\n2. Concepts: y")).toBe("1. Request: do x\n2. Concepts: y") })
  test("no tags: the text, without an analysis", () => {
    expect(summaryOf("The user asked for x.")).toBe("The user asked for x.")
    expect(summaryOf("<analysis>notes</analysis>\nThe user asked for x.")).toBe("The user asked for x.")
  })
  test("an analysis that is never closed is not a summary", () => { expect(summaryOf("<analysis>notes that go on")).toBe("") })
  test("nothing: nothing", () => { expect(summaryOf("   ")).toBe("") })
})

describe("compactPrompt", () => {
  test("asks for the sections, with no tools", () => {
    const p = compactPrompt()
    for (const s of ["Primary request and intent", "Files and code", "Errors and fixes", "All user messages", "Pending tasks", "Current work", "Next step"]) expect(p).toContain(s)
    expect(p).toContain("Do not call any tools")
    expect(p).not.toContain("Additional instructions")
  })
  test("what the user typed after /compact is added", () => { expect(compactPrompt("  focus on the tests ")).toContain("Additional instructions from the user for this summary:\nfocus on the tests") })
})

describe("continuationText", () => {
  test("holds the summary", () => { expect(continuationText("S1")).toContain("\nS1\n") })
})

describe("compactCommand: /compact and what follows it", () => {
  test("the command, with or without a focus", () => {
    expect(compactCommand("/compact")).toEqual({ focus: "" })
    expect(compactCommand("  /compact  ")).toEqual({ focus: "" })
    expect(compactCommand("/compact focus on the tests\nand the build")).toEqual({ focus: "focus on the tests\nand the build" })
    expect(compactCommand("/COMPACT x")).toEqual({ focus: "x" })
  })
  test("not the command: another name, text before it, no space after it", () => {
    expect(compactCommand("/compacting")).toBeNull()
    expect(compactCommand("/pdf-skill")).toBeNull()
    expect(compactCommand("please /compact")).toBeNull()
    expect(compactCommand("")).toBeNull()
  })
})

describe("when and how much", () => {
  test("shouldCompact: at the share of the context, counting the prompt that is coming", () => {
    expect(shouldCompact(9499, 0, 10000)).toBe(false)
    expect(shouldCompact(9500, 0, 10000)).toBe(true)
    expect(shouldCompact(8500, 1000, 10000)).toBe(true)
    expect(COMPACT_AT).toBe(0.95)
  })
  test("no context size known: never by itself", () => { expect(shouldCompact(1e9, 0, 0)).toBe(false) })
  test("summaryRoom: what is left, up to the most a summary may take", () => {
    expect(summaryRoom(32768, 20000, 600)).toBe(MAX_SUMMARY)
    expect(summaryRoom(32768, 30000, 600)).toBe(32768 - 30000 - 600 - 24)
    expect(summaryRoom(8000, 7900, 600)).toBeLessThan(0)
    expect(summaryRoom(0, 1e6, 1e6)).toBe(MAX_SUMMARY)
  })
  test("estimateTokens: about 2.6 characters a token, objects by their text", () => {
    expect(estimateTokens("a".repeat(26))).toBe(10)
    expect(estimateTokens([{ role: "user", content: "hi" }])).toBeGreaterThan(5)
    expect(estimateTokens("")).toBe(0)
  })
})
