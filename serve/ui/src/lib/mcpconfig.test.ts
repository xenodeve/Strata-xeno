import { describe, expect, test } from "bun:test"
import { MASK, bodyFor, emptyDraft, entryOf, entryFromRow, limitsOf, parseKv, parsePaste, toDraft, type McpRow } from "./mcpconfig"

const row = (over: Partial<McpRow>): McpRow => ({ name: "fs", kind: "program", command: "npx", disabled: false, source: "config", editable: true, status: "ready", error: null, tools: [], ...over })

describe("key=value lines", () => {
  test("the first = splits, blanks are skipped, a line with no key is reported", () => {
    expect(parseKv("A=1\n\nB=x=y\r\nC=")).toEqual({ map: { A: "1", B: "x=y", C: "" }, bad: [] })
    expect(parseKv("A=1\n=2\nnoequals").bad).toEqual([2, 3])
  })
})

describe("a draft to an entry", () => {
  test("a program: arguments one per line, env as lines, empty parts left out", () => {
    const d = { ...emptyDraft(), name: "fs", command: " npx ", args: "-y\n@x/server\n\n", env: "TOKEN=abc" }
    expect(entryOf(d)).toEqual({ entry: { command: "npx", args: ["-y", "@x/server"], env: { TOKEN: "abc" } }, problems: [] })
  })
  test("an address with headers", () => {
    const d = { ...emptyDraft(), name: "web", kind: "address" as const, url: "https://h/mcp", headers: "Authorization=Bearer q" }
    expect(entryOf(d).entry).toEqual({ url: "https://h/mcp", headers: { Authorization: "Bearer q" } })
  })
  test("each wrong part is named by its field", () => {
    const fields = (d: Parameters<typeof entryOf>[0], taken: string[] = []) => entryOf(d, taken).problems.map((p) => p.field)
    expect(fields({ ...emptyDraft(), name: "two words", command: "x" })).toEqual(["name"])
    expect(fields({ ...emptyDraft(), name: "fs", command: "x" }, ["fs"])).toEqual(["name"])
    expect(fields({ ...emptyDraft(), name: "fs" })).toEqual(["command"])
    expect(fields({ ...emptyDraft(), name: "fs", kind: "address", url: "ftp://h" })).toEqual(["url"])
    expect(fields({ ...emptyDraft(), name: "fs", command: "x", env: "oops" })).toEqual(["env"])
    expect(fields({ ...emptyDraft(), name: "fs", command: "x", args: Array(65).fill("a").join("\n") })).toEqual(["args"])
  })
  test("editing a server: its masked secret goes back as the mask, which keeps the stored one", () => {
    const r = row({ env: { TOKEN: MASK }, args: ["-y"] })
    const d = toDraft(r)
    expect(d.env).toBe(`TOKEN=${MASK}`)
    expect(entryOf(d).entry.env).toEqual({ TOKEN: MASK })
  })
})

describe("the body of a save", () => {
  const rows = [row({ name: "a" }), row({ name: "b", disabled: true, env: { K: MASK } }), row({ name: "theirs", source: "file", editable: false })]
  test("only the run config's own servers, with their fields and no state", () => {
    expect(bodyFor(rows)).toEqual({ servers: { a: { command: "npx" }, b: { command: "npx", env: { K: MASK }, disabled: true } } })
    expect(entryFromRow(row({ status: "failed", error: "x", tools: [{ tool: "t" }] }))).toEqual({ command: "npx" })
  })
  test("a change adds, replaces or deletes one, and the limits go along when given", () => {
    expect(Object.keys(bodyFor(rows, { name: "c", entry: { url: "http://h/mcp" } }).servers)).toEqual(["a", "b", "c"])
    expect(Object.keys(bodyFor(rows, { name: "a", entry: null }).servers)).toEqual(["b"])
    expect(bodyFor(rows, undefined, { max_rounds: 3 }).settings).toEqual({ max_rounds: 3 })
  })
  test("a disabled server stays disabled when another one is changed", () => {
    expect(bodyFor(rows, { name: "a", entry: null }).servers.b.disabled).toBe(true)
  })
})

describe("the Claude Desktop block", () => {
  test("with and without the wrapper", () => {
    const inner = { fs: { command: "npx", args: ["-y", "pkg"], env: { A: "1" } }, web: { url: "http://h/mcp" } }
    for (const text of [JSON.stringify({ mcpServers: inner }), JSON.stringify(inner)]) {
      const p = parsePaste(text)
      expect(p.problem).toBeNull()
      expect(p.entries).toEqual({ fs: { command: "npx", args: ["-y", "pkg"], env: { A: "1" } }, web: { url: "http://h/mcp" } })
    }
  })
  test("what is not a block says why, and nothing is taken", () => {
    expect(parsePaste("{ not json").problem).toContain("not valid JSON")
    expect(parsePaste("{}").problem).toContain("no server")
    expect(parsePaste("[1]").problem).toContain("no server")
    expect(parsePaste(JSON.stringify({ mcpServers: { a: "x" } })).problem).toContain("not a server")
    expect(parsePaste(JSON.stringify({ mcpServers: { "a b": { command: "x" } } })).problem).toContain("a b:")
    expect(parsePaste(JSON.stringify({ mcpServers: { fs: { command: "x" } } }), ["fs"]).problem).toContain("fs:")
    expect(parsePaste(JSON.stringify({ mcpServers: { fs: {} } })).entries).toEqual({})
  })
})

describe("the limits", () => {
  test("numbers or the first that is not one", () => {
    expect(limitsOf({ timeout_s: "30", max_result_chars: "5000", max_rounds: "4" })).toEqual({ limits: { timeout_s: 30, max_result_chars: 5000, max_rounds: 4 }, bad: null })
    expect(limitsOf({ timeout_s: "30", max_result_chars: "", max_rounds: "4" })).toEqual({ limits: null, bad: "max_result_chars" })
    expect(limitsOf({ timeout_s: "x", max_result_chars: "1", max_rounds: "1" }).bad).toBe("timeout_s")
  })
})
