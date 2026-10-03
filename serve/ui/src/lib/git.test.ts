import { describe, expect, test } from "bun:test"
import { baseName, parseDiff } from "./git"

describe("parseDiff: a unified diff as numbered lines", () => {
  const diff = [
    "diff --git a/a.txt b/a.txt", "index 111..222 100644", "--- a/a.txt", "+++ b/a.txt",
    "@@ -1,3 +1,3 @@ function f()", " one", "-two", "+TWO", " three",
    "@@ -10,2 +10,3 @@", " ten", "+eleven", " twelve", "\\ No newline at end of file", "",
  ].join("\n")
  test("the header is left out, a hunk says where it is, lines are numbered in the old and the new file", () => {
    const rows = parseDiff(diff)
    expect(rows[0]).toEqual({ kind: "hunk", text: "@@ function f()" })
    expect(rows.slice(1, 5)).toEqual([
      { kind: "ctx", text: "one", old: 1, now: 1 }, { kind: "del", text: "two", old: 2 }, { kind: "add", text: "TWO", now: 2 }, { kind: "ctx", text: "three", old: 3, now: 3 },
    ])
  })
  test("a second hunk starts its own numbers, and the \"no newline\" note is a note", () => {
    const rows = parseDiff(diff)
    const second = rows.findIndex((r, i) => i > 0 && r.kind === "hunk")
    expect(rows[second]).toEqual({ kind: "hunk", text: "@@" })
    expect(rows[second + 1]).toEqual({ kind: "ctx", text: "ten", old: 10, now: 10 })
    expect(rows[second + 2]).toEqual({ kind: "add", text: "eleven", now: 11 })
    expect(rows[second + 3]).toEqual({ kind: "ctx", text: "twelve", old: 11, now: 12 })
    expect(rows.at(-1)).toEqual({ kind: "note", text: "No newline at end of file" })
  })
  test("an added file (/dev/null) counts from one", () => {
    const rows = parseDiff("--- /dev/null\n+++ b/n.txt\n@@ -0,0 +1,2 @@\n+fresh\n+lines\n")
    expect(rows.map((r) => [r.kind, r.now])).toEqual([["hunk", undefined], ["add", 1], ["add", 2]])
  })
  test("nothing, or text with no hunk: no lines", () => {
    expect(parseDiff("")).toEqual([])
    expect(parseDiff("Binary files a/x and b/x differ\n")).toEqual([])
  })
  test("a line that starts with --- inside a hunk is a removed line, not a header", () => {
    const rows = parseDiff("@@ -1,2 +1,1 @@\n---- not a header\n keep\n")
    expect(rows[1]).toEqual({ kind: "del", text: "--- not a header", old: 1 })
  })
})

describe("baseName", () => {
  test("the last part of a path, either separator, a trailing one ignored", () => {
    expect(baseName("C:\\work\\app")).toBe("app")
    expect(baseName("/home/me/app/")).toBe("app")
    expect(baseName("app")).toBe("app")
  })
})
