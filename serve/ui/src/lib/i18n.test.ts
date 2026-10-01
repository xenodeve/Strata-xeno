import { afterEach, describe, expect, test } from "bun:test"
import { readdirSync, readFileSync, statSync } from "node:fs"
import { join } from "node:path"
import { getLang, setLang, t, tn } from "./i18n"
import { th } from "../i18n/th"

afterEach(() => setLang("en"))

describe("t", () => {
  test("English is the key; {name} is filled in, and a name that is not given stays as written", () => {
    expect(t("Chat")).toBe("Chat")
    expect(t("{n} of {m} tokens", { n: 5, m: 9 })).toBe("5 of 9 tokens")
    expect(t("{n} of {m} tokens", { n: 5 })).toBe("5 of {m} tokens")
  })
  test("Thai when the language is Thai, and the English for a text that has no Thai yet", () => {
    setLang("th")
    expect(getLang()).toBe("th")
    expect(t("Chat")).toBe("แชท")
    expect(t("A text nobody has translated {x}", { x: 1 })).toBe("A text nobody has translated 1")
  })
  test("tn picks the form for n; a Thai entry may be the same for both", () => {
    expect(tn(1, "{n} tool call", "{n} tool calls")).toBe("1 tool call")
    expect(tn(3, "{n} tool call", "{n} tool calls")).toBe("3 tool calls")
  })
})

// The completeness gate: every string the source passes to t / tn / msg has a Thai entry, every Thai entry is used, and the
// {names} in a Thai text are the ones in its English.
const SRC = join(import.meta.dir, "..")
function files(dir: string): string[] {
  return readdirSync(dir).flatMap((f) => {
    const p = join(dir, f)
    if (statSync(p).isDirectory()) return f === "vendor" || f === "i18n" ? [] : files(p)
    return /\.tsx?$/.test(f) && !/\.test\./.test(f) && !f.endsWith(".d.ts") ? [p] : []
  })
}
// a string literal: "..." or '...' or `...` without ${}
const LIT = "(\"(?:[^\"\\\\\\n]|\\\\.)*\"|'(?:[^'\\\\\\n]|\\\\.)*'|`(?:[^`\\\\$]|\\\\.)*`)"
const unq = (lit: string) => (lit[0] === '"' ? (JSON.parse(lit) as string) : lit.slice(1, -1).replace(/\\(.)/g, "$1"))
const used = new Map<string, string>()
for (const f of files(SRC)) {
  const src = readFileSync(f, "utf8").replace(/\/\*[\s\S]*?\*\//g, "").replace(/(^|\s)\/\/.*$/gm, "$1")      // comments may show examples: they are not strings the app shows
  for (const m of src.matchAll(new RegExp("(?<![\\w.$])(?:t|msg)\\(\\s*" + LIT, "g"))) used.set(unq(m[1]), f)
  for (const m of src.matchAll(new RegExp("(?<![\\w.$])tn\\(\\s*[^,()]+,\\s*" + LIT + "\\s*,\\s*" + LIT, "g"))) { used.set(unq(m[1]), f); used.set(unq(m[2]), f) }
}
const names = (s: string) => [...s.matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort().join(",")

describe("the Thai catalog", () => {
  test("the scan finds the strings", () => { expect(used.size).toBeGreaterThan(0) })
  test("every string the source shows has a Thai entry", () => {
    const missing = [...used].filter(([k]) => !(k in th)).map(([k, f]) => `${f.slice(SRC.length + 1)}: ${k}`)
    expect(missing).toEqual([])
  })
  test("every Thai entry is a string the source shows", () => {
    expect(Object.keys(th).filter((k) => !used.has(k))).toEqual([])
  })
  test("a Thai text keeps the {names} of its English", () => {
    const bad = Object.entries(th).filter(([k, v]) => names(k) !== names(v)).map(([k]) => k)
    expect(bad).toEqual([])
  })
})
