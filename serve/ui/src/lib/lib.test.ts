import { describe, expect, test } from "bun:test"
import { apiMessages, type Message } from "./chat"
import { markdown } from "./markdown"
import { parse } from "./router"

describe("markdown", () => {
  test("escapes HTML before formatting", () => {
    const html = markdown('<img src=x onerror=alert(1)> **b**')
    expect(html).not.toContain("<img")
    expect(html).toContain("&lt;img")
    expect(html).toContain("<strong>b</strong>")
  })
  test("links are http(s) only and cannot leave the attribute", () => {
    expect(markdown("[x](javascript:alert(1))")).not.toContain("<a ")
    const html = markdown('[x](https://a.test/"onmouseover="alert(1))')
    const tag = html.match(/<a [^>]*>/)![0]                       // the quote is escaped: one href, no attribute of its own
    expect(tag).toMatch(/^<a href="[^"]*" target="_blank" rel="noopener noreferrer">$/)
    expect(tag.match(/=/g)!.length).toBe(3 + 1)                   // href, target, rel + the escaped "=" inside the href
  })
  test("an unfinished code fence still renders while streaming", () => {
    expect(markdown("```py\nprint(1)")).toContain("<pre><code>print(1)</code></pre>")
  })
  test("a horizontal rule needs three of the same mark", () => {
    expect(markdown("---")).toContain("<hr>")
    expect(markdown("***")).toContain("<hr>")
    expect(markdown("--")).not.toContain("<hr>")
  })
  test("Thai text passes through", () => {
    expect(markdown("สวัสดี **ครับ**")).toBe("<p>สวัสดี <strong>ครับ</strong></p>")
  })
})

describe("apiMessages", () => {
  const user = (text: string): Message => ({ role: "user", text, time: 1 })
  test("a file is fenced with more backticks than it contains", () => {
    const m: Message = { ...user("see"), files: [{ name: "a.md", text: "```js\nx\n```" }] }
    const c = (apiMessages([m])[0] as { content: string }).content
    expect(c).toContain("File: a.md\n````\n```js")
  })
  test("an answer that used a tool goes back as the model wrote it", () => {
    const a: Message = {
      role: "assistant", time: 2, text: "Looking. Found it.",
      tools: [{ id: "c1", name: "fs__read", at: 8, rat: 0, state: "done", round: 1, arguments: { p: 1 }, result: "data" }],
    }
    const out = apiMessages([user("q"), a])
    expect(out.map((x) => x.role)).toEqual(["user", "assistant", "tool", "assistant"])
    expect((out[1] as { content: string }).content).toBe("Looking.")
    expect((out[3] as { content: string }).content).toBe("Found it.")
  })
  test("a failed answer is not sent back", () => {
    expect(apiMessages([user("q"), { role: "assistant", text: "", time: 2, error: "boom" }]).length).toBe(1)
  })
})

describe("router", () => {
  test("pages, params, and the default", () => {
    expect(parse("#/requests/r1")).toEqual({ page: "requests", params: ["r1"] })
    expect(parse("#/hardware/gpu/0")).toEqual({ page: "hardware", params: ["gpu", "0"] })
    expect(parse("")).toEqual({ page: "chat", params: [] })
    expect(parse("#/nope")).toEqual({ page: "chat", params: [] })
  })
})
