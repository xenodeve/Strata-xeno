import { describe, expect, test } from "bun:test"
import { rootOf } from "./api"
import { promptSplit } from "./metrics"
import { apiMessages, type Message } from "./chat"
import { archOf, parseArch } from "./arch"
import { markdown } from "./markdown"
import { attribute, overlapOpportunityMs } from "./stall"
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

describe("arch", () => {
  const list = parseArch("sm120@NVIDIA_GeForce_RTX_5060_Ti,sm89@NVIDIA_GeForce_RTX_4070_SUPER")
  test("a card is matched by name, not by position", () => {
    expect(archOf(list, "NVIDIA GeForce RTX 4070 SUPER")).toBe("sm89")        // NVML 0, CUDA 1
    expect(archOf(list, "NVIDIA GeForce RTX 5060 Ti")).toBe("sm120")
  })
  test("an unknown card or an old engine is null, not a guess", () => {
    expect(archOf(list, "Some Other Card")).toBeNull()
    expect(parseArch("")).toEqual([])
    expect(parseArch("none")).toEqual([])
  })
})

describe("stall attribution", () => {
  const stats = { ms_cpu: 800, ms_gpu_wait: 800, ms_pool: 1000, ms_plan: 40, ms_actq: 60, ms_jobs: 70, ms_stage: 120, ms_commit: 60, ms_draft: 90, nvme_ms: 83 }
  test("each resource's buckets add up to the decode time, never more", () => {
    for (const a of attribute(stats, 2400)!) {
      expect(a.buckets.reduce((x, b) => x + b.ms, 0)).toBeCloseTo(2400, 5)
      expect(a.buckets.every((b) => b.ms >= 0)).toBe(true)
    }
  })
  test("the CPU waits on the GPU, the GPU waits on the CPU", () => {
    const [cpu, gpu] = attribute(stats, 2400)!
    expect(cpu.buckets.find((b) => b.key === "ms_gpu_wait")!.kind).toBe("wait")
    expect(gpu.buckets.find((b) => b.key === "ms_pool")!.kind).toBe("wait")
    expect(gpu.buckets.find((b) => b.key === "ms_gpu_wait")!.kind).toBe("busy")
  })
  test("stages that overlap past the wall time are scaled down, not shown as more than 100%", () => {
    const [cpu] = attribute({ ms_cpu: 2000, ms_gpu_wait: 2000 }, 2000)!
    expect(cpu.buckets.reduce((x, b) => x + b.ms, 0)).toBeCloseTo(2000, 5)
  })
  test("overhead is not a number until it is measured", () => {
    expect(attribute(stats, 2400)![0].overhead).toBeNull()
  })
  test("no counters or no time: nothing to attribute", () => {
    expect(attribute(null, 2400)).toBeNull()
    expect(attribute(stats, 0)).toBeNull()
    expect(attribute(stats, null)).toBeNull()
  })
  test("the overlap opportunity is the SSD wait, an upper bound, and null when not measured", () => {
    expect(overlapOpportunityMs(stats)).toBe(83)
    expect(overlapOpportunityMs({})).toBeNull()
    expect(overlapOpportunityMs(null)).toBeNull()
  })
})

describe("api root", () => {
  test("from where the page is served, relative to any proxy prefix", () => {
    expect(rootOf("/")).toBe("/")
    expect(rootOf("/next/")).toBe("/")
    expect(rootOf("/p/next/")).toBe("/p/")
    expect(rootOf("/p/index.html")).toBe("/p/")
  })
  test("a path that starts with // can never point at another origin", () => {
    expect(rootOf("//evil.test/next/")).toBe("/evil.test/")
    expect(rootOf("///x/")).toBe("/x/")
  })
})

describe("prompt split", () => {
  test("read and cached are separate and add up to the prompt", () => {
    expect(promptSplit(1000, 250)).toEqual({ read: 750, cached: 250, cachedShare: 0.25 })
  })
  test("a cache bigger than the prompt, or none, never goes negative", () => {
    expect(promptSplit(10, 99)).toEqual({ read: 0, cached: 10, cachedShare: 1 })
    expect(promptSplit(10, null)).toEqual({ read: 10, cached: 0, cachedShare: 0 })
    expect(promptSplit(0, 0).cachedShare).toBe(0)
  })
})
