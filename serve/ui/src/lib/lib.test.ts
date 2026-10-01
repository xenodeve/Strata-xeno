import { describe, expect, test } from "bun:test"
import { rootOf } from "./api"
import { gpuDesign, latticePattern, nextDesign, ORB_DESIGNS, phaseKind, replyDesign, serverDesign, toolDesign } from "./orbs"
import { clientName } from "./format"
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
    const html = markdown("```py\nprint(1)")
    expect(html).toMatch(/<pre><code[^>]*>.*<\/code><\/pre>/s)
    expect(html.replace(/<[^>]+>/g, "")).toContain("print(1)")           // coloured (lib/markdown.test.ts), and the text is the same
    expect(markdown("```\nprint(1)")).toContain("<pre><code>print(1)</code></pre>")      // with no language it stays plain
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

describe("client name", () => {
  test("a raw User-Agent becomes the name a person knows", () => {
    expect(clientName("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/153.0.0.0 Safari/537.36")).toBe("Chrome")
    expect(clientName("Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 Chrome/120 Safari/537.36 Edg/120.0")).toBe("Edge")
    expect(clientName("Mozilla/5.0 (X11; Linux) Gecko/20100101 Firefox/121.0")).toBe("Firefox")
    expect(clientName("claude-cli/2.1.281 (external, cli)")).toBe("Claude Code")
    expect(clientName("curl/8.19.0")).toBe("curl")
    expect(clientName("open-webui/0.6")).toBe("Open WebUI")
  })
  test("anything else is its first product token, and nothing is nothing", () => {
    expect(clientName("python-httpx/0.27.0")).toBe("python-httpx")
    expect(clientName("MyApp 3.2 (Linux)")).toBe("MyApp")
    expect(clientName("")).toBe("")
  })
})

describe("which orb says what", () => {
  const live = (o: Partial<{ state: "reading" | "generating" | "idle" | "unloaded"; queued: number; tok_s: number | null; phase: string | null }> = {}) => ({ state: "idle" as const, queued: 0, tok_s: null, ...o })
  test("the server: each state has its own form; idle shows the searching orb, slowly; an unloaded model is dormant (shaping, slowly); no orb is a frozen picture", () => {
    expect(serverDesign(live({ state: "reading" }))).toMatchObject({ design: "listening", moving: true })        // the prompt is taken in
    expect(serverDesign(live({ state: "generating", tok_s: 120 }))).toMatchObject({ design: "composing", moving: true })
    expect(serverDesign(live())).toEqual({ design: "searching", moving: true, speed: 0.5, rest: true })      // every display frame: the one orb people look at stays smooth
    expect(serverDesign(live({ state: "unloaded" }))).toEqual({ design: "shaping", moving: true, speed: 0.5, fps: 30, rest: true })        // dormant, but never a frozen picture
    expect(serverDesign(live({ queued: 2 }))).toMatchObject({ design: "connecting", moving: true })
    expect(serverDesign(live(), true)).toMatchObject({ design: "connecting", moving: true })        // not answering: still trying
  })
  test("the server's phase names what it is doing, so thinking is not 'writing'", () => {
    expect(phaseKind("reading the prompt")).toEqual({ kind: "reading" })
    expect(phaseKind("thinking")).toEqual({ kind: "thinking" })
    expect(phaseKind("answering")).toEqual({ kind: "answering" })
    expect(phaseKind("writing a tool call: fs__read_file")).toEqual({ kind: "tool", tool: "fs__read_file" })
    expect(phaseKind("tool call complete")).toEqual({ kind: "toolDone" })
    expect(phaseKind(null)).toEqual({ kind: "other" })
    expect(phaseKind("something new")).toEqual({ kind: "other" })
  })
  test("while the server generates, the orb takes the form of the phase", () => {
    expect(serverDesign(live({ state: "generating", phase: "thinking", tok_s: 120 }))).toMatchObject({ design: "solving", moving: true })
    expect(serverDesign(live({ state: "generating", phase: "answering", tok_s: 120 }))).toMatchObject({ design: "composing", moving: true })
    expect(serverDesign(live({ state: "generating", phase: "writing a tool call: web_search" }))).toMatchObject({ design: "searching" })
    expect(serverDesign(live({ state: "generating", phase: "writing a tool call: fs__read" }))).toMatchObject({ design: "connecting" })
    expect(serverDesign(live({ state: "generating", phase: "tool call complete" }))).toMatchObject({ design: "weaving" })
    expect(serverDesign(live({ state: "generating", phase: null }))).toMatchObject({ design: "composing" })        // a server that does not say
  })
  test("the writing orb runs at the pace of the writing, within limits", () => {
    expect(serverDesign(live({ state: "generating", tok_s: 0 })).speed).toBeCloseTo(0.7, 5)
    expect(serverDesign(live({ state: "generating", tok_s: 150 })).speed).toBeCloseTo(1.1, 5)
    expect(serverDesign(live({ state: "generating", tok_s: 9000 })).speed).toBeCloseTo(1.5, 5)
    expect(serverDesign(live({ state: "generating", tok_s: null })).speed).toBe(1)
  })
  test("a GPU: rests, works, works hard, or struggles when it is held back", () => {
    expect(gpuDesign({ util: 1, throttle: [] })).toEqual({ design: "searching", moving: true, speed: 0.5, fps: 30, rest: true })
    expect(gpuDesign({ util: 12, throttle: [] })).toEqual({ design: "working", moving: true })
    expect(gpuDesign({ util: 85, throttle: [] })).toEqual({ design: "weaving", moving: true })
    expect(gpuDesign({ util: 30, throttle: ["power cap"] })).toEqual({ design: "solving", moving: true })
    expect(gpuDesign({ util: null, throttle: null })).toEqual({ design: "searching", moving: true, speed: 0.5, fps: 30, rest: true })
  })
  test("a tool call: connecting, or searching when the tool looks things up", () => {
    expect(toolDesign("fs__read_file")).toBe("connecting")
    expect(toolDesign("brave__web_search")).toBe("searching")
    expect(toolDesign("github__find_issues")).toBe("searching")
    expect(toolDesign("")).toBe("connecting")
  })
  test("a reply: waiting, thinking, a tool running, planning the next step after a tool, writing, nothing once done", () => {
    expect(replyDesign({ streaming: true, reasoning: "", text: "", tools: [] })).toBe("breathing")
    expect(replyDesign({ streaming: true, reasoning: "hm", text: "", tools: [] })).toBe("solving")
    expect(replyDesign({ streaming: true, reasoning: "hm", text: "", tools: [{ name: "x__search", state: "running" }] })).toBe("searching")
    expect(replyDesign({ streaming: true, reasoning: "hm", text: "", tools: [{ name: "x__read", state: "writing" }] })).toBe("connecting")
    expect(replyDesign({ streaming: true, reasoning: "hm", text: "", tools: [{ name: "x__read", state: "done" }] })).toBe("weaving")
    expect(replyDesign({ streaming: true, reasoning: "", text: "", tools: [{ name: "x__read", state: "done" }] })).toBe("weaving")
    expect(replyDesign({ streaming: true, reasoning: "hm", text: "Hel", tools: [] })).toBe("composing")
    expect(replyDesign({ streaming: true, reasoning: "", text: "Hi", tools: [{ name: "x__read", state: "done" }] })).toBe("composing")
    expect(replyDesign({ streaming: false, reasoning: "", text: "Hi", tools: [] })).toBeNull()
  })
  test("the empty chat's orb picks another of the nine forms, never the one it has", () => {
    expect(ORB_DESIGNS).toHaveLength(9)
    for (const prev of ORB_DESIGNS) {
      const seen = new Set<string>()
      for (let i = 0; i < 100; i++) { const d = nextDesign(prev, i / 100); expect(d).not.toBe(prev); seen.add(d) }
      expect(seen.size).toBe(8)                      // every other form can come up
    }
    expect(nextDesign("working", 0.999999)).toBe("shaping")
    expect(nextDesign(null, 0)).toBe("working")      // nothing shown yet: any of the nine
  })
  test("the lattice beside a thought runs the pattern of what the agent is doing", () => {
    expect(latticePattern("solving")).toBe("orbit")
    expect(latticePattern("breathing")).toBe("orbit")
    expect(latticePattern("searching")).toBe("ripple")
    expect(latticePattern("connecting")).toBe("snake")
    expect(latticePattern("weaving")).toBe("spiral")
    expect(latticePattern(null)).toBe("orbit")
  })
})
