import { beforeAll, describe, expect, test } from "bun:test"

// The chat's real path: POST /v1/chat/completions, an SSE stream (reasoning, text, MCP tool events, usage), and the next
// request carrying the tool rounds back. fetch is replaced by a stream cut at awkward places, as a network does.
let ChatController: typeof import("./chat").ChatController

beforeAll(async () => {
  const g = globalThis as Record<string, unknown>
  g.localStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} }
  g.requestAnimationFrame = (f: () => void) => setTimeout(f, 0)
  g.cancelAnimationFrame = clearTimeout
  g.location = { pathname: "/next/", search: "" }
  ;({ ChatController } = await import("./chat"))
})

const enc = new TextEncoder()
const sse = (...events: unknown[]) => events.map((e) => (typeof e === "string" ? e : `data: ${JSON.stringify(e)}\n\n`)).join("")

/** A body that arrives in `size`-byte pieces, so lines and UTF-8 characters are split across reads. */
function stream(text: string, size = 7): ReadableStream<Uint8Array> {
  const bytes = enc.encode(text)
  let i = 0
  return new ReadableStream({
    pull(c) {
      if (i >= bytes.length) return c.close()
      c.enqueue(bytes.slice(i, i + size))
      i += size
    },
  })
}

const ctx = { health: { model: "m", images: false, max_context: 4096 }, mcp: { servers: [], tools: 2 }, projectionLoaded: false }
const delta = (d: Record<string, unknown>) => ({ choices: [{ delta: d }] })

function mockFetch(bodies: string[], seen: Record<string, unknown>[]) {
  let n = 0
  ;(globalThis as Record<string, unknown>).fetch = async (_u: string, init: { body: string }) => {
    seen.push(JSON.parse(init.body))
    return new Response(stream(bodies[n++]), { status: 200 })
  }
}

describe("chat stream", () => {
  test("reasoning, a tool round, the answer, and the usage line, over a stream cut into 7-byte pieces", async () => {
    const seen: Record<string, unknown>[] = []
    mockFetch([
      sse(": keep-alive\n\n", delta({ reasoning_content: "I should look. " }),
        { strata_mcp: { event: "start", id: "c1", name: "fs__read" } },
        { strata_mcp: { event: "call", id: "c1", name: "fs__read", server: "fs", tool: "read", arguments: { p: "a.txt" }, round: 1 } },
        { strata_mcp: { event: "result", id: "c1", ok: true, text: "file says สวัสดี", chars: 16, ms: 120 } },
        delta({ reasoning_content: "Got it." }), delta({ content: "The file says " }), delta({ content: "สวัสดี." }),
        { choices: [], usage: { completion_tokens: 40 } }, "data: [DONE]\n\n"),
    ], seen)
    const c = new ChatController()
    await c.send("read a.txt", [], ctx)
    const m = c.messages[1]
    expect(m.text).toBe("The file says สวัสดี.")
    expect(m.reasoning).toBe("I should look. \n\nGot it.")             // a new round after a tool starts a new paragraph
    expect(m.tools).toHaveLength(1)
    expect(m.tools![0]).toMatchObject({ id: "c1", state: "done", server: "fs", tool: "read", round: 1, result: "file says สวัสดี", ok: true })
    expect(m.meta).toContain("40 tokens")
    expect(m.meta).toContain("1 tool call")
    expect(c.busy).toBeNull()
    expect(seen[0]).toMatchObject({ model: "m", stream: true, reasoning_effort: "high", strata_mcp: true })
  })

  test("the next request carries the tool round back as the model wrote it", async () => {
    const seen: Record<string, unknown>[] = []
    mockFetch([
      sse({ strata_mcp: { event: "call", id: "c1", name: "fs__read", server: "fs", tool: "read", arguments: { p: "a" }, round: 1 } },
        { strata_mcp: { event: "result", id: "c1", ok: true, text: "DATA", chars: 4, ms: 1 } },
        delta({ content: "Done." }), { choices: [], usage: { completion_tokens: 3 } }),
      sse(delta({ content: "ok" }), { choices: [], usage: { completion_tokens: 1 } }),
    ], seen)
    const c = new ChatController()
    await c.send("first", [], ctx)
    await c.send("second", [], ctx)
    const roles = (seen[1].messages as { role: string }[]).map((x) => x.role)
    expect(roles).toEqual(["user", "assistant", "tool", "assistant", "user"])
    const toolCalls = (seen[1].messages as { tool_calls?: { id: string; function: { name: string; arguments: string } }[] }[])[1].tool_calls!
    expect(toolCalls[0]).toMatchObject({ id: "c1", function: { name: "fs__read", arguments: '{"p":"a"}' } })
    expect((seen[1].messages as { content: string }[])[2].content).toBe("DATA")
  })

  test("a tool still running when the stream ends is shown as not run, not as running forever", async () => {
    const seen: Record<string, unknown>[] = []
    mockFetch([sse({ strata_mcp: { event: "start", id: "c9", name: "x__y" } }, delta({ content: "partial" }))], seen)
    const c = new ChatController()
    await c.send("go", [], ctx)
    expect(c.messages[1].tools![0].state).toBe("skipped")
  })

  test("an error line in the stream becomes the answer's error, and a 401 says what to do", async () => {
    const seen: Record<string, unknown>[] = []
    mockFetch([sse(delta({ content: "half" }), { error: { message: "the engine stopped" } })], seen)
    const c = new ChatController()
    let toast = ""
    c.onError = (_t, text) => { toast = text }
    await c.send("go", [], ctx)
    expect(c.messages[1].error).toBe("the engine stopped")
    expect(toast).toBe("the engine stopped")
    ;(globalThis as Record<string, unknown>).fetch = async () => new Response(JSON.stringify({ error: { message: "missing" } }), { status: 401 })
    await c.send("again", [], ctx)
    expect(c.messages[3].error).toContain("API key")
  })

  test("stop ends the answer as stopped and keeps what was written", async () => {
    const c = new ChatController()
    ;(globalThis as Record<string, unknown>).fetch = async (_u: string, init: { signal: AbortSignal }) => {
      const body = new ReadableStream<Uint8Array>({
        start(ctl) {
          ctl.enqueue(enc.encode(sse(delta({ content: "so far" }))))
          init.signal.addEventListener("abort", () => ctl.error(Object.assign(new Error("aborted"), { name: "AbortError" })))
        },
      })
      return new Response(body, { status: 200 })
    }
    const p = c.send("go", [], ctx)
    await new Promise((r) => setTimeout(r, 30))
    c.stop()
    await p
    expect(c.messages[1]).toMatchObject({ text: "so far", stopped: true, meta: "Stopped" })
    expect(c.messages[1].error).toBeUndefined()
  })

  // The speed at which the prompt is read, under the prompt: sampled from the server's progress while it reads, replaced
  // by the engine's own mean (the final chunk's timings) when the request is done.
  const held = (final: unknown[]) => {
    let release!: () => void
    ;(globalThis as Record<string, unknown>).fetch = async () => new Response(new ReadableStream<Uint8Array>({
      start(ctl) { release = () => { ctl.enqueue(enc.encode(sse(...final))); ctl.close() } },
    }), { status: 200 })
    return () => release()
  }

  test("prefill: a speed from the last second while the prompt is read, the engine's mean when it is done", async () => {
    const release = held([delta({ content: "ok" }),
      { choices: [], usage: { completion_tokens: 3 }, timings: { prompt_n: 2400, prompt_ms: 2000, prompt_per_second: 1200, cache_n: 5000 } }])
    const c = new ChatController()
    const p = c.send("hi", [], ctx)
    c.samplePrefill({ state: "reading", prompt_read: 5000 }, 0)
    expect(c.messages[0].prefill).toMatchObject({ state: "reading", rate: null })
    c.samplePrefill({ state: "reading", prompt_read: 5600 }, 500)
    expect(c.messages[0].prefill).toMatchObject({ state: "reading", rate: 1200 })
    c.samplePrefill({ state: "generating", prompt_read: null, prefill_tok_s_mean: 1190 }, 900)     // the read is over: the engine's mean now, at once
    expect(c.messages[0].prefill).toEqual({ state: "done", rate: null, mean: 1190, read: null, cached: null })
    c.samplePrefill({ state: "reading", prompt_read: 9000 }, 1400)         // a later look (another request) changes nothing
    expect(c.messages[0].prefill!.state).toBe("done")
    release(); await p
    expect(c.messages[0].prefill).toEqual({ state: "done", rate: null, mean: 1200, read: 2400, cached: 5000 })
  })

  test("prefill: with no timings the mean comes from the samples; a cache that held it all has no speed", async () => {
    let release = held([delta({ content: "ok" }), { choices: [], usage: { completion_tokens: 1 } }])
    const c = new ChatController()
    let p = c.send("hi", [], ctx)
    c.samplePrefill({ state: "reading", prompt_read: 100 }, 0)
    c.samplePrefill({ state: "reading", prompt_read: 1100 }, 1000)
    c.samplePrefill({ state: "generating", prompt_read: null, prefill_tok_s_mean: null }, 1500)   // no engine mean: the samples' own
    expect(c.messages[0].prefill).toEqual({ state: "done", rate: null, mean: 1000, read: null, cached: null })
    release(); await p
    expect(c.messages[0].prefill).toEqual({ state: "done", rate: null, mean: 1000, read: null, cached: null })
    release = held([delta({ content: "ok" }), { choices: [], usage: { completion_tokens: 1 }, timings: { prompt_n: 0, prompt_ms: 0, prompt_per_second: null, cache_n: 800 } }])
    p = c.send("again", [], ctx)
    release(); await p
    expect(c.messages[2].prefill).toEqual({ state: "done", rate: null, mean: null, read: 0, cached: 800 })
  })

  test("prefill: after a tool round the final timings are the last round's, so the mean is the first read's", async () => {
    const release = held([{ strata_mcp: { event: "call", id: "c1", name: "fs__read", server: "fs", tool: "read", arguments: {}, round: 1 } },
      { strata_mcp: { event: "result", id: "c1", ok: true, text: "x", chars: 1, ms: 1 } }, delta({ content: "ok" }),
      { choices: [], usage: { completion_tokens: 1 }, timings: { prompt_n: 50, prompt_ms: 100, prompt_per_second: 500, cache_n: 3000 } }])
    const c = new ChatController()
    const p = c.send("hi", [], ctx)
    c.samplePrefill({ state: "reading", prompt_read: 0 }, 0)
    c.samplePrefill({ state: "reading", prompt_read: 2000 }, 1000)
    release(); await p
    expect(c.messages[0].prefill).toMatchObject({ state: "done", mean: 2000, read: null })
  })
})
