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
})
