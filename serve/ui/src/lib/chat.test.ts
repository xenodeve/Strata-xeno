import { beforeAll, describe, expect, test } from "bun:test"

// The chat's real path: POST /v1/chat/completions, an SSE stream (reasoning, text, MCP tool events, usage), and the next
// request carrying the tool rounds back. fetch is replaced by a stream cut at awkward places, as a network does.
let ChatController: typeof import("./chat").ChatController
let metaText: typeof import("./chat").metaText
let mcpRequest: typeof import("./chat").mcpRequest
let DEFAULTS: typeof import("./chat").DEFAULTS
let store: typeof import("./store").store
let addRule: typeof import("./agent").addRule
let rulesOf: typeof import("./agent").rulesOf
type Message = import("./chat").Message

beforeAll(async () => {
  const g = globalThis as Record<string, unknown>
  g.localStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} }
  g.requestAnimationFrame = (f: () => void) => setTimeout(f, 0)
  g.cancelAnimationFrame = clearTimeout
  g.location = { pathname: "/next/", search: "" }
  ;({ ChatController, metaText, mcpRequest, DEFAULTS } = await import("./chat"))
  ;({ store } = await import("./store"))
  ;({ addRule, rulesOf } = await import("./agent"))
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
    expect(metaText(m)).toContain("40 tokens")
    expect(metaText(m)).toContain("1 tool call")
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
    expect(c.messages[1]).toMatchObject({ text: "so far", stopped: true })
    expect(metaText(c.messages[1])).toBe("Stopped")
    expect(c.messages[1].error).toBeUndefined()
  })

  test("the line under an answer is worded when it is shown: the numbers are stored, an older stored text still shows", () => {
    const a = { role: "assistant" as const, text: "x", time: 1 }
    expect(metaText({ ...a, stats: { tokens: 40, tokS: 38.24, tools: 2, projection: "on" } })).toBe("40 tokens · 38.2 tok/s · projection on · 2 tool calls")
    expect(metaText({ ...a, stats: { tokens: 1, stopped: true, projection: "off" } })).toBe("1 tokens · stopped · projection off")
    expect(metaText({ ...a, stats: { tokens: 5, tools: 1, limit: 3 } })).toBe("5 tokens · 1 tool call · stopped at the limit of 3 tool rounds (mcp.max_rounds)")
    expect(metaText({ ...a, meta: "12 tokens · 3.0 tok/s" })).toBe("12 tokens · 3.0 tok/s")
    expect(metaText(a)).toBe("")
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

  // Taking a prompt back (undo) and rewriting one (edit): both end the conversation at that prompt.
  const two = async () => {
    const seen: Record<string, unknown>[] = []
    mockFetch([sse(delta({ content: "one" }), { choices: [], usage: { completion_tokens: 1 } }), sse(delta({ content: "two" }), { choices: [], usage: { completion_tokens: 1 } })], seen)
    const c = new ChatController()
    await c.send("first", [{ kind: "file", name: "a.txt", text: "AAA" }], ctx)
    await c.send("second", [{ kind: "image", name: "p.png", url: "data:image/png;base64,xx" }], ctx)
    return { c, seen }
  }

  test("undo takes the last prompt back with its answer, and returns what was sent", async () => {
    const { c } = await two()
    const back = c.undoLast()
    expect(back).toMatchObject({ text: "second", attachments: [{ kind: "image", name: "p.png", url: "data:image/png;base64,xx" }] })
    expect(back!.removed.map((m) => m.text)).toEqual(["second", "two"])               // what left, for the page to close up gently
    expect(c.messages.map((m) => m.text)).toEqual(["first", "one"])
    expect(c.undoLast()!.text).toBe("first")
    expect(c.messages).toHaveLength(0)
    expect(c.undoLast()).toBeNull()
  })

  test("undo keeps only the attachments that still hold their data (a reload keeps names alone)", () => {
    const c = new ChatController()
    c.messages = [{ role: "user", text: "q", time: 1, images: [{ name: "gone.png" }], files: [{ name: "gone.txt" }, { name: "kept.txt", text: "K" }] }, { role: "assistant", text: "a", time: 2 }]
    expect(c.undoLast()).toMatchObject({ text: "q", attachments: [{ kind: "file", name: "kept.txt", text: "K" }] })
  })

  test("nothing is undone or edited while an answer is being written", async () => {
    const release = held([delta({ content: "ok" })])
    const c = new ChatController()
    const p = c.send("hi", [], ctx)
    expect(c.undoLast()).toBeNull()
    expect(await c.edit(0, "other", ctx)).toBe(false)
    expect(c.messages).toHaveLength(2)
    release(); await p
  })

  test("edit rewrites an earlier prompt: what came after it is replaced, and only the new text goes to the model", async () => {
    const { c, seen } = await two()
    expect(await c.edit(0, "first, reworded", ctx)).toBe(true)
    const sent = seen[2].messages as { role: string; content: unknown }[]
    expect(sent).toHaveLength(1)
    expect(String(sent[0].content)).toContain("first, reworded")
    expect(String(sent[0].content)).toContain("AAA")                  // its attachment goes with it
    expect(c.messages.map((m) => m.role)).toEqual(["user", "assistant"])
    expect(c.messages[0].text).toBe("first, reworded")
  })

  test("edit refuses an answer, an index that is not there, and an empty prompt with nothing attached", async () => {
    const { c } = await two()
    expect(await c.edit(1, "x", ctx)).toBe(false)
    expect(await c.edit(9, "x", ctx)).toBe(false)
    expect(await c.edit(2, "   ", ctx)).toBe(true)                    // the second prompt still has its image: allowed
    const c2 = new ChatController()
    c2.messages = [{ role: "user", text: "q", time: 1 }, { role: "assistant", text: "a", time: 2 }]
    expect(await c2.edit(0, "  ", ctx)).toBe(false)
    expect(c2.messages).toHaveLength(2)
  })
})

describe("which MCP servers a request uses", () => {
  const sv = (name: string, tools: number, status = "ready") => ({ name, transport: "stdio", status, tools: Array.from({ length: tools }, (_, i) => ({ tool: `t${i}` })) })
  const mcp = { servers: [sv("files", 3), sv("web", 2), sv("down", 0, "failed")], tools: 5 }
  test("all on: strata_mcp and no list", () => {
    expect(mcpRequest(DEFAULTS, mcp)).toEqual({ strata_mcp: true })
  })
  test("one switched off: it is named, the others still run", () => {
    expect(mcpRequest({ ...DEFAULTS, mcpOff: ["web"] }, mcp)).toEqual({ strata_mcp: true, strata_mcp_off: ["web"] })
  })
  test("every server that has tools switched off: the request does not ask for MCP at all", () => {
    expect(mcpRequest({ ...DEFAULTS, mcpOff: ["files", "web"] }, mcp)).toEqual({})
  })
  test("the master switch off: nothing, whatever the list says", () => {
    expect(mcpRequest({ ...DEFAULTS, mcp: false }, mcp)).toEqual({})
  })
  test("a name that is no longer a server is not sent, and a saved value that is not a list is ignored", () => {
    expect(mcpRequest({ ...DEFAULTS, mcpOff: ["gone", "web"] }, mcp)).toEqual({ strata_mcp: true, strata_mcp_off: ["web"] })
    expect(mcpRequest({ ...DEFAULTS, mcpOff: "web" as unknown as string[] }, mcp)).toEqual({ strata_mcp: true })
  })
  test("no server has tools: nothing", () => {
    expect(mcpRequest(DEFAULTS, { servers: [sv("down", 0, "failed")], tools: 0 })).toEqual({})
  })
  test("a list that has not arrived yet (servers empty, tools counted) still asks, as before", () => {
    expect(mcpRequest(DEFAULTS, { servers: [], tools: 2 })).toEqual({ strata_mcp: true })
  })
})

// "/name" in the prompt (a skill in use) makes the request ask the server to load that skill.
describe("a message that starts with a skill's name", () => {
  const ok = () => sse(delta({ content: "ok" }), "data: [DONE]\n\n")
  test("names the skill in the request when it is one in use", async () => {
    const seen: Record<string, unknown>[] = []
    mockFetch([ok()], seen)
    await new ChatController().send("/tdd write the tests", [], { ...ctx, skills: ["tdd", "pdf-tools"] })
    expect(seen[0].strata_skill).toBe("tdd")
  })
  test("does not when the name is not a skill in use, or the page has no list of skills", async () => {
    const seen: Record<string, unknown>[] = []
    mockFetch([ok(), ok(), ok()], seen)
    await new ChatController().send("/nope hi", [], { ...ctx, skills: ["tdd"] })
    await new ChatController().send("/tdd hi", [], ctx)
    await new ChatController().send("/tdd hi", [], { ...ctx, skills: [] })
    expect(seen.map((b) => "strata_skill" in b)).toEqual([false, false, false])
  })
  test("only the last prompt counts: a later plain one does not ask again", async () => {
    const seen: Record<string, unknown>[] = []
    mockFetch([ok(), ok()], seen)
    const c = new ChatController()
    await c.send("/tdd write the tests", [], { ...ctx, skills: ["tdd"] })
    await c.send("and then?", [], { ...ctx, skills: ["tdd"] })
    expect([seen[0].strata_skill, "strata_skill" in seen[1]]).toEqual(["tdd", false])
  })
})

// The coding tools (issue #96): the request names the folder, mode and chat; the stream brings cards, verdicts and todo lists; the page answers.
describe("the coding tools in the chat", () => {
  const ON = { available: true, allowed: true, shell: "bash", tools: ["Read", "Write", "Bash"] }
  const call = (id = "c1", name = "Bash", args: unknown = { command: "npm test" }) => [
    { strata_mcp: { event: "start", id, name } },
    { strata_mcp: { event: "call", id, name, server: "agent", tool: name, arguments: args, round: 1 } },
  ]
  const finish = [delta({ content: "ok" }), { choices: [], usage: { completion_tokens: 1 } }, "data: [DONE]\n\n"]
  function keep() {
    const data = new Map<string, string>()
    ;(globalThis as Record<string, unknown>).localStorage = {
      getItem: (k: string) => data.get(k) ?? null, setItem: (k: string, v: string) => { data.set(k, v) }, removeItem: (k: string) => { data.delete(k) },
    }
    return data
  }

  test("the request names the folder, the mode and the chat, and no more than that", async () => {
    keep()
    const seen: Record<string, unknown>[] = []
    mockFetch([sse(...finish)], seen)
    const c = new ChatController()
    c.setSettings({ ...c.settings, agentMode: "plan" })
    await c.send("hi", [], { ...ctx, agent: ON, folder: ["C:/work/app", "C:/work/app-wt2"] })
    expect(seen[0].strata_agent).toEqual({ cwd: "C:/work/app", dirs: ["C:/work/app-wt2"], mode: "plan", session: c.index.active, allow: [] })
  })

  test("without the tools (switched off, or the server has none) the request has no strata_agent", async () => {
    keep()
    const seen: Record<string, unknown>[] = []
    mockFetch([sse(...finish), sse(...finish)], seen)
    const c = new ChatController()
    await c.send("hi", [], ctx)
    c.setSettings({ ...c.settings, agent: false })
    await c.send("again", [], { ...ctx, agent: ON })
    expect(seen.map((b) => "strata_agent" in b)).toEqual([false, false])
  })

  test("what the user allowed for this chat goes with the next request", async () => {
    keep()
    const seen: Record<string, unknown>[] = []
    mockFetch([sse(...finish), sse(...finish)], seen)
    const c = new ChatController()
    await c.send("one", [], { ...ctx, agent: ON })
    addRule(store, c.index.active!, "Bash(git status:*)")
    await c.send("two", [], { ...ctx, agent: ON })
    expect((seen[1].strata_agent as { allow: string[] }).allow).toEqual(["Bash(git status:*)"])
    expect((seen[0].strata_agent as { allow: string[] }).allow).toEqual([])
  })

  test("a question is a card on its call; the call waits until it is answered and then finishes", async () => {
    keep()
    mockFetch([sse(...call(), { strata_mcp: { event: "permission", id: "q1", call_id: "c1", tool: "Bash", arguments: { command: "npm test" }, why: "a command asks every time", danger: false, rule: "Bash(npm test:*)" } },
      { strata_mcp: { event: "result", id: "c1", ok: true, text: "passed", chars: 6, truncated: false, ms: 900 } }, ...finish)], [])
    const c = new ChatController()
    await c.send("test it", [], { ...ctx, agent: ON })
    const t = c.messages[1].tools![0]
    expect(t.ask).toMatchObject({ id: "q1", why: "a command asks every time", danger: false, rule: "Bash(npm test:*)" })
    expect(t.server).toBe("agent")
    expect(t.state).toBe("done")                                              // the result came after
  })

  test("while it waits the call says so", async () => {
    keep()
    let release!: () => void
    const first = sse(...call(), { strata_mcp: { event: "permission", id: "q1", call_id: "c1", tool: "Bash", arguments: {}, why: "w", danger: true, rule: null } })
    ;(globalThis as Record<string, unknown>).fetch = async () => new Response(new ReadableStream<Uint8Array>({
      start(ctl) { ctl.enqueue(enc.encode(first)); release = () => { ctl.enqueue(enc.encode(sse(...finish))); ctl.close() } },
    }), { status: 200 })
    const c = new ChatController()
    const p = c.send("go", [], { ...ctx, agent: ON })
    await new Promise((r) => setTimeout(r, 30))
    const t = c.messages[1].tools![0]
    expect(t.state).toBe("asking")
    expect(t.ask?.danger).toBe(true)
    release()
    await p
  })

  test("answering sends the id and the choice; Allow for this chat also remembers the rule", async () => {
    keep()
    const c = new ChatController()
    c.messages = [{ role: "user", text: "x", time: 1 }, { role: "assistant", text: "", time: 2, tools: [{ id: "c1", name: "Bash", at: 0, rat: 0, state: "asking", server: "agent", ask: { id: "q1", tool: "Bash", why: "w", danger: false, rule: "Bash(npm test:*)" } }] }]
    c.index = { ...c.index, active: "chat9" }
    const posts: { url: string; body: unknown }[] = []
    ;(globalThis as Record<string, unknown>).fetch = async (u: string, init: { body: string }) => { posts.push({ url: u, body: JSON.parse(init.body) }); return new Response('{"ok":true}', { status: 200 }) }
    expect(await c.answer("c1", "allow_chat")).toBe(true)
    expect(posts).toHaveLength(1)
    expect(posts[0].url).toContain("agent/permission")
    expect(posts[0].body).toEqual({ id: "q1", decision: "allow_chat" })
    expect(c.messages[1].tools![0].ask?.answer).toBe("allow_chat")
    expect(c.messages[1].tools![0].state).toBe("running")
    expect(rulesOf(store, "chat9")).toEqual(["Bash(npm test:*)"])
  })

  test("Deny and Allow once remember nothing", async () => {
    keep()
    const c = new ChatController()
    const mk = (id: string): Message => ({ role: "assistant", text: "", time: 2, tools: [{ id, name: "Bash", at: 0, rat: 0, state: "asking", ask: { id: "q" + id, tool: "Bash", why: "w", danger: false, rule: "Bash(ls:*)" } }] })
    c.messages = [{ role: "user", text: "x", time: 1 }, mk("a"), mk("b")]
    c.index = { ...c.index, active: "chat9" }
    ;(globalThis as Record<string, unknown>).fetch = async () => new Response('{"ok":true}', { status: 200 })
    await c.answer("a", "allow")
    await c.answer("b", "deny")
    expect(rulesOf(store, "chat9")).toEqual([])
    expect(c.messages[2].tools![0].state).toBe("running")                      // the server says how it ended: with the result
  })

  test("an answer the server did not take leaves the card to answer again, and says why", async () => {
    keep()
    const c = new ChatController()
    const errors: string[] = []
    c.onError = (title, text) => errors.push(`${title}: ${text}`)
    c.messages = [{ role: "user", text: "x", time: 1 }, { role: "assistant", text: "", time: 2, tools: [{ id: "c1", name: "Bash", at: 0, rat: 0, state: "asking", ask: { id: "q1", tool: "Bash", why: "w", danger: false, rule: null } }] }]
    ;(globalThis as Record<string, unknown>).fetch = async () => new Response(JSON.stringify({ error: { message: "no question with that id is waiting" } }), { status: 404 })
    expect(await c.answer("c1", "allow")).toBe(false)
    expect(c.messages[1].tools![0].ask?.answer).toBeUndefined()
    expect(c.messages[1].tools![0].state).toBe("asking")
    expect(errors[0]).toContain("no question with that id is waiting")
  })

  test("answering a call that is not there does nothing", async () => {
    keep()
    const c = new ChatController()
    ;(globalThis as Record<string, unknown>).fetch = async () => { throw new Error("must not be called") }
    expect(await c.answer("nope", "allow")).toBe(false)
  })

  test("the auto-mode check is shown on the call: what it found", async () => {
    keep()
    mockFetch([sse(...call("c1", "Read", { file_path: "../x" }), { strata_mcp: { event: "judging", id: "q1", call_id: "c1", tool: "Read" } },
      { strata_mcp: { event: "judged", id: "q1", call_id: "c1", verdict: "allow", severity: 1 } },
      { strata_mcp: { event: "result", id: "c1", ok: true, text: "x", chars: 1, truncated: false, ms: 5 } }, ...finish)], [])
    const c = new ChatController()
    await c.send("go", [], { ...ctx, agent: ON })
    expect(c.messages[1].tools![0].judge).toEqual({ verdict: "allow", severity: 1 })
    expect(c.messages[1].tools![0].ask).toBeUndefined()
  })

  test("a todo list from the stream is the answer's list, and a later one replaces it", async () => {
    keep()
    const t1 = [{ content: "a", status: "in_progress", activeForm: "Doing a" }]
    const t2 = [{ content: "a", status: "completed", activeForm: "Doing a" }, { content: "b", status: "in_progress", activeForm: "Doing b" }]
    mockFetch([sse({ strata_mcp: { event: "todos", call_id: "c1", todos: t1 } }, { strata_mcp: { event: "todos", call_id: "c2", todos: t2 } }, ...finish)], [])
    const c = new ChatController()
    await c.send("plan it", [], { ...ctx, agent: ON })
    expect(c.messages[1].todos).toEqual(t2)
  })

  test("approving a plan switches the page back to the default mode", async () => {
    keep()
    mockFetch([sse({ strata_mcp: { event: "mode", call_id: "c1", mode: "ask" } }, ...finish)], [])
    const c = new ChatController()
    c.setSettings({ ...c.settings, agentMode: "plan" })
    await c.send("go", [], { ...ctx, agent: ON })
    expect(c.settings.agentMode).toBe("ask")
  })

  test("the folder is the project's, else the default one for chats that are in no project", () => {
    keep()
    const c = new ChatController()
    c.setSettings({ ...c.settings, agentFolder: "C:/default" })
    expect(c.folder()).toBe("C:/default")                                      // a chat that is not saved yet is in no project
    c.index = { active: "s1", items: [{ id: "s1", title: "t", time: 1 }], projects: [{ id: "p1", name: "Work" }] }
    expect(c.folder()).toBe("C:/default")
    c.move("s1", "p1")
    expect(c.folder()).toBe("C:/default")                                      // the project has no folder yet
    c.setProjectFolders("p1", [" C:/work/app ", "C:/work/app-wt2"])
    expect(c.folder()).toBe("C:/work/app")                                     // the first is the main one
    expect(c.folders()).toEqual(["C:/work/app", "C:/work/app-wt2"])
    c.setProjectFolders("p1", [])
    expect(c.folder()).toBe("C:/default")
    expect(c.folders()).toEqual(["C:/default"])
    c.setSettings({ ...c.settings, agentFolder: "  " })
    expect(c.folder()).toBeNull()
  })

  test("a project cannot be made without a folder, and it is the folder its chats' tools work in", () => {
    keep()
    const c = new ChatController()
    expect(c.addProject("Work", [])).toBeNull()
    expect(c.addProject("Work", ["   "])).toBeNull()
    expect(c.index.projects).toEqual([])
    expect(c.addProject("", ["C:/work/app"])).toBeNull()                       // a name is needed too
    const id = c.addProject("Work", ["  C:/work/app ", "C:/work/app-wt2"])!
    expect(c.index.projects).toEqual([{ id, name: "Work", folders: ["C:/work/app", "C:/work/app-wt2"] }])
    c.index = { ...c.index, active: "s1", items: [{ id: "s1", title: "t", time: 1, project: id }] }
    expect(c.folder()).toBe("C:/work/app")
    expect(c.folders()).toEqual(["C:/work/app", "C:/work/app-wt2"])
    expect(new ChatController().index.projects[0].folders).toEqual(["C:/work/app", "C:/work/app-wt2"])
  })

  test("a project's folders are kept with the conversations", () => {
    keep()
    const c = new ChatController()
    c.index = { active: null, items: [], projects: [{ id: "p1", name: "Work" }] }
    c.setProjectFolders("p1", ["C:/work/app"])
    expect(new ChatController().index.projects[0].folders).toEqual(["C:/work/app"])
  })

  test("an event this page does not know is ignored", async () => {
    keep()
    mockFetch([sse(...call(), { strata_mcp: { event: "something-new", id: "zz", call_id: "c1" } }, { strata_mcp: { event: "result", id: "c1", ok: true, text: "x", chars: 1, truncated: false, ms: 1 } }, ...finish)], [])
    const c = new ChatController()
    await c.send("go", [], { ...ctx, agent: ON })
    expect(c.messages[1].tools).toHaveLength(1)
    expect(c.messages[1].tools![0].state).toBe("done")
  })
})

// Many conversations (issue #92) through the controller, over a localStorage that really keeps things.
describe("conversations", () => {
  function keep(limit = Infinity) {
    const data = new Map<string, string>()
    ;(globalThis as Record<string, unknown>).localStorage = {
      getItem: (k: string) => data.get(k) ?? null,
      setItem: (k: string, v: string) => { if (v.length > limit) throw new Error("QuotaExceededError"); data.set(k, v) },
      removeItem: (k: string) => { data.delete(k) },
    }
    return data
  }
  const answer = (text: string) => sse(delta({ content: text }), { choices: [], usage: { completion_tokens: 1 } }, "data: [DONE]\n\n")

  test("a conversation is added with its first prompt, and New chat keeps it in the list", async () => {
    keep()
    mockFetch([answer("a"), answer("b")], [])
    const c = new ChatController()
    expect(c.index.items).toEqual([])
    await c.send("first question", [], ctx)
    expect(c.index.items).toHaveLength(1)
    expect(c.index.items[0].title).toBe("first question")
    expect(c.index.active).toBe(c.index.items[0].id)
    expect(c.newSession()).toBe(true)
    expect(c.messages).toEqual([])
    expect(c.index.items).toHaveLength(1)
    await c.send("second question", [], ctx)
    expect(c.index.items.map((i) => i.title).sort()).toEqual(["first question", "second question"])
  })

  test("opening one restores its messages, and a reload comes back to the one that was open", async () => {
    keep()
    mockFetch([answer("a"), answer("b")], [])
    const c = new ChatController()
    await c.send("one", [], ctx)
    const first = c.index.active!
    c.newSession()
    await c.send("two", [], ctx)
    expect(c.open(first)).toBe(true)
    expect(c.messages.map((m) => m.text)).toEqual(["one", "a"])
    const again = new ChatController()
    expect(again.index.active).toBe(first)
    expect(again.messages.map((m) => m.text)).toEqual(["one", "a"])
    expect(again.index.items).toHaveLength(2)
  })

  test("nothing changes while an answer is being written", async () => {
    keep()
    mockFetch([answer("a")], [])
    const c = new ChatController()
    await c.send("one", [], ctx)
    const first = c.index.active!
    c.busy = { abort: new AbortController(), msg: c.messages[1] }
    expect(c.newSession()).toBe(false)
    expect(c.open(first)).toBe(true)                       // the one that is open is already open
    expect(c.remove(first)).toBe(false)
    expect(c.messages).toHaveLength(2)
    c.busy = null
  })

  test("a conversation the app already had becomes the first one", () => {
    const data = keep()
    data.set("strata.chat", JSON.stringify([{ role: "user", text: "from before", time: 5 }, { role: "assistant", text: "yes", time: 6 }]))
    const c = new ChatController()
    expect(c.index.items).toHaveLength(1)
    expect(c.index.items[0].title).toBe("from before")
    expect(c.messages).toHaveLength(2)
  })

  test("deleting, renaming, and projects reach the controller's index and the storage", async () => {
    const data = keep()
    mockFetch([answer("a")], [])
    const c = new ChatController()
    await c.send("one", [], ctx)
    const id = c.index.active!
    c.rename(id, "Kept")
    const p = c.addProject("Work", ["C:/work"])!
    c.move(id, p)
    expect(c.index.items[0]).toMatchObject({ title: "Kept", project: p })
    expect(JSON.parse(data.get("strata.chats")!).projects).toEqual([{ id: p, name: "Work", folders: ["C:/work"] }])
    c.removeProject(p)
    expect(c.index.items[0].project).toBeUndefined()
    expect(c.remove(id)).toBe(true)
    expect(c.messages).toEqual([])
    expect(c.index.items).toEqual([])
  })

  test("storage that is full is said once, not kept silently", async () => {
    keep(300)
    mockFetch([answer("x".repeat(50))], [])
    const c = new ChatController()
    const said: string[] = []
    c.onError = (title) => said.push(title)
    await c.send("q".repeat(400), [], ctx)
    expect(said.length).toBe(1)
    expect(said[0]).toContain("storage is full")
  })
})

describe("a conversation appears when its prompt is sent, and taking back the first prompt is announced", () => {
  function keep() {
    const data = new Map<string, string>()
    ;(globalThis as Record<string, unknown>).localStorage = {
      getItem: (k: string) => data.get(k) ?? null, setItem: (k: string, v: string) => { data.set(k, v) }, removeItem: (k: string) => { data.delete(k) },
    }
    return data
  }
  const answer = (text: string) => sse(delta({ content: text }), { choices: [], usage: { completion_tokens: 1 } }, "data: [DONE]\n\n")

  test("it is in the list as soon as the prompt is sent, not when the answer ends; the answer that is still being written is not stored", async () => {
    const data = keep()
    let release!: () => void
    ;(globalThis as Record<string, unknown>).fetch = async () => new Response(new ReadableStream<Uint8Array>({
      start(ctl) { release = () => { ctl.enqueue(enc.encode(answer("ok"))); ctl.close() } },
    }), { status: 200 })
    const c = new ChatController()
    const p = c.send("hello there", [], ctx)
    expect(c.busy).not.toBeNull()                                   // the answer has not come
    expect(c.index.items).toHaveLength(1)
    expect(c.index.items[0].title).toBe("hello there")
    expect(c.index.active).toBe(c.index.items[0].id)
    expect((JSON.parse(data.get("strata.chat")!) as { role: string }[]).map((m) => m.role)).toEqual(["user"])
    expect(JSON.parse(data.get("strata.chats")!).items).toHaveLength(1)
    release()
    await p
    expect((JSON.parse(data.get("strata.chat")!) as { role: string }[]).map((m) => m.role)).toEqual(["user", "assistant"])
    expect(c.index.items).toHaveLength(1)
  })

  test("a rewrite of the first prompt keeps one conversation", async () => {
    keep()
    mockFetch([answer("a"), answer("b")], [])
    const c = new ChatController()
    await c.send("first", [], ctx)
    await c.edit(0, "first, reworded", ctx)
    expect(c.index.items).toHaveLength(1)
    expect(c.index.items[0].title).toBe("first, reworded")
  })

  test("rewriting the first prompt keeps the conversation: its name by hand, its project and its place", async () => {
    keep()
    mockFetch([answer("a"), answer("b")], [])
    const c = new ChatController()
    await c.send("first", [], ctx)
    const id = c.index.active!
    c.rename(id, "Kept")
    const project = c.addProject("Work", ["C:/work"])!
    c.move(id, project)
    await c.edit(0, "first, reworded", ctx)
    expect(c.index.items).toHaveLength(1)
    expect(c.index.items[0]).toMatchObject({ id, title: "Kept", project })
    expect(c.index.active).toBe(id)
  })

  test("taking back the only prompt would empty the conversation, and the controller says so first", async () => {
    keep()
    mockFetch([answer("a"), answer("b")], [])
    const c = new ChatController()
    expect(c.undoWouldEmpty()).toBe(false)                          // nothing to take back
    await c.send("one", [], ctx)
    expect(c.undoWouldEmpty()).toBe(true)
    await c.send("two", [], ctx)
    expect(c.undoWouldEmpty()).toBe(false)                          // there is an earlier prompt
    c.undoLast()
    expect(c.undoWouldEmpty()).toBe(true)
    const back = c.undoLast()!
    expect(back.text).toBe("one")
    expect(c.index.items).toEqual([])                                // the conversation is gone, as the question said it would be
    expect(c.index.active).toBeNull()
  })

  test("while an answer is being written there is nothing to ask about", async () => {
    keep()
    mockFetch([answer("a")], [])
    const c = new ChatController()
    await c.send("one", [], ctx)
    c.busy = { abort: new AbortController(), msg: c.messages[1] }
    expect(c.undoWouldEmpty()).toBe(false)
    c.busy = null
  })
})

// Compacting (lib/compact.ts): /compact, and a conversation that nears the end of the context being summarised before the next prompt.
describe("compacting the conversation", () => {
  function keep() {
    const data = new Map<string, string>()
    ;(globalThis as Record<string, unknown>).localStorage = {
      getItem: (k: string) => data.get(k) ?? null, setItem: (k: string, v: string) => { data.set(k, v) }, removeItem: (k: string) => { data.delete(k) },
    }
    return data
  }
  const said = (text: string, usage: Record<string, number> = { completion_tokens: 1 }) => sse(delta({ content: text }), { choices: [], usage }, "data: [DONE]\n\n")
  const SUMMARY = "<analysis>notes</analysis><summary>1. Primary request: build the thing\n9. Next step: tests</summary>"
  const small = { ...ctx, health: { model: "m", images: false, max_context: 1000 } }
  type Req = { messages: { role: string; content: unknown }[]; [k: string]: unknown }
  const waits = () => (globalThis as Record<string, unknown>).fetch = (_u: string, init: { signal: AbortSignal }) => new Promise((_res, rej) => init.signal.addEventListener("abort", () => rej(Object.assign(new Error("aborted"), { name: "AbortError" }))))

  async function twoExchanges(c: InstanceType<typeof ChatController>, usage = { prompt_tokens: 20, completion_tokens: 5 }) {
    mockFetch([said("answer one", usage), said("answer two", usage)], [])
    await c.send("first prompt", [], ctx)
    await c.send("second prompt", [], ctx)
  }

  test("/compact: the model is asked for a summary, with no tools, and the summary takes the place of the messages", async () => {
    keep()
    const c = new ChatController()
    await twoExchanges(c)
    const seen: Record<string, unknown>[] = []
    mockFetch([said(SUMMARY)], seen)
    expect(await c.compact(ctx)).toBe(true)
    const req = seen[0] as Req
    expect(req.stream).toBe(true)
    expect("strata_mcp" in req || "strata_agent" in req || "strata_skill" in req).toBe(false)
    expect(req.reasoning_effort).toBe("low")
    expect(req.messages.map((m) => m.role)).toEqual(["user", "assistant", "user", "assistant", "user"])
    expect(String(req.messages.at(-1)!.content)).toContain("Primary request and intent")
    expect(c.messages).toHaveLength(1)
    expect(c.messages[0].role).toBe("user")
    expect(c.messages[0].text).toContain("1. Primary request: build the thing")
    expect(c.messages[0].text).not.toContain("<analysis>")
    expect(c.messages[0].compact).toMatchObject({ auto: false })
    expect(c.messages[0].compact!.before).toBe(25)                       // what the last answer reported: its prompt and its own tokens
    expect(c.busy).toBeNull()
    expect(c.compacting).toBe(false)
  })

  test("what is typed after /compact is passed on as what to focus on", async () => {
    keep()
    const c = new ChatController()
    await twoExchanges(c)
    const seen: Record<string, unknown>[] = []
    mockFetch([said(SUMMARY)], seen)
    await c.compact(ctx, "the failing tests")
    expect(String((seen[0] as Req).messages.at(-1)!.content)).toContain("the failing tests")
    expect(c.messages[0].compact!.focus).toBe("the failing tests")
  })

  test("the next prompt goes after the summary, which the model reads as the earlier part", async () => {
    keep()
    const c = new ChatController()
    await twoExchanges(c)
    mockFetch([said(SUMMARY)], [])
    await c.compact(ctx)
    const seen: Record<string, unknown>[] = []
    mockFetch([said("after")], seen)
    await c.send("third prompt", [], ctx)
    const msgs = (seen[0] as Req).messages
    expect(msgs.map((m) => m.role)).toEqual(["user", "user"])
    expect(String(msgs[0].content)).toContain("1. Primary request: build the thing")
    expect(msgs[1].content).toBe("third prompt")
  })

  test("there is nothing to compact in an empty conversation, or one that is only a summary", async () => {
    keep()
    const c = new ChatController()
    expect(c.canCompact()).toBe(false)
    expect(await c.compact(ctx)).toBe(false)
    await twoExchanges(c)
    expect(c.canCompact()).toBe(true)
    mockFetch([said(SUMMARY)], [])
    await c.compact(ctx)
    expect(c.canCompact()).toBe(false)
  })

  test("a conversation that nears the end of the context is compacted before the next prompt, which and its answer stay out of the summary", async () => {
    keep()
    const c = new ChatController()
    mockFetch([said("answer one", { prompt_tokens: 700, completion_tokens: 150 })], [])
    await c.send("first prompt", [], small)
    expect(c.messages[1].stats!.ctx).toBe(850)
    const seen: Record<string, unknown>[] = []
    mockFetch([said(SUMMARY), said("answer two")], seen)
    await c.send("second prompt", [], small)
    expect(seen).toHaveLength(2)
    expect(String((seen[0] as Req).messages.at(-1)!.content)).toContain("Primary request and intent")
    expect(JSON.stringify((seen[0] as Req).messages)).not.toContain("second prompt")
    expect((seen[1] as Req).messages.map((m) => m.role)).toEqual(["user", "user"])
    expect(String((seen[1] as Req).messages[0].content)).toContain("build the thing")
    expect((seen[1] as Req).messages[1].content).toBe("second prompt")
    expect(c.messages.map((m) => m.role)).toEqual(["user", "user", "assistant"])
    expect(c.messages[0].compact).toMatchObject({ auto: true, before: 850 })
    expect(c.messages[2].text).toBe("answer two")
  })

  test("a conversation that is not near the end is left alone", async () => {
    keep()
    const c = new ChatController()
    mockFetch([said("answer one", { prompt_tokens: 100, completion_tokens: 50 })], [])
    await c.send("first prompt", [], small)
    const seen: Record<string, unknown>[] = []
    mockFetch([said("answer two")], seen)
    await c.send("second prompt", [], small)
    expect(seen).toHaveLength(1)
    expect(c.messages.some((m) => m.compact)).toBe(false)
  })

  test("with automatic compacting switched off nothing is compacted by itself", async () => {
    keep()
    const c = new ChatController()
    c.setSettings({ ...c.settings, autoCompact: false })
    mockFetch([said("answer one", { prompt_tokens: 900, completion_tokens: 50 })], [])
    await c.send("first prompt", [], small)
    const seen: Record<string, unknown>[] = []
    mockFetch([said("answer two")], seen)
    await c.send("second prompt", [], small)
    expect(seen).toHaveLength(1)
  })

  test("a prompt with a big file counts: the conversation is compacted before it is sent", async () => {
    keep()
    const c = new ChatController()
    mockFetch([said("answer one", { prompt_tokens: 300, completion_tokens: 50 })], [])
    await c.send("first prompt", [], small)
    const seen: Record<string, unknown>[] = []
    mockFetch([said(SUMMARY), said("answer two")], seen)
    await c.send("read this", [{ kind: "file", name: "big.txt", text: "x".repeat(2000) }], small)
    expect(seen).toHaveLength(2)
    expect(c.messages[0].compact?.auto).toBe(true)
  })

  test("when the summary cannot be had, the conversation is as it was and the prompt is still sent", async () => {
    keep()
    const c = new ChatController()
    const errors: string[] = []
    c.onError = (title) => errors.push(title)
    mockFetch([said("answer one", { prompt_tokens: 700, completion_tokens: 150 })], [])
    await c.send("first prompt", [], small)
    let n = 0
    ;(globalThis as Record<string, unknown>).fetch = async () => (n++ === 0 ? new Response(JSON.stringify({ error: { message: "the engine is busy" } }), { status: 500 }) : new Response(stream(said("answer two")), { status: 200 }))
    await c.send("second prompt", [], small)
    expect(errors).toContain("Could not compact the conversation")
    expect(c.messages.some((m) => m.compact)).toBe(false)
    expect(c.messages.map((m) => m.text)).toEqual(["first prompt", "answer one", "second prompt", "answer two"])
  })

  test("a reply with no summary in it is an error, and the conversation stays", async () => {
    keep()
    const c = new ChatController()
    const errors: string[] = []
    c.onError = (title) => errors.push(title)
    await twoExchanges(c)
    mockFetch([said("<analysis>only notes")], [])
    expect(await c.compact(ctx)).toBe(false)
    expect(errors).toEqual(["Could not compact the conversation"])
    expect(c.messages).toHaveLength(4)
  })

  test("when the conversation does not fit with the request, the oldest prompts are left out and the model is asked again", async () => {
    keep()
    const c = new ChatController()
    await twoExchanges(c)
    mockFetch([said("answer three")], [])
    await c.send("third prompt", [], ctx)
    const seen: Record<string, unknown>[] = []
    let n = 0
    ;(globalThis as Record<string, unknown>).fetch = async (_u: string, init: { body: string }) => {
      seen.push(JSON.parse(init.body))
      return n++ === 0 ? new Response(JSON.stringify({ error: { message: "prompt (5000 tokens) leaves no room to answer in the context (4096)" } }), { status: 400 }) : new Response(stream(said(SUMMARY)), { status: 200 })
    }
    expect(await c.compact(ctx)).toBe(true)
    expect(seen).toHaveLength(2)
    expect((seen[1] as Req).messages.length).toBeLessThan((seen[0] as Req).messages.length)
    expect((seen[1] as Req).messages[0].role).toBe("user")
    expect(c.messages).toHaveLength(1)
  })

  test("Stop ends it, with no error, and the conversation is as it was", async () => {
    keep()
    const c = new ChatController()
    const errors: string[] = []
    c.onError = (title) => errors.push(title)
    await twoExchanges(c)
    waits()
    const running = c.compact(ctx)
    await new Promise((r) => setTimeout(r, 5))
    expect(c.compacting).toBe(true)
    expect(c.busy).not.toBeNull()
    c.stop()
    expect(await running).toBe(false)
    expect(errors).toEqual([])
    expect(c.messages).toHaveLength(4)
    expect(c.busy).toBeNull()
    expect(c.compacting).toBe(false)
  })

  test("while it runs nothing else can be sent", async () => {
    keep()
    const c = new ChatController()
    await twoExchanges(c)
    waits()
    const running = c.compact(ctx)
    await new Promise((r) => setTimeout(r, 5))
    await c.send("another", [], ctx)
    expect(c.messages).toHaveLength(4)
    c.stop()
    await running
  })

  test("taking a prompt back never takes the summary back, and a summary is not a prompt to rewrite", async () => {
    keep()
    const c = new ChatController()
    await twoExchanges(c)
    mockFetch([said(SUMMARY)], [])
    await c.compact(ctx)
    expect(c.undoLast()).toBeNull()                                      // only the summary is left: no prompt to take back
    expect(await c.edit(0, "x", ctx)).toBe(false)
    mockFetch([said("after")], [])
    await c.send("third prompt", [], ctx)
    expect(c.undoWouldEmpty()).toBe(false)                              // the summary stays, so the conversation does not empty
    const back = c.undoLast()
    expect(back?.text).toBe("third prompt")
    expect(c.messages).toHaveLength(1)
    expect(c.messages[0].compact).toBeDefined()
  })

  test("the summary is kept with the conversation, and the conversation is not named after it", async () => {
    keep()
    const c = new ChatController()
    await twoExchanges(c)
    mockFetch([said(SUMMARY)], [])
    await c.compact(ctx)
    const again = new ChatController()
    expect(again.messages[0].compact).toMatchObject({ auto: false })
    expect(again.index.items[0].title).toBe("first prompt")
  })
})

// A conversation started inside a project (issue #99): it is in the project from its first prompt, and the tools work in the project's folders before that.
describe("a new chat in a project", () => {
  function keep() {
    const data = new Map<string, string>()
    ;(globalThis as Record<string, unknown>).localStorage = {
      getItem: (k: string) => data.get(k) ?? null, setItem: (k: string, v: string) => { data.set(k, v) }, removeItem: (k: string) => { data.delete(k) },
    }
    return data
  }
  const said = (text: string) => sse(delta({ content: text }), { choices: [], usage: { completion_tokens: 1 } }, "data: [DONE]\n\n")

  test("it is in the project from its first prompt, and the project's folders are its folders at once", async () => {
    keep()
    const c = new ChatController()
    const p = c.addProject("Work", ["C:/work/app", "C:/work/app-wt2"])!
    expect(c.newSession(p)).toBe(true)
    expect(c.currentProject()).toBe(p)
    expect(c.folders()).toEqual(["C:/work/app", "C:/work/app-wt2"])              // before any prompt
    const seen: Record<string, unknown>[] = []
    mockFetch([said("hi")], seen)
    await c.send("hello", [], { ...ctx, agent: { available: true, allowed: true, shell: null, tools: [] }, folder: c.folders() })
    expect(c.index.items[0].project).toBe(p)
    expect(c.pendingProject).toBeNull()
    expect(c.currentProject()).toBe(p)
    expect((seen[0].strata_agent as { cwd: string }).cwd).toBe("C:/work/app")
  })

  test("a chat that is open is kept in the list, and the new one starts in the project", async () => {
    keep()
    const c = new ChatController()
    const p = c.addProject("Work", ["C:/w"])!
    mockFetch([said("one")], [])
    await c.send("first", [], ctx)
    expect(c.index.items[0].project).toBeUndefined()
    expect(c.newSession(p)).toBe(true)
    expect(c.messages).toHaveLength(0)
    mockFetch([said("two")], [])
    await c.send("second", [], ctx)
    expect(c.index.items.map((i) => i.project)).toEqual([p, undefined])
  })

  test("no project: no project; an unknown one is ignored", () => {
    keep()
    const c = new ChatController()
    c.addProject("Work", ["C:/w"])
    expect(c.newSession("nope")).toBe(true)
    expect(c.currentProject()).toBeUndefined()
    expect(c.newSession()).toBe(true)
    expect(c.currentProject()).toBeUndefined()
  })

  test("another new chat without a project takes it back, and so does opening a conversation, and deleting the project", async () => {
    keep()
    const c = new ChatController()
    const p = c.addProject("Work", ["C:/w"])!
    mockFetch([said("one")], [])
    await c.send("first", [], ctx)
    const id = c.index.active!
    c.newSession(p)
    expect(c.currentProject()).toBe(p)
    c.newSession()
    expect(c.currentProject()).toBeUndefined()
    c.newSession(p)
    expect(c.open(id)).toBe(true)
    expect(c.currentProject()).toBeUndefined()
    c.newSession(p)
    c.removeProject(p)
    expect(c.pendingProject).toBeNull()
  })

  test("not while an answer is being written", () => {
    keep()
    const c = new ChatController()
    const p = c.addProject("Work", ["C:/w"])!
    c.busy = { abort: new AbortController(), msg: { role: "assistant", text: "", time: 1 } }
    expect(c.newSession(p)).toBe(false)
    c.busy = null
  })
})
