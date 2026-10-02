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
let loadPerms: typeof import("./perms").loadPerms
let addPerm: typeof import("./perms").addPerm
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
  ;({ loadPerms, addPerm } = await import("./perms"))
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

  test("the request names the folder, the mode, the chat and the prompt's checkpoint, and no more than that", async () => {
    keep()
    const seen: Record<string, unknown>[] = []
    mockFetch([sse(...finish)], seen)
    const c = new ChatController()
    c.setSettings({ ...c.settings, agentMode: "plan" })
    await c.send("hi", [], { ...ctx, agent: ON, folder: ["C:/work/app", "C:/work/app-wt2"] })
    expect(seen[0].strata_agent).toEqual({ cwd: "C:/work/app", dirs: ["C:/work/app-wt2"], mode: "plan", session: c.index.active, allow: [], checkpoint: String(c.messages[0].time), run: expect.any(String) })      // the prompt's time is its checkpoint; the run id is how the answer is read again after a refresh
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

  test("while an answer is being written the chat cannot be deleted, and a chat that is open is already open", async () => {
    keep()
    mockFetch([answer("a")], [])
    const c = new ChatController()
    await c.send("one", [], ctx)
    const first = c.index.active!
    c.busy = { abort: new AbortController(), msg: c.messages[1] }
    expect(c.open(first)).toBe(true)                       // the one that is open is already open
    expect(c.remove(first)).toBe(false)
    expect(c.messages).toHaveLength(2)
    expect(c.newSession()).toBe(true)                      // another can be started: this one goes on answering
    expect(c.runningIds()).toEqual([first])
    expect(c.open(first)).toBe(true)
    expect(c.busy).not.toBeNull()
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
    mockFetch([said("answer one", { prompt_tokens: 900, completion_tokens: 50 })], [])
    await c.send("first prompt", [], small)
    expect(c.messages[1].stats!.ctx).toBe(950)
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
    expect(c.messages[0].compact).toMatchObject({ auto: true, before: 950 })
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
    mockFetch([said("answer one", { prompt_tokens: 900, completion_tokens: 50 })], [])
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

  test("an answer that is being written does not stop a new chat in the project", () => {
    keep()
    const c = new ChatController()
    const p = c.addProject("Work", ["C:/w"])!
    c.busy = { abort: new AbortController(), msg: { role: "assistant", text: "", time: 1 } }
    expect(c.newSession(p)).toBe(true)
    expect(c.currentProject()).toBe(p)
    c.busy = null
  })

  test("the project of a chat that has not started can be chosen, and changed, and taken away", () => {
    keep()
    const c = new ChatController()
    const p = c.addProject("Work", ["C:/work/app"])!
    const q = c.addProject("Home", ["C:/home"])!
    expect(c.setPendingProject(p)).toBe(true)
    expect(c.currentProject()).toBe(p)
    expect(c.folders()).toEqual(["C:/work/app"])
    expect(c.setPendingProject(q)).toBe(true)
    expect(c.folders()).toEqual(["C:/home"])
    expect(c.setPendingProject(null)).toBe(true)
    expect(c.currentProject()).toBeUndefined()
    expect(c.folders()).toEqual([])
  })

  test("a project that is not there, a chat that has started, and an answer being written are refused", async () => {
    keep()
    const c = new ChatController()
    const p = c.addProject("Work", ["C:/w"])!
    expect(c.setPendingProject("nope")).toBe(false)
    c.busy = { abort: new AbortController(), msg: { role: "assistant", text: "", time: 1 } }
    expect(c.setPendingProject(p)).toBe(false)
    c.busy = null
    mockFetch([said("one")], [])
    await c.send("first", [], ctx)
    expect(c.setPendingProject(p)).toBe(false)                          // it is in the list now: moving it is the sidebar's
    expect(c.index.items[0].project).toBeUndefined()
  })

  test("the chosen project is the project from the first prompt", async () => {
    keep()
    const c = new ChatController()
    const p = c.addProject("Work", ["C:/w"])!
    c.setPendingProject(p)
    mockFetch([said("one")], [])
    await c.send("first", [], ctx)
    expect(c.index.items[0].project).toBe(p)
  })
})

// Chats that answer at the same time (issue #99): another chat can be opened or started while one is answering. The one that was left goes on in the background (the server takes the
// requests one after the other) and is saved in its own place; coming back to it shows what has been written.
describe("chats that answer at the same time", () => {
  function keep() {
    const data = new Map<string, string>()
    ;(globalThis as Record<string, unknown>).localStorage = {
      getItem: (k: string) => data.get(k) ?? null, setItem: (k: string, v: string) => { data.set(k, v) }, removeItem: (k: string) => { data.delete(k) },
    }
    return data
  }
  const tick = () => new Promise((r) => setTimeout(r, 8))
  type Call = { body: Record<string, any>; chunk: (text: string) => void; end: () => void; aborted: boolean }
  /** A server that answers when it is told to: each request is a `Call` that is fed text and then ended. */
  function gated(): Call[] {
    const calls: Call[] = []
    ;(globalThis as Record<string, unknown>).fetch = (_u: string, init: { body: string; signal: AbortSignal }) => {
      let ctl!: ReadableStreamDefaultController<Uint8Array>
      const body = new ReadableStream<Uint8Array>({ start(c) { ctl = c } })
      const call: Call = {
        body: JSON.parse(init.body), aborted: false,
        chunk: (text) => ctl.enqueue(enc.encode(sse(delta({ content: text })))),
        end: () => { ctl.enqueue(enc.encode(sse({ choices: [], usage: { prompt_tokens: 50, completion_tokens: 5 } }, "data: [DONE]\n\n"))); ctl.close() },
      }
      init.signal.addEventListener("abort", () => { call.aborted = true; try { ctl.error(Object.assign(new Error("aborted"), { name: "AbortError" })) } catch { /* closed */ } })
      calls.push(call)
      return Promise.resolve(new Response(body, { status: 200 }))
    }
    return calls
  }

  test("a new chat can be started and answered while another is answering", async () => {
    keep()
    const calls = gated()
    const c = new ChatController()
    const a = c.send("question A", [], ctx)
    await tick()
    const idA = c.index.active!
    expect(c.busy).not.toBeNull()
    expect(c.newSession()).toBe(true)
    expect(c.busy).toBeNull()                                       // the new chat is not answering
    expect(c.runningIds()).toEqual([idA])
    const b = c.send("question B", [], ctx)
    await tick()
    expect(calls).toHaveLength(2)                                   // both requests are out; the server takes them one after the other
    expect(calls[1].body.messages.map((m: { content: string }) => m.content)).toEqual(["question B"])
    expect(calls[0].body.messages.map((m: { content: string }) => m.content)).toEqual(["question A"])
    calls[1].chunk("answer B"); calls[1].end()
    await b
    expect(c.messages.map((m) => m.text)).toEqual(["question B", "answer B"])
    expect(c.runningIds()).toEqual([idA])
    calls[0].chunk("answer A"); calls[0].end()
    await a
    expect(c.runningIds()).toEqual([])
    expect(c.messages.map((m) => m.text)).toEqual(["question B", "answer B"])        // the page still shows the one that is open
    expect(c.open(idA)).toBe(true)
    expect(c.messages.map((m) => m.text)).toEqual(["question A", "answer A"])       // and the other has its answer, stored in its own place
  })

  test("another chat can be opened while one answers, and the live answer is there when coming back", async () => {
    keep()
    const calls = gated()
    const c = new ChatController()
    calls.length = 0
    const first = c.send("old one", [], ctx)
    await tick(); calls[0].chunk("old answer"); calls[0].end(); await first
    const old = c.index.active!
    c.newSession()
    const run = c.send("long one", [], ctx)
    await tick()
    const long = c.index.active!
    calls[1].chunk("part one, ")
    await tick()
    expect(c.open(old)).toBe(true)
    expect(c.messages.map((m) => m.text)).toEqual(["old one", "old answer"])
    expect(c.busy).toBeNull()
    calls[1].chunk("part two")
    await tick()
    expect(c.open(long)).toBe(true)
    expect(c.messages.map((m) => m.text)).toEqual(["long one", "part one, part two"])      // what has been written, live
    expect(c.busy?.msg).toBe(c.messages[1])
    calls[1].chunk(" and the end"); calls[1].end()
    await run
    expect(c.messages[1].text).toBe("part one, part two and the end")
    expect(c.busy).toBeNull()
    expect(new ChatController().messages.map((m) => m.text)).toEqual(["long one", "part one, part two and the end"])      // kept, in the open conversation's place
  })

  test("the answer of a chat that is not open is stored in its place, and it moves up the list", async () => {
    const data = keep()
    const calls = gated()
    const c = new ChatController()
    const first = c.send("first", [], ctx)
    await tick()
    const idFirst = c.index.active!
    c.newSession()
    const second = c.send("second", [], ctx)
    await tick()
    const idSecond = c.index.active!
    calls[1].chunk("two"); calls[1].end(); await second
    expect([...c.index.items].sort((x, y) => y.time - x.time)[0].id).toBe(idSecond)
    await tick(); await tick()                                      // a later moment than the other finished
    calls[0].chunk("one, finished later"); calls[0].end(); await first
    expect(c.index.active).toBe(idSecond)
    const stored = JSON.parse(data.get("strata.chat." + idFirst)!) as { text: string }[]
    expect(stored.map((m) => m.text)).toEqual(["first", "one, finished later"])
    expect([...c.index.items].sort((x, y) => y.time - x.time)[0].id).toBe(idFirst)          // it finished last, so it is the newest (the list is shown by time)
    expect(c.index.items.find((i) => i.id === idFirst)!.title).toBe("first")
  })

  test("a chat answers one question at a time, but another chat can ask", async () => {
    keep()
    const calls = gated()
    const c = new ChatController()
    const a = c.send("in A", [], ctx)
    await tick()
    await c.send("again in A", [], ctx)                             // refused: A is answering
    expect(calls).toHaveLength(1)
    expect(c.messages).toHaveLength(2)
    c.newSession()
    const b = c.send("in B", [], ctx)
    await tick()
    expect(calls).toHaveLength(2)
    calls[0].end(); calls[1].end()
    await Promise.all([a, b])
  })

  test("Stop ends the open chat's answer only", async () => {
    keep()
    const calls = gated()
    const c = new ChatController()
    const a = c.send("A", [], ctx)
    await tick()
    c.newSession()
    const b = c.send("B", [], ctx)
    await tick()
    c.stop()
    await b
    expect(calls[1].aborted).toBe(true)
    expect(calls[0].aborted).toBe(false)
    expect(c.messages[1].stopped).toBe(true)
    expect(c.runningIds()).toHaveLength(1)
    calls[0].chunk("fine"); calls[0].end()
    await a
  })

  test("a chat that is answering cannot be deleted until it is done", async () => {
    keep()
    const calls = gated()
    const c = new ChatController()
    const a = c.send("A", [], ctx)
    await tick()
    const idA = c.index.active!
    c.newSession()
    expect(c.remove(idA)).toBe(false)
    expect(c.index.items.some((i) => i.id === idA)).toBe(true)
    calls[0].end()
    await a
    expect(c.remove(idA)).toBe(true)
    expect(c.index.items.some((i) => i.id === idA)).toBe(false)
  })

  test("a chat that waits for the user's answer to a question is told apart", async () => {
    keep()
    const calls = gated()
    const c = new ChatController()
    const a = c.send("A", [], ctx)
    await tick()
    const idA = c.index.active!
    c.messages[1].tools = [{ id: "t", name: "Bash", at: 0, rat: 0, state: "asking" }]
    expect(c.askingIds()).toEqual([idA])
    c.newSession()
    expect(c.askingIds()).toEqual([idA])
    expect(c.runningIds()).toEqual([idA])
    calls[0].end()
    await a
    expect(c.askingIds()).toEqual([])
  })

  test("a chat that is summarised in the background keeps its summary in place", async () => {
    keep()
    const c = new ChatController()
    const said = (text: string) => sse(delta({ content: text }), { choices: [], usage: { completion_tokens: 1 } }, "data: [DONE]\n\n")
    mockFetch([said("answer one"), said("answer two")], [])
    await c.send("first", [], ctx)
    await c.send("second", [], ctx)
    const id = c.index.active!
    const calls = gated()
    const summary = c.compact(ctx)
    await tick()
    expect(c.compacting).toBe(true)
    c.newSession()
    expect(c.compacting).toBe(false)                                 // the new chat is not being summarised
    calls[0].chunk("<summary>1. Primary request: kept</summary>"); calls[0].end()
    await summary
    expect(c.open(id)).toBe(true)
    expect(c.messages).toHaveLength(1)
    expect(c.messages[0].compact).toBeDefined()
    expect(c.messages[0].text).toContain("1. Primary request: kept")
  })
})

// Rules that last (issue #99): an answer kept for a project or for everywhere, and the rules that go with every request.
describe("rules that last", () => {
  function keep() {
    const data = new Map<string, string>()
    ;(globalThis as Record<string, unknown>).localStorage = {
      getItem: (k: string) => data.get(k) ?? null, setItem: (k: string, v: string) => { data.set(k, v) }, removeItem: (k: string) => { data.delete(k) },
    }
    return data
  }
  const finish = [delta({ content: "ok" }), { choices: [], usage: { completion_tokens: 1 } }, "data: [DONE]\n\n"]
  const ON = { available: true, allowed: true, shell: "bash", tools: ["Read", "Bash"] }
  const asking = (rule: string | null = "Bash(npm test:*)"): Message[] => [{ role: "user", text: "x", time: 1 }, { role: "assistant", text: "", time: 2, tools: [{ id: "c1", name: "Bash", at: 0, rat: 0, state: "asking", server: "agent", ask: { id: "q1", tool: "Bash", why: "w", danger: false, rule } }] }]
  const posts = () => {
    const sent: { body: { id: string; decision: string } }[] = []
    ;(globalThis as Record<string, unknown>).fetch = async (_u: string, init: { body: string }) => { sent.push({ body: JSON.parse(init.body) }); return new Response('{"ok":true}', { status: 200 }) }
    return sent
  }

  test("\"always allow in this project\" is allowed for the rest of this request and kept for the project", async () => {
    keep()
    const c = new ChatController()
    const p = c.addProject("Work", ["C:/w"])!
    c.messages = asking()
    c.index = { ...c.index, active: "chat9" }
    const sent = posts()
    expect(await c.answer("c1", "allow_chat", { scope: { kind: "project", id: p }, effect: "allow" })).toBe(true)
    expect(sent[0].body).toEqual({ id: "q1", decision: "allow_chat" })
    expect(c.messages[1].tools![0].ask).toMatchObject({ answer: "allow_chat", kept: { scope: "project", effect: "allow" } })
    expect(loadPerms(store).projects[p].allow).toEqual(["Bash(npm test:*)"])
    expect(rulesOf(store, "chat9")).toEqual([])                            // not kept for the chat as well: it is kept for the project
  })

  test("\"never anywhere\" is a no for now and a deny rule for good", async () => {
    keep()
    const c = new ChatController()
    c.messages = asking()
    c.index = { ...c.index, active: "chat9" }
    const sent = posts()
    await c.answer("c1", "allow", { scope: { kind: "everywhere" }, effect: "deny" })
    expect(sent[0].body.decision).toBe("deny")
    expect(loadPerms(store).everywhere.deny).toEqual(["Bash(npm test:*)"])
    expect(c.messages[1].tools![0].ask?.kept).toEqual({ scope: "everywhere", effect: "deny" })
  })

  test("a call with no rule (a dangerous command) keeps nothing", async () => {
    keep()
    const c = new ChatController()
    c.messages = asking(null)
    c.index = { ...c.index, active: "chat9" }
    posts()
    await c.answer("c1", "allow", { scope: { kind: "everywhere" }, effect: "allow" })
    expect(loadPerms(store).everywhere.allow).toEqual([])
    expect(c.messages[1].tools![0].ask?.kept).toBeUndefined()
  })

  test("a server that did not take the answer keeps nothing", async () => {
    keep()
    const c = new ChatController()
    c.messages = asking()
    c.index = { ...c.index, active: "chat9" }
    ;(globalThis as Record<string, unknown>).fetch = async () => new Response(JSON.stringify({ error: { message: "no such question" } }), { status: 404 })
    c.onError = () => {}
    expect(await c.answer("c1", "allow_chat", { scope: { kind: "everywhere" }, effect: "allow" })).toBe(false)
    expect(loadPerms(store).everywhere.allow).toEqual([])
    expect(c.messages[1].tools![0].ask?.answer).toBeUndefined()
  })

  test("a request carries the chat's, the project's and everywhere's allowed rules and the refused ones", async () => {
    keep()
    const c = new ChatController()
    const p = c.addProject("Work", ["C:/w"])!
    c.newSession(p)
    addPerm(store, { kind: "everywhere" }, "allow", "Read")
    addPerm(store, { kind: "everywhere" }, "deny", "Bash(rm:*)")
    addPerm(store, { kind: "project", id: p }, "allow", "Edit(src/**)")
    addPerm(store, { kind: "project", id: p }, "deny", "Bash(curl:*)")
    const seen: Record<string, unknown>[] = []
    mockFetch([sse(...finish)], seen)
    await c.send("go", [], { ...ctx, agent: ON, folder: c.folders() })
    expect(seen[0].strata_agent).toMatchObject({ allow: ["Edit(src/**)", "Read"], deny: ["Bash(curl:*)", "Bash(rm:*)"] })
  })

  test("a chat that is in no project gets only everywhere's rules", async () => {
    keep()
    const c = new ChatController()
    c.addProject("Work", ["C:/w"])
    addPerm(store, { kind: "everywhere" }, "allow", "Read")
    addPerm(store, { kind: "project", id: "other" }, "deny", "Bash")
    const seen: Record<string, unknown>[] = []
    mockFetch([sse(...finish)], seen)
    await c.send("go", [], { ...ctx, agent: ON })
    expect((seen[0].strata_agent as { allow: string[]; deny?: string[] }).allow).toEqual(["Read"])
    expect("deny" in (seen[0].strata_agent as object)).toBe(false)
  })

  test("deleting a project takes its rules with it", () => {
    keep()
    const c = new ChatController()
    const p = c.addProject("Work", ["C:/w"])!
    addPerm(store, { kind: "project", id: p }, "allow", "Read")
    c.removeProject(p)
    expect(loadPerms(store).projects[p]).toBeUndefined()
  })
})

// Messages typed while an answer is being written wait in line and go when it ends (issue #99).
describe("messages that wait while an answer is written", () => {
  function keep() {
    const data = new Map<string, string>()
    ;(globalThis as Record<string, unknown>).localStorage = {
      getItem: (k: string) => data.get(k) ?? null, setItem: (k: string, v: string) => { data.set(k, v) }, removeItem: (k: string) => { data.delete(k) },
    }
    return data
  }
  const tick = () => new Promise((r) => setTimeout(r, 8))
  type Call = { body: Record<string, any>; chunk: (text: string) => void; end: () => void }
  function gated(): Call[] {
    const calls: Call[] = []
    ;(globalThis as Record<string, unknown>).fetch = (_u: string, init: { body: string; signal: AbortSignal }) => {
      let ctl!: ReadableStreamDefaultController<Uint8Array>
      const body = new ReadableStream<Uint8Array>({ start(c) { ctl = c } })
      const call: Call = {
        body: JSON.parse(init.body),
        chunk: (text) => ctl.enqueue(enc.encode(sse(delta({ content: text })))),
        end: () => { ctl.enqueue(enc.encode(sse({ choices: [], usage: { prompt_tokens: 50, completion_tokens: 5 } }, "data: [DONE]\n\n"))); ctl.close() },
      }
      init.signal.addEventListener("abort", () => { try { ctl.error(Object.assign(new Error("aborted"), { name: "AbortError" })) } catch { /* closed */ } })
      calls.push(call)
      return Promise.resolve(new Response(body, { status: 200 }))
    }
    return calls
  }
  const file = { kind: "file" as const, name: "a.py", text: "x = 1" }

  test("a message typed while the answer is written is kept, and one typed when it is not is not", async () => {
    keep()
    const calls = gated()
    const c = new ChatController()
    expect(c.queue("early", [], ctx)).toBe(false)                         // nothing is answering: it is to be sent, not kept
    const a = c.send("first", [], ctx)
    await tick()
    expect(c.queue("  second  ", [file], ctx)).toBe(true)
    expect(c.queue("   ", [], ctx)).toBe(false)                           // nothing to keep
    expect(c.queuedOf().map((q) => [q.text, q.files.length])).toEqual([["second", 1]])
    calls[0].chunk("one"); calls[0].end()
    await a
  })

  test("when the answer ends well the next one goes by itself, then the next, in order, each after the one before", async () => {
    keep()
    const calls = gated()
    const c = new ChatController()
    const a = c.send("first", [], ctx)
    await tick()
    c.queue("second", [], ctx)
    c.queue("third", [file], ctx)
    expect(calls).toHaveLength(1)
    calls[0].chunk("answer 1"); calls[0].end()
    await a
    await tick()
    expect(calls).toHaveLength(2)                                         // the second went when the first ended
    expect(calls[1].body.messages.map((m: { content: unknown }) => (typeof m.content === "string" ? m.content : "?")).slice(-1)).toEqual(["second"])
    expect(c.queuedOf().map((q) => q.text)).toEqual(["third"])
    calls[1].chunk("answer 2"); calls[1].end()
    await tick(); await tick()
    expect(calls).toHaveLength(3)
    const last = calls[2].body.messages.at(-1).content
    expect(String(last)).toContain("third")
    expect(String(last)).toContain("a.py")                                // its file went with it
    calls[2].chunk("answer 3"); calls[2].end()
    await tick(); await tick()
    expect(c.messages.map((m) => m.text)).toEqual(["first", "answer 1", "second", "answer 2", "third", "answer 3"])
    expect(c.queuedOf()).toEqual([])
  })

  test("Stop sends nothing: the messages stay, and can be sent, taken back or dropped", async () => {
    keep()
    const calls = gated()
    const c = new ChatController()
    const a = c.send("first", [], ctx)
    await tick()
    c.queue("second", [], ctx)
    c.queue("third", [], ctx)
    c.stop()
    await a
    await tick()
    expect(calls).toHaveLength(1)
    expect(c.queuedOf().map((q) => q.text)).toEqual(["second", "third"])
    const back = c.unqueue(c.queuedOf()[1].id)
    expect(back?.text).toBe("third")
    expect(c.unqueue("nope")).toBeNull()
    const sending = c.sendQueued(c.queuedOf()[0].id)
    await tick()
    expect(calls).toHaveLength(2)
    expect(c.queuedOf()).toEqual([])
    calls[1].end()
    expect(await sending).toBe(true)
  })

  test("an answer that failed sends nothing either", async () => {
    keep()
    ;(globalThis as Record<string, unknown>).fetch = async () => new Response(JSON.stringify({ error: { message: "the engine is busy" } }), { status: 500 })
    const c = new ChatController()
    c.onError = () => {}
    const a = c.send("first", [], ctx)
    c.queue("second", [], ctx)
    await a
    expect(c.queuedOf().map((q) => q.text)).toEqual(["second"])
  })

  test("each conversation has its own line, and one that is not open goes on when it is opened", async () => {
    keep()
    const calls = gated()
    const c = new ChatController()
    const a = c.send("in A", [], ctx)
    await tick()
    const idA = c.index.active!
    c.queue("waits in A", [], ctx)
    c.newSession()
    expect(c.queuedOf()).toEqual([])                                      // B has none
    calls[0].chunk("answer A"); calls[0].end()
    await a
    await tick()
    expect(calls).toHaveLength(1)                                         // A is not open: its waiting message does not go by itself
    expect(c.open(idA)).toBe(true)
    await tick()
    expect(calls).toHaveLength(2)                                         // it goes when A is opened
    expect(String(calls[1].body.messages.at(-1).content)).toBe("waits in A")
    calls[1].end()
    await tick()
  })

  test("a conversation that is deleted takes its line with it", async () => {
    keep()
    const calls = gated()
    const c = new ChatController()
    const a = c.send("in A", [], ctx)
    await tick()
    const idA = c.index.active!
    c.queue("waits", [], ctx)
    c.newSession()
    calls[0].end()
    await a
    expect(c.remove(idA)).toBe(true)
    expect(c.open(idA)).toBe(false)
    await tick()
    expect(calls).toHaveLength(1)
  })

  test("no more than twenty wait", async () => {
    keep()
    const calls = gated()
    const c = new ChatController()
    const a = c.send("first", [], ctx)
    await tick()
    for (let i = 0; i < 25; i++) c.queue("m" + i, [], ctx)
    expect(c.queuedOf()).toHaveLength(20)
    expect(c.queuedOf()[0].text).toBe("m5")                                // the oldest were let go
    c.stop()
    await a
    await tick()
    expect(calls).toHaveLength(1)
  })
})

// Rewind (issue #99): the conversation is cut at a prompt, and each request carries its prompt's checkpoint.
describe("rewinding a conversation", () => {
  function keep() {
    const data = new Map<string, string>()
    ;(globalThis as Record<string, unknown>).localStorage = {
      getItem: (k: string) => data.get(k) ?? null, setItem: (k: string, v: string) => { data.set(k, v) }, removeItem: (k: string) => { data.delete(k) },
    }
    return data
  }
  const said = (text: string) => sse(delta({ content: text }), { choices: [], usage: { completion_tokens: 1 } }, "data: [DONE]" + String.fromCharCode(10) + String.fromCharCode(10))
  const ON = { available: true, allowed: true, shell: "bash", tools: ["Write"] }
  const three = async () => {
    const c = new ChatController()
    mockFetch([said("a1"), said("a2"), said("a3")], [])
    await c.send("one", [], ctx)
    await c.send("two", [{ kind: "file", name: "f.txt", text: "ftext" }], ctx)
    await c.send("three", [], ctx)
    return c
  }

  test("the chat is cut before the prompt, which comes back with the attachments that are held", async () => {
    keep()
    const c = await three()
    const back = c.rewindTo(2)
    expect(back?.text).toBe("two")
    expect(back?.attachments.map((a) => a.name)).toEqual(["f.txt"])
    expect(back?.removed.map((m) => m.text)).toEqual(["two", "a2", "three", "a3"])
    expect(c.messages.map((m) => m.text)).toEqual(["one", "a1"])
    expect(new ChatController().messages.map((m) => m.text)).toEqual(["one", "a1"])        // kept
  })

  test("to the first prompt: nothing is left, and the conversation leaves the list", async () => {
    keep()
    const c = await three()
    expect(c.rewindTo(0)?.text).toBe("one")
    expect(c.messages).toEqual([])
    expect(c.index.items).toHaveLength(0)
  })

  test("only a prompt can be gone back to, and not while an answer is being written", async () => {
    keep()
    const c = await three()
    expect(c.rewindTo(1)).toBeNull()                                            // an answer
    expect(c.rewindTo(99)).toBeNull()
    expect(c.rewindTo(-1)).toBeNull()
    expect(c.messages).toHaveLength(6)
    c.busy = { abort: new AbortController(), msg: c.messages[5] }
    expect(c.rewindTo(2)).toBeNull()
    c.busy = null
  })

  test("the checkpoint of a prompt is its time, and a summary has none", async () => {
    keep()
    const c = await three()
    expect(c.checkpointOf(2)).toBe(String(c.messages[2].time))
    expect(c.checkpointOf(1)).toBeNull()
    c.messages.unshift({ role: "user", text: "summary", time: 5, compact: { before: 1, after: 1, auto: false } })
    expect(c.checkpointOf(0)).toBeNull()
  })

  test("each request carries the checkpoint of its own prompt, so the files the tools change in it can be put back", async () => {
    keep()
    const c = new ChatController()
    const seen: Record<string, unknown>[] = []
    mockFetch([said("a"), said("b")], seen)
    await c.send("first", [], { ...ctx, agent: ON, folder: "C:/w" })
    await c.send("second", [], { ...ctx, agent: ON, folder: "C:/w" })
    const cps = seen.map((b) => (b.strata_agent as { checkpoint: string }).checkpoint)
    expect(cps).toEqual([String(c.messages[0].time), String(c.messages[2].time)])
    expect(new Set(cps).size).toBe(2)
  })
})

// The model asks the user (AskUserQuestion): a form on its call, and the answers go back (issue #99).
describe("questions the model asks", () => {
  function keep() {
    const data = new Map<string, string>()
    ;(globalThis as Record<string, unknown>).localStorage = {
      getItem: (k: string) => data.get(k) ?? null, setItem: (k: string, v: string) => { data.set(k, v) }, removeItem: (k: string) => { data.delete(k) },
    }
    return data
  }
  const questions = [{ question: "Which library?", header: "Library", multiSelect: false, options: [{ label: "requests", description: "usual" }, { label: "httpx", description: "async" }] }]
  const finish = [delta({ content: "ok" }), { choices: [], usage: { completion_tokens: 1 } }, "data: [DONE]" + String.fromCharCode(10) + String.fromCharCode(10)]

  test("the question is a form on the call, which waits for the user", async () => {
    keep()
    let release!: () => void
    const held = new Promise<void>((r) => { release = r })
    ;(globalThis as Record<string, unknown>).fetch = async () => {
      const first = sse({ strata_mcp: { event: "start", id: "c1", name: "AskUserQuestion" } }, { strata_mcp: { event: "call", id: "c1", name: "AskUserQuestion", server: "agent", tool: "AskUserQuestion", arguments: { questions }, round: 1 } },
        { strata_mcp: { event: "question", id: "q1", call_id: "c1", questions } })
      return new Response(new ReadableStream({ async start(c) { c.enqueue(enc.encode(first)); await held; c.enqueue(enc.encode(sse(...finish))); c.close() } }), { status: 200 })
    }
    const c = new ChatController()
    const p = c.send("pick", [], ctx)
    await new Promise((r) => setTimeout(r, 20))
    const call = c.messages[1].tools![0]
    expect(call.state).toBe("asking")
    expect(call.question).toMatchObject({ id: "q1", questions: [{ header: "Library", multiSelect: false }] })
    expect(call.question!.answers).toBeUndefined()
    release()
    await p
  })

  test("answering sends the id and the answers, and the form says it is answered", async () => {
    keep()
    const c = new ChatController()
    c.messages = [{ role: "user", text: "x", time: 1 }, { role: "assistant", text: "", time: 2, tools: [{ id: "c1", name: "AskUserQuestion", at: 0, rat: 0, state: "asking", question: { id: "q1", questions } }] }]
    const posts: { url: string; body: unknown }[] = []
    ;(globalThis as Record<string, unknown>).fetch = async (u: string, init: { body: string }) => { posts.push({ url: u, body: JSON.parse(init.body) }); return new Response('{"ok":true}', { status: 200 }) }
    expect(await c.answerQuestion("c1", { "Which library?": ["httpx"] })).toBe(true)
    expect(posts[0].url).toContain("agent/question")
    expect(posts[0].body).toEqual({ id: "q1", answers: { "Which library?": ["httpx"] } })
    expect(c.messages[1].tools![0].question!.answers).toEqual({ "Which library?": ["httpx"] })
    expect(c.messages[1].tools![0].state).toBe("running")
    expect(await c.answerQuestion("c1", null)).toBe(false)                          // answered once
  })

  test("skipping sends null; a server that did not take the answer leaves the form for another try", async () => {
    keep()
    const c = new ChatController()
    const errors: string[] = []
    c.onError = (title) => errors.push(title)
    c.messages = [{ role: "user", text: "x", time: 1 }, { role: "assistant", text: "", time: 2, tools: [{ id: "c1", name: "AskUserQuestion", at: 0, rat: 0, state: "asking", question: { id: "q1", questions } }] }]
    ;(globalThis as Record<string, unknown>).fetch = async () => new Response(JSON.stringify({ error: { message: "no such question" } }), { status: 404 })
    expect(await c.answerQuestion("c1", null)).toBe(false)
    expect(errors).toEqual(["The answer was not taken"])
    expect(c.messages[1].tools![0].question!.answers).toBeUndefined()
    const posts: unknown[] = []
    ;(globalThis as Record<string, unknown>).fetch = async (_u: string, init: { body: string }) => { posts.push(JSON.parse(init.body)); return new Response('{"ok":true}', { status: 200 }) }
    expect(await c.answerQuestion("c1", null)).toBe(true)
    expect(posts).toEqual([{ id: "q1", answers: null }])
    expect(c.messages[1].tools![0].question!.answers).toBeNull()
  })
})

// The user's hooks (issue #99): what they did comes in the stream, on the call they were about or on the answer.
describe("what the user's hooks did, in the chat", () => {
  function keep() {
    const data = new Map<string, string>()
    ;(globalThis as Record<string, unknown>).localStorage = {
      getItem: (k: string) => data.get(k) ?? null, setItem: (k: string, v: string) => { data.set(k, v) }, removeItem: (k: string) => { data.delete(k) },
    }
  }
  const finish = [delta({ content: "ok" }), { choices: [], usage: { completion_tokens: 1 } }, "data: [DONE]" + String.fromCharCode(10) + String.fromCharCode(10)]
  const note = (more: Record<string, unknown>) => ({ event: "hook", hook: "h0123abcd", command: "./check.sh", ok: false, code: 2, blocked: false, timeout: false, error: null, text: "", ms: 5, ...more })

  test("a hook that stopped a call is a note on that call; a prompt or stop hook is a note on the answer", async () => {
    keep()
    ;(globalThis as Record<string, unknown>).fetch = async () => new Response(sse(
      { strata_mcp: note({ on: "prompt", call_id: null, ok: true, code: 0, text: "branch is main" }) },
      { strata_mcp: { event: "start", id: "c1", name: "Bash" } }, { strata_mcp: { event: "call", id: "c1", name: "Bash", server: "agent", tool: "Bash", arguments: { command: "git commit" }, round: 1 } },
      { strata_mcp: note({ on: "before_tool", call_id: "c1", tool: "Bash", blocked: true, text: "no commits to main" }) },
      { strata_mcp: { event: "result", id: "c1", ok: false, text: "Bash was stopped by a hook: no commits to main", chars: 40, truncated: false, ms: 9 } },
      { strata_mcp: note({ on: "stop", call_id: null, ok: true, code: 0, text: "checked" }) },
      ...finish), { status: 200 })
    const c = new ChatController()
    await c.send("commit", [], ctx)
    const m = c.messages[1]
    expect(m.hooks!.map((h) => [h.on, h.text])).toEqual([["prompt", "branch is main"], ["stop", "checked"]])
    expect(m.tools![0].hooks!).toHaveLength(1)
    expect(m.tools![0].hooks![0]).toMatchObject({ on: "before_tool", blocked: true, text: "no commits to main", code: 2 })
  })

  test("what is not a hook's report, and a call that is not shown, do not break the answer", async () => {
    keep()
    ;(globalThis as Record<string, unknown>).fetch = async () => new Response(sse({ strata_mcp: { event: "hook", on: "sometime" } }, { strata_mcp: note({ on: "after_tool", call_id: "gone" }) }, ...finish), { status: 200 })
    const c = new ChatController()
    await c.send("hi", [], ctx)
    expect(c.messages[1].text).toBe("ok")
    expect(c.messages[1].hooks!.map((h) => h.on)).toEqual(["after_tool"])                  // no such call: it is kept on the answer
  })

  test("without hooks the message has none", async () => {
    keep()
    ;(globalThis as Record<string, unknown>).fetch = async () => new Response(sse(...finish), { status: 200 })
    const c = new ChatController()
    await c.send("hi", [], ctx)
    expect(c.messages[1].hooks).toBeUndefined()
  })
})

// Sub-agents (issue #99): what a helper did comes in the stream as steps on the Task call it works for.
describe("what a helper did, on the Task call", () => {
  function keep() {
    const data = new Map<string, string>()
    ;(globalThis as Record<string, unknown>).localStorage = {
      getItem: (k: string) => data.get(k) ?? null, setItem: (k: string, v: string) => { data.set(k, v) }, removeItem: (k: string) => { data.delete(k) },
    }
  }
  const finish = [delta({ content: "ok" }), { choices: [], usage: { completion_tokens: 1 } }, "data: [DONE]" + String.fromCharCode(10) + String.fromCharCode(10)]
  const task = [
    { strata_mcp: { event: "start", id: "t1", name: "Task" } },
    { strata_mcp: { event: "call", id: "t1", name: "Task", server: "agent", tool: "Task", arguments: { description: "find hello", prompt: "find where hello is said" }, round: 1 } },
  ]

  test("steps are listed under the Task, each with its result, and the helper's start and end are kept", async () => {
    keep()
    ;(globalThis as Record<string, unknown>).fetch = async () => new Response(sse(...task,
      { strata_mcp: { event: "helper", state: "start", kind: "explore", description: "find hello", call_id: "t1" } },
      { strata_mcp: { event: "step", id: "s1", name: "Grep", arguments: { pattern: "hello" }, call_id: "t1" } },
      { strata_mcp: { event: "step_result", id: "s1", ok: true, chars: 40, text: "a.txt:1:hello", call_id: "t1" } },
      { strata_mcp: { event: "step", id: "s2", name: "Read", arguments: { file_path: "../x" }, call_id: "t1" } },
      { strata_mcp: { event: "step_result", id: "s2", ok: false, chars: 10, text: "denied", call_id: "t1" } },
      { strata_mcp: { event: "helper", state: "end", ok: true, steps: 2, call_id: "t1" } },
      { strata_mcp: { event: "result", id: "t1", ok: true, text: "[Report of the helper (explore, 2 steps).]\n\na.txt", chars: 50, truncated: false, ms: 9 } },
      ...finish), { status: 200 })
    const c = new ChatController()
    await c.send("look", [], ctx)
    const call = c.messages[1].tools![0]
    expect(call.steps!.map((s) => [s.name, s.state])).toEqual([["Grep", "done"], ["Read", "error"]])
    expect(call.steps![0]).toMatchObject({ arguments: { pattern: "hello" }, text: "a.txt:1:hello" })
    expect(call.helper).toEqual({ kind: "explore", description: "find hello", state: "done", steps: 2 })
    expect(call.state).toBe("done")
  })

  test("a helper that did not finish is marked, and a step of no known call or a bad one is left out", async () => {
    keep()
    ;(globalThis as Record<string, unknown>).fetch = async () => new Response(sse(...task,
      { strata_mcp: { event: "helper", state: "start", kind: "general", description: "d", call_id: "t1" } },
      { strata_mcp: { event: "step", id: "s1", name: "Bash", arguments: { command: "ls" }, call_id: "nobody" } },
      { strata_mcp: { event: "step_result", id: "unknown", ok: true, call_id: "t1" } },
      { strata_mcp: { event: "helper", state: "end", ok: false, call_id: "t1" } },
      ...finish), { status: 200 })
    const c = new ChatController()
    await c.send("look", [], ctx)
    const call = c.messages[1].tools![0]
    expect(call.helper!.state).toBe("failed")
    expect(call.steps).toEqual([])
  })

  test("a call that is not a Task has no helper", async () => {
    keep()
    ;(globalThis as Record<string, unknown>).fetch = async () => new Response(sse(
      { strata_mcp: { event: "start", id: "c1", name: "Read" } }, { strata_mcp: { event: "call", id: "c1", name: "Read", server: "agent", tool: "Read", arguments: { file_path: "a" }, round: 1 } }, ...finish), { status: 200 })
    const c = new ChatController()
    await c.send("look", [], ctx)
    expect(c.messages[1].tools![0].helper).toBeUndefined()
    expect(c.messages[1].tools![0].steps).toBeUndefined()
  })
})

// A refresh while the agent works (issue #99 follow-up): the answer goes on in the server, the cut read is not an error, and the page that comes back reads the run again from its start.
describe("a refresh while the agent is working", () => {
  const ON = { available: true, allowed: true, shell: "bash", tools: ["Read"] }
  const finish = [delta({ content: "all done" }), { choices: [], usage: { completion_tokens: 3 } }, "data: [DONE]" + String.fromCharCode(10) + String.fromCharCode(10)]
  const toolCall = [
    { strata_mcp: { event: "start", id: "c1", name: "Read" } },
    { strata_mcp: { event: "call", id: "c1", name: "Read", server: "agent", tool: "Read", arguments: { file_path: "a.txt" }, round: 1 } },
    { strata_mcp: { event: "result", id: "c1", ok: true, text: "contents", chars: 8, truncated: false, ms: 5 } },
  ]
  function keep() {
    const data = new Map<string, string>()
    ;(globalThis as Record<string, unknown>).localStorage = {
      getItem: (k: string) => data.get(k) ?? null, setItem: (k: string, v: string) => { data.set(k, v) }, removeItem: (k: string) => { data.delete(k) },
    }
  }
  const pending = () => store.get<Record<string, { run: string; prompt: string; time: number }>>("runs", {})
  /** A request whose stream gives its first events and then fails, as a refreshed page's does. */
  function cutAfterFirst(c: InstanceType<typeof ChatController>, calls: { url: string; body?: string }[]) {
    let cut!: () => void
    const held = new Promise<void>((r) => { cut = r })
    ;(globalThis as Record<string, unknown>).fetch = async (u: string, init?: { body?: string }) => {
      calls.push({ url: String(u), body: init?.body })
      return new Response(new ReadableStream({
        async start(ctl) { ctl.enqueue(enc.encode(sse(delta({ reasoning_content: "thinking first" })))); await held; ctl.error(new TypeError("network error")) },
      }), { status: 200 })
    }
    return () => { ;(c as unknown as { leaving: boolean }).leaving = true; cut() }
  }

  test("a read cut by leaving the page is not an error, and the run is remembered by its id", async () => {
    keep()
    const c = new ChatController()
    const calls: { url: string; body?: string }[] = []
    const leave = cutAfterFirst(c, calls)
    const p = c.send("work on it", [], { ...ctx, agent: ON, folder: "C:/proj" })
    await new Promise((r) => setTimeout(r, 20))
    leave()
    await p
    expect(JSON.parse(calls[0].body!).strata_agent.run).toMatch(/^[A-Za-z0-9_-]{8,64}$/)
    expect(c.messages[1].error).toBeUndefined()
    expect(Object.values(pending())).toHaveLength(1)
    expect(pending()[c.index.active!].run).toBe(JSON.parse(calls[0].body!).strata_agent.run)
  })

  test("the page that comes back reads the run again from its start and the conversation is whole", async () => {
    keep()
    const c = new ChatController()
    const calls: { url: string; body?: string }[] = []
    const leave = cutAfterFirst(c, calls)
    const p = c.send("work on it", [], { ...ctx, agent: ON, folder: "C:/proj" })
    await new Promise((r) => setTimeout(r, 20))
    leave()
    await p
    const runId = JSON.parse(calls[0].body!).strata_agent.run

    const back = new ChatController()                                                   // the page after the refresh: it restores what was stored
    expect(back.messages).toHaveLength(1)                                              // the prompt; the answer that was being written is not stored half-empty
    const seen: string[] = []
    ;(globalThis as Record<string, unknown>).fetch = async (u: string) => { seen.push(String(u)); return new Response(sse(delta({ reasoning_content: "thinking first" }), ...toolCall, ...finish), { status: 200 }) }
    await back.resumeRuns()
    expect(seen).toEqual([expect.stringContaining(`agent/run?id=${runId}&from=0`)])
    expect(back.messages).toHaveLength(2)
    const m = back.messages[1]
    expect(m.error).toBeUndefined()
    expect(m.text).toBe("all done")
    expect(m.reasoning).toBe("thinking first")
    expect(m.tools![0]).toMatchObject({ name: "Read", state: "done", ok: true })
    expect(m.stats?.tokS).toBeUndefined()                                                // read again at once: its speed is not what it was written at
    expect(pending()).toEqual({})
    expect(back.busy).toBeNull()
    expect(new ChatController().messages).toHaveLength(2)                                // and it is stored: another refresh finds the answer
  })

  test("while it is read again the conversation shows the answer being written, and Stop stops it in the server", async () => {
    keep()
    const c = new ChatController()
    const calls: { url: string; body?: string }[] = []
    const leave = cutAfterFirst(c, calls)
    const p = c.send("work on it", [], { ...ctx, agent: ON, folder: "C:/proj" })
    await new Promise((r) => setTimeout(r, 20))
    leave()
    await p
    const runId = JSON.parse(calls[0].body!).strata_agent.run

    const back = new ChatController()
    const posts: { url: string; body: string }[] = []
    let release!: () => void
    const held = new Promise<void>((r) => { release = r })
    ;(globalThis as Record<string, unknown>).fetch = async (u: string, init?: { body?: string; signal?: AbortSignal }) => {
      if (String(u).includes("agent/cancel")) { posts.push({ url: String(u), body: init!.body! }); return new Response('{"ok":true}', { status: 200 }) }
      return new Response(new ReadableStream({
        async start(ctl) {
          ctl.enqueue(enc.encode(sse(delta({ content: "part" }))))
          init?.signal?.addEventListener("abort", () => ctl.error(new DOMException("aborted", "AbortError")))
          await held
          ctl.close()
        },
      }), { status: 200 })
    }
    const done = back.resumeRuns()
    await new Promise((r) => setTimeout(r, 30))
    expect(back.busy).not.toBeNull()
    expect(back.messages[1].text).toBe("part")
    back.stop()
    await done
    expect(posts).toHaveLength(1)
    expect(posts[0].url).toContain("agent/cancel")
    expect(JSON.parse(posts[0].body)).toEqual({ id: runId })
    expect(back.messages[1].stopped).toBe(true)
    expect(pending()).toEqual({})
    release()
  })

  test("a run that the server no longer has is said so, in the conversation, and the message is kept", async () => {
    keep()
    const c = new ChatController()
    const calls: { url: string; body?: string }[] = []
    const leave = cutAfterFirst(c, calls)
    const p = c.send("work on it", [], { ...ctx, agent: ON, folder: "C:/proj" })
    await new Promise((r) => setTimeout(r, 20))
    leave()
    await p
    const back = new ChatController()
    const errors: string[] = []
    back.onError = (title) => errors.push(title)
    ;(globalThis as Record<string, unknown>).fetch = async () => new Response(JSON.stringify({ error: { message: "there is no such run" } }), { status: 404 })
    await back.resumeRuns()
    expect(back.messages[0].text).toBe("work on it")
    expect(back.messages[1].error).toContain("could not be recovered")
    expect(errors).toEqual([])                                                           // no pop-up: it is in the conversation
    expect(pending()).toEqual({})
  })

  test("a record of a conversation that is gone, or of another prompt, is dropped and nothing is read", async () => {
    keep()
    store.set("runs", { nobody: { run: "run-xxxxxxxx", prompt: "1", time: 2 } })
    let fetched = 0
    ;(globalThis as Record<string, unknown>).fetch = async () => { fetched++; return new Response("", { status: 200 }) }
    const c = new ChatController()
    await c.resumeRuns()
    expect(fetched).toBe(0)
    expect(pending()).toEqual({})
    // a conversation that has moved on
    const d = new ChatController()
    ;(globalThis as Record<string, unknown>).fetch = async () => new Response(sse(...finish), { status: 200 })
    await d.send("first", [], { ...ctx, agent: ON, folder: "C:/proj" })
    store.set("runs", { [d.index.active!]: { run: "run-yyyyyyyy", prompt: "12345", time: 2 } })
    fetched = 0
    ;(globalThis as Record<string, unknown>).fetch = async () => { fetched++; return new Response("", { status: 200 }) }
    await new ChatController().resumeRuns()
    expect(fetched).toBe(0)
    expect(pending()).toEqual({})
  })

  test("an answer that ends as usual leaves no record, and a request without the coding tools never has one", async () => {
    keep()
    ;(globalThis as Record<string, unknown>).fetch = async () => new Response(sse(...finish), { status: 200 })
    const c = new ChatController()
    await c.send("with tools", [], { ...ctx, agent: ON, folder: "C:/proj" })
    expect(pending()).toEqual({})
    const bodies: string[] = []
    ;(globalThis as Record<string, unknown>).fetch = async (_u: string, init?: { body?: string }) => { bodies.push(init!.body!); return new Response(sse(...finish), { status: 200 }) }
    const d = new ChatController()
    await d.send("plain", [], ctx)
    expect(JSON.parse(bodies[0]).strata_agent).toBeUndefined()
    expect(pending()).toEqual({})
  })

  test("a real failure of the read (not leaving the page) is still an error", async () => {
    keep()
    ;(globalThis as Record<string, unknown>).fetch = async () => { throw new TypeError("network error") }
    const c = new ChatController()
    const errors: string[] = []
    c.onError = (title) => errors.push(title)
    await c.send("hi", [], { ...ctx, agent: ON, folder: "C:/proj" })
    expect(c.messages[1].error).toBe("network error")
    expect(errors).toEqual(["The request failed"])
    expect(pending()).toEqual({})
  })
})

describe("a question that was answered before the page read the run again", () => {
  const finish = [delta({ content: "ok" }), { choices: [], usage: { completion_tokens: 1 } }, "data: [DONE]" + String.fromCharCode(10) + String.fromCharCode(10)]
  const questions = [{ question: "Which?", header: "Library", multiSelect: false, options: [{ label: "a", description: "x" }, { label: "b", description: "y" }] }]
  function keep() {
    const data = new Map<string, string>()
    ;(globalThis as Record<string, unknown>).localStorage = {
      getItem: (k: string) => data.get(k) ?? null, setItem: (k: string, v: string) => { data.set(k, v) }, removeItem: (k: string) => { data.delete(k) },
    }
  }
  test("its card is not left waiting: the permission and the question show as answered", async () => {
    keep()
    ;(globalThis as Record<string, unknown>).fetch = async () => new Response(sse(
      { strata_mcp: { event: "start", id: "c1", name: "Bash" } }, { strata_mcp: { event: "call", id: "c1", name: "Bash", server: "agent", tool: "Bash", arguments: { command: "make" }, round: 1 } },
      { strata_mcp: { event: "permission", id: "p1", call_id: "c1", tool: "Bash", arguments: { command: "make" }, why: "asks", danger: false, rule: null } },
      { strata_mcp: { event: "answered", id: "p1", call_id: "c1", answer: "allow" } },
      { strata_mcp: { event: "start", id: "c2", name: "AskUserQuestion" } }, { strata_mcp: { event: "call", id: "c2", name: "AskUserQuestion", server: "agent", tool: "AskUserQuestion", arguments: { questions }, round: 1 } },
      { strata_mcp: { event: "question", id: "q1", call_id: "c2", questions } },
      { strata_mcp: { event: "question_answered", id: "q1", call_id: "c2", answers: { "Which?": ["b"] } } },
      ...finish), { status: 200 })
    const c = new ChatController()
    await c.send("go", [], ctx)
    const [bash, ask] = c.messages[1].tools!
    expect(bash.ask?.answer).toBe("allow")
    expect(bash.state === "asking").toBe(false)
    expect(ask.question?.answers).toEqual({ "Which?": ["b"] })
    expect(ask.state === "asking").toBe(false)
  })

  test("an answer that is not one, or for a card that is not there, is ignored", async () => {
    keep()
    ;(globalThis as Record<string, unknown>).fetch = async () => new Response(sse(
      { strata_mcp: { event: "start", id: "c1", name: "Bash" } }, { strata_mcp: { event: "call", id: "c1", name: "Bash", server: "agent", tool: "Bash", arguments: { command: "make" }, round: 1 } },
      { strata_mcp: { event: "permission", id: "p1", call_id: "c1", tool: "Bash", arguments: {}, why: "asks", danger: false, rule: null } },
      { strata_mcp: { event: "answered", id: "p1", answer: "maybe" } }, { strata_mcp: { event: "answered", id: "nobody", answer: "allow" } },
      ...finish), { status: 200 })
    const c = new ChatController()
    await c.send("go", [], ctx)
    expect(c.messages[1].tools![0].ask?.answer).toBeUndefined()
  })
})

