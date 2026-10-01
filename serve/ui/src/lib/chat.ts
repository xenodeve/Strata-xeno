// The chat's state and its conversation with POST /v1/chat/completions (the classic app's logic, ported).
// A plain controller outside React: the stream mutates the answer in place and React repaints once per frame.
import { useSyncExternalStore } from "react"
import { apiHeaders, errorMessage, url, type Health, type McpInfo } from "./api"
import { fmt } from "./format"
import { store } from "./store"
import { PrefillMeter, type Prefill } from "./prefill"

export interface ToolCall {
  id: string; name: string; at: number; rat: number
  state: "writing" | "running" | "done" | "error" | "skipped"
  server?: string; tool?: string; arguments?: unknown; round?: number
  result?: string; ok?: boolean; chars?: number; truncated?: boolean; ms?: number | null; open?: boolean
}
export interface Attachment { kind: "image" | "file"; name: string; url?: string; text?: string }
export interface Message {
  role: "user" | "assistant"; text: string; time: number
  prefill?: Prefill                                              // on a prompt: the speed it was read at
  reasoning?: string; thinkSecs?: number | null; meta?: string; error?: string; stopped?: boolean; limit?: number
  images?: { name: string; url?: string }[]; files?: { name: string; text?: string }[]; tools?: ToolCall[]
}
export interface Settings {
  thinking: "none" | "low" | "medium" | "high"; temperature: number; top_p: number; top_k: number
  max: string; seed: string; show: boolean; esp: boolean; mcp: boolean; prefill: boolean
}
export const DEFAULTS: Settings = { thinking: "high", temperature: 0.6, top_p: 0.95, top_k: 20, max: "", seed: "", show: true, esp: true, mcp: true, prefill: true }

// a file's text in the message, fenced with more backticks than it contains itself
const fileBlock = (f: { name: string; text: string }) => {
  const longest = Math.max(2, ...(f.text.match(/`+/g) || []).map((s) => s.length))
  const fence = "`".repeat(longest + 1)
  return `File: ${f.name}\n${fence}\n${f.text}\n${fence}`
}
const userText = (m: Message) =>
  [m.text, ...(m.files || []).filter((f) => f.text != null).map((f) => fileBlock(f as { name: string; text: string }))]
    .filter((s) => s).join("\n\n")

type ApiMessage = Record<string, unknown>

// An answer that used MCP tools goes back as the model wrote it: per round the text before the calls, the calls and
// their results (as the model read them), then the rest - so the next question can build on what the tools found.
function assistantMessages(m: Message): ApiMessage[] {
  const ran = (m.tools || []).filter((t) => t.round != null && t.result != null && t.state !== "skipped")
  if (!ran.length) return m.text ? [{ role: "assistant", content: m.text }] : []
  const out: ApiMessage[] = []
  let pos = 0
  for (const r of [...new Set(ran.map((t) => t.round))]) {
    const calls = ran.filter((t) => t.round === r)
    const at = Math.min(Math.max(pos, calls[0].at || 0), m.text.length)
    out.push({
      role: "assistant", content: m.text.slice(pos, at).trim(),
      tool_calls: calls.map((t) => ({ id: t.id, type: "function", function: { name: t.name, arguments: JSON.stringify(t.arguments || {}) } })),
    })
    for (const t of calls) out.push({ role: "tool", tool_call_id: t.id, content: t.result })
    pos = at
  }
  const rest = m.text.slice(pos).trim()
  if (rest) out.push({ role: "assistant", content: rest })
  return out
}

export function apiMessages(messages: Message[]): ApiMessage[] {
  const out: ApiMessage[] = []
  for (const m of messages) {
    if (m.role === "user") {
      const imgs = (m.images || []).filter((i) => i.url)
      const text = userText(m)
      out.push({ role: "user", content: imgs.length ? [{ type: "text", text }, ...imgs.map((i) => ({ type: "image_url", image_url: { url: i.url } }))] : text })
    } else if (!m.error) {
      out.push(...assistantMessages(m))
    }
  }
  return out
}

interface ToolEvent {
  event: string; id: string; name: string; server?: string; tool?: string; arguments?: unknown; round?: number
  text?: string; ok?: boolean; chars?: number; truncated?: boolean; ms?: number; skipped?: boolean; max_rounds?: number
}

// a tool event from the stream (the `strata_mcp` field of a chunk)
function onTool(m: Message, x: ToolEvent) {
  if (x.event === "limit") { m.limit = x.max_rounds; return }
  m.tools = m.tools || []
  let t = m.tools.find((y) => y.id === x.id)
  if (!t) { t = { id: x.id, name: x.name, at: m.text.length, rat: (m.reasoning || "").length, state: "writing" }; m.tools.push(t) }
  if (x.event === "call") {
    Object.assign(t, { name: x.name, server: x.server, tool: x.tool, arguments: x.arguments, round: x.round, state: "running" })
  } else if (x.event === "result") {
    Object.assign(t, { result: x.text, ok: x.ok, chars: x.chars, truncated: x.truncated, ms: x.ms, state: x.skipped ? "skipped" : x.ok ? "done" : "error" })
  }
}

export function exportMarkdown(messages: Message[], model: string): string {
  const tools = (m: Message) => (m.tools || []).filter((t) => t.result != null).map((t) =>
    `<details><summary>Tool ${t.server ? `${t.server} / ` : ""}${t.tool || t.name}${t.ok ? "" : " (error)"}</summary>\n\n` +
    `\`\`\`json\n${JSON.stringify(t.arguments || {}, null, 2)}\n\`\`\`\n\n\`\`\`\n${t.result}\n\`\`\`\n\n</details>\n\n`).join("")
  return messages.map((m) => m.role === "user" ? `## You\n\n${m.text}\n` :
    `## ${model}\n\n${m.reasoning ? `<details><summary>Thinking</summary>\n\n${m.reasoning}\n\n</details>\n\n` : ""}${tools(m)}${m.text || m.error || ""}\n`).join("\n")
}

export interface SendContext { health: Health; mcp: McpInfo; projectionLoaded: boolean }

export class ChatController {
  // a read that was cut off by closing the page is not still reading
  messages: Message[] = store.get<Message[]>("chat", []).map((m) => (m.prefill?.state === "reading" ? { ...m, prefill: { ...m.prefill, state: "done" as const, rate: null } } : m))
  settings: Settings = { ...DEFAULTS, ...store.get<Partial<Settings>>("sampling", {}) }
  busy: { abort: AbortController; msg: Message } | null = null
  onError: (title: string, text: string) => void = () => {}
  private meter: PrefillMeter | null = null
  private version = 0
  private listeners = new Set<() => void>()
  private frame = 0

  subscribe = (cb: () => void) => { this.listeners.add(cb); return () => { this.listeners.delete(cb) } }
  getVersion = () => this.version
  notify() { this.version++; this.listeners.forEach((l) => l()) }
  private paint() { if (!this.frame) this.frame = requestAnimationFrame(() => { this.frame = 0; this.notify() }) }

  save() {
    store.set("chat", this.messages.map((m) => ({ ...m, images: (m.images || []).map((i) => ({ name: i.name })), files: (m.files || []).map((f) => ({ name: f.name })) })))
  }
  setSettings(s: Settings) { this.settings = s; store.set("sampling", s); this.notify() }

  stop() { this.busy?.abort.abort() }

  /** One look at /metrics while a request runs: the engine's position in the prompt gives the speed under the prompt that
   *  was sent (the mean over the last second) while it is read, and the engine's own mean as soon as it is read. The tokens
   *  read and cached come with the final timings, at the end of the answer. */
  samplePrefill(live: { state: string; prompt_read: number | null; prefill_tok_s_mean?: number | null }, ms: number) {
    const b = this.busy, meter = this.meter
    if (!b || !meter) return
    const um = this.messages[this.messages.indexOf(b.msg) - 1]
    if (!um || um.role !== "user" || um.prefill?.state !== "reading") return
    if (live.state === "generating") {
      um.prefill = { state: "done", rate: null, mean: live.prefill_tok_s_mean ?? meter.mean(), read: null, cached: null }
    } else if (live.state === "reading" && live.prompt_read != null) {
      meter.push(ms, live.prompt_read)
      um.prefill = { state: "reading", rate: meter.rate(), mean: null, read: null, cached: null }
    } else return
    this.notify()
  }

  /** Clears the chat; returns what undoes it. */
  clear(): (() => void) | null {
    if (this.busy || !this.messages.length) return null
    const backup = this.messages
    this.messages = []
    this.save(); this.notify()
    return () => { this.messages = backup; this.save(); this.notify() }
  }

  async send(text: string, attachments: Attachment[], ctx: SendContext) {
    text = text.trim()
    if ((!text && !attachments.length) || this.busy) return
    this.messages.push({
      role: "user", text, time: Date.now(),
      images: attachments.filter((a) => a.kind === "image").map((a) => ({ name: a.name, url: a.url })),
      files: attachments.filter((a) => a.kind === "file").map((a) => ({ name: a.name, text: a.text })),
    })
    const um = this.messages[this.messages.length - 1]
    um.prefill = { state: "reading", rate: null, mean: null, read: null, cached: null }
    this.meter = new PrefillMeter()
    const m: Message = { role: "assistant", text: "", reasoning: "", time: Date.now() }
    this.messages.push(m)
    const abort = new AbortController()
    this.busy = { abort, msg: m }
    this.notify()

    const s = this.settings
    const body: Record<string, unknown> = { model: ctx.health.model, messages: apiMessages(this.messages), stream: true, reasoning_effort: s.thinking }
    if (s.temperature > 0) Object.assign(body, { temperature: +s.temperature, top_p: +s.top_p, top_k: +s.top_k })
    else body.temperature = 0
    if (s.seed) body.seed = +s.seed
    if (s.max) body.max_tokens = +s.max
    if (ctx.projectionLoaded) body.experimental_speed_projection = !!s.esp
    if (s.mcp !== false && ctx.mcp.tools > 0) body.strata_mcp = true      // this server may run MCP tools for it

    let firstAt: number | null = null
    let thinkStart: number | null = null
    let usage: { completion_tokens?: number } | null = null
    let timings: { prompt_n?: number; prompt_per_second?: number | null; cache_n?: number } | null = null
    try {
      const r = await fetch(url("v1/chat/completions"), { method: "POST", headers: apiHeaders(true), body: JSON.stringify(body), signal: abort.signal })
      if (!r.ok) throw new Error(await errorMessage(r))
      const reader = r.body!.getReader()
      const dec = new TextDecoder()
      let buf = ""
      for (;;) {
        const { value, done } = await reader.read()
        if (done) break
        buf += dec.decode(value, { stream: true })
        let nl: number
        while ((nl = buf.indexOf("\n")) >= 0) {
          const line = buf.slice(0, nl).trim()
          buf = buf.slice(nl + 1)
          if (!line.startsWith("data:")) continue                 // ": keep-alive" comments while a long prompt is read
          const data = line.slice(5).trim()
          if (data === "[DONE]") continue
          let j: any
          try { j = JSON.parse(data) } catch { continue }
          if (j.error) throw new Error(j.error.message || "the engine reported an error")
          if (j.usage) usage = j.usage
          if (j.timings) timings = j.timings
          if (j.strata_mcp) onTool(m, j.strata_mcp)
          const d = j.choices?.[0]?.delta || {}
          const lastTool = m.tools?.length ? m.tools[m.tools.length - 1] : null    // a new round after a tool
          if (d.reasoning_content) {
            firstAt ??= performance.now()
            thinkStart ??= performance.now()
            if (lastTool && m.reasoning && lastTool.rat === m.reasoning.length) m.reasoning += "\n\n"
            m.reasoning += d.reasoning_content
          }
          if (d.content) {
            firstAt ??= performance.now()
            if (thinkStart && m.thinkSecs == null) m.thinkSecs = (performance.now() - thinkStart) / 1000
            if (lastTool && m.text && lastTool.at === m.text.length) m.text += "\n\n"
            m.text += d.content
          }
          this.paint()
        }
      }
    } catch (e) {
      const err = e as Error
      if (err.name === "AbortError") m.stopped = true
      else { m.error = err.message || String(err); this.onError("The request failed", m.error) }
    }
    if (thinkStart && m.thinkSecs == null) m.thinkSecs = (performance.now() - thinkStart) / 1000
    // The read is over: the engine's own mean replaces the live speed. After a tool round the final timings are the last
    // round's (a different prompt), so then the mean is the one measured while the first prompt was read.
    const sampled = this.meter?.mean() ?? null
    const own = timings && !(m.tools || []).length ? timings : null
    um.prefill = { state: "done", rate: null, mean: own ? own.prompt_per_second ?? null : sampled, read: own?.prompt_n ?? null, cached: own?.cache_n ?? null }
    this.meter = null
    const n = usage?.completion_tokens ?? null
    if (n && firstAt) {
      const secs = (performance.now() - firstAt) / 1000
      m.meta = `${fmt(n)} tokens${secs > 0.25 ? ` · ${fmt(n / secs, 1)} tok/s` : ""}${m.stopped ? " · stopped" : ""}` +
        (ctx.projectionLoaded ? (s.esp ? " · projection on" : " · projection off") : "")
    } else if (m.stopped) {
      m.meta = "Stopped"
    }
    for (const t of m.tools || []) if (t.state === "writing" || t.state === "running") { t.state = "skipped"; t.ms = null }
    const ran = (m.tools || []).filter((t) => t.state === "done" || t.state === "error").length
    if (ran) m.meta = `${m.meta ? `${m.meta} · ` : ""}${ran} tool call${ran > 1 ? "s" : ""}`
    if (m.limit) m.meta = `${m.meta || ""} · stopped at the limit of ${m.limit} tool rounds (mcp.max_rounds)`
    this.busy = null
    if (this.frame) { cancelAnimationFrame(this.frame); this.frame = 0 }
    this.save()
    this.notify()
  }
}

export const chat = new ChatController()
export const useChatVersion = () => useSyncExternalStore(chat.subscribe, chat.getVersion)
