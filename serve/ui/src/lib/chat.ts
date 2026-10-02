// The chat's state and its conversation with POST /v1/chat/completions (the classic app's logic, ported).
// A plain controller outside React: the stream mutates the answer in place and React repaints once per frame.
import { useSyncExternalStore } from "react"
import { apiHeaders, errorMessage, postPermission, url, type Health, type McpInfo } from "./api"
import { fmt } from "./format"
import { store } from "./store"
import { PrefillMeter, type Prefill } from "./prefill"
import { t, tn } from "./i18n"
import { skillOfMessage } from "./slash"
import { addRule, agentRequest, NO_AGENT, rulesOf, type AgentInfo } from "./agent"
import { addProject, loadIndex, moveSession, newSession, openSession, persistIndex, removeProject, removeSession, renameProject, renameSession, saveActive, folderOf, setProjectFolder, type SessionIndex, type StoredMessage } from "./sessions"

/** The question a coding tool has put to the user (a card), and what the user answered; the server runs the call only after "allow". */
export interface Ask { id: string; tool: string; why: string; danger: boolean; rule: string | null; arguments?: unknown; answer?: "allow" | "allow_chat" | "deny" }
export interface Todo { content: string; status: "pending" | "in_progress" | "completed"; activeForm: string }
export interface ToolCall {
  id: string; name: string; at: number; rat: number
  state: "writing" | "asking" | "running" | "done" | "error" | "skipped"
  ask?: Ask; judge?: { verdict: string; severity: number | null }                // the coding tools: a question for the user, and what auto mode found
  server?: string; tool?: string; arguments?: unknown; round?: number
  result?: string; ok?: boolean; chars?: number; truncated?: boolean; ms?: number | null; open?: boolean
}
export interface Attachment { kind: "image" | "file"; name: string; url?: string; text?: string }
/** What the line under an answer says, as numbers: it is turned into words by `metaText` when it is shown, so that the line
 *  follows the language in use (a stored answer has no text of its own in either language). */
export interface Stats { tokens?: number; tokS?: number | null; stopped?: boolean; tools?: number; limit?: number; projection?: "on" | "off" }
export interface Message {
  role: "user" | "assistant"; text: string; time: number
  prefill?: Prefill                                              // on a prompt: the speed it was read at
  reasoning?: string; thinkSecs?: number | null; stats?: Stats; meta?: string /* legacy: the line as text, from an older version */
  error?: string; stopped?: boolean; limit?: number
  images?: { name: string; url?: string }[]; files?: { name: string; text?: string }[]; tools?: ToolCall[]
  todos?: Todo[]                                                 // the coding tools' list of steps, as the model last sent it
}

/** The line under an answer ("40 tokens · 38.2 tok/s · 1 tool call"), in the language in use now. An answer stored by an
 *  older version has only the text, which is shown as it was written. */
export function metaText(m: Message): string {
  const s = m.stats
  if (!s) return m.meta || ""
  const parts: string[] = []
  if (s.tokens) {
    parts.push(t("{n} tokens", { n: fmt(s.tokens) }))
    if (s.tokS != null) parts.push(`${fmt(s.tokS, 1)} tok/s`)
    if (s.stopped) parts.push(t("stopped"))
    if (s.projection) parts.push(s.projection === "on" ? t("projection on") : t("projection off"))
  } else if (s.stopped) {
    parts.push(t("Stopped"))
  }
  if (s.tools) parts.push(tn(s.tools, "{n} tool call", "{n} tool calls"))
  if (s.limit) parts.push(t("stopped at the limit of {n} tool rounds (mcp.max_rounds)", { n: s.limit }))
  return parts.join(" · ")
}
export interface Settings {
  thinking: string; temperature: number; top_p: number; top_k: number
  max: string; seed: string; show: boolean; esp: boolean; mcp: boolean; mcpOff: string[]; prefill: boolean
  agent: boolean; agentMode: string; agentFolder: string        // the coding tools: on (the default), ask / plan / auto, and the folder for chats that are in no project
}
/** What a request says about MCP: nothing when the tools are off (or no server that is not switched off has any), else `strata_mcp`
 *  and, when the + menu's list switched some servers off for this chat, their names. */
export function mcpRequest(s: Settings, mcp: McpInfo): { strata_mcp?: true; strata_mcp_off?: string[] } {
  if (s.mcp === false) return {}
  const off = (Array.isArray(s.mcpOff) ? s.mcpOff : []).filter((n) => typeof n === "string" && mcp.servers.some((x) => x.name === n))
  const usable = mcp.servers.length ? mcp.servers.filter((x) => !off.includes(x.name) && x.tools.length > 0).length : mcp.tools      // the list may not have arrived: the count says
  if (usable === 0 || (!mcp.servers.length && mcp.tools === 0)) return {}
  return off.length ? { strata_mcp: true, strata_mcp_off: off } : { strata_mcp: true }
}
export const DEFAULTS: Settings = { thinking: "high", temperature: 0.6, top_p: 0.95, top_k: 20, max: "", seed: "", show: true, esp: true, mcp: true, mcpOff: [], prefill: true, agent: true, agentMode: "ask", agentFolder: "" }

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
  call_id?: string; why?: string; danger?: boolean; rule?: string | null; verdict?: string; severity?: number | null; todos?: Todo[]; mode?: string      // the coding tools
}

// a tool event from the stream (the `strata_mcp` field of a chunk)
function onTool(m: Message, x: ToolEvent) {
  if (x.event === "limit") { m.limit = x.max_rounds; return }
  if (x.event === "todos") { if (Array.isArray(x.todos)) m.todos = x.todos; return }
  if (x.event === "permission" || x.event === "judging" || x.event === "judged") {          // about a call that is already shown: x.call_id is its id
    const c = (m.tools || []).find((y) => y.id === x.call_id)
    if (!c) return
    if (x.event === "permission") { c.ask = { id: x.id, tool: String(x.tool ?? c.name), why: x.why ?? "", danger: !!x.danger, rule: x.rule ?? null, arguments: x.arguments }; c.state = "asking" }
    else if (x.event === "judged" && (x.verdict === "allow" || x.verdict === "block" || x.verdict === "ask")) c.judge = { verdict: x.verdict, severity: x.severity ?? null }
    return
  }
  if (x.event !== "start" && x.event !== "call" && x.event !== "result") return                // a kind of event this page does not know
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

/** The attachments of a sent message that can be sent again: a reload keeps only their names (the data is not stored). */
function attachmentsOf(m: Message): { kept: Attachment[]; lost: number } {
  const kept: Attachment[] = []
  let lost = 0
  for (const i of m.images || []) i.url ? kept.push({ kind: "image", name: i.name, url: i.url }) : lost++
  for (const f of m.files || []) f.text != null ? kept.push({ kind: "file", name: f.name, text: f.text }) : lost++
  return { kept, lost }
}

/** Messages as they come back from storage: a read that was cut off by closing the page is not still reading. */
function restore(msgs: Message[]): Message[] {
  return msgs.map((m) => (m.prefill?.state === "reading" ? { ...m, prefill: { ...m.prefill, state: "done" as const, rate: null } } : m))
}

export interface SendContext { health: Health; mcp: McpInfo; projectionLoaded: boolean; skills?: string[]; agent?: AgentInfo; folder?: string | null }      // skills: the names of the skills in use, for "/name"

export class ChatController {
  // a read that was cut off by closing the page is not still reading
  index: SessionIndex = loadIndex(store, Date.now())             // the conversations, and which is open (lib/sessions.ts)
  messages: Message[] = restore(store.get<Message[]>("chat", []))   // the open one
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

  private fullNoted = false
  /** The open conversation as it is stored: attachments are kept by name only. */
  private stored(): StoredMessage[] {
    return this.messages.filter((m) => m !== this.busy?.msg)               // the answer that is still being written is not stored half-empty
      .map((m) => ({ ...m, images: (m.images || []).map((i) => ({ name: i.name })), files: (m.files || []).map((f) => ({ name: f.name })) }))
  }
  /** The browser refused a write: said once, until a write works again. */
  private storageFull() {
    if (this.fullNoted) return
    this.fullNoted = true
    this.onError(t("This browser's storage is full"), t("The conversation may not be kept. Delete some from Recents to make room."))
  }
  save() {
    const r = saveActive(store, this.index, this.stored(), Date.now())
    this.index = r.index
    if (r.ok) this.fullNoted = false
    else this.storageFull()
  }
  private saveIndex() { if (persistIndex(store, this.index)) this.fullNoted = false; else this.storageFull(); this.notify() }

  /** Starts an empty conversation; the one that was open stays in Recents. Not while an answer is being written. */
  newSession(): boolean {
    if (this.busy) return false
    if (!this.messages.length) return true
    const next = newSession(store, this.index, this.stored())
    if (next === this.index) { this.storageFull(); return false }
    this.index = next
    this.messages = []
    this.notify()
    return true
  }
  /** Opens a conversation of the list. False when there is none with that id, an answer is being written, or the open one could not be kept. */
  open(id: string): boolean {
    if (id === this.index.active) return true              // already open (also fine while it is being written)
    if (this.busy) return false
    const r = openSession(store, this.index, this.stored(), id)
    if (!r) return false
    if (!r.ok) { this.storageFull(); return false }
    this.index = r.index
    this.messages = restore(r.messages as Message[])
    this.notify()
    return true
  }
  remove(id: string): boolean {
    if (this.busy && this.index.active === id) return false
    const r = removeSession(store, this.index, id)
    this.index = r.index
    if (r.clearedActive) this.messages = []
    this.notify()
    return true
  }
  rename(id: string, title: string) { this.index = renameSession(this.index, id, title); this.saveIndex() }
  move(id: string, project: string | undefined) { this.index = moveSession(this.index, id, project); this.saveIndex() }
  /** A project needs a folder: it is where its chats' coding tools work. Null (and nothing made) without a name or without a folder. */
  addProject(name: string, folder: string): string | null {
    if (!folder.trim()) return null
    const before = this.index.projects.length
    const id = Math.random().toString(36).slice(2, 10)
    this.index = addProject(this.index, name, id, folder)
    this.saveIndex()
    return this.index.projects.length > before ? id : null
  }
  renameProject(id: string, name: string) { this.index = renameProject(this.index, id, name); this.saveIndex() }
  removeProject(id: string) { this.index = removeProject(this.index, id); this.saveIndex() }
  setSettings(s: Settings) { this.settings = s; store.set("sampling", s); this.notify() }
  setProjectFolder(id: string, folder: string) { this.index = setProjectFolder(this.index, id, folder); this.saveIndex() }
  /** The folder the coding tools work in for the open conversation: its project's, else the default one (Settings); null when there is none. */
  folder(): string | null {
    const mine = this.index.items.find((i) => i.id === this.index.active)
    return folderOf(this.index, mine?.project) ?? (this.settings.agentFolder?.trim() || null)
  }

  stop() { this.busy?.abort.abort() }

  /** The user's answer to a card of the coding tools. The call goes on (or is refused) at the server; the card stays until the server says
   *  it took the answer, so a lost one can be given again. "Allow for this chat" also keeps the rule for the next requests of this chat. */
  async answer(callId: string, decision: "allow" | "allow_chat" | "deny"): Promise<boolean> {
    const call = this.messages.flatMap((m) => m.tools ?? []).find((c) => c.id === callId && c.ask && !c.ask.answer)
    if (!call?.ask) return false
    const r = await postPermission(call.ask.id, decision)
    if ("error" in r) { this.onError(t("The answer was not taken"), r.error); return false }
    call.ask.answer = decision
    call.state = "running"
    if (decision === "allow_chat" && call.ask.rule) addRule(store, this.index.active ?? "new", call.ask.rule)
    this.notify()
    return true
  }

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

  private lostNote(lost: number) {
    if (lost) this.onError(t("Attachments not restored"), tn(lost, "{n} attachment was not kept: only the names of attachments are stored, so a file or an image is gone after the page is reloaded.", "{n} attachments were not kept: only the names of attachments are stored, so a file or an image is gone after the page is reloaded."))
  }

  /** Whether taking the last prompt back would leave the conversation empty (it is the first prompt), which deletes the conversation: the page
   *  asks first. Never while an answer is being written (nothing can be taken back then). */
  undoWouldEmpty(): boolean {
    if (this.busy) return false
    const at = this.messages.map((m) => m.role).lastIndexOf("user")
    return at === 0
  }

  /** Takes the last prompt back, with its answer: both leave the chat and the prompt (with the attachments that are still
   *  held) is returned to be put in the composer. Not while an answer is being written. */
  undoLast(): { text: string; attachments: Attachment[]; removed: Message[] } | null {
    if (this.busy) return null
    let at = -1
    for (let i = this.messages.length - 1; i >= 0 && at < 0; i--) if (this.messages[i].role === "user") at = i
    if (at < 0) return null
    const m = this.messages[at]
    const { kept, lost } = attachmentsOf(m)
    const removed = this.messages.slice(at)
    this.messages = this.messages.slice(0, at)
    this.save(); this.notify()
    this.lostNote(lost)
    return { text: m.text, attachments: kept, removed }
  }

  /** Rewrites the prompt at `index`: it and everything after it are replaced by the new prompt (with its own attachments
   *  that are still held) and a new answer. False when it cannot be done: an answer is being written, the message is not a
   *  prompt, or nothing would be sent. */
  async edit(index: number, text: string, ctx: SendContext): Promise<boolean> {
    const m = this.messages[index]
    if (this.busy || !m || m.role !== "user") return false
    const { kept, lost } = attachmentsOf(m)
    if (!text.trim() && !kept.length) return false
    this.messages = this.messages.slice(0, index)
    this.notify()                                             // not saved here: send() saves, and an empty save would delete the conversation (its name, its project) and make it again
    this.lostNote(lost)
    await this.send(text, kept, ctx)
    return true
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
    this.save()                                               // the conversation is in the list now, not when its answer ends
    this.notify()

    const s = this.settings
    const body: Record<string, unknown> = { model: ctx.health.model, messages: apiMessages(this.messages), stream: true, reasoning_effort: s.thinking }
    if (s.temperature > 0) Object.assign(body, { temperature: +s.temperature, top_p: +s.top_p, top_k: +s.top_k })
    else body.temperature = 0
    if (s.seed) body.seed = +s.seed
    if (s.max) body.max_tokens = +s.max
    if (ctx.projectionLoaded) body.experimental_speed_projection = !!s.esp
    Object.assign(body, mcpRequest(s, ctx.mcp))                            // this server may run MCP tools for it (the ones not switched off)
    Object.assign(body, agentRequest(s, ctx.agent ?? NO_AGENT, ctx.folder, this.index.active, rulesOf(store, this.index.active ?? "new")))      // the coding tools, when they are on and reachable
    const lastPrompt = [...this.messages].reverse().find((x) => x.role === "user")
    const skill = lastPrompt ? skillOfMessage(lastPrompt.text, ctx.skills ?? []) : null
    if (skill) body.strata_skill = skill                                   // "/name": the server loads that skill for this message

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
          if (j.error) throw new Error(j.error.message || t("the engine reported an error"))
          if (j.usage) usage = j.usage
          if (j.timings) timings = j.timings
          if (j.strata_mcp) {
            onTool(m, j.strata_mcp)
            if (j.strata_mcp.event === "mode" && (j.strata_mcp.mode === "ask" || j.strata_mcp.mode === "plan" || j.strata_mcp.mode === "auto")) this.setSettings({ ...this.settings, agentMode: j.strata_mcp.mode })      // the plan was approved
          }
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
      else { m.error = err.message || String(err); this.onError(t("The request failed"), m.error) }
    }
    if (thinkStart && m.thinkSecs == null) m.thinkSecs = (performance.now() - thinkStart) / 1000
    // The read is over: the engine's own mean replaces the live speed. After a tool round the final timings are the last
    // round's (a different prompt), so then the mean is the one measured while the first prompt was read.
    const sampled = this.meter?.mean() ?? null
    const own = timings && !(m.tools || []).length ? timings : null
    um.prefill = { state: "done", rate: null, mean: own ? own.prompt_per_second ?? null : sampled, read: own?.prompt_n ?? null, cached: own?.cache_n ?? null }
    this.meter = null
    const n = usage?.completion_tokens ?? null
    // what the line under the answer says, kept as numbers (metaText words it when it is shown)
    const stats: Stats = {}
    if (n && firstAt) {
      const secs = (performance.now() - firstAt) / 1000
      stats.tokens = n
      if (secs > 0.25) stats.tokS = n / secs
      if (m.stopped) stats.stopped = true
      if (ctx.projectionLoaded) stats.projection = s.esp ? "on" : "off"
    } else if (m.stopped) {
      stats.stopped = true
    }
    for (const tc of m.tools || []) if (tc.state === "writing" || tc.state === "running") { tc.state = "skipped"; tc.ms = null }
    const ran = (m.tools || []).filter((tc) => tc.state === "done" || tc.state === "error").length
    if (ran) stats.tools = ran
    if (m.limit) stats.limit = m.limit
    if (Object.keys(stats).length) m.stats = stats
    this.busy = null
    if (this.frame) { cancelAnimationFrame(this.frame); this.frame = 0 }
    this.save()
    this.notify()
  }
}

export const chat = new ChatController()
export const useChatVersion = () => useSyncExternalStore(chat.subscribe, chat.getVersion)
