// The chat's state and its conversation with POST /v1/chat/completions (the classic app's logic, ported).
// A plain controller outside React: the stream mutates the answer in place and React repaints once per frame.
import { useSyncExternalStore } from "react"
import { apiHeaders, errorMessage, postPermission, postQuestionAnswer, url, type Health, type McpInfo } from "./api"
import { fmt } from "./format"
import { store } from "./store"
import { PrefillMeter, type Prefill } from "./prefill"
import { t, tn } from "./i18n"
import { skillOfMessage } from "./slash"
import { compactPrompt, continuationText, estimateTokens, MIN_SUMMARY, shouldCompact, summaryOf, summaryRoom } from "./compact"
import { addRule, agentRequest, NO_AGENT, rulesOf, type AgentInfo } from "./agent"
import { addPerm, forgetPerms, permsFor, type Effect, type Scope } from "./perms"
import { forgetCheckpoints } from "./rewind"
import { noteFrom, type HookNote } from "./hooks"
import { addProject, loadIndex, moveSession, newSession, openSession, persistIndex, removeProject, removeSession, renameProject, renameSession, saveActive, saveBackground, foldersOf, setProjectFolders, type SessionIndex, type StoredMessage } from "./sessions"

/** The question a coding tool has put to the user (a card), and what the user answered; the server runs the call only after "allow". */
export interface Ask { id: string; tool: string; why: string; danger: boolean; rule: string | null; arguments?: unknown; answer?: "allow" | "allow_chat" | "deny"; kept?: { scope: "project" | "everywhere"; effect: Effect } }
export interface Todo { content: string; status: "pending" | "in_progress" | "completed"; activeForm: string }
/** What the model asked the user (AskUserQuestion): one to four questions with two to four choices each; `answers` once the user answered (null: skipped). */
export interface AskedQuestion {
  id: string
  questions: { question: string; header: string; multiSelect: boolean; options: { label: string; description: string }[] }[]
  answers?: Record<string, string[]> | null
}
export interface ToolCall {
  id: string; name: string; at: number; rat: number
  state: "writing" | "asking" | "running" | "done" | "error" | "skipped"
  question?: AskedQuestion
  ask?: Ask; judge?: { verdict: string; severity: number | null }                // the coding tools: a question for the user, and what auto mode found
  hooks?: HookNote[]                                                                 // what the user's hooks did about this call
  server?: string; tool?: string; arguments?: unknown; round?: number
  result?: string; ok?: boolean; chars?: number; truncated?: boolean; ms?: number | null; open?: boolean
}
export interface Attachment { kind: "image" | "file"; name: string; url?: string; text?: string }
/** What the line under an answer says, as numbers: it is turned into words by `metaText` when it is shown, so that the line
 *  follows the language in use (a stored answer has no text of its own in either language). */
export interface Stats { ctx?: number; tokens?: number; tokS?: number | null; stopped?: boolean; tools?: number; limit?: number; projection?: "on" | "off" }
/** On the message that stands for what was compacted: how many tokens the conversation used before, about how many the summary takes, who started it, what it was asked to focus on. */
export interface CompactInfo { before: number; after: number; auto: boolean; focus?: string }
export interface Message {
  role: "user" | "assistant"; text: string; time: number
  compact?: CompactInfo                                          // this message is the summary that took the place of the earlier ones: shown as a line, read by the model as the earlier part
  prefill?: Prefill                                              // on a prompt: the speed it was read at
  reasoning?: string; thinkSecs?: number | null; stats?: Stats; meta?: string /* legacy: the line as text, from an older version */
  error?: string; stopped?: boolean; limit?: number
  images?: { name: string; url?: string }[]; files?: { name: string; text?: string }[]; tools?: ToolCall[]
  todos?: Todo[]                                                 // the coding tools' list of steps, as the model last sent it
  hooks?: HookNote[]                                             // what the user's prompt and stop hooks did (the ones that belong to no call)
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
  autoCompact: boolean                                          // a conversation that nears the end of the context is summarised before the next prompt (on by default)
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
export const DEFAULTS: Settings = { thinking: "high", temperature: 0.6, top_p: 0.95, top_k: 20, max: "", seed: "", show: true, esp: true, mcp: true, mcpOff: [], prefill: true, agent: true, agentMode: "ask", agentFolder: "", autoCompact: true }

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
  on?: string; hook?: string; command?: string; code?: number | null; blocked?: boolean; timeout?: boolean; error?: string | null
  call_id?: string; why?: string; danger?: boolean; rule?: string | null; verdict?: string; severity?: number | null; todos?: Todo[]; mode?: string; questions?: AskedQuestion["questions"]      // the coding tools
}

// a tool event from the stream (the `strata_mcp` field of a chunk)
function onTool(m: Message, x: ToolEvent) {
  if (x.event === "limit") { m.limit = x.max_rounds; return }
  if (x.event === "todos") { if (Array.isArray(x.todos)) m.todos = x.todos; return }
  if (x.event === "hook") {                                                                  // a hook of the user's ran: on the call it was about, else on the answer
    const note = noteFrom(x as unknown as Record<string, unknown>)
    if (!note) return
    const c = x.call_id ? (m.tools || []).find((y) => y.id === x.call_id) : undefined
    const list = c ? (c.hooks = c.hooks || []) : (m.hooks = m.hooks || [])
    if (list.length < 20) list.push(note)
    return
  }
  if (x.event === "question") {                                                              // the model asks the user: a form on its call
    const c = (m.tools || []).find((y) => y.id === x.call_id)
    if (c && Array.isArray(x.questions)) { c.question = { id: x.id, questions: x.questions }; c.state = "asking" }
    return
  }
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

/** A message the user wrote: not the summary that a compaction left in place of earlier ones. */
export const isPrompt = (m: Message): boolean => m.role === "user" && !m.compact

export function exportMarkdown(messages: Message[], model: string): string {
  const tools = (m: Message) => (m.tools || []).filter((t) => t.result != null).map((t) =>
    `<details><summary>Tool ${t.server ? `${t.server} / ` : ""}${t.tool || t.name}${t.ok ? "" : " (error)"}</summary>\n\n` +
    `\`\`\`json\n${JSON.stringify(t.arguments || {}, null, 2)}\n\`\`\`\n\n\`\`\`\n${t.result}\n\`\`\`\n\n</details>\n\n`).join("")
  return messages.map((m) => m.compact ? `## Summary of the earlier conversation\n\n${m.text}\n` : m.role === "user" ? `## You\n\n${m.text}\n` :
    `## ${model}\n\n${m.reasoning ? `<details><summary>Thinking</summary>\n\n${m.reasoning}\n\n</details>\n\n` : ""}${tools(m)}${m.text || m.error || ""}\n`).join("\n")
}

/** The words of an answer that is streamed (the reasoning is not part of them). */
async function readText(r: Response): Promise<string> {
  const reader = r.body!.getReader()
  const dec = new TextDecoder()
  let buf = "", out = ""
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buf += dec.decode(value, { stream: true })
    let nl: number
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trim()
      buf = buf.slice(nl + 1)
      if (!line.startsWith("data:")) continue
      const data = line.slice(5).trim()
      if (data === "[DONE]") continue
      let j: any
      try { j = JSON.parse(data) } catch { continue }
      if (j.error) throw new Error(j.error.message || t("the engine reported an error"))
      out += j.choices?.[0]?.delta?.content || ""
    }
  }
  return out
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

export interface SendContext { health: Health; mcp: McpInfo; projectionLoaded: boolean; skills?: string[]; agent?: AgentInfo; folder?: string | string[] | null }      // skills: the names of the skills in use, for "/name"

/** An answer that is being written (or a conversation that is being summarised) for one conversation. Several can run at once, each for its own conversation: the page shows one,
 *  and the others go on in the background (the server takes their requests one after the other). `messages` is the live list of that conversation. */
interface Run { abort: AbortController; msg: Message; messages: Message[]; meter: PrefillMeter | null; compacting: boolean }
/** A message typed while the conversation was answering: it waits, and is sent when the answer ends. */
export interface Queued { id: string; text: string; files: Attachment[]; ctx: SendContext }
const MAX_QUEUED = 20
const NEW_KEY = "new"                                         // a conversation that is not in the list yet (only before its first prompt is saved)

/** A conversation as it is stored: attachments are kept by name only, and an answer that is still being written (`skip`) is left out. */
function storedOf(messages: Message[], skip?: Message): StoredMessage[] {
  return messages.filter((m) => m !== skip)
    .map((m) => ({ ...m, images: (m.images || []).map((i) => ({ name: i.name })), files: (m.files || []).map((f) => ({ name: f.name })) }))
}

export class ChatController {
  // a read that was cut off by closing the page is not still reading
  index: SessionIndex = loadIndex(store, Date.now())             // the conversations, and which is open (lib/sessions.ts)
  messages: Message[] = restore(store.get<Message[]>("chat", []))   // the open one
  settings: Settings = { ...DEFAULTS, ...store.get<Partial<Settings>>("sampling", {}) }
  private runs = new Map<string, Run>()                       // the conversations that are being answered, by id
  private queues = new Map<string, Queued[]>()                 // what was typed meanwhile, by conversation
  /** The open conversation's answer that is being written, or null. (Another conversation may be answering in the background: see `runningIds`.) */
  get busy(): { abort: AbortController; msg: Message } | null {
    const r = this.runs.get(this.index.active ?? NEW_KEY)
    return r ? { abort: r.abort, msg: r.msg } : null
  }
  set busy(v: { abort: AbortController; msg: Message } | null) {
    const key = this.index.active ?? NEW_KEY
    if (v) this.runs.set(key, { abort: v.abort, msg: v.msg, messages: this.messages, meter: null, compacting: false })
    else this.runs.delete(key)
  }
  /** The open conversation is being summarised. */
  get compacting(): boolean { return !!this.runs.get(this.index.active ?? NEW_KEY)?.compacting }
  /** The messages of the open conversation that wait for its answer to end. */
  queuedOf(): Queued[] { return this.queues.get(this.index.active ?? NEW_KEY) ?? [] }
  /** Keeps a message to be sent when the open conversation's answer ends. False when it is not answering (it is sent then, not kept) or there is nothing to keep. */
  queue(text: string, files: Attachment[], ctx: SendContext): boolean {
    const typed = text.trim()
    if (!this.busy || (!typed && !files.length)) return false
    const key = this.index.active ?? NEW_KEY
    this.queues.set(key, [...(this.queues.get(key) ?? []), { id: Math.random().toString(36).slice(2, 10), text: typed, files, ctx }].slice(-MAX_QUEUED))
    this.notify()
    return true
  }
  /** Takes a waiting message back (to be edited, or dropped). */
  unqueue(id: string): Queued | null {
    const key = this.index.active ?? NEW_KEY
    const q = this.queues.get(key) ?? []
    const at = q.findIndex((x) => x.id === id)
    if (at < 0) return null
    const [gone] = q.splice(at, 1)
    if (!q.length) this.queues.delete(key)
    this.notify()
    return gone
  }
  /** Sends a waiting message now (when the answer was stopped, so nothing sent it). */
  async sendQueued(id: string): Promise<boolean> {
    if (this.busy) return false
    const q = this.unqueue(id)
    if (!q) return false
    await this.send(q.text, q.files, q.ctx)
    return true
  }
  /** The next waiting message of a conversation goes when its answer ended well and the conversation is the one that is open; an answer that was stopped, or failed, sends nothing. */
  private drain(key: string) {
    const q = this.queues.get(key)
    if (!q?.length || this.index.active !== key || this.busy) return
    const last = this.messages[this.messages.length - 1]
    if (last && last.role === "assistant" && (last.stopped || last.error)) return
    const next = q.shift()!
    if (!q.length) this.queues.delete(key)
    this.notify()
    void this.send(next.text, next.files, next.ctx)
  }
  /** The conversations that are being answered now (the open one too), and those among them that wait for the user's answer to a question of the coding tools. */
  runningIds(): string[] { return [...this.runs.keys()].filter((k) => k !== NEW_KEY) }
  askingIds(): string[] { return [...this.runs].filter(([k, r]) => k !== NEW_KEY && (r.msg.tools || []).some((c) => c.state === "asking")).map(([k]) => k) }
  pendingProject: string | null = null                          // the project a new conversation (one that is not in the list yet) was started in
  onError: (title: string, text: string) => void = () => {}
  private version = 0
  private listeners = new Set<() => void>()
  private frame = 0

  subscribe = (cb: () => void) => { this.listeners.add(cb); return () => { this.listeners.delete(cb) } }
  getVersion = () => this.version
  notify() { this.version++; this.listeners.forEach((l) => l()) }
  private paint() { if (!this.frame) this.frame = requestAnimationFrame(() => { this.frame = 0; this.notify() }) }

  private fullNoted = false
  /** The open conversation as it is stored: the answer that is still being written is not stored half-empty. */
  private stored(skip: Message | undefined = this.busy?.msg): StoredMessage[] { return storedOf(this.messages, skip) }
  /** The browser refused a write: said once, until a write works again. */
  private storageFull() {
    if (this.fullNoted) return
    this.fullNoted = true
    this.onError(t("This browser's storage is full"), t("The conversation may not be kept. Delete some from Recents to make room."))
  }
  save(skip?: Message) {
    const r = saveActive(store, this.index, this.stored(skip ?? this.busy?.msg), Date.now(), this.pendingProject ?? undefined)
    this.index = r.index
    if (this.index.active !== null) this.pendingProject = null
    if (r.ok) this.fullNoted = false
    else this.storageFull()
  }
  private saveIndex() { if (persistIndex(store, this.index)) this.fullNoted = false; else this.storageFull(); this.notify() }

  /** Starts an empty conversation; the one that was open stays in Recents (and goes on answering, if it was). */
  newSession(project?: string): boolean {
    const into = project && this.index.projects.some((p) => p.id === project) ? project : null       // started inside a project: its folders are where the tools work from the first prompt
    if (!this.messages.length) { this.pendingProject = into; this.notify(); return true }
    const next = newSession(store, this.index, this.stored())
    if (next === this.index) { this.storageFull(); return false }
    this.index = next
    this.messages = []
    this.pendingProject = into
    this.notify()
    return true
  }
  /** The project a conversation that has not started is to work in (null: none). Only before its first prompt, and not while something runs; false otherwise or for a project that is not there. */
  setPendingProject(project: string | null): boolean {
    if (this.busy || this.messages.length || this.index.active !== null) return false
    if (project !== null && !this.index.projects.some((p) => p.id === project)) return false
    this.pendingProject = project
    this.notify()
    return true
  }
  /** The project the open conversation is in, or the one a new conversation was started in. */
  currentProject(): string | undefined {
    const mine = this.index.items.find((i) => i.id === this.index.active)
    return mine ? mine.project : this.pendingProject ?? undefined
  }
  /** Opens a conversation of the list; the one that was open stays as it is (and goes on answering, if it was). False when there is none with that id, or the open one could not be kept. */
  open(id: string): boolean {
    if (id === this.index.active) return true              // already open
    const r = openSession(store, this.index, this.stored(), id)
    if (!r) return false
    if (!r.ok) { this.storageFull(); return false }
    this.index = r.index
    const live = this.runs.get(id)                            // one that is being answered shows its live messages, not what was stored when it was left
    this.messages = live ? live.messages : restore(r.messages as Message[])
    this.pendingProject = null
    this.notify()
    this.drain(id)                                            // what was typed in it while it answered in the background goes now
    return true
  }
  remove(id: string): boolean {
    if (this.runs.has(id)) return false                       // not while it is being answered
    this.queues.delete(id)
    forgetCheckpoints(id)                                     // the way back for its files goes with it
    const r = removeSession(store, this.index, id)
    this.index = r.index
    if (r.clearedActive) this.messages = []
    this.notify()
    return true
  }
  rename(id: string, title: string) { this.index = renameSession(this.index, id, title); this.saveIndex() }
  move(id: string, project: string | undefined) { this.index = moveSession(this.index, id, project); this.saveIndex() }
  /** A project needs a folder, and may have more (other worktrees): they are where its chats' coding tools work. Null (and nothing made) without a name or without a folder. */
  addProject(name: string, folders: string[]): string | null {
    if (!folders.some((f) => f.trim())) return null
    const before = this.index.projects.length
    const id = Math.random().toString(36).slice(2, 10)
    this.index = addProject(this.index, name, id, folders)
    this.saveIndex()
    return this.index.projects.length > before ? id : null
  }
  renameProject(id: string, name: string) { this.index = renameProject(this.index, id, name); this.saveIndex() }
  removeProject(id: string) { this.index = removeProject(this.index, id); if (this.pendingProject === id) this.pendingProject = null; forgetPerms(store, id); this.saveIndex() }
  setSettings(s: Settings) { this.settings = s; store.set("sampling", s); this.notify() }
  setProjectFolders(id: string, folders: string[]) { this.index = setProjectFolders(this.index, id, folders); this.saveIndex() }
  /** The folders the coding tools work in for the open conversation: its project's (the first is the main one), else the default one (Settings); none when there is none. */
  folders(): string[] {
    const own = foldersOf(this.index, this.currentProject())
    return own.length ? own : this.settings.agentFolder?.trim() ? [this.settings.agentFolder.trim()] : []
  }
  /** The main folder: where commands run. */
  folder(): string | null { return this.folders()[0] ?? null }

  stop() { this.busy?.abort.abort() }

  // ------------------------------------------------------------------------------------------------ compacting (lib/compact.ts)
  /** How many tokens the conversation uses: what the last answer reported (its prompt after every tool round, and its own tokens), else a guess from the text. */
  contextUsed(): number {
    const last = this.messages[this.messages.length - 1]
    if (last && last.role === "assistant" && !last.error && last.stats?.ctx) return last.stats.ctx
    return estimateTokens(apiMessages(this.messages))
  }
  /** What the last answer said the conversation uses (its prompt after every tool round, and its own tokens), or null when it said nothing (no answer yet, or messages came and went since). */
  contextReported(): number | null {
    const last = this.messages[this.messages.length - 1]
    return last && last.role === "assistant" && !last.error && last.stats?.ctx ? last.stats.ctx : null
  }
  /** Whether there is something to summarise: a prompt of the user's. */
  canCompact(): boolean { return !this.busy && this.messages.some(isPrompt) }
  private shouldAutoCompact(ctx: SendContext, text: string, attachments: Attachment[]): boolean {
    if (this.settings.autoCompact === false || !this.messages.some(isPrompt)) return false
    const incoming = estimateTokens(text) + attachments.reduce((n, a) => n + (a.kind === "file" ? estimateTokens(a.text ?? "") : 0), 0)
    return shouldCompact(this.contextUsed(), incoming, ctx.health.max_context)
  }

  /** `/compact`: the model summarises the conversation and the summary takes the place of its messages. False when there was nothing to do or it did not work (the
   *  conversation is then as it was; the page says why). Stop ends it. */
  async compact(ctx: SendContext, focus = ""): Promise<boolean> {
    if (!this.canCompact()) return false
    const abort = new AbortController()
    const key = this.index.active ?? NEW_KEY
    const run: Run = { abort, msg: { role: "assistant", text: "", time: Date.now() }, messages: this.messages, meter: null, compacting: false }
    this.runs.set(key, run)
    const ok = await this.runCompact(key, run, ctx, focus, false, 0, this.contextUsed())
    this.runs.delete(key)
    this.notify()
    return ok
  }

  /** Saves what a run changed: in the open conversation's place, or in its own slot when another conversation is open. */
  private persist(key: string, run: Run) {
    if (this.index.active === key || key === NEW_KEY) { this.save(run.msg); return }
    const r = saveBackground(store, this.index, key, storedOf(run.messages, run.msg), Date.now())
    this.index = r.index
    if (!r.ok) this.storageFull()
  }

  /** Summarises all but the last `tail` messages (the prompt that is being sent, and its answer, stay out of it) and puts the summary in their place. */
  private async runCompact(key: string, run: Run, ctx: SendContext, focus: string, auto: boolean, tail: number, before: number): Promise<boolean> {
    run.compacting = true
    this.notify()
    let ok = false
    try {
      const head = run.messages.slice(0, run.messages.length - tail)
      const summary = summaryOf(await this.summarize(ctx, head, focus, run.abort.signal, before))
      if (!summary) throw new Error(t("The model sent no summary."))
      const note: Message = { role: "user", text: continuationText(summary), time: Date.now(), compact: { before, after: estimateTokens(summary), auto, ...(focus ? { focus } : {}) } }
      const rest = run.messages.slice(run.messages.length - tail)
      run.messages.splice(0, run.messages.length, note, ...rest)           // in place: the page may be showing another conversation, and this one's list must stay the same list
      this.persist(key, run)
      ok = true
    } catch (e) {
      if ((e as Error).name !== "AbortError") this.onError(t("Could not compact the conversation"), (e as Error).message || String(e))
    }
    run.compacting = false
    this.notify()
    return ok
  }

  /** The model's summary of `head`, as it wrote it. When the conversation does not fit with the request, the oldest part is left out and it is asked again. */
  private async summarize(ctx: SendContext, head: Message[], focus: string, signal: AbortSignal, used = 0): Promise<string> {
    const prompt = compactPrompt(focus)
    let from = 0
    for (;;) {
      const history = apiMessages(head.slice(from))
      const room = summaryRoom(ctx.health.max_context, from === 0 && used > 0 ? used : estimateTokens(history), estimateTokens(prompt))      // what the server reported is the safe figure for the whole conversation
      const next = this.nextStart(head, from)
      if (room < MIN_SUMMARY && next !== null) { from = next; continue }
      const body: Record<string, unknown> = { model: ctx.health.model, messages: [...history, { role: "user", content: prompt }], stream: true, reasoning_effort: "low", temperature: 0.3, max_tokens: Math.max(64, room) }
      if (ctx.projectionLoaded) body.experimental_speed_projection = !!this.settings.esp
      const r = await fetch(url("v1/chat/completions"), { method: "POST", headers: apiHeaders(true), body: JSON.stringify(body), signal })
      if (!r.ok) {
        const why = await errorMessage(r)
        if (next !== null && /context|room/i.test(why)) { from = next; continue }
        throw new Error(why)
      }
      return readText(r)
    }
  }
  /** Where a shorter part to summarise starts: at a prompt, a quarter of what is left further on; null when there is no later prompt. */
  private nextStart(head: Message[], from: number): number | null {
    const step = Math.max(1, Math.ceil((head.length - from) / 4))
    for (let i = from + step; i < head.length; i++) if (isPrompt(head[i])) return i
    return null
  }

  /** The user's answer to a card of the coding tools. The call goes on (or is refused) at the server; the card stays until the server says
   *  it took the answer, so a lost one can be given again. "Allow for this chat" also keeps the rule for the next requests of this chat. */
  async answer(callId: string, decision: "allow" | "allow_chat" | "deny", keep?: { scope: Scope; effect: Effect }): Promise<boolean> {
    const call = this.messages.flatMap((m) => m.tools ?? []).find((c) => c.id === callId && c.ask && !c.ask.answer)
    if (!call?.ask) return false
    const sent = keep ? (keep.effect === "allow" ? "allow_chat" : "deny") : decision          // "always allow" is allowed for the rest of this request too; "never" is a no
    const r = await postPermission(call.ask.id, sent)
    if ("error" in r) { this.onError(t("The answer was not taken"), r.error); return false }
    call.ask.answer = sent
    call.state = "running"
    if (keep && call.ask.rule) {
      if (addPerm(store, keep.scope, keep.effect, call.ask.rule)) call.ask.kept = { scope: keep.scope.kind, effect: keep.effect }
    } else if (decision === "allow_chat" && call.ask.rule) addRule(store, this.index.active ?? "new", call.ask.rule)
    this.notify()
    return true
  }

  /** The user's answers to what the model asked (null: skipped). The call goes on at the server; the form stays until the server says it took the answer, so a lost one can be given again. */
  async answerQuestion(callId: string, answers: Record<string, string[]> | null): Promise<boolean> {
    const call = this.messages.flatMap((m) => m.tools ?? []).find((c) => c.id === callId && c.question && c.question.answers === undefined)
    if (!call?.question) return false
    const r = await postQuestionAnswer(call.question.id, answers)
    if ("error" in r) { this.onError(t("The answer was not taken"), r.error); return false }
    call.question.answers = answers
    call.state = "running"
    this.notify()
    return true
  }

  /** One look at /metrics while a request runs: the engine's position in the prompt gives the speed under the prompt that
   *  was sent (the mean over the last second) while it is read, and the engine's own mean as soon as it is read. The tokens
   *  read and cached come with the final timings, at the end of the answer. */
  samplePrefill(live: { state: string; prompt_read: number | null; prefill_tok_s_mean?: number | null }, ms: number) {
    const run = this.runs.get(this.index.active ?? NEW_KEY), meter = run?.meter
    if (!run || !meter) return
    const um = run.messages[run.messages.indexOf(run.msg) - 1]
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
    const at = this.messages.map((m) => isPrompt(m)).lastIndexOf(true)
    return at === 0
  }

  /** Takes the last prompt back, with its answer: both leave the chat and the prompt (with the attachments that are still
   *  held) is returned to be put in the composer. Not while an answer is being written. */
  undoLast(): { text: string; attachments: Attachment[]; removed: Message[] } | null {
    if (this.busy) return null
    let at = -1
    for (let i = this.messages.length - 1; i >= 0 && at < 0; i--) if (isPrompt(this.messages[i])) at = i
    if (at < 0) return null
    const m = this.messages[at]
    const { kept, lost } = attachmentsOf(m)
    const removed = this.messages.slice(at)
    this.messages = this.messages.slice(0, at)
    this.save(); this.notify()
    this.lostNote(lost)
    return { text: m.text, attachments: kept, removed }
  }

  private lastStamp = 0
  /** A time for a prompt that no other prompt has: the clock, or one more than the last, so that two prompts sent in the same millisecond still have their own checkpoints. */
  private stamp(): number { this.lastStamp = Math.max(Date.now(), this.lastStamp + 1); return this.lastStamp }

  /** The id of a prompt's checkpoint (the server keeps the files the tools changed in it), or null for a message that is not a prompt. */
  checkpointOf(index: number): string | null {
    const m = this.messages[index]
    return m && isPrompt(m) ? String(m.time) : null
  }

  /** Takes the conversation back to before the prompt at `index`: it and everything after it leave the chat, and the prompt (with the attachments that are still held) is returned to be
   *  put in the composer. Not while an answer is being written. */
  rewindTo(index: number): { text: string; attachments: Attachment[]; removed: Message[] } | null {
    if (this.busy) return null
    const m = this.messages[index]
    if (!m || !isPrompt(m)) return null
    const { kept, lost } = attachmentsOf(m)
    const removed = this.messages.slice(index)
    this.messages = this.messages.slice(0, index)
    this.save(); this.notify()
    this.lostNote(lost)
    return { text: m.text, attachments: kept, removed }
  }

  /** Rewrites the prompt at `index`: it and everything after it are replaced by the new prompt (with its own attachments
   *  that are still held) and a new answer. False when it cannot be done: an answer is being written, the message is not a
   *  prompt, or nothing would be sent. */
  async edit(index: number, text: string, ctx: SendContext): Promise<boolean> {
    const m = this.messages[index]
    if (this.busy || !m || !isPrompt(m)) return false
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
    const compactFirst = this.shouldAutoCompact(ctx, text, attachments)       // measured on the conversation so far, before this prompt is part of it
    const before = compactFirst ? this.contextUsed() : 0
    this.messages.push({
      role: "user", text, time: this.stamp(),
      images: attachments.filter((a) => a.kind === "image").map((a) => ({ name: a.name, url: a.url })),
      files: attachments.filter((a) => a.kind === "file").map((a) => ({ name: a.name, text: a.text })),
    })
    const um = this.messages[this.messages.length - 1]
    um.prefill = { state: "reading", rate: null, mean: null, read: null, cached: null }
    const m: Message = { role: "assistant", text: "", reasoning: "", time: Date.now() }
    this.messages.push(m)
    const abort = new AbortController()
    const conv = this.messages                                // this conversation's list: it stays the same list when the page shows another conversation meanwhile
    this.save(m)                                              // the conversation is in the list now (it has its id), not when its answer ends
    const id = this.index.active ?? NEW_KEY
    const run: Run = { abort, msg: m, messages: conv, meter: new PrefillMeter(), compacting: false }
    this.runs.set(id, run)
    this.notify()
    if (compactFirst) await this.runCompact(id, run, ctx, "", true, 2, before)       // near the end of the context: the earlier messages become a summary; this prompt and its answer are not part of it

    const s = this.settings
    const body: Record<string, unknown> = { model: ctx.health.model, messages: apiMessages(conv), stream: true, reasoning_effort: s.thinking }
    if (s.temperature > 0) Object.assign(body, { temperature: +s.temperature, top_p: +s.top_p, top_k: +s.top_k })
    else body.temperature = 0
    if (s.seed) body.seed = +s.seed
    if (s.max) body.max_tokens = +s.max
    if (ctx.projectionLoaded) body.experimental_speed_projection = !!s.esp
    Object.assign(body, mcpRequest(s, ctx.mcp))                            // this server may run MCP tools for it (the ones not switched off)
    Object.assign(body, agentRequest(s, ctx.agent ?? NO_AGENT, ctx.folder, id === NEW_KEY ? null : id, permsFor(store, rulesOf(store, id), this.currentProject()), String(um.time)))      // the prompt's time is its checkpoint: the files the tools change in it can be put back      // the coding tools, when they are on and reachable
    const lastPrompt = [...conv].reverse().find((x) => x.role === "user")
    const skill = lastPrompt ? skillOfMessage(lastPrompt.text, ctx.skills ?? []) : null
    if (skill) body.strata_skill = skill                                   // "/name": the server loads that skill for this message

    let firstAt: number | null = null
    let thinkStart: number | null = null
    let usage: { completion_tokens?: number; prompt_tokens?: number } | null = null
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
    const sampled = run.meter?.mean() ?? null
    const own = timings && !(m.tools || []).length ? timings : null
    um.prefill = { state: "done", rate: null, mean: own ? own.prompt_per_second ?? null : sampled, read: own?.prompt_n ?? null, cached: own?.cache_n ?? null }
    run.meter = null
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
    if (usage?.prompt_tokens && !m.error) stats.ctx = usage.prompt_tokens + (usage.completion_tokens ?? 0)       // how much of the context the conversation uses now (after every tool round)
    for (const tc of m.tools || []) if (tc.state === "writing" || tc.state === "running") { tc.state = "skipped"; tc.ms = null }
    const ran = (m.tools || []).filter((tc) => tc.state === "done" || tc.state === "error").length
    if (ran) stats.tools = ran
    if (m.limit) stats.limit = m.limit
    if (Object.keys(stats).length) m.stats = stats
    this.runs.delete(id)
    if (this.frame) { cancelAnimationFrame(this.frame); this.frame = 0 }
    if (this.index.active === id || id === NEW_KEY) this.save()
    else {                                                    // another conversation is open: this one is saved in its own place
      const r = saveBackground(store, this.index, id, storedOf(conv), Date.now())
      this.index = r.index
      if (!r.ok) this.storageFull()
    }
    this.notify()
    this.drain(id)                                            // what was typed while it answered goes next
  }
}

export const chat = new ChatController()
export const useChatVersion = () => useSyncExternalStore(chat.subscribe, chat.getVersion)
