// The page's side of the chat's coding tools (issue #96): what a request says about them, the rules a user chose "allow for this chat" for,
// and short forms of a call for showing it. The server decides what may run and asks; this only carries the user's choices and shows
// what happened. Nothing here can make a call run without the server's say-so.
import type { Backing } from "./sessions"

export interface AgentInfo { available: boolean; allowed: boolean; reason?: string; shell: string | null; tools: string[] }
export const NO_AGENT: AgentInfo = { available: false, allowed: false, shell: null, tools: [] }
export type AgentMode = "ask" | "plan" | "auto"
export const MODES: AgentMode[] = ["ask", "plan", "auto"]

export interface AgentRequest { cwd?: string; dirs?: string[]; mode: AgentMode; session: string; allow: string[]; deny?: string[] }

/** What a request says: `strata_agent` when the coding tools are on (the default), reachable from here and the server has them; else nothing.
 *  An unknown mode is the default one (the server also ignores it). */
export function agentRequest(s: { agent?: boolean; agentMode?: string }, info: AgentInfo, folders: string | string[] | null | undefined, session: string | null, rules: string[] | { allow: string[]; deny?: string[] }): { strata_agent?: AgentRequest } {
  if (s.agent === false || !info.available || !info.allowed) return {}
  const mode: AgentMode = s.agentMode === "plan" || s.agentMode === "auto" ? s.agentMode : "ask"
  const list = (Array.isArray(folders) ? folders : [folders]).filter((f): f is string => typeof f === "string" && f.trim().length > 0).map((f) => f.trim())
  const [cwd, ...dirs] = list                                                  // the first is the main folder; the others are the project's other folders
  const allow = (Array.isArray(rules) ? rules : rules.allow).slice(0, 200)
  const deny = (Array.isArray(rules) ? [] : rules.deny ?? []).slice(0, 200)
  return { strata_agent: { ...(cwd ? { cwd } : {}), ...(cwd && dirs.length ? { dirs } : {}), mode, session: session || "new", allow, ...(deny.length ? { deny } : {}) } }
}

// ------------------------------------------------------------------------------------------------ the rules of a chat
const KEY = "agent.rules"
const MAX_CHATS = 50
const MAX_RULES = 100
interface RuleStore { order: string[]; rules: Record<string, string[]> }

function load(b: Backing): RuleStore {
  const raw = b.get<unknown>(KEY, null)
  const out: RuleStore = { order: [], rules: {} }
  if (!raw || typeof raw !== "object") return out
  const r = raw as { order?: unknown; rules?: unknown }
  if (!Array.isArray(r.order) || !r.rules || typeof r.rules !== "object") return out
  for (const id of r.order) {
    const list = (r.rules as Record<string, unknown>)[id as string]
    if (typeof id === "string" && Array.isArray(list)) { out.order.push(id); out.rules[id] = list.filter((x): x is string => typeof x === "string" && x.length > 0 && x.length <= 500) }
  }
  return out
}

export const rulesOf = (b: Backing, session: string): string[] => load(b).rules[session] ?? []

/** Remember "allow for this chat": once, newest last; a chat keeps at most 100, and the chats used longest ago are forgotten past 50. */
export function addRule(b: Backing, session: string, rule: string): void {
  const s = load(b)
  const list = s.rules[session] ?? []
  if (!list.includes(rule)) list.push(rule)
  while (list.length > MAX_RULES) list.shift()
  s.rules[session] = list
  s.order = [...s.order.filter((x) => x !== session), session]
  while (s.order.length > MAX_CHATS) delete s.rules[s.order.shift()!]
  b.set(KEY, s)
}

export function forgetRules(b: Backing, session: string): void {
  const s = load(b)
  delete s.rules[session]
  s.order = s.order.filter((x) => x !== session)
  b.set(KEY, s)
}

// ------------------------------------------------------------------------------------------------ showing a call
const cut = (s: string, n = 120) => (s.length > n ? s.slice(0, n) + "…" : s)
const str = (x: unknown) => (typeof x === "string" ? x : "")

/** A call in a few words (what the row says next to the tool's name). */
export function toolSummary(tool: string, args: unknown): string {
  const a = args && typeof args === "object" ? (args as Record<string, unknown>) : null
  if (!a) return ""
  switch (tool) {
    case "Read": case "Write": case "Edit": return cut(str(a.file_path))
    case "NotebookEdit": return cut(str(a.notebook_path))
    case "Glob": return cut(str(a.pattern))
    case "Grep": return cut(str(a.path) ? `${str(a.pattern)} in ${str(a.path)}` : str(a.pattern))
    case "Bash": return cut(str(a.description).trim() || str(a.command).split("\n")[0])
    case "TodoWrite": return Array.isArray(a.todos) ? `${a.todos.length} ${a.todos.length === 1 ? "step" : "steps"}` : ""
    case "BashOutput": return str(a.bash_id)
    case "KillShell": return str(a.shell_id)
    default: return ""
  }
}

export interface DiffRow { kind: "same" | "del" | "add" | "more"; text: string }
const MAX_ROWS = 200

/** An edit as lines: the same lines at the start and end, the old ones removed, the new ones added (cut when it is long). */
export function diffRows(oldText: string, newText: string): DiffRow[] {
  const a = oldText === "" ? [] : oldText.split("\n")
  const b = newText === "" ? [] : newText.split("\n")
  let head = 0
  while (head < a.length && head < b.length && a[head] === b[head]) head++
  let tail = 0
  while (tail < a.length - head && tail < b.length - head && a[a.length - 1 - tail] === b[b.length - 1 - tail]) tail++
  const rows: DiffRow[] = [
    ...a.slice(0, head).map((text) => ({ kind: "same" as const, text })),
    ...a.slice(head, a.length - tail).map((text) => ({ kind: "del" as const, text })),
    ...b.slice(head, b.length - tail).map((text) => ({ kind: "add" as const, text })),
    ...a.slice(a.length - tail).map((text) => ({ kind: "same" as const, text })),
  ]
  return rows.length > MAX_ROWS ? [...rows.slice(0, MAX_ROWS), { kind: "more", text: `… ${rows.length - MAX_ROWS} more lines` }] : rows
}
