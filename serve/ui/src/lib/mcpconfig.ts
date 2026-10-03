import { t } from "./i18n"

// Setting up MCP servers from the page (issue #79). The server (serve/mcp_admin.py) decides and checks everything; this is
// the form's side: a draft (all text) to the entry the server takes, the Claude Desktop block people paste, and the body of
// a save. The values of `env` and `headers` come back from the server as MASK and are sent back as MASK to keep them.

export const MASK = "********"
export const NAME = /^[A-Za-z0-9_.-]{1,40}$/

export interface McpEntry { command?: string; args?: string[]; env?: Record<string, string>; cwd?: string; url?: string; headers?: Record<string, string>; disabled?: boolean }
export interface McpTool { tool: string; name?: string; description?: string }
export interface McpRow extends McpEntry {
  name: string; kind: string; source: "config" | "file"; editable: boolean; disabled: boolean
  shadowed?: boolean; overridden?: boolean; file?: string
  status: string; error: string | null; tools: McpTool[]; info?: { name?: string; version?: string }
}
export interface McpLimits { timeout_s: number; max_result_chars: number; max_rounds: number }
export interface McpConfigView { servers: McpRow[]; editable: boolean; reason: string; settings: McpLimits; config_file: string | null; tools: number }
export interface Problem { field: string; message: string }

export interface Draft { name: string; kind: "program" | "address"; command: string; args: string; env: string; cwd: string; url: string; headers: string }
export const emptyDraft = (): Draft => ({ name: "", kind: "program", command: "", args: "", env: "", cwd: "", url: "", headers: "" })

const kv = (m?: Record<string, string>) => Object.entries(m ?? {}).map(([k, v]) => `${k}=${v}`).join("\n")

export function toDraft(e: McpRow): Draft {
  return { name: e.name, kind: e.url ? "address" : "program", command: e.command ?? "", args: (e.args ?? []).join("\n"), env: kv(e.env), cwd: e.cwd ?? "", url: e.url ?? "", headers: kv(e.headers) }
}

/** "KEY=value" lines to an object (a blank line is skipped, the first = splits); `bad`: the line numbers (from 1) with no key. */
export function parseKv(text: string): { map: Record<string, string>; bad: number[] } {
  const map: Record<string, string> = {}
  const bad: number[] = []
  text.split("\n").forEach((raw, i) => {
    const line = raw.replace(/\r$/, "")
    if (!line.trim()) return
    const at = line.indexOf("=")
    const key = at < 0 ? "" : line.slice(0, at).trim()
    if (!key) bad.push(i + 1)
    else map[key] = line.slice(at + 1)
  })
  return { map, bad }
}

/** The entry the server takes from a draft, and what is wrong with it (the server checks again; this is the quick answer). */
export function entryOf(d: Draft, taken: string[] = []): { entry: McpEntry; problems: Problem[] } {
  const problems: Problem[] = []
  if (!NAME.test(d.name)) problems.push({ field: "name", message: t("A name is 1-40 letters, digits, dots, dashes or underscores, with no spaces.") })
  else if (taken.includes(d.name)) problems.push({ field: "name", message: t("There is already a server with this name.") })
  const entry: McpEntry = {}
  if (d.kind === "address") {
    if (!/^https?:\/\/\S+$/.test(d.url.trim())) problems.push({ field: "url", message: t("The address must start with http:// or https://.") })
    else entry.url = d.url.trim()
    const h = parseKv(d.headers)
    if (h.bad.length) problems.push({ field: "headers", message: t("Line {n} has no name before the = sign.", { n: h.bad[0] }) })
    if (Object.keys(h.map).length) entry.headers = h.map
  } else {
    if (!d.command.trim()) problems.push({ field: "command", message: t("The program to start is required.") })
    else entry.command = d.command.trim()
    const args = d.args.split("\n").map((a) => a.replace(/\r$/, "")).filter((a) => a !== "")
    if (args.length > 64) problems.push({ field: "args", message: t("At most 64 arguments.") })
    else if (args.length) entry.args = args
    const e = parseKv(d.env)
    if (e.bad.length) problems.push({ field: "env", message: t("Line {n} has no name before the = sign.", { n: e.bad[0] }) })
    if (Object.keys(e.map).length) entry.env = e.map
    if (d.cwd.trim()) entry.cwd = d.cwd.trim()
  }
  return { entry, problems }
}

/** What a server is saved as: its own fields only, nothing the page added (state, tools, source). */
export function entryFromRow(e: McpRow): McpEntry {
  const out: McpEntry = {}
  for (const k of ["command", "args", "env", "cwd", "url", "headers"] as const) if (e[k] !== undefined && e[k] !== null) (out as Record<string, unknown>)[k] = e[k]
  if (e.disabled) out.disabled = true
  return out
}

/** The whole list the run config will hold, as the POST body takes it: the entries from the run config (those from --mcp-config
 *  are not ours to write), each by name, with `change` applied: a new or replaced entry, or null to delete one. */
export function bodyFor(rows: McpRow[], change?: Change | Change[], settings?: Partial<McpLimits>) {
  const servers: Record<string, McpEntry> = {}
  for (const r of rows) if (r.source === "config") servers[r.name] = entryFromRow(r)
  for (const c of change === undefined ? [] : Array.isArray(change) ? change : [change]) {
    if (c.entry === null) delete servers[c.name]
    else servers[c.name] = c.entry
  }
  return settings ? { servers, settings } : { servers }
}
export interface Change { name: string; entry: McpEntry | null }

/** A block of Claude Desktop's format ({"mcpServers": {...}}), or the same without the wrapper ({"name": {...}}). */
export function parsePaste(text: string, taken: string[] = []): { entries: Record<string, McpEntry>; problem: string | null } {
  let data: unknown
  try { data = JSON.parse(text) } catch { return { entries: {}, problem: t("This is not valid JSON. Paste the block as it is, from the opening { to the closing }.") } }
  const block = data && typeof data === "object" && !Array.isArray(data)
    ? ((data as Record<string, unknown>).mcpServers ?? (data as Record<string, unknown>).mcp_servers ?? data)
    : null
  if (!block || typeof block !== "object" || Array.isArray(block) || !Object.keys(block).length) return { entries: {}, problem: t("There is no server in it. Expected {\"mcpServers\": {\"name\": {...}}}.") }
  const entries: Record<string, McpEntry> = {}
  for (const [name, raw] of Object.entries(block as Record<string, unknown>)) {
    if (!raw || typeof raw !== "object" || Array.isArray(raw)) return { entries: {}, problem: t("\"{name}\" is not a server: it needs a \"command\" or a \"url\".", { name }) }
    const r = raw as Record<string, unknown>
    const draft = emptyDraft()
    draft.name = name
    if (r.url) {
      draft.kind = "address"
      draft.url = String(r.url)
      draft.headers = kv(r.headers as Record<string, string> | undefined)
    } else {
      draft.command = typeof r.command === "string" ? r.command : ""
      draft.args = Array.isArray(r.args) ? r.args.map(String).join("\n") : ""
      draft.env = kv(r.env as Record<string, string> | undefined)
      draft.cwd = typeof r.cwd === "string" ? r.cwd : ""
    }
    const { entry, problems } = entryOf(draft, taken)
    if (problems.length) return { entries: {}, problem: `${name}: ${problems[0].message}` }
    entries[name] = entry
  }
  return { entries, problem: null }
}

/** The limits as the form holds them (text) to numbers, or the name of the first one that is not a number. */
export function limitsOf(text: Record<keyof McpLimits, string>): { limits: McpLimits | null; bad: keyof McpLimits | null } {
  const out = {} as McpLimits
  for (const k of ["timeout_s", "max_result_chars", "max_rounds"] as const) {
    const n = Number(text[k])
    if (!text[k].trim() || !Number.isFinite(n)) return { limits: null, bad: k }
    out[k] = n
  }
  return { limits: out, bad: null }
}

/** A server's state as one word the page has a name for. */
export const stateOf = (r: McpRow) => (r.disabled ? "disabled" : r.status)
