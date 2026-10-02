import { store } from "./store"
import { t } from "./i18n"
import type { McpConfigView } from "./mcpconfig"
import type { ImportView } from "./importer"
import type { HooksView } from "./hooks"
import type { AgentInfo } from "./agent"

export const apiHeaders = (json = false): Record<string, string> => {
  const h: Record<string, string> = {}
  const key = store.get("apikey", "")
  if (key) h.Authorization = "Bearer " + key
  if (json) h["Content-Type"] = "application/json"
  return h
}

// The server's root, from where this page is served: "/", "/p/" behind a path-prefixed reverse proxy, and the same
// with "next/" taken off while the app lives at /next/. Never an absolute path baked into the build.
export const rootOf = (pathname: string) => pathname.replace(/next\/$/, "").replace(/[^/]*$/, "").replace(/^\/+/, "/")   // never "//host/": that is another origin
export const root = () => rootOf(location.pathname)
export const url = (path: string) => root() + path

export interface Health { model: string; images: boolean; max_context: number }
export const NO_HEALTH: Health = { model: "strata", images: false, max_context: 0 }

export async function getHealth(): Promise<Health> {
  const r = await fetch(url("health"))
  return (await r.json()) as Health
}

export interface McpTool { tool: string; description?: string }
export interface McpServer {
  name: string; transport: string; status: string; tools: McpTool[]; error?: string; info?: { name?: string; version?: string }
}
export interface McpInfo { servers: McpServer[]; tools: number }

export async function getMcp(): Promise<McpInfo | null> {
  try {
    const r = await fetch(url("mcp"), { headers: apiHeaders() })
    return r.ok ? ((await r.json()) as McpInfo) : null
  } catch { return null }
}

/** The chat's coding tools (serve/agent.py): whether the server has them, and whether this caller may use them (only from this PC, or with the key). */
export async function getAgent(): Promise<AgentInfo | null> {
  try {
    const r = await fetch(url("agent"), { headers: apiHeaders() })
    return r.ok ? ((await r.json()) as AgentInfo) : null
  } catch { return null }
}

export interface HelpersView { on: boolean; available: boolean; editable: boolean; config_file: string | null }

/** Sub-agents for the coding tools (serve/subagent.py): whether they are on. null when the server cannot be asked (not this PC, no such route). */
export async function getHelpers(): Promise<HelpersView | null> {
  try {
    const r = await fetch(url("agent/helpers"), { headers: apiHeaders() })
    return r.ok ? ((await r.json()) as HelpersView) : null
  } catch { return null }
}

export async function postHelpers(on: boolean): Promise<{ view: HelpersView } | { error: string }> {
  try {
    const r = await fetch(url("agent/helpers"), { method: "POST", headers: apiHeaders(true), body: JSON.stringify({ on }) })
    return r.ok ? { view: (await r.json()) as HelpersView } : { error: await errorMessage(r) }
  } catch (e) { return { error: e instanceof Error ? e.message : String(e) } }
}

export interface WebView { on: boolean; provider: "searxng" | "brave"; searxng_url: string; brave_key_set: boolean; available: boolean; editable: boolean; config_file: string | null }

/** Web access for the coding tools (serve/web.py): whether it is on, the search provider, whether there is a Brave key (never the key). null when the server cannot be asked (not this PC, no such route). */
export async function getWeb(): Promise<WebView | null> {
  try {
    const r = await fetch(url("agent/web"), { headers: apiHeaders() })
    return r.ok ? ((await r.json()) as WebView) : null
  } catch { return null }
}

/** Change web access: any of `on`, `provider`, `searxng_url`, `brave_key` (the others stay). The answer is the new state, or why not. */
export async function postWeb(body: Record<string, unknown>): Promise<{ view: WebView } | { error: string }> {
  try {
    const r = await fetch(url("agent/web"), { method: "POST", headers: apiHeaders(true), body: JSON.stringify(body) })
    return r.ok ? { view: (await r.json()) as WebView } : { error: await errorMessage(r) }
  } catch (e) { return { error: e instanceof Error ? e.message : String(e) } }
}

/** The user's hooks (serve/hooks.py), read only, with whether each is on; null when the server cannot be asked (not this PC, no such route). */
export async function getHooks(): Promise<HooksView | null> {
  try {
    const r = await fetch(url("agent/hooks"), { headers: apiHeaders() })
    return r.ok ? ((await r.json()) as HooksView) : null
  } catch { return null }
}

/** Switch hooks off: `off` is the ids of all that are off (the rest are on). The answer is the new list, or why not. */
export async function postHooksOff(off: string[]): Promise<{ view: HooksView } | { error: string }> {
  try {
    const r = await fetch(url("agent/hooks"), { method: "POST", headers: apiHeaders(true), body: JSON.stringify({ off }) })
    return r.ok ? { view: (await r.json()) as HooksView } : { error: await errorMessage(r) }
  } catch (e) { return { error: e instanceof Error ? e.message : String(e) } }
}

export interface FolderView { path: string; parent: string | null; dirs: { name: string; path: string }[]; truncated: boolean }

/** The folders of this PC (serve/folders.py), to choose a project's folder from: where `path` is and the folders in it; blank is the home folder, "@drives" the drives
 *  (Windows). `unknown` when the server cannot be asked (another PC, no such route): then the path cannot be checked, only used as typed. */
export async function getFolders(path: string): Promise<FolderView | { error: string } | "unknown"> {
  try {
    const r = await fetch(url("agent/folders?path=" + encodeURIComponent(path)), { headers: apiHeaders() })
    if (!r.ok) return "unknown"
    const body = await r.json()
    if (body?.ok === true) return body as FolderView
    return typeof body?.error === "string" ? { error: body.error } : "unknown"
  } catch { return "unknown" }
}

/** The user's answer to a question of the coding tools (a card in the chat); false with the reason when the server did not take it. */
export async function postPermission(id: string, decision: "allow" | "allow_chat" | "deny"): Promise<{ ok: true } | { error: string }> {
  try {
    const r = await fetch(url("agent/permission"), { method: "POST", headers: apiHeaders(true), body: JSON.stringify({ id, decision }) })
    return r.ok ? { ok: true } : { error: await errorMessage(r) }
  } catch (e) { return { error: e instanceof Error ? e.message : String(e) } }
}

/** The user's answers to the questions the model asked (AskUserQuestion): {question: [choices or own words]}, or null to skip them. */
export async function postQuestionAnswer(id: string, answers: Record<string, string[]> | null): Promise<{ ok: true } | { error: string }> {
  try {
    const r = await fetch(url("agent/question"), { method: "POST", headers: apiHeaders(true), body: JSON.stringify({ id, answers }) })
    return r.ok ? { ok: true } : { error: await errorMessage(r) }
  } catch (e) { return { error: e instanceof Error ? e.message : String(e) } }
}

export async function getMcpConfig(): Promise<McpConfigView | null> {
  try {
    const r = await fetch(url("mcp/config"), { headers: apiHeaders() })
    return r.ok ? ((await r.json()) as McpConfigView) : null
  } catch { return null }
}

/** Save the MCP servers (and limits); the answer is the new state, or why not (the server names the field it refused). */
export async function saveMcpConfig(body: unknown): Promise<{ view: McpConfigView } | { error: string }> {
  try {
    const r = await fetch(url("mcp/config"), { method: "POST", headers: apiHeaders(true), body: JSON.stringify(body) })
    if (r.ok) return { view: (await r.json()) as McpConfigView }
    return { error: await errorMessage(r) }
  } catch (e) { return { error: e instanceof Error ? e.message : String(e) } }
}

/** The skills and MCP servers of the other apps on this PC (issue #94); `rescan` reads their folders again. */
export async function getImport(rescan = false): Promise<ImportView | null> {
  try {
    const r = await fetch(url("import" + (rescan ? "?rescan=1" : "")), { headers: apiHeaders() })
    return r.ok ? ((await r.json()) as ImportView) : null
  } catch { return null }
}

/** Change the skill switches, import one MCP server, or rescan; the answer is the new state, or why not. */
export async function postImport(body: unknown): Promise<{ view: ImportView } | { error: string }> {
  try {
    const r = await fetch(url("import"), { method: "POST", headers: apiHeaders(true), body: JSON.stringify(body) })
    if (r.ok) return { view: (await r.json()) as ImportView }
    return { error: await errorMessage(r) }
  } catch (e) { return { error: e instanceof Error ? e.message : String(e) } }
}

export async function errorMessage(r: Response): Promise<string> {
  let msg = `HTTP ${r.status}`
  try { msg = ((await r.json()) as { error?: { message?: string } }).error?.message || msg } catch { /* not json */ }
  if (r.status === 401) msg = t("This server needs an API key: add it under Settings.")
  return msg
}
