import { store } from "./store"
import { t } from "./i18n"
import type { McpConfigView } from "./mcpconfig"
import type { ImportView } from "./importer"
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

/** The user's answer to a question of the coding tools (a card in the chat); false with the reason when the server did not take it. */
export async function postPermission(id: string, decision: "allow" | "allow_chat" | "deny"): Promise<{ ok: true } | { error: string }> {
  try {
    const r = await fetch(url("agent/permission"), { method: "POST", headers: apiHeaders(true), body: JSON.stringify({ id, decision }) })
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
