import { store } from "./store"
import { t } from "./i18n"

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

export async function errorMessage(r: Response): Promise<string> {
  let msg = `HTTP ${r.status}`
  try { msg = ((await r.json()) as { error?: { message?: string } }).error?.message || msg } catch { /* not json */ }
  if (r.status === 401) msg = t("This server needs an API key: add it under About > Settings.")
  return msg
}
