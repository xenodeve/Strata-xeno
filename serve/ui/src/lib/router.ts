import { useSyncExternalStore } from "react"

// Hash routing with relative paths only: the app works behind a path-prefixed reverse proxy.
export const PAGES = ["chat", "dashboard", "live", "requests", "hardware", "settings", "about"] as const
export type Page = (typeof PAGES)[number]

export interface Route { page: Page; params: string[] }

export function parse(hash: string): Route {
  const [head, ...params] = hash.replace(/^#\/?/, "").split("/").filter(Boolean).map(decodeURIComponent)
  return { page: (PAGES as readonly string[]).includes(head) ? (head as Page) : "chat", params }
}

const subscribe = (cb: () => void) => {
  addEventListener("hashchange", cb)
  return () => removeEventListener("hashchange", cb)
}

export const href = (page: Page, ...params: string[]) => "#/" + [page, ...params.map(encodeURIComponent)].join("/")

let last: { hash: string; route: Route } | null = null
const snapshot = (): Route => {
  if (!last || last.hash !== location.hash) last = { hash: location.hash, route: parse(location.hash) }
  return last.route
}

export const useRoute = (): Route => useSyncExternalStore(subscribe, snapshot, () => parse(""))
