// This browser's runner (#310 part 2): the runner is the part of Strata Desktop on the engineer's own PC that runs the coding tools, MCP servers
// and skills there for the server. Its id is kept here as `strata.runner` (Strata Desktop sets it; Settings > This browser's runner takes a
// pasted one), and a request or page route for a project on that PC names it. Knowing an id grants nothing: the server only routes to a runner
// of the caller's own key. A page with no runner id behaves as before.
import { msg, t } from "./i18n"
import { store } from "./store"

export type Machine = "my" | "server"
export const RUNNER_RE = /^r[0-9a-f]{16}$/
const KEY = "runner"

/** The runner id of this browser, or "" when there is none (or what is kept is not one). Kept JSON-encoded like every strata.* value; a bare id
 *  written by another program is read too. */
export function runnerId(): string {
  let v: unknown = store.get<unknown>(KEY, null)
  if (v === null) {
    try { v = localStorage.getItem("strata." + KEY) } catch { v = null }              // not JSON: a bare id
  }
  return typeof v === "string" && RUNNER_RE.test(v) ? v : ""
}

/** Keeps `id` (spaces around it are dropped) as this browser's runner; "" forgets it. False (and nothing kept) for what is not a runner id. */
export function setRunnerId(id: string): boolean {
  const v = id.trim()
  if (!v) { store.remove(KEY); return true }
  if (!RUNNER_RE.test(v)) return false
  return store.set(KEY, v)
}

/** `path` with `runner=<id>` added to its query (or starting one); unchanged without a runner. */
export const withRunner = (path: string, runner?: string): string => (runner ? `${path}${path.includes("?") ? "&" : "?"}runner=${encodeURIComponent(runner)}` : path)

/** Which machine ran a call: the event's `machine` ("my" is the engineer's own PC); a server MCP may be named my_x, so the name never decides. */
export const machineOf = (call: { server?: string; machine?: string }): Machine => (call.machine === "my" ? "my" : "server")

/** The coding tools' group: the server's ("agent") or the runner's ("my_agent", or "my_agent_2" when a server MCP took that name). */
export const isCodingServer = (server?: string): boolean => server === "agent" || (!!server && /^my_agent(_\d+)?$/.test(server))
/** The skills server of either machine. */
export const isSkillsServer = (server?: string): boolean => server === "skills" || (!!server && /^my_skills(_\d+)?$/.test(server))

/** Where a chat's coding tools work. A project keeps its machine (a My PC project is never quietly run with the server's tools). A chat in no
 *  project: the Server PC for an owner (GET /agent `allowed`), else My PC when this browser has a runner; without a runner id, the Server PC
 *  as before. */
export function chatMachine(c: { project: Machine | undefined; serverAllowed: boolean; runner: string }): Machine {
  if (c.project) return c.project
  return c.runner && !c.serverAllowed ? "my" : "server"
}

/** The labels of the two machines. */
export const MACHINE_LABEL: Record<Machine, string> = { my: msg("My PC"), server: msg("Server PC") }

// Why the server closed a card (an approval or the model's questions) during a call on My PC: `answered` / `question_answered` with a `reason`
// (serve/runner_hub.py: REPLACED, REVOKED, DEVICE, CLOSED, and the watcher's "offline", "ended", "cancelled", "closed").
const CLOSED: Record<string, string> = {
  offline: msg("Your PC went offline, so this was closed."),
  "your PC's runner restarted": msg("Your PC's runner restarted, so this was closed."),
  "the key was revoked": msg("The key of this chat was revoked, so this was closed."),
  "the owner refused or cleared this PC's device": msg("The owner refused or cleared your PC as a device, so this was closed."),
  "the server is stopping": msg("The server is stopping, so this was closed."),
  closed: msg("The server is stopping, so this was closed."),
  ended: msg("Your PC's runner stopped, so this was closed."),
  cancelled: msg("The answer stopped before you answered."),
}

/** The words for a card the server closed (null: none was given, so the card shows what the user did). */
export function closeReason(reason: string | null | undefined): string | null {
  if (!reason) return null
  return CLOSED[reason] ? t(CLOSED[reason]) : t("This was closed: {reason}", { reason })
}
