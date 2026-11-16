// This browser's runner (#310 part 2): the runner is the part of Strata Desktop on the engineer's own PC that runs the coding tools, MCP servers
// and skills there for the server. Its id is kept here as `strata.runner` (Strata Desktop sets it; Settings > This browser's runner takes a
// pasted one), and a request or page route for a project on that PC names it. Knowing an id grants nothing: the server only routes to a runner
// of the caller's own key. A page with no runner id behaves as before.
import { msg, t } from "./i18n"
import { store } from "./store"
import type { AgentInfo } from "./agent"

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

/** A call of the coding tools: the server's group "agent", or - only when the event says the call ran on My PC (`machine: "my"`) - the runner's
 *  "my_agent" ("my_agent_2" when a server MCP took that name). A server MCP literally named my_agent stays an MCP card: the name never decides alone. */
export const isCodingCall = (call: { server?: string; machine?: string }): boolean =>
  call.machine === "my" ? !!call.server && /^my_agent(_\d+)?$/.test(call.server) : call.server === "agent"
/** A call of the skills server of either machine, by the same rule. */
export const isSkillsCall = (call: { server?: string; machine?: string }): boolean =>
  call.machine === "my" ? !!call.server && /^my_skills(_\d+)?$/.test(call.server) : call.server === "skills"

/** Where a chat's coding tools work. A project keeps its machine (a My PC project is never quietly run with the server's tools). A chat in no
 *  project: the Server PC for an owner (GET /agent `allowed`), else My PC when this browser has a runner; without a runner id, the Server PC
 *  as before. `serverAllowed` undefined (GET /agent not answered, or it failed) is the Server PC too: an owner's chat never moves to the other
 *  machine on a network blip (with no answer the Server PC's tools are not offered either, as before). */
export function chatMachine(c: { project: Machine | undefined; serverAllowed: boolean | undefined; runner: string }): Machine {
  if (c.project) return c.project
  return c.runner && c.serverAllowed === false ? "my" : "server"
}

export type RunnerState = AgentInfo | "unknown" | null | undefined          // GET /agent?runner=; "unknown": no such runner for this key (404); null: the ask failed; undefined: not answered yet

/** The coding tools of a chat on My PC, from what this page knows of its runner, and what stops them (null: nothing).
 *  - No runner id, or a runner the server does not know for this key: nothing can reach the PC. The chat must not send a prompt that silently
 *    goes without tools: `block` says why, and the page refuses the send (with the coding tools on).
 *  - Otherwise (online, offline, not answered yet, or one failed ask) the request goes with `runner`: the server knows best, and an offline PC is
 *    answered with its own words (409 runner_offline), never a tool-less answer. */
export function myPcCoding(runner: string, state: RunnerState): { info: AgentInfo; block: "no-runner" | "unknown" | null } {
  if (!runner) return { info: { available: true, allowed: false, shell: null, tools: [] }, block: "no-runner" }
  if (state === "unknown") return { info: { available: true, allowed: false, shell: null, tools: [] }, block: "unknown" }
  if (state && typeof state === "object") return { info: { ...state, allowed: true }, block: null }
  return { info: { available: true, allowed: true, shell: null, tools: [] }, block: null }
}

/** What a blocked My PC chat says: in full (next to the prompt, on a refused send, in the controls) and short (the + menu's row). `server` is this
 *  server's address, for the command that starts the runner. */
export function blockText(block: "no-runner" | "unknown", server: string): { full: string; short: string } {
  return block === "no-runner"
    ? { full: t("This project is on My PC, and this browser has no runner. Set its id in Settings, under This browser's runner, then send again."), short: t("My PC: this browser has no runner") }
    : { full: t("Your PC's runner is not connected for your key. Start it on your PC with python -m serve.runner --server {server} --key YOUR-KEY, then send again.", { server }), short: t("My PC: the runner is not connected") }
}

/** The labels of the two machines. */
export const MACHINE_LABEL: Record<Machine, string> = { my: msg("My PC"), server: msg("Server PC") }

// Why the server closed a card (an approval or the model's questions) during a call on My PC: `answered` / `question_answered` with a `reason`,
// the card watcher's `abort.why` (serve/runner_hub.py `_card`: `r.ended or "ended"`, "offline", "cancelled"). `Runner.ended` is one of the short
// words given to `Runner.end`: "replaced" (a newer runner on the PC, or a key rename that took its id), "revoked", "device" (refused or cleared),
// "closed" (the server stops, or a runner registered while it stops).
const CLOSED: Record<string, string> = {
  offline: msg("Your PC went offline, so this was closed."),
  replaced: msg("Your PC's runner restarted, so this was closed."),
  revoked: msg("The key of this chat was revoked, so this was closed."),
  device: msg("The owner refused or cleared your PC as a device, so this was closed."),
  closed: msg("The server is stopping, so this was closed."),
  ended: msg("Your PC's runner stopped, so this was closed."),
  cancelled: msg("The answer stopped before you answered."),
}

/** The words for a card the server closed (null: none was given, so the card shows what the user did). A reason this page does not know gets a
 *  sentence of its own, never the server's bare word. */
export function closeReason(reason: string | null | undefined): string | null {
  if (!reason) return null
  return t(CLOSED[reason] ?? msg("The server closed this before you answered."))
}
