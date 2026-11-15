// Named keys and their devices (#310, serve/identity.py): what the owner's Settings shows and changes, the devices waiting for approval,
// whether this server is in team mode, and who holds the engine. Never a key or a hash: the server sends a new key's text once, in the
// answer to its create, and keeps only its sha256.
import { useEffect, useSyncExternalStore } from "react"
import { apiHeaders, errorMessage, url } from "./api"
import { noteGate } from "./device"

export type Role = "owner" | "guest"
export interface DeviceRow {
  device: string; label: string; addr: string; status: "pending" | "approved" | "refused"
  first_seen: number | null; last_seen: number | null; decided_by: string | null
}
export interface KeyRow { name: string; role: Role; created: number | null; devices: DeviceRow[]; legacy?: boolean }
export interface LogRow { t: number; what: string; key: string; device: string; by: string }
export interface KeysView { keys: KeyRow[]; log: LogRow[]; retired: string[]; team: boolean; keys_file: string | null }
export interface PendingDevice { key: string; device: string; label: string; addr: string; code: string; first_seen: number | null }

/** What GET /keys answered, read for the page: the owner's view; a guest of a team-mode server (403 "forbidden"); someone who is not the
 *  owner of a server outside team mode (403 with the reason, as the other owner-only routes); an error; or nothing (a server without the
 *  route). A device 403 never gets here (the waiting screen takes it). */
export type KeysAnswer =
  | { kind: "view"; view: KeysView } | { kind: "guest" } | { kind: "not-owner"; message: string } | { kind: "error"; message: string } | { kind: "none" }

export function keysAnswer(status: number, body: unknown): KeysAnswer {
  const err = (body as { error?: { type?: unknown; message?: unknown } } | null)?.error
  if (status === 200 && body && Array.isArray((body as KeysView).keys)) {
    const v = body as Partial<KeysView>
    return { kind: "view", view: { keys: v.keys!, log: v.log ?? [], retired: v.retired ?? [], team: v.team === true, keys_file: v.keys_file ?? null } }
  }
  if (status === 403 && err?.type === "forbidden") return { kind: "guest" }
  if (status === 403) return { kind: "not-owner", message: typeof err?.message === "string" ? err.message : "" }
  if (status === 404) return { kind: "none" }
  return { kind: "error", message: typeof err?.message === "string" && err.message ? err.message : `HTTP ${status}` }
}

/** Whether the server is in team mode, as far as this answer tells: true, false, or null (it does not say). */
export function teamOf(a: KeysAnswer): boolean | null {
  return a.kind === "view" ? a.view.team : a.kind === "guest" ? true : a.kind === "not-owner" ? false : null
}

export async function getKeys(): Promise<KeysAnswer> {
  try {
    const r = await fetch(url("keys"), { headers: apiHeaders() })
    if (await noteGate(r)) return { kind: "none" }
    if (r.status === 401) return { kind: "error", message: await errorMessage(r) }
    let body: unknown = null
    try { body = await r.json() } catch { /* not json */ }
    const a = keysAnswer(r.status, body)
    const team = teamOf(a)
    if (team !== null) setTeam(team)
    return a
  } catch (e) { return { kind: "error", message: e instanceof Error ? e.message : String(e) } }
}

export type KeysAction =
  | { action: "create"; name: string; role: Role } | { action: "rename"; name: string; to: string }
  | { action: "role"; name: string; role: Role } | { action: "revoke"; name: string }

/** A change to the keys. A create answers the new key's text, this once. */
export async function postKeys(body: KeysAction): Promise<{ ok: true; key?: string; runs_stopped?: number; prompts_closed?: number } | { error: string }> {
  try {
    const r = await fetch(url("keys"), { method: "POST", headers: apiHeaders(true), body: JSON.stringify(body) })
    if (!r.ok) { await noteGate(r); return { error: await errorMessage(r) } }
    const got = (await r.json()) as { key?: string; runs_stopped?: number; prompts_closed?: number }
    return { ok: true, ...got }
  } catch (e) { return { error: e instanceof Error ? e.message : String(e) } }
}

/** The devices waiting for approval that this caller may decide; null when the server cannot be asked. */
export async function getPending(): Promise<PendingDevice[] | null> {
  try {
    const r = await fetch(url("devices/pending"), { headers: apiHeaders() })
    if (!r.ok) { await noteGate(r); return null }
    const body = (await r.json()) as { devices?: PendingDevice[] }
    return Array.isArray(body.devices) ? body.devices : null
  } catch { return null }
}

/** Approve or refuse a device of a key, or clear it (forgotten: next time it is seen it waits again). */
export async function postDevice(key: string, device: string, decision: "approve" | "refuse" | "clear"): Promise<{ ok: true } | { error: string }> {
  const body = decision === "clear" ? { key, device, action: "clear" } : { key, device, approve: decision === "approve" }
  try {
    const r = await fetch(url("devices"), { method: "POST", headers: apiHeaders(true), body: JSON.stringify(body) })
    if (!r.ok) { await noteGate(r); return { error: await errorMessage(r) } }
    return { ok: true }
  } catch (e) { return { error: e instanceof Error ? e.message : String(e) } }
}

// Team mode, as GET /keys last told it (asked once, the first time a part of the page wants it; again after a change of keys or of the
// API key). Until it is known the page is drawn as outside team mode, which is how it looked before #310.
let team = false
let asked = false
const listeners = new Set<() => void>()
function setTeam(v: boolean) { if (v !== team) { team = v; listeners.forEach((f) => f()) } }
export function refreshTeam() { asked = true; void getKeys() }
export function useTeam(): boolean {
  useEffect(() => { if (!asked) refreshTeam() }, [])
  return useSyncExternalStore((cb) => { listeners.add(cb); return () => { listeners.delete(cb) } }, () => team, () => false)
}

// Who holds the engine and who waits for it (/metrics live, #310).
export interface Who { who: string; since: number }
export interface LiveWho { holder?: string | null; waiting_who?: Who[]; slots?: { state: string; who?: string | null }[] }

/** The identities whose requests run on the engine now: the holder, or with "parallel" each busy slot's (once each, in slot order). */
export function holdersOf(l: LiveWho): string[] {
  if (l.holder) return [l.holder]
  const names = (l.slots ?? []).filter((s) => s.state !== "idle" && s.who).map((s) => s.who as string)
  return [...new Set(names)]
}

/** The identities waiting for the engine, in arrival order, each with its place: "ann (1), bob (2)". Read from waiting_who only
 *  (live.waiting is #465's count of requests waiting for a slot, with no names). */
export function waitingText(l: LiveWho): string {
  return (l.waiting_who ?? []).map((w, i) => `${w.who} (${i + 1})`).join(", ")
}

/** A log entry's action, as the server writes it ("created", "renamed from ann", "role:guest", ...): the kind, and the name or role it carries. */
export function logKind(what: string): { kind: string; arg: string } {
  if (what.startsWith("renamed from ")) return { kind: "renamed", arg: what.slice(13) }
  if (what.startsWith("role:")) return { kind: "role", arg: what.slice(5) }
  return { kind: what, arg: "" }
}
