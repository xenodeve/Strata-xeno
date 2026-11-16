// Named keys and their devices (#310, serve/identity.py): what the owner's Settings shows and changes, the devices waiting for approval,
// whether this server is in team mode, and who holds the engine. Never a key or a hash: the server sends a new key's text once, in the
// answer to its create, and keeps only its sha256.
import { useEffect, useSyncExternalStore } from "react"
import { apiHeaders, errorMessage, url } from "./api"
import { noteGate } from "./device"
import { t } from "./i18n"

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

// The runners connected to this server (GET /runners, #310 part 2). The caller's own runners come in full, and every runner for the owner;
// a guest gets another identity's runner as `who` and `online` only, so the page has nothing more of it to show.
export interface RunnerRow {
  who: string; online: boolean
  id?: string; device?: string; machine?: string | null; platform?: string | null; shell?: string | null; home?: string | null; since?: number
}

/** The runners this caller may see; null when the server cannot be asked (no such route, a caller who may not, an error). */
export async function getRunners(): Promise<RunnerRow[] | null> {
  try {
    const r = await fetch(url("runners"), { headers: apiHeaders() })
    if (!r.ok) { await noteGate(r); return null }
    const body = (await r.json()) as { runners?: RunnerRow[] }
    return Array.isArray(body.runners) ? body.runners : null
  } catch { return null }
}

/** The runners as one line: "ann (ANN-LAPTOP) online, bob offline". Only the name, the machine's name when the row has one, and whether
 *  it is online: never the id, the home folder or the shell, which the owner's full rows also carry. */
export function runnersText(rows: RunnerRow[]): string {
  return rows.map((r) => `${r.who}${r.machine ? ` (${r.machine})` : ""} ${r.online ? t("online") : t("offline")}`).join(", ")
}

/** A log entry's action, as the server writes it ("created", "renamed from ann", "role:guest", ...): the kind, and the name or role it carries. */
export function logKind(what: string): { kind: string; arg: string } {
  if (what.startsWith("renamed from ")) return { kind: "renamed", arg: what.slice(13) }
  if (what.startsWith("role:")) return { kind: "role", arg: what.slice(5) }
  return { kind: what, arg: "" }
}

// A new key's text, shown once (I1 of the task 6 review). It is kept here, not in the Settings page's state, and drawn above the waiting
// screen (App), so it survives anything that takes the page away before it is copied: the first key puts this device in team mode, where
// it may wait for approval at once. Gone when the user dismisses it, or with the page (never written to storage).
/** What became of this browser's own API key: it took the new key, it could not keep it, it kept the key it had, or it kept a key that
 *  may no longer be valid (the server could not be asked whether it had an API key before the first key). */
export type BrowserKey = "stored" | "failed" | "unchanged" | "uncertain"
export interface FreshKey { secret: string; name: string; browser: BrowserKey }
let fresh: FreshKey | null = null
const freshListeners = new Set<() => void>()
export const getFreshKey = () => fresh
export function subscribeFreshKey(f: () => void) { freshListeners.add(f); return () => { freshListeners.delete(f) } }
export function showFreshKey(k: FreshKey) { fresh = k; freshListeners.forEach((f) => f()) }
export function dismissFreshKey() { fresh = null; freshListeners.forEach((f) => f()) }
export const useFreshKey = (): FreshKey | null => useSyncExternalStore(subscribeFreshKey, getFreshKey, () => null)

/** Whether this browser takes a new key as its own API key. Only the first key (it turns team mode on, and from then on this PC needs a
 *  key too): when the browser has none; or when it has one the server never asked for (no API key, /health "api_key": false), which is
 *  not valid once team mode is on. A browser holding the server's API key keeps it (it stays valid); so does one whose server could not
 *  be asked (serverHasKey null). */
export function browserKeyAfterCreate(o: { first: boolean; stored: string; serverHasKey: boolean | null }): "store" | "keep" {
  if (!o.first) return "keep"
  if (!o.stored.trim()) return "store"
  return o.serverHasKey === false ? "store" : "keep"
}

export interface CreateIO {
  serverHasKey: () => Promise<boolean | null>                       // /health "api_key"
  post: (b: KeysAction) => Promise<{ ok: true; key?: string } | { error: string }>
  stored: () => string                                              // this browser's API key
  keep: (key: string) => boolean                                    // store it as this browser's API key; false when refused
}

/** Make a key, and decide whether this browser takes it. /health is read BEFORE the create: its "api_key" is the server's key_required(),
 *  true as soon as the first key has made the keys file, so only the state before the create tells a stale stored key from the server's own
 *  API key. */
export async function createKey(body: { name: string; role: Role }, first: boolean, io: CreateIO): Promise<{ fresh: FreshKey } | { error: string }> {
  const before = first ? await io.serverHasKey() : null
  const r = await io.post({ action: "create", name: body.name, role: body.role })
  if ("error" in r) return { error: r.error }
  if (!r.key) return { error: "the server did not send the new key" }
  const stored = io.stored()
  const take = browserKeyAfterCreate({ first, stored, serverHasKey: before })
  const browser: BrowserKey = take === "store" ? (io.keep(r.key) ? "stored" : "failed") : first && before === null && stored.trim() ? "uncertain" : "unchanged"
  return { fresh: { secret: r.key, name: body.name.trim(), browser } }
}
