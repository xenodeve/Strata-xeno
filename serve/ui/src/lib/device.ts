// The device this browser is, for a server in team mode (#310): an id made once and kept in this browser, and a name the owner reads
// when the device asks to be let in. Both go with every request (apiHeaders). A key used from a new device waits until the owner (or
// the key's holder, from a device already let in) approves it: the server then answers 403 device_pending with a 4-character code
// the two compare, or 403 device_refused. Outside team mode the server does not look at either header.
import { useSyncExternalStore } from "react"
import { store } from "./store"

const ID = /^[A-Za-z0-9_-]{22}$/
let id: string | null = null                 // kept here too: a browser that refuses to store it still sends one id all session

/** This browser's device id: 16 random bytes in base64url (22 characters), made once and kept as strata.device. */
export function deviceId(): string {
  if (id) return id
  const kept = store.get<unknown>("device", null)
  if (typeof kept === "string" && ID.test(kept)) return (id = kept)
  const b = new Uint8Array(16)
  crypto.getRandomValues(b)
  id = btoa(String.fromCharCode(...b)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "")
  store.set("device", id)
  return id
}

/** "Chrome on Windows": what the owner sees for a device that was not given a name. */
export function uaSummary(ua: string): string {
  const browser = /Edg(e|A|iOS)?\//.test(ua) ? "Edge" : /OPR\/|Opera/.test(ua) ? "Opera" : /Firefox\/|FxiOS\//.test(ua) ? "Firefox"
    : /Chrome\/|CriOS\//.test(ua) ? "Chrome" : /Safari\//.test(ua) ? "Safari" : ""
  const system = /iPhone/.test(ua) ? "iPhone" : /iPad/.test(ua) ? "iPad" : /Android/.test(ua) ? "Android" : /Windows/.test(ua) ? "Windows"
    : /CrOS/.test(ua) ? "ChromeOS" : /Mac OS X|Macintosh/.test(ua) ? "macOS" : /Linux/.test(ua) ? "Linux" : ""
  return browser && system ? `${browser} on ${system}` : browser || system || "A browser"
}

/** What a header can carry (fetch refuses anything past Latin-1): printable ASCII, spaces folded, at most 80 characters. */
export const headerSafe = (s: string) => s.replace(/[^\x20-\x7E]+/g, " ").replace(/\s+/g, " ").trim().slice(0, 80).trim()

const browserName = () => uaSummary(typeof navigator !== "undefined" ? navigator.userAgent || "" : "")

/** The name of this device as the owner sees it: the one given in Settings (strata.deviceName), else the browser's summary. */
export function deviceName(): string {
  const given = store.get<unknown>("deviceName", "")
  return typeof given === "string" && given.trim() ? given.trim() : browserName()
}

/** False when the browser refused to keep it. A blank name goes back to the browser's summary. */
export function setDeviceName(name: string): boolean {
  return name.trim() ? store.set("deviceName", name.trim().slice(0, 80)) : (store.remove("deviceName"), true)
}

/** The device headers apiHeaders adds. */
export function deviceHeaders(): Record<string, string> {
  return { "X-Strata-Device": deviceId(), "X-Strata-Device-Name": headerSafe(deviceName()) || headerSafe(browserName()) || "A browser" }
}

export type Gate = { kind: "pending"; code: string } | { kind: "refused" } | null

/** What a response says about this device: waiting for approval (with its code; "----" when the server's keys file is broken), refused,
 *  or nothing (any other answer). Reads a copy of the body, so the caller can still read it. */
export async function deviceGate(r: Response): Promise<Gate> {
  if (r.status !== 403) return null
  try {
    const e = ((await r.clone().json()) as { error?: { type?: unknown; code?: unknown } })?.error
    if (e?.type === "device_pending") return { kind: "pending", code: typeof e.code === "string" && e.code ? e.code : "----" }
    if (e?.type === "device_refused") return { kind: "refused" }
  } catch { /* not json */ }
  return null
}

/** The code of a 403 device_pending, else null. */
export async function isDevicePending(r: Response): Promise<string | null> {
  const g = await deviceGate(r)
  return g?.kind === "pending" ? g.code : null
}

// The device's state as last seen, for the waiting screen: any part of the app that gets a 403 for the device says so here.
let gate: Gate = null
const listeners = new Set<() => void>()
export const getGate = () => gate
export function setGate(g: Gate) {
  if (g === gate || (g && gate && g.kind === gate.kind && (g.kind !== "pending" || (gate.kind === "pending" && g.code === gate.code)))) return
  gate = g
  listeners.forEach((f) => f())
}
/** Notes a response's device state; true when it was a device 403 (the caller stops there: the waiting screen takes over). */
export async function noteGate(r: Response): Promise<boolean> {
  const g = await deviceGate(r)
  if (g) setGate(g)
  return g !== null
}
export const useGate = (): Gate => useSyncExternalStore((cb) => { listeners.add(cb); return () => { listeners.delete(cb) } }, getGate, () => null)

/** For tests: forget the id read into memory, as a reload does. */
export function _resetDevice() { id = null; gate = null }
