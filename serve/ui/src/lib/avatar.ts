// What stands for a status in the app: the orbs (nine forms, one per kind of thing happening), a plain loading mark (a ring that
// turns), or the bots of Libraries.dev (bot-avatars, MIT, vendored): the same states told by a shape and a mood. The choice is kept in the browser, and the bots are loaded
// only when chosen.
import { useSyncExternalStore } from "react"
import { store } from "./store"
import type { OrbDesign } from "./orbs"

export type AvatarKind = "orbs" | "loading" | "bots"
/** The eighteen shapes bot-avatars has. */
export const BOT_TYPES = ["clover", "flower", "triangle", "square", "blob", "ghost", "circle", "drop", "star", "droid", "mech", "alien", "hexagon", "cat", "cloud", "pill", "pebble", "puddle"] as const
export type BotType = (typeof BOT_TYPES)[number]
/** Which avatar: a shape for each status ("auto"), a shape drawn at random for each place ("random", it stays until the page is
 *  opened again), or one chosen shape for all of them. */
export type BotChoice = "auto" | "random" | BotType

/** One of the eighteen, by a number in [0, 1). */
export const randomBot = (r: number): BotType => BOT_TYPES[Math.min(BOT_TYPES.length - 1, Math.floor(r * BOT_TYPES.length))]
export type BotState = "default" | "working" | "sleeping"

const saved = store.get<string>("avatar", "")
let kind: AvatarKind = saved === "bots" || saved === "loading" ? saved : "orbs"
const listeners = new Set<() => void>()
export const getAvatar = () => kind
export function setAvatar(k: AvatarKind) {
  kind = k
  store.set("avatar", k)
  listeners.forEach((f) => f())
}
/** The next choice in the round: orbs, a plain loading mark, bots. */
export const nextAvatar = (k: AvatarKind): AvatarKind => (k === "orbs" ? "loading" : k === "loading" ? "bots" : "orbs")

let choice: BotChoice = (() => { const v = store.get<string>("avatar.type", ""); return v === "random" || (BOT_TYPES as readonly string[]).includes(v) ? (v as BotChoice) : "auto" })()
const choiceListeners = new Set<() => void>()
export const getBotChoice = () => choice
export function setBotChoice(c: BotChoice) {
  choice = c
  store.set("avatar.type", c)
  choiceListeners.forEach((f) => f())
}
export function useBotChoice(): BotChoice {
  return useSyncExternalStore((cb) => { choiceListeners.add(cb); return () => { choiceListeners.delete(cb) } }, getBotChoice)
}

export function useAvatar(): AvatarKind {
  return useSyncExternalStore((cb) => { listeners.add(cb); return () => { listeners.delete(cb) } }, getAvatar)
}

const TYPES: Record<OrbDesign, BotType> = {
  working: "mech", searching: "droid", solving: "cat", listening: "cloud", connecting: "flower",
  weaving: "star", composing: "pill", breathing: "circle", shaping: "triangle",
}

/** The bot for an orb design: a shape for each of the nine forms (or the one the person chose, for all of them); at work it hops and laughs, at rest (an idle server, waiting) it looks
 *  around, dormant (an unloaded model) or paused it sleeps. */
export function botFor(design: OrbDesign, look: { moving: boolean; rest?: boolean }, pick: BotChoice = "auto", drawn?: BotType): { type: BotType; state: BotState } {
  const type = pick === "auto" ? TYPES[design] : pick === "random" ? drawn ?? TYPES[design] : pick
  if (!look.moving) return { type, state: "sleeping" }
  if (look.rest) return { type, state: design === "shaping" ? "sleeping" : "default" }
  return { type, state: design === "breathing" ? "default" : "working" }
}

/** How the plain loading mark behaves for a design: a ring that turns while something is at work, one that breathes while it waits or
 *  an idle server rests, and a still one when paused or dormant. (It does not tell the forms apart: the words beside it do.) */
export function plainLook(design: OrbDesign, look: { moving: boolean; rest?: boolean }): "spin" | "pulse" | "still" {
  if (!look.moving) return "still"
  if (look.rest) return design === "shaping" ? "still" : "pulse"
  return design === "breathing" ? "pulse" : "spin"
}
