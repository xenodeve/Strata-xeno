// The thinking levels a model accepts. The server lists them (what the model's chat template renders without an error:
// `engine.efforts` in /metrics, "none" first when thinking can be turned off); the chat offers those, in that order, the
// highest last. A level saved for another model is mapped onto one the model has.
import { msg } from "./i18n"

const NAMES: Record<string, string> = { none: msg("Off"), low: msg("Low"), medium: msg("Medium"), high: msg("High"), xhigh: msg("XHigh") }
const ALWAYS = ["none", "low", "medium", "high"]          // what an older server (one that does not say) is assumed to take

export interface EffortChoice { value: string; label: string }

/** The levels to offer; `label` is English text to show with t(). */
export function effortChoices(efforts: readonly string[] | undefined): EffortChoice[] {
  const list = efforts && efforts.length ? efforts : ALWAYS
  return list.map((v) => ({ value: v, label: NAMES[v] ?? v }))
}

/** `saved` as the model in use can take it: itself when offered; "high" and "xhigh" stand for each other; otherwise the
 *  model's default, or its highest level. With no levels known, the saved one is left alone. */
export function settleEffort(saved: string, offered: readonly string[], fallback: string | null | undefined): string {
  if (!offered.length || offered.includes(saved)) return saved
  const twin = saved === "high" ? "xhigh" : saved === "xhigh" ? "high" : null
  if (twin && offered.includes(twin)) return twin
  return fallback && offered.includes(fallback) ? fallback : offered[offered.length - 1]
}
