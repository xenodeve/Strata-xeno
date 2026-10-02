// How much of the model's context window a conversation uses (issue #99): the figure the last answer reported (its prompt after every tool round, and its own tokens), else a
// guess from the text; what it is made of; and where it will be compacted by itself. Pure numbers and text; the page draws them (the meter on the prompt bar, `/context`).
import type { Message } from "./chat"
import { COMPACT_AT, estimateTokens } from "./compact"

export type PartKey = "conversation" | "tools" | "summary" | "system" | "free"
export interface Part { key: PartKey; tokens: number }
export interface ContextView {
  known: boolean              // the server said how big the window is
  max: number
  used: number
  exact: boolean              // `used` is what the server reported, not a guess
  pct: number                 // 0..100, rounded; more than 100 is shown as 100
  autoAt: number | null       // the tokens at which it is compacted by itself, or null when that is switched off
  level: "ok" | "warn" | "full"       // under 60 %, under the compacting point, at it or over
  parts: Part[]
}

const textOf = (m: Message) => [m.text, ...(m.files || []).map((f) => f.text ?? "")].join("\n")

/** What the messages are made of, by a guess from their text: what was said, what the tools did (calls and results), and the summary that stands for compacted messages. */
export function weigh(messages: Message[]): { conversation: number; tools: number; summary: number } {
  let conversation = 0, tools = 0, summary = 0
  for (const m of messages) {
    if (m.compact) { summary += estimateTokens(m.text); continue }
    if (m.error) continue
    conversation += estimateTokens(textOf(m))
    for (const c of m.tools || []) if (c.result != null && c.state !== "skipped") tools += estimateTokens(c.result) + estimateTokens(c.arguments ?? {})
  }
  return { conversation, tools, summary }
}

/** The view of the context. `reported` is what the last answer said the conversation uses (null: none yet, so a guess is used); `max` the window (0: not known). */
export function contextView(messages: Message[], reported: number | null, max: number, autoCompact: boolean): ContextView {
  const w = weigh(messages)
  const guess = w.conversation + w.tools + w.summary
  const exact = reported != null && reported > 0
  const used = exact ? reported! : guess
  const system = exact ? Math.max(0, used - guess) : 0                        // what the server adds: its instructions, the tools, the memory
  const known = max > 0
  const pct = known ? Math.min(100, Math.round((used / max) * 100)) : 0
  const autoAt = known && autoCompact ? Math.floor(max * COMPACT_AT) : null
  const level = !known ? "ok" : autoAt !== null && used >= autoAt ? "full" : pct >= 60 ? "warn" : "ok"
  const parts: Part[] = [
    { key: "conversation", tokens: w.conversation }, { key: "tools", tokens: w.tools }, { key: "summary", tokens: w.summary },
    { key: "system", tokens: system }, { key: "free", tokens: known ? Math.max(0, max - used) : 0 },
  ]
  return { known, max, used, exact, pct, autoAt, level, parts }
}

/** `/context`: the command typed alone. */
export const isContextCommand = (text: string): boolean => /^\s*\/context\s*$/i.test(text)
