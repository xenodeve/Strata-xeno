// Which orb design says what the app is doing: the "map your agent's state to a design" recipe of haplollc/ThinkingOrbs
// (a SwiftUI port of Jakub Antalik's thinking-orbs, MIT), applied to this app's own states.
export type OrbDesign = "working" | "searching" | "solving" | "listening" | "connecting" | "weaving" | "composing" | "breathing" | "shaping"

/** The server's state: a prompt being read breathes, an answer being written flows; idle or unloaded rests (a still ring). */
export function serverDesign(state: "reading" | "generating" | "idle" | "unloaded"): { design: OrbDesign; moving: boolean } {
  if (state === "generating") return { design: "composing", moving: true }
  if (state === "reading") return { design: "breathing", moving: true }
  return { design: "breathing", moving: false }
}

/** A tool call is a connection; one that looks things up is a search. */
export function toolDesign(name: string): OrbDesign {
  return /search|find|query|lookup|fetch|browse|grep/i.test(name || "") ? "searching" : "connecting"
}

/** What an assistant message is doing right now, as an orb; null once it is done. Waiting for the model, thinking, a tool
 *  running, writing: the five phases of the recipe (idle is the resting ring of serverDesign). */
export function replyDesign(m: {
  streaming: boolean; reasoning?: string; text: string; tools?: { name: string; state: string }[]
}): OrbDesign | null {
  if (!m.streaming) return null
  const running = (m.tools || []).filter((t) => t.state === "running" || t.state === "writing")
  if (running.length) return toolDesign(running[running.length - 1].name)
  if (m.text) return "composing"
  return m.reasoning ? "solving" : "breathing"
}
