// Which orb design says what a thing is doing. The nine designs of thinking-orbs (Libraries.dev, MIT) each have a meaning;
// this maps the app's own states onto them, so the form of an orb tells the state before a word is read.
//   working: general work · searching: a scan or a lookup, and the idle look of the server and a GPU · solving: reasoning, or struggling · listening: taking input
//   connecting: a call, a queue, reaching out · weaving: a multi-step plan, heavy parallel work · composing: writing
//   breathing: at rest, waiting · shaping: dormant, a different mode
export type OrbDesign = "working" | "searching" | "solving" | "listening" | "connecting" | "weaving" | "composing" | "breathing" | "shaping"

export interface OrbLook { design: OrbDesign; moving: boolean; speed?: number; fps?: number }

/** Idle is not a still picture: the searching globe keeps turning, slowly, so the orb is seen and reads as alive and calm. */
const IDLE_SPEED = 0.5
const GPU_IDLE_FPS = 30                              // the small orbs that idle (a GPU card, a dormant model, an empty list) draw at half the display rate, to spare the page; the server's idle orb is never capped
export const SLOW = { speed: IDLE_SPEED, fps: GPU_IDLE_FPS }

/** The server. An answer being written flows, at the pace of its tokens; a prompt being read is taken in (listening); a request
 *  waiting its turn or a server that does not answer is reaching out; an unloaded model is dormant; idle is the searching
 *  globe, slowly. */
export function serverDesign(
  live: { state: "reading" | "generating" | "idle" | "unloaded"; queued?: number; tok_s?: number | null },
  stale = false,
): OrbLook & { speed: number } {
  if (stale) return { design: "connecting", moving: true, speed: 1 }
  if (live.state === "generating") {
    const t = live.tok_s
    return { design: "composing", moving: true, speed: t == null ? 1 : Math.max(0.7, Math.min(1.5, 0.7 + (t / 150) * 0.4)) }
  }
  if (live.state === "reading") return { design: "listening", moving: true, speed: 1 }
  if (live.state === "unloaded") return { design: "shaping", moving: true, speed: IDLE_SPEED, fps: GPU_IDLE_FPS }
  if ((live.queued ?? 0) > 0) return { design: "connecting", moving: true, speed: 1 }
  return { design: "searching", moving: true, speed: IDLE_SPEED }
}

/** A GPU: idle (the searching globe, slowly), at work, working hard (parallel strands), or held back by a limit (struggling). */
export function gpuDesign(g: { util: number | null | undefined; throttle?: string[] | null }): OrbLook {
  if (g.throttle && g.throttle.length) return { design: "solving", moving: true }
  const u = g.util ?? 0
  if (u >= 60) return { design: "weaving", moving: true }
  if (u >= 5) return { design: "working", moving: true }
  return { design: "searching", moving: true, speed: IDLE_SPEED, fps: GPU_IDLE_FPS }
}

/** A tool call is a connection; one that looks things up is a search. */
export function toolDesign(name: string): OrbDesign {
  return /search|find|query|lookup|fetch|browse|grep/i.test(name || "") ? "searching" : "connecting"
}

/** What an assistant message is doing right now, as an orb; null once it is done. Waiting for the model, thinking, a tool
 *  running, planning the next step after a tool, writing. */
export function replyDesign(m: {
  streaming: boolean; reasoning?: string; text: string; tools?: { name: string; state: string }[]
}): OrbDesign | null {
  if (!m.streaming) return null
  const tools = m.tools || []
  const running = tools.filter((t) => t.state === "running" || t.state === "writing")
  if (running.length) return toolDesign(running[running.length - 1].name)
  if (m.text) return "composing"
  if (tools.some((t) => t.state === "done" || t.state === "error")) return "weaving"      // a tool has answered: the agent plans the next step
  return m.reasoning ? "solving" : "breathing"
}

export type LatticePattern = "orbit" | "ripple" | "snake" | "spiral"

/** The lattice that sits beside a thought (components/thought.tsx) runs the pattern of what the agent is doing: thinking
 *  circles, a lookup ripples outward, a call to a tool snakes through, planning the next step spirals in. */
export function latticePattern(phase: OrbDesign | null): LatticePattern {
  if (phase === "searching") return "ripple"
  if (phase === "connecting") return "snake"
  if (phase === "weaving") return "spiral"
  return "orbit"
}

export const ORB_DESIGNS: readonly OrbDesign[] = ["working", "searching", "solving", "listening", "connecting", "weaving", "composing", "breathing", "shaping"]

/** The words for a form, for the orb's accessible name when it is shown for its own sake (the empty chat). */
export const orbLabel = (d: OrbDesign) => d[0].toUpperCase() + d.slice(1)

/** One of the other forms (any of the nine when none is shown yet); `r` in [0, 1) picks it. */
export function nextDesign(prev: OrbDesign | null, r: number): OrbDesign {
  const rest = ORB_DESIGNS.filter((d) => d !== prev)
  return rest[Math.min(rest.length - 1, Math.floor(r * rest.length))]
}
