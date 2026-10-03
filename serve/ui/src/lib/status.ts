import { afterToolStep, toolDesign, type OrbDesign } from "./orbs"
import { t } from "./i18n"

// What the agent is doing at this moment, in words, as one status that never goes quiet (issue #99 follow-up).
//
// The principle: other harnesses leave the last tool's card showing "done" while the model reads its result, or say only "reading the prompt", so that the user sees a series of separate
// notices with gaps between them and the work feels slow. Here there is always one line for what is going on now, told from what the page knows (the order of the events of the answer and
// what the server is doing): the call being written, the tool or the helper running, a hook of the user's running, auto mode checking, the question waiting for the user, the result being read,
// the next step being planned, the thinking, the writing. A change from one to the next is a change of words, not a gap.

/** What the status needs of a tool call. `at` and `rat` are the lengths of the text and of the thinking when the call began. */
export interface StatusTool { name: string; state: string; at?: number; rat?: number; judging?: boolean; hookRunning?: string | null }
export interface StatusInput {
  streaming: boolean
  reasoning?: string
  text: string
  tools?: StatusTool[]
  /** What the server is doing for this answer: "reading" the prompt, "generating", ... (its live status). */
  serverState?: string | null
  /** A hook of the user's that belongs to the answer and not to a call (a prompt or an end hook) is running. */
  hookRunning?: string | null
}
export type StatusKind = "asking" | "hook" | "judging" | "writing" | "running" | "reading" | "planning" | "thinking" | "composing"
export interface AgentStatus { kind: StatusKind; design: OrbDesign; label: string }

const status = (kind: StatusKind, design: OrbDesign, label: string): AgentStatus => ({ kind, design, label })

/** How far the server is in reading a prompt (the prompt of the first turn, or the one that now holds a tool's result): tokens read of tokens to read, as a share. While it reads it never says
 *  100 %: the read is over when the server starts to write. Null when the server does not say. */
export function readProgress(read: number | null | undefined, total: number | null | undefined): { read: number; total: number; percent: number } | null {
  if (typeof read !== "number" || typeof total !== "number" || !(total > 0) || read < 0) return null
  const r = Math.min(read, total)
  return { read: r, total, percent: Math.min(99, Math.floor((r / total) * 100)) }
}

/** The status of an answer that is being written, or null when it is not (or nothing has happened yet: the page's own waiting line covers that). */
export function agentStatus(i: StatusInput): AgentStatus | null {
  if (!i.streaming) return null
  const tools = i.tools ?? []
  const waiting = tools.find((c) => c.state === "asking")
  if (waiting) return status("asking", "listening", t("Waiting for you…"))
  const hooked = tools.find((c) => c.hookRunning && (c.state === "running" || c.state === "writing" || c.state === "done" || c.state === "error"))
  if (hooked || i.hookRunning) return status("hook", "working", t("Running your hook…"))
  const judged = tools.find((c) => c.judging)
  if (judged) return status("judging", "solving", t("Auto mode is checking the call…"))
  const doing = [...tools].reverse().find((c) => c.state === "running" || c.state === "writing")
  if (doing) {
    const design = toolDesign(doing.name)
    if (doing.state === "writing") return status("writing", design, t("Writing the call to {tool}…", { tool: doing.name }))
    return status("running", design, doing.name === "Task" ? t("A helper is working…") : t("Running {tool}…", { tool: doing.name }))
  }
  const text = i.text ?? "", reasoning = i.reasoning ?? ""
  const last = tools[tools.length - 1]
  if (last) {                                                                       // a tool has answered: what has the model done since?
    if (text.slice(last.at ?? text.length).trim()) return status("composing", "composing", t("Answering…"))
    if (reasoning.slice(last.rat ?? reasoning.length).trim()) return status("thinking", "solving", t("Thinking…"))
    return afterToolStep(i.serverState) === "planning"
      ? status("planning", "weaving", t("Planning the next step…"))
      : status("reading", "listening", t("Reading the tool's result…"))
  }
  if (text.trim()) return status("composing", "composing", t("Answering…"))
  if (reasoning.trim()) return status("thinking", "solving", t("Thinking…"))
  return null
}
