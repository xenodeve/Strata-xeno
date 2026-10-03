// What the Chat's right panel shows besides Git (issue #99): the plan (the model's to-do list and the plan it sent for approval), the skills used in the conversation, and
// where the panel is left open. Derived from the messages, so a stored conversation has them too.
import type { Message, Todo } from "./chat"
import { skillOfMessage } from "./slash"

export type PanelTab = "git" | "plan" | "skills" | "memory" | "context"
export const PANEL_TABS: PanelTab[] = ["git", "plan", "skills", "memory", "context"]
export interface PanelState { open: boolean; tab: PanelTab }
export const PANEL_DEFAULT: PanelState = { open: false, tab: "git" }

/** The state as it was kept: anything else is the default. */
export function panelState(raw: unknown): PanelState {
  const r = raw as Partial<PanelState> | null
  return { open: r?.open === true, tab: PANEL_TABS.includes(r?.tab as PanelTab) ? (r!.tab as PanelTab) : "git" }
}

// ------------------------------------------------------------------------------------------------ the plan
export interface Plan { todos: Todo[]; done: number; total: number; current: Todo | null; approved: string | null }

/** The to-do list as the model last sent it, how far it is, and the plan it sent for approval last (ExitPlanMode). */
export function planOf(messages: Message[]): Plan {
  let todos: Todo[] = []
  let approved: string | null = null
  for (const m of messages) {
    if (m.todos?.length) todos = m.todos
    for (const c of m.tools || []) {
      const plan = c.name === "ExitPlanMode" && c.arguments && typeof c.arguments === "object" ? (c.arguments as { plan?: unknown }).plan : undefined
      if (typeof plan === "string" && plan.trim()) approved = plan
    }
  }
  const done = todos.filter((t) => t.status === "completed").length
  return { todos, done, total: todos.length, current: todos.find((t) => t.status === "in_progress") ?? null, approved }
}

// ------------------------------------------------------------------------------------------------ skills
export interface SkillCall { kind: "use" | "file" | "find"; name?: string }

/** What a tool call of the skills server is: loading a skill, reading a file of one, or looking for one. Null for any other call. */
export function skillCall(call: { server?: string; tool?: string; arguments?: unknown }): SkillCall | null {
  if (call.server !== "skills") return null
  const name = call.arguments && typeof call.arguments === "object" && typeof (call.arguments as { name?: unknown }).name === "string" ? (call.arguments as { name: string }).name : undefined
  if (call.tool === "use_skill") return { kind: "use", name }
  if (call.tool === "read_skill_file") return { kind: "file", name }
  if (call.tool === "find_skills") return { kind: "find" }
  return null
}

export interface SkillUse { name: string; by: "you" | "model"; uses: number; files: number }

/** The skills used in the conversation, in the order they were first used: the ones the user asked for with `/name` (`names` are the skills in use) and the ones the model loaded. */
export function skillsUsed(messages: Message[], names: string[]): SkillUse[] {
  const seen = new Map<string, SkillUse>()
  const bump = (name: string, by: "you" | "model", file = false) => {
    const u = seen.get(name) ?? { name, by, uses: 0, files: 0 }
    if (file) u.files++; else u.uses++
    seen.set(name, u)
  }
  for (const m of messages) {
    if (m.role === "user" && !m.compact) { const n = skillOfMessage(m.text, names); if (n) bump(n, "you") }
    for (const c of m.tools || []) {
      if (c.state === "skipped" || c.state === "writing") continue
      const s = skillCall(c)
      if (s?.name && s.kind === "use") bump(s.name, "model")
      else if (s?.name && s.kind === "file") bump(s.name, "model", true)
    }
  }
  return [...seen.values()]
}

/** A skill call as the line the chat shows for it. */
export function skillLine(call: SkillCall): { key: "use" | "file" | "find"; name?: string } { return { key: call.kind, name: call.name } }
