import { msg, t } from "./i18n"

// The user's hooks (issue #99, serve/hooks.py): commands of their own that the server runs before and after a tool call, when a prompt is sent and when the model has finished. They are written
// in the server's run config, never here; this page lists them (Settings > Hooks) with a switch each, and shows in the chat what each one did.

export type HookEvent = "before_tool" | "after_tool" | "prompt" | "stop"

/** What one hook did, as the server's stream tells it (a `hook` event). */
export interface HookNote {
  hook: string; on: HookEvent; command: string; ok: boolean; code: number | null; blocked: boolean; timeout: boolean; error?: string | null; text: string; ms: number
}

/** The hooks as the server lists them (GET /agent/hooks). */
export interface HookItem { id: string; event: HookEvent; matcher: string | null; command: string; timeout: number; on: boolean }
export interface HooksView { hooks: HookItem[]; problems: string[]; config_file: string | null; shell: boolean; editable: boolean }

const EVENT_WORDS: Record<HookEvent, string> = { before_tool: msg("Before a tool call"), after_tool: msg("After a tool call"), prompt: msg("When you send a prompt"), stop: msg("When the model has finished") }
export const eventText = (e: HookEvent): string => t(EVENT_WORDS[e] ?? e)

/** The note from the stream, checked: anything that is not a hook's report is left out. */
export function noteFrom(x: Record<string, unknown>): HookNote | null {
  const on = x.on
  if (on !== "before_tool" && on !== "after_tool" && on !== "prompt" && on !== "stop") return null
  return {
    hook: typeof x.hook === "string" ? x.hook : "", on, command: typeof x.command === "string" ? x.command : "", ok: x.ok === true, code: typeof x.code === "number" ? x.code : null,
    blocked: x.blocked === true, timeout: x.timeout === true, error: typeof x.error === "string" ? x.error : null, text: typeof x.text === "string" ? x.text : "", ms: typeof x.ms === "number" ? x.ms : 0,
  }
}

/** What a hook did, in a few words. */
export function noteText(n: HookNote): string {
  if (n.blocked) return t("A hook stopped this call")
  if (n.timeout) return t("A hook ran out of time and was stopped")
  if (n.error) return t("A hook could not run")
  if (n.ok) return t("A hook ran")
  return t("A hook finished with exit code {n}", { n: n.code ?? "?" })
}
