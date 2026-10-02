// "/" at the start of a message opens the skills in use, to be picked by name (like Claude Code's slash commands). A message that starts
// with the name of a skill in use asks the server to load that skill (`strata_skill`); the server checks the name again.
import type { SkillItem } from "./importer"
import { t } from "./i18n"

export interface Command {
  name: string; description: string
  from: string                   // the app it comes from (Claude Code, Codex CLI, ...)
  plugin: string | null          // the plugin of that app, for a skill a plugin brings
  builtin?: boolean              // a command of Strata itself (/compact), not a skill
}

/** The commands Strata has of its own, listed with the skills when "/" is typed. A skill of the same name does not hide them. */
export const builtinCommands = (): Command[] => [
  { name: "compact", description: t("Summarise the conversation so far to free up context. Add after it what the summary should focus on."), from: "Strata", plugin: null, builtin: true },
  { name: "context", description: t("Show how much of the context window the conversation uses, and what it is made of."), from: "Strata", plugin: null, builtin: true },
  { name: "memory", description: t("Show the notes and instruction files the chat reads for this project."), from: "Strata", plugin: null, builtin: true },
  { name: "init", description: t("Have the model look at the project and write a CLAUDE.md for it."), from: "Strata", plugin: null, builtin: true },
  { name: "permissions", description: t("Open the rules of the coding tools: what is allowed, and what is never allowed, for good."), from: "Strata", plugin: null, builtin: true },
  { name: "clear", description: t("Start a new chat. This one stays in Recents."), from: "Strata", plugin: null, builtin: true },
]

/** The query when the caret is in a slash command at the very start of the message (`/pd|`), else null: a path, a URL, a `/` further in
 *  or a space after the name (the arguments are being typed) is not one. Only what is before the caret counts. */
export function slashQuery(text: string, caret: number): string | null {
  const m = /^\/([^\s/]*)$/.exec(text.slice(0, caret))
  return m ? m[1] : null
}

/** The skills that can be picked: the copy in use of each name, by name. `labels` names the apps by their id. */
export function commandsOf(items: SkillItem[], labels: Record<string, string> = {}): Command[] {
  const seen = new Map<string, Command>()
  for (const i of items) {
    if (!i.used || i.off || seen.has(i.name)) continue
    seen.set(i.name, { name: i.name, description: i.description, from: labels[i.harness] ?? i.harness, plugin: i.origin.startsWith("plugin:") ? i.origin.slice(7) : null })
  }
  return [...seen.values()].sort((a, b) => a.name.localeCompare(b.name))
}

/** The commands for what is typed after the slash: the exact name, then names that start with it, names that contain it, then a
 *  description that does; each group by name. Nothing typed lists them all. */
export function matchCommands(all: Command[], query: string): Command[] {
  const q = query.trim().toLowerCase()
  const rank = (c: Command) => {
    if (!q) return 0
    const n = c.name.toLowerCase()
    return n === q ? 0 : n.startsWith(q) ? 1 : n.includes(q) ? 2 : c.description.toLowerCase().includes(q) ? 3 : -1
  }
  return all.map((c) => ({ c, r: rank(c) })).filter((x) => x.r >= 0)
    .sort((a, b) => a.r - b.r || a.c.name.localeCompare(b.c.name)).map((x) => x.c)
}

/** The message after one is picked: its first word becomes `/name`, what follows is kept after one space; the caret goes after that space. */
export function pickCommand(text: string, name: string): { text: string; caret: number } {
  const rest = text.replace(/^\S*/, "").replace(/^\s+/, "")
  const head = `/${name} `
  return { text: head + rest, caret: head.length }
}

/** The skill a message asks for: its first word is `/` and the name of a skill in use. */
export function skillOfMessage(text: string, names: string[]): string | null {
  const m = /^\s*\/([^\s/]+)(?=\s|$)/.exec(text)
  return m && names.includes(m[1]) ? m[1] : null
}

/** Where the slash command of a message is (`at` is its `/`, `end` is after its last letter), and which skill it is: for showing it in
 *  bold with what it is. Null when the message does not start with `/name` of a skill in use. */
export function markOf(text: string, commands: Command[]): { at: number; end: number; cmd: Command } | null {
  const m = /^(\s*)\/([^\s/]+)(?=\s|$)/.exec(text)
  const cmd = m ? commands.find((c) => c.name === m[2]) : undefined
  return m && cmd ? { at: m[1].length, end: m[1].length + 1 + m[2].length, cmd } : null
}
