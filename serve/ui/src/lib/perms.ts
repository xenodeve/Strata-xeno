// The rules the user set for the chat's coding tools (issue #99), in Claude Code's syntax (`Tool(specifier)`, e.g. `Bash(npm test:*)`, `Read(src/**)`): which ones to allow, and which ones
// to refuse, for a project or for everything. "Allow for this chat" (lib/agent.ts) stays with the chat; these last. They are kept in this browser and sent with every request; the server
// decides: a deny rule beats an allow rule, and no rule settles a secret, a change in .git or a dangerous command - those ask every time.
import type { Backing } from "./sessions"

export interface RuleSet { allow: string[]; deny: string[] }
export interface Perms { everywhere: RuleSet; projects: Record<string, RuleSet> }
export type Scope = { kind: "everywhere" } | { kind: "project"; id: string }
export type Effect = "allow" | "deny"

const KEY = "agent.perms"
export const MAX_RULES = 200                      // of one kind in one place
const MAX_PROJECTS = 100
const MAX_LEN = 500

/** Whether text is a rule: a tool's name, and between brackets, what it is limited to. */
export const validRule = (text: string): boolean => text.length > 0 && text.length <= MAX_LEN && /^[A-Za-z_][A-Za-z0-9_]*(\([\s\S]+\))?$/.test(text)

const strings = (x: unknown): string[] => {
  const out: string[] = []
  for (const r of Array.isArray(x) ? x : []) if (typeof r === "string" && validRule(r) && !out.includes(r)) out.push(r)
  return out.slice(0, MAX_RULES)
}
const set = (x: unknown): RuleSet => { const o = x && typeof x === "object" ? (x as Record<string, unknown>) : {}; return { allow: strings(o.allow), deny: strings(o.deny) } }
const empty = (): RuleSet => ({ allow: [], deny: [] })

export function loadPerms(b: Backing): Perms {
  const raw = b.get<unknown>(KEY, null)
  const o = raw && typeof raw === "object" ? (raw as Record<string, unknown>) : {}
  const projects: Record<string, RuleSet> = {}
  if (o.projects && typeof o.projects === "object") {
    for (const [id, v] of Object.entries(o.projects as Record<string, unknown>).slice(0, MAX_PROJECTS)) {
      const s = set(v)
      if (s.allow.length || s.deny.length) projects[id] = s
    }
  }
  return { everywhere: set(o.everywhere), projects }
}

const where = (p: Perms, scope: Scope): RuleSet => (scope.kind === "everywhere" ? p.everywhere : (p.projects[scope.id] ??= empty()))

export const ruleSetOf = (p: Perms, scope: Scope): RuleSet => (scope.kind === "everywhere" ? p.everywhere : p.projects[scope.id] ?? empty())

/** Adds a rule (newest last) to the allow or the deny list of a scope. False when it is not a rule, is there already, or the list is full. A rule is never in both lists of a scope:
 *  the other list loses it. */
export function addPerm(b: Backing, scope: Scope, effect: Effect, rule: string): boolean {
  const text = rule.trim()
  if (!validRule(text)) return false
  const p = loadPerms(b)
  const s = where(p, scope)
  if (s[effect].includes(text) || s[effect].length >= MAX_RULES) return false
  s[effect].push(text)
  const other: Effect = effect === "allow" ? "deny" : "allow"
  s[other] = s[other].filter((r) => r !== text)
  b.set(KEY, p)
  return true
}

export function removePerm(b: Backing, scope: Scope, effect: Effect, rule: string): void {
  const p = loadPerms(b)
  const s = where(p, scope)
  s[effect] = s[effect].filter((r) => r !== rule)
  if (scope.kind === "project" && !s.allow.length && !s.deny.length) delete p.projects[scope.id]
  b.set(KEY, p)
}

/** A project that is gone takes its rules with it. */
export function forgetPerms(b: Backing, projectId: string): void {
  const p = loadPerms(b)
  if (!(projectId in p.projects)) return
  delete p.projects[projectId]
  b.set(KEY, p)
}

/** What a request carries: the rules allowed for this chat, for its project and everywhere, and the ones refused for its project and everywhere. */
export function permsFor(b: Backing, chatAllow: string[], projectId: string | undefined): RuleSet {
  const p = loadPerms(b)
  const mine = projectId ? p.projects[projectId] ?? empty() : empty()
  const uniq = (...lists: string[][]) => [...new Set(lists.flat())]
  return { allow: uniq(chatAllow, mine.allow, p.everywhere.allow), deny: uniq(mine.deny, p.everywhere.deny) }
}

/** How many rules there are in all, for a line that says so. */
export const countRules = (p: Perms): number => p.everywhere.allow.length + p.everywhere.deny.length + Object.values(p.projects).reduce((n, s) => n + s.allow.length + s.deny.length, 0)
