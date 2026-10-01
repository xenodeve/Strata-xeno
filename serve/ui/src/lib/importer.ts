// The skills and the MCP servers of the other coding apps (issue #94), as the page holds them: the types of GET /import, the skill switches
// (kept as off-lists, so a skill found later is on) and the filter over a long list. The server decides and checks everything.

export interface SkillSettings { enabled: boolean; harness_off: string[]; off: Record<string, string[]> }
export interface SkillItem {
  id: string; harness: string; name: string; description: string; origin: string
  off: boolean                   // its own switch is off
  used: boolean                  // it is the copy that is in use
  same_as: string | null         // the id of the copy that is in use, when this is another app's copy of it
}
export interface HarnessInfo {
  id: string; label: string; found: boolean; files: string[]; errors: string[]
  skills: number; skills_used: number; skills_off: boolean; servers: number
}
export interface Candidate {
  id: string; harness: string; name: string; kind: "program" | "address" | null
  state: "available" | "already" | "unsupported"; reason: string | null; note: string | null
  disabled_in_source: boolean; already_as: string | null
  command?: string; args?: string[]; cwd?: string; url?: string; env?: Record<string, string>; headers?: Record<string, string>      // only for a caller who may change the settings
}
export interface ImportView {
  available: boolean
  editable?: boolean; reason?: string; config_file?: string | null
  harnesses?: HarnessInfo[]
  skills?: { settings: SkillSettings; used: number; total: number; items: SkillItem[] }
  mcp?: { harnesses: { id: string; label: string; servers: Candidate[] }[] }
}

export const harnessOn = (s: SkillSettings, harness: string) => !s.harness_off.includes(harness)
export const itemOn = (s: SkillSettings, harness: string, name: string) => !(s.off[harness] ?? []).includes(name)

export const setMaster = (s: SkillSettings, on: boolean): SkillSettings => ({ ...s, enabled: on })

export function setHarness(s: SkillSettings, harness: string, on: boolean): SkillSettings {
  const rest = s.harness_off.filter((h) => h !== harness)
  return { ...s, harness_off: on ? rest : [...rest, harness] }
}

export function setItem(s: SkillSettings, harness: string, name: string, on: boolean): SkillSettings {
  if (on === itemOn(s, harness, name)) return s                   // already so: nothing moves
  const names = (s.off[harness] ?? []).filter((n) => n !== name)
  const off = { ...s.off }
  const next = on ? names : [...names, name]
  if (next.length) off[harness] = next
  else delete off[harness]
  return { ...s, off }
}

/** The items whose name, description or app name has every word of `query` (any case); all of them for an empty query. */
export function filterItems(items: SkillItem[], query: string, labels: Record<string, string> = {}): SkillItem[] {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean)
  if (!words.length) return items
  return items.filter((i) => { const hay = `${i.name} ${i.description} ${labels[i.harness] ?? i.harness}`.toLowerCase(); return words.every((w) => hay.includes(w)) })
}

export function skillSummary(items: SkillItem[]): { used: number; total: number; apps: number } {
  return { used: items.filter((i) => i.used).length, total: items.length, apps: new Set(items.map((i) => i.harness)).size }
}
