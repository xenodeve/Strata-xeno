// The memory and instruction files the chat reads (issue #99, serve/memory.py), as the page holds them: the sources for a project, the text of one, and the switches for what other apps
// wrote down (off until switched on). `/init` is the command that has the model write the project's instruction file.
import { apiHeaders, url } from "./api"
import { postImport } from "./api"

export interface MemorySource {
  id: string                      // "project:0", "claude:instructions", "claude:memory"
  app: string; label: string
  kind: "project" | "instructions" | "memory"
  shown: string[]                 // the files, as the user knows them (relative to the project folder, or ~/...)
  bytes: number | null
  on: boolean
}
export interface MemoryList { max: number; sources: MemorySource[] }

const query = (folders: string[]) => `path=${encodeURIComponent(folders[0] ?? "")}${folders.slice(1).map((f) => "&dirs=" + encodeURIComponent(f)).join("")}`

/** The sources there are for a project's folders (the first is the main one); null when the server cannot be asked. */
export async function getMemory(folders: string[]): Promise<MemoryList | null> {
  try {
    const r = await fetch(url("agent/memory?" + query(folders)), { headers: apiHeaders() })
    return r.ok ? ((await r.json()) as MemoryList) : null
  } catch { return null }
}

/** The text of one file of a source (`n` is its place in `shown`). */
export async function getMemoryFile(folders: string[], id: string, n: number): Promise<{ name: string; text: string; cut: boolean } | null> {
  try {
    const r = await fetch(url(`agent/memory/file?${query(folders)}&id=${encodeURIComponent(id)}&n=${n}`), { headers: apiHeaders() })
    if (!r.ok) return null
    const j = await r.json()
    return j?.ok === true ? { name: String(j.name), text: String(j.text), cut: !!j.cut } : null
  } catch { return null }
}

/** The sources that are switched on, after `id` is switched. Only another app's sources can be switched; a project's own are always on. */
export function withSwitch(on: string[], id: string, value: boolean): string[] {
  const rest = on.filter((x) => x !== id)
  return value ? [...rest, id] : rest
}

/** Saves which of the other apps' sources are on (from this PC; the server checks). The answer is why not, or nothing. */
export async function saveMemorySwitches(on: string[]): Promise<string | null> {
  const r = await postImport({ memory: { on } })
  return "error" in r ? r.error : null
}

/** A size as a person reads it. */
export const sizeText = (bytes: number | null): string => (bytes == null ? "" : bytes < 1024 ? `${bytes} B` : `${(bytes / 1024).toFixed(bytes < 10240 ? 1 : 0)} KB`)

/** What `/init` sends: the model looks at the project and writes its instruction file, as Claude Code's own does. It asks before it writes, like any write. */
export const INIT_PROMPT = [
  "Please look through this project and write an instruction file for it, `CLAUDE.md`, in the project folder. It will be given to you at the start of every future chat in this project, so it should hold what you would need to work in the code without asking.",
  "",
  "Include: the commands to build, run, test and lint (and how to run one test); the architecture in a few lines (the big pieces and how they fit, only what needs several files to see); the conventions that are not obvious (naming, error handling, formatting) and anything the project says in its README or in other instruction files (for example AGENTS.md or Cursor rules) that matters.",
  "",
  "Do not list every file, do not repeat what is obvious from reading the code, do not invent sections that have nothing to say, and do not put in generic advice. If `CLAUDE.md` or `AGENTS.md` already exists, read it and improve it instead of starting again. Keep it short: a page or two.",
  "",
  "Look at the project first (its README, its build and test files, a few of the main source files), then write the file.",
].join("\n")

/** `/init` and `/memory` typed alone. */
export const isInitCommand = (text: string): boolean => /^\s*\/init\s*$/i.test(text)
export const isMemoryCommand = (text: string): boolean => /^\s*\/memory\s*$/i.test(text)
