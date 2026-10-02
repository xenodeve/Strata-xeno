// The Git state of a project's folder, as the server reads it (serve/gitview.py, read only) and as the Chat's right panel shows it (issue #99).
import { apiHeaders, url } from "./api"
import { msg } from "./i18n"

export interface GitFile { path: string; status: string; from?: string }
export interface GitBranch { name: string; current: boolean; upstream?: string; track?: string }
export interface GitWorktree { path: string; head?: string; branch?: string; detached?: boolean; bare?: boolean; current: boolean }
export interface GitCommit { sha: string; author: string; time: number; subject: string }
export interface GitInfo {
  ok: true; git: boolean; repo: boolean
  root?: string; folder?: string; error?: string
  branch?: string | null; detached?: boolean; oid?: string | null; upstream?: string | null; ahead?: number; behind?: number
  counts?: { staged: number; unstaged: number; untracked: number; conflicted: number }; truncated?: boolean
  staged?: GitFile[]; unstaged?: GitFile[]; untracked?: GitFile[]; conflicted?: GitFile[]
  branches?: GitBranch[]; moreBranches?: boolean; worktrees?: GitWorktree[]; commits?: GitCommit[]
}
export type GitResult = GitInfo | { ok: false; error: string } | "unavailable"      // unavailable: the server cannot be asked (another PC, an older server)

/** The state of `folder` (the repository it is in, if any). */
export async function getGit(folder: string): Promise<GitResult> {
  try {
    const r = await fetch(url("agent/git?path=" + encodeURIComponent(folder)), { headers: apiHeaders() })
    if (!r.ok) return "unavailable"
    return (await r.json()) as GitResult
  } catch { return "unavailable" }
}

export interface GitDiff { ok: true; diff: string; binary: boolean; truncated: boolean }
export async function getGitDiff(folder: string, file: string, staged: boolean, untracked: boolean): Promise<GitDiff | { ok: false; error: string } | "unavailable"> {
  try {
    const q = `path=${encodeURIComponent(folder)}&file=${encodeURIComponent(file)}${staged ? "&staged=1" : ""}${untracked ? "&untracked=1" : ""}`
    const r = await fetch(url("agent/git/diff?" + q), { headers: apiHeaders() })
    if (!r.ok) return "unavailable"
    return (await r.json()) as GitDiff
  } catch { return "unavailable" }
}

/** A file's change as a word: the letter git gives it (M, A, D, R, C, T, U) or `?` for a file that is not tracked. */
export const STATUS_WORD: Record<string, string> = { M: msg("modified"), A: msg("added"), D: msg("deleted"), R: msg("renamed"), C: msg("copied"), T: msg("type changed"), U: msg("conflict"), "?": msg("new") }

export interface DiffLine { kind: "hunk" | "add" | "del" | "ctx" | "note"; text: string; old?: number; now?: number }

/** A unified diff as lines, with the numbers of the lines in the old and the new file; the header lines before the first hunk are left out. */
export function parseDiff(text: string): DiffLine[] {
  const out: DiffLine[] = []
  let old = 0, now = 0, inHunk = false
  for (const raw of text.split("\n")) {
    const h = /^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@\s?(.*)$/.exec(raw)
    if (h) { old = +h[1]; now = +h[2]; inHunk = true; out.push({ kind: "hunk", text: h[3] ? `@@ ${h[3]}` : "@@" }); continue }
    if (!inHunk) continue
    if (raw.startsWith("\\")) out.push({ kind: "note", text: raw.slice(2) })
    else if (raw.startsWith("+")) out.push({ kind: "add", text: raw.slice(1), now: now++ })
    else if (raw.startsWith("-")) out.push({ kind: "del", text: raw.slice(1), old: old++ })
    else if (raw.startsWith(" ")) out.push({ kind: "ctx", text: raw.slice(1), old: old++, now: now++ })
  }
  return out
}

/** The last part of a path, for a tab or a row. */
export const baseName = (p: string): string => p.replace(/[\\/]+$/, "").split(/[\\/]/).pop() || p
