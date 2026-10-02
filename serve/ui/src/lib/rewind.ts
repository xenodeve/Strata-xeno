// Rewind (issue #99): every prompt is a checkpoint. The server keeps the files the chat's tools change (serve/checkpoints.py); this is the page's side: what a rewind to a prompt would do,
// doing it, and forgetting a chat's checkpoints. What the shell tool does to files is not tracked, so it is not undone.
import { apiHeaders, errorMessage, url } from "./api"

export interface RewindFile { path: string; action: "restore" | "delete" | "skip"; changed: boolean; present?: boolean; why?: string }
export interface RewindPreview { ok: true; known: boolean; files: RewindFile[] }
export interface RewindResult { ok: true; restored: number; deleted: number; kept: string[]; failed: string[]; skipped: string[] }

const post = async <T,>(path: string, body: unknown): Promise<T | { error: string }> => {
  try {
    const r = await fetch(url(path), { method: "POST", headers: apiHeaders(true), body: JSON.stringify(body) })
    return r.ok ? ((await r.json()) as T) : { error: await errorMessage(r) }
  } catch (e) { return { error: e instanceof Error ? e.message : String(e) } }
}

/** What putting the files back to before this prompt would do. */
export const previewRewind = (session: string, checkpoint: string) => post<RewindPreview>("agent/rewind", { session, checkpoint })

/** Puts them back. A file that was changed since by someone else is left, unless `includeChanged`. */
export const applyRewind = (session: string, checkpoint: string, includeChanged: boolean) => post<RewindResult>("agent/rewind", { session, checkpoint, apply: true, include_changed: includeChanged })

/** A chat is gone: its checkpoints go too (not waited for, not reported). */
export function forgetCheckpoints(session: string): void { void post("agent/checkpoints/forget", { session }) }

export type RewindWhat = "both" | "files" | "conversation"

/** The line the page says after files were put back. */
export function rewindSummary(r: RewindResult): { restored: number; deleted: number; kept: number; failed: number } {
  return { restored: r.restored, deleted: r.deleted, kept: r.kept.length, failed: r.failed.length }
}

/** How a path is shown in the list: the last two parts of it. */
export const shortPath = (p: string): string => p.replace(/\\/g, "/").split("/").slice(-2).join("/")

/** `/rewind` typed alone. */
export const isRewindCommand = (text: string): boolean => /^\s*\/rewind\s*$/i.test(text)
