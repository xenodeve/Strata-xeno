import { useImperativeHandle, useRef, useState, type Ref } from "react"
import { HugeiconsIcon } from "@hugeicons/react"
import { ArrowUp01Icon, Cancel01Icon, Folder01Icon } from "@hugeicons/core-free-icons"
import { getFolders, type FolderView } from "../lib/api"
import { cn } from "../lib/cn"
import { FOLDERS_MAX } from "../lib/sessions"
import { t } from "../lib/i18n"
import { Button, inputCls } from "./ui"

// The folders of a project (issue #96): the first is the main one, where the coding tools run commands; the others are its other folders (other git worktrees of the same
// repository, a library next to the app) where files are as free as in the main one, as Claude Code's added directories. A folder is typed or chosen from the folders of this
// PC (the server lists them) and is checked with the server before it is added; the path that is kept is the one the server resolved. When the server cannot be asked
// (another PC) the path is used as typed.

export type Listing = FolderView | "unknown" | null

/** The path a typed folder leads to, or why it is not one. */
export async function checkFolder(path: string): Promise<{ path: string } | { error: string }> {
  const r = await getFolders(path.trim())
  if (r === "unknown") return { path: path.trim() }
  if ("error" in r) return { error: t("That is not a folder on this PC.") }
  return { path: r.path }
}

/** A folder's path with the end kept in view: the name is what tells worktrees of one repository apart, so it is the part that is never cut (the folder above it is). */
function FolderName({ path }: { path: string }) {
  const m = /^(.*[\\/])([^\\/]+)[\\/]*$/.exec(path)
  return (
    <span className="flex min-w-0 flex-1 font-mono text-[12px]" title={path}>
      {m && <span className="min-w-0 truncate text-ink-3">{m[1]}</span>}
      <span className="shrink-0 font-medium">{m ? m[2] : path}</span>
    </span>
  )
}

export interface FoldersApi {
  /** Adds what is typed in the field (when something is) and gives the folders; null when what is typed is not a folder (the field says so). */
  commit: () => Promise<string[] | null>
  focus: () => void
}

export function FoldersEditor({ folders, onChange, min = 0, onDraft, onSubmit, api }: {
  folders: string[]
  onChange: (next: string[]) => void
  min?: number                                  // the fewest folders: the last one(s) cannot be taken away
  onDraft?: (text: string) => void              // what is typed in the field, as it is typed
  onSubmit?: () => void                         // Enter in the field with nothing typed
  api?: Ref<FoldersApi>
}) {
  const [draft, setDraftState] = useState("")
  const [browsing, setBrowsing] = useState(false)
  const [listing, setListing] = useState<Listing>(null)
  const [problem, setProblem] = useState<string | null>(null)
  const [checking, setChecking] = useState(false)
  const field = useRef<HTMLInputElement>(null)
  const setDraft = (v: string) => { setDraftState(v); setProblem(null); onDraft?.(v) }

  const look = async (path: string, follow = true) => {                     // list a folder; the field follows where the list is
    const r = await getFolders(path)
    if (r === "unknown") { setListing("unknown"); return }
    if ("error" in r) { if (path.trim()) return look("", follow); setListing("unknown"); return }
    setListing(r)
    if (follow && r.path) setDraft(r.path)
  }
  const browse = async () => {
    if (browsing) { setBrowsing(false); return }
    setBrowsing(true)
    if (draft.trim()) { void look(draft.trim(), false); return }
    const last = folders.at(-1)
    if (last) {                                                              // another folder next to the last one: start in the folder above it
      const r = await getFolders(last)
      if (r !== "unknown" && !("error" in r) && r.parent !== null) { void look(r.parent, false); return }
    }
    void look("", false)
  }

  const add = async (): Promise<string[] | null> => {                        // what is typed becomes a folder of the project
    const text = draft.trim()
    if (!text) return folders
    if (folders.length >= FOLDERS_MAX) { setProblem(t("A project can have up to {n} folders.", { n: FOLDERS_MAX })); return null }
    setChecking(true)
    const r = await checkFolder(text)
    setChecking(false)
    if ("error" in r) { setProblem(r.error); field.current?.focus(); return null }
    const next = folders.includes(r.path) ? folders : [...folders, r.path]
    onChange(next)
    setDraftState(""); onDraft?.("")
    return next
  }
  useImperativeHandle(api, () => ({ commit: add, focus: () => field.current?.focus() }))                          // eslint-disable-line react-hooks/exhaustive-deps

  const remove = (i: number) => onChange(folders.filter((_, j) => j !== i))
  const makeMain = (i: number) => onChange([folders[i], ...folders.filter((_, j) => j !== i)])

  return (
    <div data-folders>
      {folders.length > 0 && (
        <ul className="mb-2 space-y-1" aria-label={t("Folders of the project")}>
          {folders.map((f, i) => (
            <li key={f} data-folder={i === 0 ? "main" : "other"} className="flex items-center gap-1.5 rounded-sm border border-line px-2 py-1">
              <HugeiconsIcon icon={Folder01Icon} size={14} aria-hidden className="shrink-0 text-ink-3" />
              <FolderName path={f} />
              {i === 0
                ? (folders.length > 1 && <span className="shrink-0 rounded-sm bg-fill px-1.5 text-[11px] text-ink-2" title={t("Commands run here")}>{t("Main")}</span>)
                : <button type="button" className="shrink-0 rounded-sm px-1.5 text-[11px] text-ink-2 transition-colors hover:bg-hover hover:text-ink" aria-label={t("Make {path} the main folder", { path: f })} onClick={() => makeMain(i)}>{t("Make main")}</button>}
              <button type="button" aria-label={t("Remove {path}", { path: f })} disabled={folders.length <= min} onClick={() => remove(i)} className="flex size-6 shrink-0 items-center justify-center rounded-sm text-ink-3 transition-colors hover:bg-hover hover:text-ink disabled:pointer-events-none disabled:opacity-30">
                <HugeiconsIcon icon={Cancel01Icon} size={13} aria-hidden />
              </button>
            </li>
          ))}
        </ul>
      )}
      <div className="flex items-center gap-2">
        <input
          ref={field}
          className={cn(inputCls, "min-w-0 flex-1 font-mono")}
          aria-label={t("Add a folder")}
          aria-invalid={problem ? true : undefined}
          value={draft}
          placeholder="C:/work/my-project"
          autoComplete="off"
          spellCheck={false}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); if (draft.trim()) void add(); else onSubmit?.() } }}
        />
        <Button onClick={() => void browse()}>{browsing ? t("Hide") : t("Browse")}</Button>
        <Button disabled={!draft.trim() || checking} onClick={() => void add()}>{t("Add")}</Button>
      </div>
      {problem && <p role="alert" className="mt-1.5 text-[12px] text-bad">{problem}</p>}
      {browsing && listing === "unknown" && <p className="mt-1.5 text-[12px] text-ink-2">{t("The folders of this PC cannot be listed from here. Type the path.")}</p>}
      {browsing && listing && listing !== "unknown" && (
        <div data-folder-list className="mt-2 max-h-[200px] overflow-y-auto rounded-sm border border-line p-1">
          {listing.parent !== null && (
            <button type="button" aria-label={t("Up one folder")} onClick={() => void look(listing.parent!)} className="flex h-8 w-full items-center gap-2 rounded-sm px-2 text-left text-[13px] text-ink-2 transition-colors hover:bg-hover hover:text-ink">
              <HugeiconsIcon icon={ArrowUp01Icon} size={14} aria-hidden /><span className="truncate">..</span>
            </button>
          )}
          {listing.dirs.map((d) => (
            <button key={d.path} type="button" data-dir onClick={() => void look(d.path)} className="flex h-8 w-full items-center gap-2 rounded-sm px-2 text-left text-[13px] transition-colors hover:bg-hover">
              <HugeiconsIcon icon={Folder01Icon} size={14} aria-hidden className="shrink-0 text-ink-3" /><span className="truncate">{d.name}</span>
            </button>
          ))}
          {listing.dirs.length === 0 && <p className="px-2 py-1.5 text-[12px] text-ink-3">{t("No folders in here.")}</p>}
          {listing.truncated && <p className="px-2 py-1.5 text-[12px] text-ink-3">{t("Only the first folders are listed. Type the path to go further.")}</p>}
        </div>
      )}
    </div>
  )
}
