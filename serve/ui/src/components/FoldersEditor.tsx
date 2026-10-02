import { useEffect, useId, useImperativeHandle, useRef, useState, type Ref } from "react"
import { HugeiconsIcon } from "@hugeicons/react"
import { ArrowUp01Icon, Cancel01Icon, Folder01Icon } from "@hugeicons/core-free-icons"
import { getFolders, type FolderView } from "../lib/api"
import { cn } from "../lib/cn"
import { matchParts, rank, sepOf, splitTyped } from "../lib/folders"
import { FOLDERS_MAX } from "../lib/sessions"
import { t } from "../lib/i18n"
import { Button, inputCls } from "./ui"

// The folders of a project (issue #96): the first is the main one, where the coding tools run commands; the others are its other folders (other git worktrees of the same
// repository, a library next to the app) where files are as free as in the main one, as Claude Code's added directories. A folder is typed or chosen from the folders of this
// PC (the server lists them; while a path is typed, the folders that go with what is typed are offered, so a rough path and one click is enough) and is checked with the server before it is added; the path that is kept is the one the server resolved. When the server cannot be asked
// (another PC) the path is used as typed.

export type Listing = FolderView | "unknown" | null
const SUGGEST_MAX = 8                                // the folders offered while a path is typed; "and n more" for the rest

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
  const listId = useId()
  const [typed, setTyped] = useState<string | null>(null)                       // what the user typed (not what a click put in the field): the suggestions go with it
  const [sugg, setSugg] = useState<{ base: string; partial: string; names: string[]; more: number } | null>(null)
  const [active, setActive] = useState(-1)
  const [focused, setFocused] = useState(false)
  const setDraft = (v: string) => { setDraftState(v); setProblem(null); onDraft?.(v) }

  useEffect(() => {                                                          // the folders in the folder that is typed, those that go with the name typed after it
    const q = typed === null ? null : splitTyped(typed)
    if (!q) { setSugg(null); return }
    let stale = false
    const timer = setTimeout(async () => {
      const r = await getFolders(q.base)
      if (stale) return
      if (r === "unknown" || "error" in r) { setSugg(null); return }
      const names = rank(r.dirs.map((d) => d.name), q.partial)
      setSugg({ base: q.base, partial: q.partial, names: names.slice(0, SUGGEST_MAX), more: Math.max(0, names.length - SUGGEST_MAX) })
      setActive(-1)
    }, 120)
    return () => { stale = true; clearTimeout(timer) }
  }, [typed])
  const open = focused && !!sugg && sugg.names.length > 0
  const pick = (name: string) => {                                           // the folder is finished in the field, and what is inside it is offered next
    if (!sugg) return
    const v = sugg.base + name + sepOf(sugg.base)
    setDraft(v)
    setTyped(v)
    field.current?.focus()
  }

  const look = async (path: string, follow = true) => {                     // list a folder; the field follows where the list is
    const r = await getFolders(path)
    if (r === "unknown") { setListing("unknown"); return }
    if ("error" in r) { if (path.trim()) return look("", follow); setListing("unknown"); return }
    setListing(r)
    if (follow && r.path) { setDraft(r.path); setTyped(null) }
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
    setDraftState(""); setTyped(null); onDraft?.("")
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
          role="combobox"
          aria-label={t("Add a folder")}
          aria-expanded={open}
          aria-controls={listId}
          aria-autocomplete="list"
          aria-activedescendant={open && active >= 0 ? `${listId}-${active}` : undefined}
          aria-invalid={problem ? true : undefined}
          value={draft}
          placeholder="C:/work/my-project"
          autoComplete="off"
          spellCheck={false}
          onChange={(e) => { setDraft(e.target.value); setTyped(e.target.value) }}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          onKeyDown={(e) => {
            if (open && sugg) {
              const n = sugg.names.length
              if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); setActive((a) => (e.key === "ArrowDown" ? (a + 1) % n : a <= 0 ? n - 1 : a - 1)); return }
              if (e.key === "Enter" && active >= 0) { e.preventDefault(); pick(sugg.names[active]); return }
              if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); setTyped(null); return }          // closes the suggestions, not the dialog
            }
            if (e.key === "Enter") { e.preventDefault(); if (draft.trim()) void add(); else onSubmit?.() }
          }}
        />
        <Button onClick={() => void browse()}>{browsing ? t("Hide") : t("Browse")}</Button>
        <Button disabled={!draft.trim() || checking} onClick={() => void add()}>{t("Add")}</Button>
      </div>
      {open && sugg && (
        <ul id={listId} role="listbox" aria-label={t("Folders that go with what is typed")} data-suggestions onMouseDown={(e) => e.preventDefault()} className="mt-1.5 max-h-[188px] overflow-y-auto rounded-sm border border-line p-1">
          {sugg.names.map((n, i) => {
            const [a, m, z] = matchParts(n, sugg.partial)
            return (
              <li key={n} id={`${listId}-${i}`} role="option" aria-selected={i === active} data-suggestion onClick={() => pick(n)} onMouseMove={() => setActive(i)} className={cn("flex h-8 cursor-pointer items-center gap-2 rounded-sm px-2 text-[13px]", i === active ? "bg-hover" : "hover:bg-hover")}>
                <HugeiconsIcon icon={Folder01Icon} size={14} aria-hidden className="shrink-0 text-ink-3" />
                <span className="truncate">{a}<b className="font-semibold">{m}</b>{z}</span>
              </li>
            )
          })}
          {sugg.more > 0 && <li aria-hidden className="px-2 py-1 text-[12px] text-ink-3">{t("and {n} more. Keep typing to narrow them.", { n: sugg.more })}</li>}
        </ul>
      )}
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
