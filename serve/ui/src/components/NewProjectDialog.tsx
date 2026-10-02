import { useEffect, useId, useRef, useState } from "react"
import { createPortal } from "react-dom"
import { HugeiconsIcon } from "@hugeicons/react"
import { ArrowUp01Icon, Folder01Icon } from "@hugeicons/core-free-icons"
import { getFolders, type FolderView } from "../lib/api"
import { cn } from "../lib/cn"
import { t } from "../lib/i18n"
import { useMounted } from "./motion"
import { Button, inputCls } from "./ui"

// A project is a name and a folder (issue #96): the folder is where the coding tools of its chats work, as Claude Code works in the folder it was started in. So a project
// cannot be made without one. The folder is typed or chosen from the folders of this PC (the server lists them); what is typed is checked with the server before the project
// is made, and the path that is kept is the one the server resolved. When the server cannot be asked (another PC), the path is used as typed.

type Listing = FolderView | "unknown" | null

export function NewProjectDialog({ open, onCreate, onCancel }: { open: boolean; onCreate: (name: string, folder: string) => void; onCancel: () => void }) {
  const mounted = useMounted(open, 220)
  const [shown, setShown] = useState(false)
  const [name, setName] = useState("")
  const [folder, setFolder] = useState("")
  const [browsing, setBrowsing] = useState(false)
  const [listing, setListing] = useState<Listing>(null)
  const [problem, setProblem] = useState<string | null>(null)
  const [checking, setChecking] = useState(false)
  const nameRef = useRef<HTMLInputElement>(null)
  const folderRef = useRef<HTMLInputElement>(null)
  const panel = useRef<HTMLDivElement>(null)
  const was = useRef<Element | null>(null)
  const head = useId()
  const cancelRef = useRef(onCancel)                                          // the page gives a new function at each render; the dialog must not start over for that
  cancelRef.current = onCancel

  useEffect(() => {
    if (!open) { setShown(false); return }
    setName(""); setFolder(""); setBrowsing(false); setListing(null); setProblem(null); setChecking(false)
    was.current = document.activeElement
    const id = requestAnimationFrame(() => { setShown(true); nameRef.current?.focus() })
    const key = (e: KeyboardEvent) => {
      if (e.key === "Escape") { e.preventDefault(); cancelRef.current() }
      else if (e.key === "Tab" && panel.current) {                           // the focus stays in the dialog
        const f = [...panel.current.querySelectorAll<HTMLElement>("button:not(:disabled), input")]
        const first = f[0], last = f[f.length - 1]
        if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus() }
        else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus() }
      }
    }
    document.addEventListener("keydown", key)
    return () => { cancelAnimationFrame(id); document.removeEventListener("keydown", key); (was.current as HTMLElement | null)?.focus?.() }
  }, [open])

  const look = async (path: string, follow = true) => {                     // list a folder; the field follows where the list is
    const r = await getFolders(path)
    if (r === "unknown") { setListing("unknown"); return }
    if ("error" in r) { if (path.trim()) return look("", follow); setListing("unknown"); return }
    setListing(r)
    if (follow && r.path) { setFolder(r.path); setProblem(null) }
  }
  const browse = () => {
    if (browsing) { setBrowsing(false); return }
    setBrowsing(true)
    void look(folder.trim(), false)
  }

  const ready = name.trim().length > 0 && folder.trim().length > 0 && !checking
  const create = async () => {
    if (!ready) { (name.trim() ? folderRef : nameRef).current?.focus(); return }
    setChecking(true)
    const r = await getFolders(folder.trim())
    setChecking(false)
    if (r !== "unknown" && "error" in r) { setProblem(t("That is not a folder on this PC.")); folderRef.current?.focus(); return }
    onCreate(name.trim(), r === "unknown" ? folder.trim() : r.path)
  }

  if (!mounted) return null
  return createPortal(
    <div className="fixed inset-0 z-50 grid place-items-center p-4">
      <div aria-hidden className={cn("absolute inset-0 bg-black/45 transition-opacity duration-200", shown && open ? "opacity-100" : "opacity-0")} onClick={onCancel} />
      <div
        ref={panel}
        role="dialog"
        aria-modal="true"
        aria-labelledby={head}
        data-new-project
        className={cn("relative w-full max-w-[440px] rounded-xl border border-line bg-surface p-5 shadow-[0_24px_64px_-16px_rgb(0_0_0/0.4)] transition-[opacity,transform] duration-200 ease-[var(--ease)]", shown && open ? "translate-y-0 scale-100 opacity-100" : "translate-y-2 scale-[0.98] opacity-0")}
      >
        <h2 id={head} className="text-[16px] font-semibold">{t("New project")}</h2>
        <p className="mt-1.5 text-[13px] leading-relaxed text-ink-2">{t("A project is a folder on this PC. The coding tools of its chats work in it: files inside it are free to read and change, everything else asks you first.")}</p>

        <label className="mt-4 block">
          <span className="text-[13px]">{t("Project name")}</span>
          <input
            ref={nameRef}
            className={cn(inputCls, "mt-1.5")}
            aria-label={t("Project name")}
            value={name}
            maxLength={60}
            autoComplete="off"
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); folderRef.current?.focus() } }}
          />
        </label>

        <div className="mt-3">
          <label htmlFor={head + "-folder"} className="text-[13px]">{t("Folder")}</label>
          <div className="mt-1.5 flex items-center gap-2">
            <input
              id={head + "-folder"}
              ref={folderRef}
              className={cn(inputCls, "flex-1 font-mono")}
              aria-label={t("Folder of the project")}
              aria-invalid={problem ? true : undefined}
              value={folder}
              placeholder="C:/work/my-project"
              autoComplete="off"
              spellCheck={false}
              onChange={(e) => { setFolder(e.target.value); setProblem(null) }}
              onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); void create() } }}
            />
            <Button onClick={browse}>{browsing ? t("Hide") : t("Browse")}</Button>
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

        <div className="mt-5 flex justify-end gap-2">
          <Button kind="quiet" onClick={onCancel}>{t("Cancel")}</Button>
          <Button kind="primary" disabled={!ready} onClick={() => void create()}>{t("Create project")}</Button>
        </div>
      </div>
    </div>,
    document.body,
  )
}
