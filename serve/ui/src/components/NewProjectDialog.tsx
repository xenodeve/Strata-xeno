import { useEffect, useId, useRef, useState } from "react"
import { createPortal } from "react-dom"
import { cn } from "../lib/cn"
import { t } from "../lib/i18n"
import { FoldersEditor, type FoldersApi } from "./FoldersEditor"
import { GlidePanel, useMounted } from "./motion"
import { Button, inputCls } from "./ui"

// A project is a name and its folders (issue #96): the folders are where the coding tools of its chats work, as Claude Code works in the folder it was started in (and in
// the other folders it was given, such as other git worktrees). So a project cannot be made without a folder. The folders are typed or chosen from the folders of this PC
// (components/FoldersEditor.tsx). The dialog stretches where its content grows (a folder added, the list of folders, what is offered while a path is typed) and closes
// up where it shrinks: the part that changed glides and the rest follows (GlidePanel), the dialog stays centered.

export function NewProjectDialog({ open, onCreate, onCancel }: { open: boolean; onCreate: (name: string, folders: string[]) => void; onCancel: () => void }) {
  const mounted = useMounted(open, 220)
  const [shown, setShown] = useState(false)
  const [name, setName] = useState("")
  const [folders, setFolders] = useState<string[]>([])
  const [pending, setPending] = useState("")                                    // what is typed in the folder field and not yet added
  const [busy, setBusy] = useState(false)
  const nameRef = useRef<HTMLInputElement>(null)
  const editor = useRef<FoldersApi>(null)
  const panel = useRef<HTMLDivElement>(null)
  const was = useRef<Element | null>(null)
  const head = useId()
  const cancelRef = useRef(onCancel)                                          // the page gives a new function at each render; the dialog must not start over for that
  cancelRef.current = onCancel

  useEffect(() => {
    if (!open) { setShown(false); return }
    setName(""); setFolders([]); setPending(""); setBusy(false)
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

  const ready = name.trim().length > 0 && (folders.length > 0 || pending.trim().length > 0) && !busy
  const create = async () => {
    if (!ready) { if (!name.trim()) nameRef.current?.focus(); else editor.current?.focus(); return }
    setBusy(true)
    const list = await editor.current?.commit()                              // what is typed counts too, once it is checked
    setBusy(false)
    if (!list || list.length === 0) return
    onCreate(name.trim(), list)
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
        className={cn("relative w-full max-w-[460px] overflow-hidden rounded-xl border border-line bg-surface shadow-[0_24px_64px_-16px_rgb(0_0_0/0.4)] transition-[opacity,transform] duration-200 ease-[var(--ease)]", shown && open ? "translate-y-0 scale-100 opacity-100" : "translate-y-2 scale-[0.98] opacity-0")}
      >
        <GlidePanel cap={() => innerHeight - 34} inner="p-5" className="[scrollbar-gutter:auto]">
          <h2 id={head} className="text-[16px] font-semibold">{t("New project")}</h2>
          <p className="mt-1.5 text-[13px] leading-relaxed text-ink-2">{t("A project is one or more folders on this PC, such as the worktrees of one repository. The coding tools of its chats work in them: files inside are free to read and change, everything else asks you first. The first folder is the main one: commands run there.")}</p>

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
              onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); editor.current?.focus() } }}
            />
          </label>

          <div className="mt-3">
            <span className="mb-1.5 block text-[13px]">{t("Folders")}</span>
            <FoldersEditor folders={folders} onChange={setFolders} onDraft={setPending} onSubmit={() => void create()} api={editor} />
          </div>

          <div className="mt-5 flex justify-end gap-2">
            <Button kind="quiet" onClick={onCancel}>{t("Cancel")}</Button>
            <Button kind="primary" disabled={!ready} onClick={() => void create()}>{t("Create project")}</Button>
          </div>
        </GlidePanel>
      </div>
    </div>,
    document.body,
  )
}
