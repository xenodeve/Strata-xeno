import { useEffect, useRef, useState, type CSSProperties, type ReactNode } from "react"
import { HugeiconsIcon } from "@hugeicons/react"
import { Add01Icon, MessageAdd01Icon, MoreHorizontalIcon, SidebarLeftIcon } from "@hugeicons/core-free-icons"
import { chat, useChatVersion } from "../lib/chat"
import type { SessionMeta } from "../lib/sessions"
import { t } from "../lib/i18n"
import { store } from "../lib/store"
import { cn } from "../lib/cn"
import { toast } from "./toast"
import { BranchedMenu, type BranchSection, type BranchTopic } from "./BranchedMenu"
import { useMounted } from "./motion"
import { NewProjectDialog } from "./NewProjectDialog"

// Recents (the conversations that are in no project, newest first, on one line from "Recents") and Projects (folders that group them) beside the Chat page, as in ChatGPT, Gemini
// and Claude, drawn as the branched menu: a project, and Recents itself, is a section, a conversation a branch, and the line is drawn
// to the one that is open. The list stretches open and shrinks to a rail. They are kept in this browser (lib/sessions.ts). While an answer is being written the other conversations cannot be
// opened (the same rule as New chat); the one that is open can always be clicked.
const SHRINK_MS = 320
const RECENTS = "recents"

type Target = { kind: "chat" | "project"; id: string }
interface Open extends Target { x: number; y: number; moving?: boolean }

export function Sidebar({ drawer = false, onClose }: { drawer?: boolean; onClose?: () => void }) {
  useChatVersion()
  const idx = chat.index
  const busy = !!chat.busy
  const [shut, setShut] = useState(() => store.get<string>("sidebar", "open") === "closed")
  const [folded, setFolded] = useState<string[]>(() => { const f = store.get<unknown>("sidebar.folded", []); return Array.isArray(f) ? f.filter((x): x is string => typeof x === "string") : [] })
  const [menu, setMenu] = useState<Open | null>(null)
  const [renaming, setRenaming] = useState<Target | null>(null)
  const [adding, setAdding] = useState(false)
  const [sure, setSure] = useState<string | null>(null)                       // what Delete was asked of once
  const sureTimer = useRef<ReturnType<typeof setTimeout>>(undefined)

  const unfiled = idx.items.filter((i) => !i.project).sort((a, b) => b.time - a.time)
  const sectionOfActive = idx.items.find((i) => i.id === idx.active)?.project ?? (idx.active ? RECENTS : undefined)       // the section the open conversation is in
  const listed = useMounted(!shut || drawer, SHRINK_MS)                        // the list stays while the sidebar shrinks around it
  const setFoldedKept = (next: string[]) => { setFolded(next); store.set("sidebar.folded", next) }
  const toggleFold = (id: string) => setFoldedKept(folded.includes(id) ? folded.filter((x) => x !== id) : [...folded, id])
  useEffect(() => {                                                           // the open conversation is never hidden in a folded section
    if (sectionOfActive && folded.includes(sectionOfActive)) setFoldedKept(folded.filter((x) => x !== sectionOfActive))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [idx.active])

  useEffect(() => {                                                           // a menu closes on a click elsewhere and on Escape
    if (!menu) return
    const down = (e: PointerEvent) => { if (!(e.target as Element | null)?.closest("[role=menu]")) setMenu(null) }
    const key = (e: KeyboardEvent) => { if (e.key === "Escape") setMenu(null) }
    document.addEventListener("pointerdown", down)
    document.addEventListener("keydown", key)
    return () => { document.removeEventListener("pointerdown", down); document.removeEventListener("keydown", key) }
  }, [menu])

  const setShutKept = (v: boolean) => { setShut(v); store.set("sidebar", v ? "closed" : "open") }
  const stillWriting = () => toast("warn", t("Still writing"), t("Stop the answer first."))
  const pick = (id: string) => {
    if (id !== idx.active && busy) { stillWriting(); return }
    if (chat.open(id)) onClose?.()
  }
  const fresh = () => {
    if (!chat.newSession()) { stillWriting(); return }
    onClose?.()
    document.querySelector<HTMLTextAreaElement>("textarea[aria-label]")?.focus()
  }
  const ask = (key: string, run: () => void) => {                              // Delete asks twice
    if (sure !== key) { setSure(key); clearTimeout(sureTimer.current); sureTimer.current = setTimeout(() => setSure(null), 4000); return }
    setSure(null)
    setMenu(null)
    run()
  }
  const openMenu = (e: React.MouseEvent<HTMLButtonElement>, target: Target) => {
    const r = e.currentTarget.getBoundingClientRect()
    setSure(null)
    setMenu(menu && menu.kind === target.kind && menu.id === target.id ? null : { ...target, x: Math.max(8, Math.min(r.right - 208, innerWidth - 216)), y: r.bottom + 4 })
  }

  const submitRename = (target: Target, value: string) => {
    setRenaming(null)
    if (target.kind === "chat") chat.rename(target.id, value)
    else chat.renameProject(target.id, value)
  }
  const renameField = (target: Target, initial: string, className: string) => (
    <input
      aria-label={t("Rename")}
      autoFocus
      defaultValue={initial}
      maxLength={80}
      onFocus={(e) => e.currentTarget.select()}
      onKeyDown={(e) => { if (e.key === "Enter") submitRename(target, e.currentTarget.value); else if (e.key === "Escape") setRenaming(null) }}
      onBlur={(e) => submitRename(target, e.currentTarget.value)}
      className={cn("h-8 min-w-0 flex-1 rounded-sm border border-accent bg-surface px-2 text-[13px] outline-none", className)}
    />
  )
  const optionsButton = (e: { target: Target; label: string }): ReactNode => (
    <button
      type="button"
      aria-label={e.label}
      aria-haspopup="menu"
      aria-expanded={menu?.kind === e.target.kind && menu.id === e.target.id}
      onClick={(ev) => openMenu(ev, e.target)}
      className="mr-1 flex size-6 shrink-0 items-center justify-center rounded-sm text-ink-3 opacity-0 transition-opacity hover:text-ink focus-visible:opacity-100 group-hover:opacity-100 aria-expanded:opacity-100 [@media(hover:none)]:opacity-100"
    >
      <HugeiconsIcon icon={MoreHorizontalIcon} size={14} aria-hidden />
    </button>
  )

  const topic = (s: SessionMeta): BranchTopic => {
    const target: Target = { kind: "chat", id: s.id }
    const title = s.title || t("Untitled chat")
    const blocked = busy && s.id !== idx.active
    const isRenaming = renaming?.kind === "chat" && renaming.id === s.id
    return {
      value: s.id, label: title, disabled: blocked, title: blocked ? t("Stop the answer first.") : title,
      custom: isRenaming ? renameField(target, s.title, "ml-10 mr-1") : undefined,
      trailing: isRenaming ? undefined : optionsButton({ target, label: t("Options for {title}", { title }) }),
    }
  }
  const projectSections: BranchSection[] = idx.projects.map((p) => {
    const target: Target = { kind: "project", id: p.id }
    const items = idx.items.filter((i) => i.project === p.id).sort((a, b) => b.time - a.time)
    const isRenaming = renaming?.kind === "project" && renaming.id === p.id
    return {
      value: p.id, label: p.name, topics: items.map(topic), empty: t("Empty"),
      custom: isRenaming ? renameField(target, p.name, "") : undefined,
      trailing: isRenaming ? undefined : (<><span className="num mr-1 shrink-0 text-[11px] text-ink-3">{items.length}</span>{optionsButton({ target, label: t("Options for project {name}", { name: p.name }) })}</>),
    }
  })
  const recentSections: BranchSection[] = [{ value: RECENTS, label: t("Recents"), topics: unfiled.map(topic), empty: idx.items.length === 0 ? t("No conversations yet") : t("Empty") }]
  const openOf = (sections: BranchSection[]) => sections.map((s) => s.value).filter((v) => !folded.includes(v))
  const menuStyle: CSSProperties | undefined = menu ? { position: "fixed", left: menu.x, top: menu.y } : undefined
  const menuItem = (label: ReactNode, onClick: () => void, opts: { danger?: boolean; key?: string } = {}) => (
    <button key={opts.key ?? String(label)} type="button" role="menuitem" onClick={onClick} className={cn("flex w-full items-center rounded-sm px-2.5 py-1.5 text-left text-[13px] transition-colors hover:bg-hover", opts.danger && "text-bad")}>{label}</button>
  )

  const content = (
    <>
      <div className="flex items-center gap-1">
        <button type="button" onClick={fresh} className="flex h-8 min-w-0 flex-1 items-center gap-2 rounded-sm px-2.5 text-[13px] font-medium transition-colors hover:bg-hover">
          <HugeiconsIcon icon={MessageAdd01Icon} size={15} strokeWidth={1.8} aria-hidden /><span className="truncate">{t("New chat")}</span>
        </button>
        {!drawer && (
          <button type="button" aria-label={t("Hide the list")} title={t("Hide the list")} onClick={() => setShutKept(true)} className="flex size-8 shrink-0 items-center justify-center rounded-sm text-ink-3 transition-colors hover:bg-hover hover:text-ink">
            <HugeiconsIcon icon={SidebarLeftIcon} size={16} strokeWidth={1.6} aria-hidden />
          </button>
        )}
      </div>

      <section data-projects>
        <div className="flex items-center justify-between px-2.5">
          <h2 className="text-[12px] font-medium text-ink-3">{t("Projects")}</h2>
          <button type="button" aria-label={t("New project")} title={t("New project")} onClick={() => setAdding(true)} className="flex size-6 items-center justify-center rounded-sm text-ink-3 transition-colors hover:bg-hover hover:text-ink">
            <HugeiconsIcon icon={Add01Icon} size={14} aria-hidden />
          </button>
        </div>
        {idx.projects.length === 0 && <p className="mt-1 px-2.5 text-[12px] text-ink-3">{t("No projects yet.")}</p>}
        <BranchedMenu fill label={t("Projects")} sections={projectSections} active={idx.active ?? ""} onSelect={pick} open={openOf(projectSections)} onToggle={toggleFold} />
      </section>

      <section data-recents>
        <BranchedMenu fill label={t("Recents")} sections={recentSections} active={idx.active ?? ""} onSelect={pick} open={openOf(recentSections)} onToggle={toggleFold} />
      </section>

      <NewProjectDialog open={adding} onCancel={() => setAdding(false)} onCreate={(name, folders) => { chat.addProject(name, folders); setAdding(false) }} />

      {menu && (
        <div role="menu" aria-label={t("Options")} style={menuStyle} className="toast-in z-50 w-52 rounded-lg border border-line bg-surface p-1 shadow-[0_14px_40px_-12px_rgb(0_0_0/0.28)]">
          {menu.kind === "chat" ? (() => {
            const s = idx.items.find((i) => i.id === menu.id)
            if (!s) return null
            return menu.moving ? (
              <>
                {s.project && menuItem(t("No project"), () => { chat.move(s.id, undefined); setMenu(null) })}
                {idx.projects.filter((p) => p.id !== s.project).map((p) => menuItem(p.name, () => { chat.move(s.id, p.id); setMenu(null) }, { key: p.id }))}
                {idx.projects.length === 0 && <p className="px-2.5 py-1.5 text-[12px] text-ink-3">{t("No projects yet.")}</p>}
              </>
            ) : (
              <>
                {menuItem(t("Rename"), () => { setRenaming({ kind: "chat", id: s.id }); setMenu(null) })}
                {menuItem(t("Move to project"), () => setMenu({ ...menu, moving: true }))}
                {menuItem(sure === "chat:" + s.id ? t("Sure? Delete") : t("Delete"), () => ask("chat:" + s.id, () => { if (!chat.remove(s.id)) stillWriting() }), { danger: true, key: "delete" })}
              </>
            )
          })() : (
            <>
              {menuItem(t("Rename"), () => { setRenaming({ kind: "project", id: menu.id }); setMenu(null) })}
              {menuItem(sure === "project:" + menu.id ? t("Sure? Delete project") : t("Delete project"), () => ask("project:" + menu.id, () => chat.removeProject(menu.id)), { danger: true, key: "delete" })}
            </>
          )}
        </div>
      )}
    </>
  )

  if (drawer) return <aside aria-label={t("Conversations")} className="flex flex-col gap-4">{content}</aside>
  return (
    // The column's width is what moves (the list inside keeps its own, so it is cut, not squeezed); the list fades with it and is
    // taken away once the rail is all that is left. The button that brings it back fades in as the list goes.
    <aside aria-label={t("Conversations")} className={cn("relative sticky top-[4.5rem] min-h-8 shrink-0 self-start overflow-hidden transition-[width] duration-300 ease-[var(--ease)]", shut ? "w-9" : "w-[250px]")}>
      {listed && (
        <div inert={shut} className={cn("side-in flex max-h-[calc(100dvh-5.5rem)] w-[250px] flex-col gap-4 overflow-y-auto pr-1 transition-opacity duration-200", shut && "pointer-events-none opacity-0")}>{content}</div>
      )}
      <button
        type="button"
        aria-label={t("Show the list")}
        title={t("Show the list")}
        aria-hidden={!shut}
        tabIndex={shut ? 0 : -1}
        onClick={() => setShutKept(false)}
        className={cn("absolute left-0.5 top-0 flex size-8 items-center justify-center rounded-sm text-ink-2 transition-[opacity,color,background-color] duration-200 hover:bg-hover hover:text-ink", shut ? "opacity-100 delay-150" : "pointer-events-none opacity-0")}
      >
        <HugeiconsIcon icon={SidebarLeftIcon} size={16} strokeWidth={1.6} aria-hidden />
      </button>
    </aside>
  )
}
