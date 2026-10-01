import { useEffect, useRef, useState, type CSSProperties } from "react"
import { HugeiconsIcon } from "@hugeicons/react"
import { Add01Icon, ArrowDown01Icon, ArrowRight01Icon, Folder01Icon, MessageAdd01Icon, MoreHorizontalIcon, SidebarLeftIcon } from "@hugeicons/core-free-icons"
import { chat, useChatVersion } from "../lib/chat"
import { groupRecents, type RecentsGroup, type SessionMeta } from "../lib/sessions"
import { msg, t } from "../lib/i18n"
import { store } from "../lib/store"
import { cn } from "../lib/cn"
import { toast } from "./toast"

// Recents (the conversations, newest first, by day) and Projects (folders that group them) beside the Chat page, as in ChatGPT, Gemini
// and Claude. They are kept in this browser (lib/sessions.ts). While an answer is being written the other conversations cannot be
// opened (the same rule as New chat); the one that is open can always be clicked.
const DAY: Record<RecentsGroup["key"], string> = { today: msg("Today"), yesterday: msg("Yesterday"), earlier: msg("Earlier") }

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
  const root = useRef<HTMLElement>(null)

  useEffect(() => {                                                           // a menu closes on a click elsewhere and on Escape
    if (!menu) return
    const down = (e: PointerEvent) => { if (!(e.target as Element | null)?.closest("[role=menu]")) setMenu(null) }
    const key = (e: KeyboardEvent) => { if (e.key === "Escape") setMenu(null) }
    document.addEventListener("pointerdown", down)
    document.addEventListener("keydown", key)
    return () => { document.removeEventListener("pointerdown", down); document.removeEventListener("keydown", key) }
  }, [menu])

  const setShutKept = (v: boolean) => { setShut(v); store.set("sidebar", v ? "closed" : "open") }
  const toggleFold = (id: string) => setFolded((f) => { const next = f.includes(id) ? f.filter((x) => x !== id) : [...f, id]; store.set("sidebar.folded", next); return next })
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

  if (shut && !drawer) {
    return (
      <aside aria-label={t("Conversations")} className="sticky top-[4.5rem] flex w-9 shrink-0 self-start justify-center">
        <button type="button" aria-label={t("Show the list")} title={t("Show the list")} onClick={() => setShutKept(false)} className="flex size-8 items-center justify-center rounded-sm text-ink-2 transition-colors hover:bg-hover hover:text-ink">
          <HugeiconsIcon icon={SidebarLeftIcon} size={16} strokeWidth={1.6} aria-hidden />
        </button>
      </aside>
    )
  }

  const submitRename = (target: Target, value: string) => {
    setRenaming(null)
    if (target.kind === "chat") chat.rename(target.id, value)
    else chat.renameProject(target.id, value)
  }
  const renameField = (target: Target, initial: string) => (
    <input
      key="rename"
      aria-label={t("Rename")}
      autoFocus
      defaultValue={initial}
      maxLength={80}
      onFocus={(e) => e.currentTarget.select()}
      onKeyDown={(e) => { if (e.key === "Enter") submitRename(target, e.currentTarget.value); else if (e.key === "Escape") setRenaming(null) }}
      onBlur={(e) => submitRename(target, e.currentTarget.value)}
      className="mx-1 h-8 w-[calc(100%-0.5rem)] rounded-sm border border-accent bg-surface px-2 text-[13px] outline-none"
    />
  )

  const chatRow = (s: SessionMeta) => {
    const active = s.id === idx.active
    const blocked = busy && !active
    const title = s.title || t("Untitled chat")
    const isRenaming = renaming?.kind === "chat" && renaming.id === s.id
    return (
      <li key={s.id} data-chat={s.id} className="group">
        {isRenaming ? renameField({ kind: "chat", id: s.id }, s.title) : (
          <div className={cn("flex items-center rounded-md transition-colors", active ? "bg-fill" : "hover:bg-hover")}>
            <button
              type="button"
              aria-current={active ? "true" : undefined}
              aria-disabled={blocked ? "true" : undefined}
              title={blocked ? t("Stop the answer first.") : title}
              onClick={() => pick(s.id)}
              className={cn("min-w-0 flex-1 truncate px-2.5 py-1.5 text-left text-[13px]", blocked ? "text-ink-3" : active ? "text-ink" : "text-ink-2")}
            >
              {title}
            </button>
            <button
              type="button"
              aria-label={t("Options for {title}", { title })}
              aria-haspopup="menu"
              aria-expanded={menu?.kind === "chat" && menu.id === s.id}
              onClick={(e) => openMenu(e, { kind: "chat", id: s.id })}
              className="mr-1 flex size-6 shrink-0 items-center justify-center rounded-sm text-ink-3 opacity-0 transition-opacity hover:text-ink focus-visible:opacity-100 group-hover:opacity-100 aria-expanded:opacity-100 [@media(hover:none)]:opacity-100"
            >
              <HugeiconsIcon icon={MoreHorizontalIcon} size={14} aria-hidden />
            </button>
          </div>
        )}
      </li>
    )
  }

  const filed = (id: string) => idx.items.filter((i) => i.project === id).sort((a, b) => b.time - a.time)
  const recents = groupRecents(idx.items.filter((i) => !i.project), Date.now())
  const menuStyle: CSSProperties | undefined = menu ? { position: "fixed", left: menu.x, top: menu.y } : undefined
  const menuItem = (label: React.ReactNode, onClick: () => void, opts: { danger?: boolean; key?: string } = {}) => (
    <button key={opts.key ?? String(label)} type="button" role="menuitem" onClick={onClick} className={cn("flex w-full items-center rounded-sm px-2.5 py-1.5 text-left text-[13px] transition-colors hover:bg-hover", opts.danger && "text-bad")}>{label}</button>
  )

  return (
    <aside ref={root} aria-label={t("Conversations")} className={cn(drawer ? "flex flex-col gap-4" : "sticky top-[4.5rem] flex max-h-[calc(100dvh-5.5rem)] w-[250px] shrink-0 flex-col gap-4 self-start overflow-y-auto pr-1")}>
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
        {adding && (
          <input
            aria-label={t("Project name")}
            placeholder={t("Project name")}
            autoFocus
            maxLength={60}
            onKeyDown={(e) => { if (e.key === "Enter") { chat.addProject(e.currentTarget.value); setAdding(false) } else if (e.key === "Escape") setAdding(false) }}
            onBlur={(e) => { if (e.currentTarget.value.trim()) chat.addProject(e.currentTarget.value); setAdding(false) }}
            className="mx-1 mt-1 h-8 w-[calc(100%-0.5rem)] rounded-sm border border-accent bg-surface px-2 text-[13px] outline-none placeholder:text-ink-3"
          />
        )}
        {idx.projects.length === 0 && !adding && <p className="mt-1 px-2.5 text-[12px] text-ink-3">{t("No projects yet.")}</p>}
        <ul className="m-0 mt-1 list-none p-0">
          {idx.projects.map((p) => {
            const items = filed(p.id)
            const open = !folded.includes(p.id)
            const isRenaming = renaming?.kind === "project" && renaming.id === p.id
            return (
              <li key={p.id} data-project={p.id}>
                {isRenaming ? renameField({ kind: "project", id: p.id }, p.name) : (
                  <div className="group flex items-center rounded-md transition-colors hover:bg-hover">
                    <button type="button" aria-expanded={open} onClick={() => toggleFold(p.id)} className="flex min-w-0 flex-1 items-center gap-1.5 px-2.5 py-1.5 text-left text-[13px] text-ink-2">
                      <HugeiconsIcon icon={open ? ArrowDown01Icon : ArrowRight01Icon} size={12} aria-hidden className="shrink-0" />
                      <HugeiconsIcon icon={Folder01Icon} size={14} aria-hidden className="shrink-0" />
                      <span className="truncate">{p.name}</span>
                      <span className="num ml-auto shrink-0 text-[11px] text-ink-3">{items.length}</span>
                    </button>
                    <button
                      type="button"
                      aria-label={t("Options for project {name}", { name: p.name })}
                      aria-haspopup="menu"
                      aria-expanded={menu?.kind === "project" && menu.id === p.id}
                      onClick={(e) => openMenu(e, { kind: "project", id: p.id })}
                      className="mr-1 flex size-6 shrink-0 items-center justify-center rounded-sm text-ink-3 opacity-0 transition-opacity hover:text-ink focus-visible:opacity-100 group-hover:opacity-100 aria-expanded:opacity-100 [@media(hover:none)]:opacity-100"
                    >
                      <HugeiconsIcon icon={MoreHorizontalIcon} size={14} aria-hidden />
                    </button>
                  </div>
                )}
                {open && !isRenaming && (
                  <ul className="m-0 ml-4 list-none border-l border-line p-0 pl-1">
                    {items.length === 0 ? <li className="px-2.5 py-1 text-[12px] text-ink-3">{t("Empty")}</li> : items.map(chatRow)}
                  </ul>
                )}
              </li>
            )
          })}
        </ul>
      </section>

      <section data-recents>
        <h2 className="px-2.5 text-[12px] font-medium text-ink-3">{t("Recents")}</h2>
        {idx.items.length === 0 && <p className="mt-1 px-2.5 text-[12px] text-ink-3">{t("No conversations yet")}</p>}
        {recents.map((g) => (
          <div key={g.key} className="mt-2 first:mt-1">
            <p className="px-2.5 pb-0.5 text-[11px] text-ink-3">{t(DAY[g.key])}</p>
            <ul className="m-0 list-none p-0">{g.items.map(chatRow)}</ul>
          </div>
        ))}
      </section>

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
    </aside>
  )
}
