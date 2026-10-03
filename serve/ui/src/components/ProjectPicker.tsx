import { useEffect, useLayoutEffect, useRef, useState } from "react"
import { createPortal } from "react-dom"
import { HugeiconsIcon } from "@hugeicons/react"
import { ArrowDown01Icon, Folder01Icon } from "@hugeicons/core-free-icons"
import type { Project } from "../lib/sessions"
import { baseName } from "../lib/git"
import { cn } from "../lib/cn"
import { t } from "../lib/i18n"

/** The project a new chat works in (issue #99): chosen on the empty chat, so a new chat does not have to be started from its project. The tools work in the project's folders from the
 *  first prompt. Without a project the chat has no folder but the default one (Settings). */
export function ProjectPicker({ projects, value, onPick }: { projects: Project[]; value: string | null; onPick: (id: string | null) => void }) {
  const [open, setOpen] = useState(false)
  const root = useRef<HTMLDivElement>(null)
  const menu = useRef<HTMLDivElement>(null)
  // The menu is drawn on the page itself, not inside the empty chat (which clips what sticks out of it): fixed, under the button, or above it when there is no room below.
  const [place, setPlace] = useState<{ left: number; top?: number; bottom?: number; max: number } | null>(null)
  useLayoutEffect(() => {
    if (!open) { setPlace(null); return }
    const put = () => {
      const r = root.current?.getBoundingClientRect()
      if (!r) return
      const below = window.innerHeight - r.bottom - 12
      const above = r.top - 12
      const want = Math.min(menu.current?.scrollHeight ?? 260, 420)
      const up = below < want && above > below
      const room = Math.max(120, up ? above : below)
      setPlace({ left: Math.min(Math.max(r.left + r.width / 2, 140), window.innerWidth - 140), ...(up ? { bottom: window.innerHeight - r.top + 4 } : { top: r.bottom + 4 }), max: room })
    }
    put()
    window.addEventListener("resize", put)
    window.addEventListener("scroll", put, true)
    return () => { window.removeEventListener("resize", put); window.removeEventListener("scroll", put, true) }
  }, [open])
  useEffect(() => {
    if (!open) return
    const down = (e: PointerEvent) => { const n = e.target as Node | null; if (!root.current?.contains(n) && !menu.current?.contains(n)) setOpen(false) }
    const key = (e: KeyboardEvent) => { if (e.key === "Escape") { e.stopPropagation(); setOpen(false) } }
    document.addEventListener("pointerdown", down)
    document.addEventListener("keydown", key)
    return () => { document.removeEventListener("pointerdown", down); document.removeEventListener("keydown", key) }
  }, [open])
  if (!projects.length) return null
  const chosen = projects.find((p) => p.id === value) ?? null
  const pick = (id: string | null) => { setOpen(false); onPick(id) }
  return (
    <div ref={root} className="relative inline-block" data-project-picker>
      <button
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={t("Project for this chat")}
        data-in-project={chosen ? chosen.name : undefined}
        onClick={() => setOpen(!open)}
        className={cn("inline-flex h-8 items-center gap-1.5 rounded-sm px-2.5 text-[13px] transition-colors hover:bg-hover", chosen ? "bg-fill font-medium" : "text-ink-2")}
      >
        <HugeiconsIcon icon={Folder01Icon} size={14} aria-hidden />
        <span>{chosen ? chosen.name : t("No project")}</span>
        <HugeiconsIcon icon={ArrowDown01Icon} size={12} aria-hidden className={cn("text-ink-3 transition-transform duration-200", open && "rotate-180")} />
      </button>
      {open && createPortal(
        <div ref={menu} role="menu" data-project-menu aria-label={t("Choose the project this chat works in")} style={place ? { left: place.left, top: place.top, bottom: place.bottom, maxHeight: place.max } : { visibility: "hidden" }} className="toast-in fixed z-50 w-64 max-w-[86vw] -translate-x-1/2 overflow-y-auto rounded-lg border border-line bg-surface p-1 text-left shadow-[0_14px_40px_-12px_rgb(0_0_0/0.28)]">
          <button type="button" role="menuitemradio" aria-checked={!chosen} onClick={() => pick(null)} className="flex w-full flex-col rounded-sm px-2.5 py-1.5 text-left text-[13px] transition-colors hover:bg-hover">
            <span className={cn(!chosen && "font-medium")}>{t("No project")}</span>
            <span className="text-[12px] text-ink-3">{t("Works in the default folder, if one is set")}</span>
          </button>
          {projects.map((p) => (
            <button key={p.id} type="button" role="menuitemradio" aria-checked={p.id === value} data-project-option={p.name} onClick={() => pick(p.id)} className="flex w-full flex-col rounded-sm px-2.5 py-1.5 text-left text-[13px] transition-colors hover:bg-hover">
              <span className={cn(p.id === value && "font-medium")}>{p.name}</span>
              <span className="truncate text-[12px] text-ink-3" title={p.folders?.join("\n")}>{p.folders?.length ? baseName(p.folders[0]) + (p.folders.length > 1 ? ` +${p.folders.length - 1}` : "") : t("No folder yet")}</span>
            </button>
          ))}
        </div>, document.body)}
    </div>
  )
}
