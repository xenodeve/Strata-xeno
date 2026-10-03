import { useEffect, useId, useRef, useState } from "react"
import { cn } from "../lib/cn"
import { t } from "../lib/i18n"
import { useMounted } from "./motion"

/** A question that has to be answered before something is lost: a dialog over a dimmed page. The safe answer (Cancel) has the focus, Escape
 *  and a click outside are that answer, Tab stays inside. It fades and rises in and out. */
export function ConfirmDialog({ open, title, text, confirmLabel, onConfirm, onCancel }: {
  open: boolean; title: string; text: string; confirmLabel: string; onConfirm: () => void; onCancel: () => void
}) {
  const mounted = useMounted(open, 220)
  const [shown, setShown] = useState(false)
  const cancel = useRef<HTMLButtonElement>(null)
  const panel = useRef<HTMLDivElement>(null)
  const was = useRef<Element | null>(null)
  const head = useId()
  const body = useId()
  useEffect(() => {
    if (!open) { setShown(false); return }
    was.current = document.activeElement
    const id = requestAnimationFrame(() => { setShown(true); cancel.current?.focus() })
    const key = (e: KeyboardEvent) => {
      if (e.key === "Escape") { e.preventDefault(); onCancel() }
      else if (e.key === "Tab" && panel.current) {                           // the focus stays in the dialog
        const f = [...panel.current.querySelectorAll<HTMLElement>("button")]
        const first = f[0], last = f[f.length - 1]
        if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus() }
        else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus() }
      }
    }
    document.addEventListener("keydown", key)
    return () => { cancelAnimationFrame(id); document.removeEventListener("keydown", key); (was.current as HTMLElement | null)?.focus?.() }
  }, [open, onCancel])
  if (!mounted) return null
  return (
    <div className="fixed inset-0 z-50 grid place-items-center p-4">
      <div aria-hidden className={cn("absolute inset-0 bg-black/45 transition-opacity duration-200", shown && open ? "opacity-100" : "opacity-0")} onClick={onCancel} />
      <div
        ref={panel}
        role="alertdialog"
        aria-modal="true"
        aria-labelledby={head}
        aria-describedby={body}
        className={cn("relative w-full max-w-[420px] rounded-xl border border-line bg-surface p-5 shadow-[0_24px_64px_-16px_rgb(0_0_0/0.4)] transition-[opacity,transform] duration-200 ease-[var(--ease)]", shown && open ? "translate-y-0 scale-100 opacity-100" : "translate-y-2 scale-[0.98] opacity-0")}
      >
        <h2 id={head} className="text-[16px] font-semibold">{title}</h2>
        <p id={body} className="mt-2 text-[13px] leading-relaxed text-ink-2">{text}</p>
        <div className="mt-5 flex justify-end gap-2">
          <button ref={cancel} type="button" onClick={onCancel} className="h-8 rounded-sm px-3 text-[13px] font-medium text-ink-2 transition-colors hover:bg-hover hover:text-ink focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--accent)]">{t("Cancel")}</button>
          <button type="button" onClick={onConfirm} className="h-8 rounded-sm bg-bad px-3 text-[13px] font-medium text-white transition-opacity hover:opacity-90 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--accent)]">{confirmLabel}</button>
        </div>
      </div>
    </div>
  )
}
