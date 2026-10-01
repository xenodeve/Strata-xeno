import { useEffect, useId, useRef, useState, type ReactNode } from "react"
import { HugeiconsIcon } from "@hugeicons/react"
import { ArrowDown01Icon } from "@hugeicons/core-free-icons"
import { cn } from "../lib/cn"

// Motion for things that open, close, appear and disappear. Height is animated through grid rows (0fr <-> 1fr), so it
// is exact for any content and needs no measuring; opacity rides along. One ease and one duration scale, everywhere.
const MS = 320

/** Mounted while open and for the length of the closing animation: a body that is expensive to build (a 20,000
 *  character tool result) exists only while it is visible. */
function useMounted(open: boolean) {
  const [mounted, setMounted] = useState(open)
  useEffect(() => {
    if (open) { setMounted(true); return }
    const t = setTimeout(() => setMounted(false), MS)
    return () => clearTimeout(t)
  }, [open])
  return open || mounted
}

export function Collapse({ open, children, className }: { open: boolean; children: ReactNode; className?: string }) {
  const mounted = useMounted(open)
  const [shown, setShown] = useState(false)          // one frame after mounting, so the opening has a "from" to animate from
  useEffect(() => {
    if (!open) { setShown(false); return }
    const id = requestAnimationFrame(() => setShown(true))
    return () => cancelAnimationFrame(id)
  }, [open])
  if (!mounted) return null
  return (
    <div className={cn("collapse-grid", shown && open && "is-open", className)}>
      <div className="min-h-0 overflow-hidden" inert={!open}>{children}</div>
    </div>
  )
}

/** A part that comes and goes with the data: `children` null closes it, and it keeps showing what it last showed
 *  while it closes, so it shrinks away rather than vanishing. */
export function Reveal({ children, className }: { children: ReactNode | null; className?: string }) {
  const last = useRef<ReactNode>(children)
  if (children != null) last.current = children
  return <Collapse open={children != null} className={className}>{last.current}</Collapse>
}

/** A heading that opens its body: collapsed until asked. */
export function Disclosure({ title, hint, children, defaultOpen = false, block = false }: { title: ReactNode; hint?: ReactNode; children: ReactNode; defaultOpen?: boolean; block?: boolean }) {
  const [open, setOpen] = useState(defaultOpen)
  const id = useId()
  return (
    <div>
      <button
        type="button"
        aria-expanded={open}
        aria-controls={id}
        onClick={() => setOpen(!open)}
        className={cn("group flex w-full items-center justify-between gap-3 rounded-sm py-1.5 text-left transition-colors hover:text-ink", block ? "text-[15px]" : "text-[13px]")}
      >
        <span className="min-w-0"><span className={block ? "font-semibold tracking-[-0.015em]" : "font-medium"}>{title}</span>{hint && <span className="ml-2.5 text-[12px] font-normal text-ink-3">{hint}</span>}</span>
        <HugeiconsIcon icon={ArrowDown01Icon} size={14} aria-hidden className={cn("shrink-0 text-ink-3 transition-transform duration-300 ease-[var(--ease)]", open && "rotate-180")} />
      </button>
      <div id={id}><Collapse open={open}>{children}</Collapse></div>
    </div>
  )
}

/** A label that changes: the new one sharpens in from a blur. It animates when `k` changes (a state), not on every
 *  edit of the text, so a counter inside the label does not flicker. */
export function Swap({ k, children }: { k: string; children: ReactNode }) {
  return <span key={k} className="swap-in inline-block">{children}</span>
}
