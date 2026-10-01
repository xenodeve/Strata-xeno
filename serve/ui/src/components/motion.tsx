import { useEffect, useId, useLayoutEffect, useRef, useState, type ReactNode } from "react"
import { HugeiconsIcon } from "@hugeicons/react"
import { ArrowDown01Icon } from "@hugeicons/core-free-icons"
import { cn } from "../lib/cn"

// Motion for things that open, close, appear and disappear. Height is animated through grid rows (0fr <-> 1fr), so it
// is exact for any content and needs no measuring; opacity rides along. One ease and one duration scale, everywhere.
const MS = 320
export const SOFT_MS = 520

/** Mounted while open and for the length of the closing animation: a body that is expensive to build (a 20,000
 *  character tool result) exists only while it is visible. */
export function useMounted(open: boolean, ms: number) {
  const [mounted, setMounted] = useState(open)
  useEffect(() => {
    if (open) { setMounted(true); return }
    const t = setTimeout(() => setMounted(false), ms)
    return () => clearTimeout(t)
  }, [open])
  return open || mounted
}

/** `instant`: when it is open at the start it is simply there (no opening), and it still closes by closing up. `soft`: a slower, even
 *  stretch and shrink (for a section that is a good part of a page or a panel). */
export function Collapse({ open, children, className, instant = false, soft = false }: { open: boolean; children: ReactNode; className?: string; instant?: boolean; soft?: boolean }) {
  const mounted = useMounted(open, soft ? SOFT_MS + 40 : MS)
  const [shown, setShown] = useState(instant && open)          // one frame after mounting, so the opening has a "from" to animate from
  useEffect(() => {
    if (!open) { setShown(false); return }
    const id = requestAnimationFrame(() => setShown(true))
    return () => cancelAnimationFrame(id)
  }, [open])
  if (!mounted) return null
  return (
    <div className={cn("collapse-grid", soft && "collapse-soft", shown && open && "is-open", className)}>
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

/** A box (full width) that takes the height of what is in it, and glides there when that changes: a bubble becomes an edit box and the
 *  box stretches open, the box closes and it shrinks back. A little padding (offset by a negative margin) keeps a focus ring
 *  from being clipped by the overflow it needs. */
export function Fit({ children, className }: { children: ReactNode; className?: string }) {
  const inner = useRef<HTMLDivElement>(null)
  const [h, setH] = useState<number | null>(null)       // unknown on the first paint: the box is as tall as it is, with no glide
  useLayoutEffect(() => {
    const el = inner.current
    if (!el) return
    const measure = () => setH(el.offsetHeight)
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  return (
    <div className="fit-box -m-1 w-[calc(100%+0.5rem)] overflow-hidden p-1" style={h == null ? undefined : { height: h + 8 }}>
      <div ref={inner} className={className}>{children}</div>
    </div>
  )
}

/** A scrolling panel whose height glides to the height of what is in it, up to a cap (then it scrolls): a section that opens inside it
 *  stretches it open, instead of the panel jumping to its largest size at once. The scrollbar's room is kept, so nothing shifts sideways
 *  when the scrollbar appears. The padding goes on `inner`: the height is measured inside it. */
export function GlidePanel({ children, className, inner = "", cap, ...rest }: { children: ReactNode; className?: string; inner?: string; cap: () => number } & React.HTMLAttributes<HTMLDivElement>) {
  const body = useRef<HTMLDivElement>(null)
  const [h, setH] = useState<number | null>(null)
  useLayoutEffect(() => {
    const el = body.current
    if (!el) return
    const measure = () => setH(Math.min(el.offsetHeight, cap()))
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    addEventListener("resize", measure)
    return () => { ro.disconnect(); removeEventListener("resize", measure) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
  return (
    <div {...rest} className={cn("glide-panel overflow-y-auto [scrollbar-gutter:stable]", className)} style={h == null ? undefined : { height: h }}>
      <div ref={body} className={inner}>{children}</div>
    </div>
  )
}

/** A label that changes: the new one sharpens in from a blur. It animates when `k` changes (a state), not on every
 *  edit of the text, so a counter inside the label does not flicker. */
export function Swap({ k, children }: { k: string; children: ReactNode }) {
  return <span key={k} className="swap-in inline-block">{children}</span>
}

/** A live status that gives way to what it becomes (the speed under an answer once it is written): the figures arrive from a little
 *  below while the status keeps its place, fading, blurring and rising away. `live` null ends the status; it is unmounted when the
 *  fade is done. The status is out of the flow, so what follows the figures does not move when it goes. */
export function Handover({ live, children }: { live: ReactNode | null; children: ReactNode }) {
  const last = useRef<ReactNode>(live)
  if (live != null) last.current = live
  const mounted = useMounted(live != null, 460)
  return (
    <span className="handover">
      <span className="handover-in" data-show={live == null ? "" : undefined}>{children}</span>
      {mounted && <span className="handover-out" data-show={live != null ? "" : undefined}>{last.current}</span>}
    </span>
  )
}
