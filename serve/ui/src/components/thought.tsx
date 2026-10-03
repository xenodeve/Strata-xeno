import { useLayoutEffect, useRef, type ReactNode } from "react"
import { HugeiconsIcon } from "@hugeicons/react"
import { ArrowDown01Icon } from "@hugeicons/core-free-icons"
import { Collapse } from "./motion"
import type { LatticePattern } from "../lib/orbs"
import { cn } from "../lib/cn"
import { t } from "../lib/i18n"

// Two React Bits components, adapted and used as a pair for the agent's thinking (MIT + Commons Clause: in this app only):
// LatticeLoader (a 3 x 3 grid of dots that runs a pattern while the agent works, then settles into a tick or a cross) and
// ThoughtLine (a line whose label shimmers while it thinks, then settles into "Thought for 2.4s", and opens to the thought).
// Changed from the originals: no `motion` package (the breathing is CSS), the lattice has no label or timer of its own (the
// line owns both), the line's open state is the caller's, and its trace is any content (here: the model's reasoning text).
// The 3 x 3 patterns below are the originals' (cells = the order they light up in, loop = cycles per loop, scale = pace).
const PATTERNS: Record<LatticePattern, { cells: number[]; loop: number; scale: number; lit?: number }> = {
  orbit: { cells: [0, 1, 2, 7, -1, 3, 6, 5, 4], loop: 8, scale: 1.2 },
  ripple: { cells: [2, 1, 2, 1, 0, 1, 2, 1, 2], loop: 4.8, scale: 1.5 },
  snake: { cells: [0, 1, 2, 5, 4, 3, 6, 7, 8], loop: 9, scale: 1, lit: 35 },
  spiral: { cells: [0, 1, 2, 7, 8, 3, 6, 5, 4], loop: 9, scale: 1.2, lit: 35 },
  arrow: { cells: [1, 2, 3, 0, 1, 2, 1, 2, 3], loop: 7.2, scale: 1 },
  dots: { cells: [0, 1, 2, 0, 1, 2, 0, 1, 2], loop: 3, scale: 2.4 },
}
const MARKS = { done: [2, 3, 5, 7], error: [0, 2, 4, 6, 8] }     // the cells that make a tick, and a cross
const STEP = 90                                                   // ms between one cell and the next

export type LatticeStatus = "working" | "done" | "error"

export function Lattice({ status, pattern = "orbit", cell, gap, paused = false }: { status: LatticeStatus; pattern?: LatticePattern; cell?: number; gap?: number; paused?: boolean }) {
  const p = PATTERNS[pattern]
  const d = STEP * p.scale
  const marks = MARKS[status === "error" ? "error" : "done"]
  return (
    <span className="lat" data-status={status} data-paused={paused ? "" : undefined} style={{ "--lat-cycle": `${Math.round(p.loop * d)}ms`, ...(cell ? { "--lat-cell": `${cell}px` } : null), ...(gap ? { "--lat-gap": `${gap}px` } : null) } as React.CSSProperties} aria-hidden>
      <span className="lat-layer lat-run">
        {p.cells.map((u, i) => (
          <span key={i} className="lat-cell" data-hole={u < 0 ? "" : undefined} data-lit={p.lit} style={u < 0 ? undefined : { animationDelay: `${Math.round(u * d)}ms` }} />
        ))}
      </span>
      <span className="lat-layer lat-mark">
        {p.cells.map((_, i) => <span key={i} className="lat-cell" data-on={marks.includes(i) ? "" : undefined} />)}
      </span>
    </span>
  )
}

const clock = (ds: number) => (ds < 600 ? `${(ds / 10).toFixed(1)}s` : `${Math.floor(ds / 600)}m ${((ds % 600) / 10).toFixed(1)}s`)

/** The thinking line. While `working` the label shimmers and a timer counts; when it stops the label becomes `doneLabel`
 *  (with the timer, "Thought for 2.4s", when the time is known) and the timer glides to the end of the shorter text.
 *  `elapsed` (seconds) fixes the timer to a known length. The thought itself is `children`, shown while `open`. */
export function Thought({ working, glyph, open, onToggle, elapsed, label, since, children }: {
  working: boolean; glyph: ReactNode; open: boolean; onToggle: () => void; elapsed?: number | null; label?: string; since?: number; children: ReactNode
}) {
  const timed = working || elapsed != null
  const doneText = timed ? t("Thought for") : t("Thoughts")
  const workText = label ?? t("Thinking…")
  const timerRef = useRef<HTMLSpanElement>(null)
  const stackRef = useRef<HTMLSpanElement>(null)
  const workRef = useRef<HTMLSpanElement>(null)
  const doneRef = useRef<HTMLSpanElement>(null)
  const dsRef = useRef(0)
  const was = useRef(working)

  const paint = (ds: number) => { dsRef.current = ds; if (timerRef.current) timerRef.current.textContent = clock(ds) }
  useLayoutEffect(() => {
    if (elapsed != null) { paint(Math.round(elapsed * 10)); return }
    if (!working) return
    const t0 = performance.now() - (since ? Math.max(0, Date.now() - since) : 0)           // counted from when the thinking began, not from when this block was drawn: it goes on when the page is left and opened again
    paint(Math.floor((performance.now() - t0) / 100))
    const id = setInterval(() => paint(Math.floor((performance.now() - t0) / 100)), 100)
    return () => clearInterval(id)
  }, [working, elapsed, since])

  // The timer sits after the wider of the two labels (they are stacked in one grid cell); shift it left by the difference
  // so it follows the one that shows. On a change it glides (a CSS transition); when only the text resizes it jumps.
  useLayoutEffect(() => {
    const t = timerRef.current, stack = stackRef.current
    if (!t || !stack) return
    const place = (glide: boolean) => {
      const active = working ? workRef.current : doneRef.current
      if (!active) return
      if (!glide) t.style.transition = "none"
      t.style.transform = `translateX(${active.offsetWidth - stack.offsetWidth}px)`
      if (!glide) { void t.offsetWidth; t.style.transition = "" }
    }
    place(was.current !== working)
    was.current = working
    const ro = new ResizeObserver(() => place(false))
    if (workRef.current) ro.observe(workRef.current)
    if (doneRef.current) ro.observe(doneRef.current)
    return () => ro.disconnect()
  }, [working, workText, doneText, timed])

  return (
    <div className="thought" data-working={working ? "" : undefined} data-open={open ? "" : undefined}>
      <button type="button" className="thought-head" aria-expanded={open} onClick={onToggle}>
        <span className="thought-glyph" aria-hidden>{glyph}</span>
        <span ref={stackRef} className="thought-label" aria-hidden>
          <span ref={workRef} className="thought-text" data-active={working ? "" : undefined}><span className="thought-shimmer">{workText}</span></span>
          <span ref={doneRef} className="thought-text thought-text--done" data-active={working ? undefined : ""}>{doneText}</span>
        </span>
        {timed && <span ref={timerRef} className="thought-timer num" data-done={working ? undefined : ""} aria-hidden>0.0s</span>}
        <HugeiconsIcon icon={ArrowDown01Icon} size={14} aria-hidden className={cn("thought-chevron", open && "rotate-180")} />
        <span className="sr-only" role="status">{working ? workText : `${doneText}${timed ? ` ${clock(dsRef.current)}` : ""}`}</span>
      </button>
      <Collapse open={open}>{children}</Collapse>
    </div>
  )
}
