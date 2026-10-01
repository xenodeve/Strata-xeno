import { useEffect, useRef, useState, type ReactNode } from "react"
import { ThinkingOrb } from "../vendor/thinking-orbs/orb"
import type { OrbDesign } from "../lib/orbs"

// thinking-orbs by Jakub Antalik (Libraries.dev, MIT), vendored in src/vendor/thinking-orbs. The orb pauses itself offscreen and
// in a hidden tab, and is a still frame under reduced motion. These three components only decide where and how it is shown.
type Size = 64 | 32 | 20

/** An orb in a fixed slot. When the design changes the old one fades out while the new one fades in (250 ms), so a state
 *  change reads as one orb changing its mind, not two orbs swapped. */
export function Orb({ design, size = 20, moving = true, label }: { design: OrbDesign; size?: Size; moving?: boolean; label?: string }) {
  const [shown, setShown] = useState(design)
  const [leaving, setLeaving] = useState<OrbDesign | null>(null)
  const was = useRef(design)
  useEffect(() => {
    if (design === was.current) return
    setLeaving(was.current)
    setShown(design)
    was.current = design
    const t = setTimeout(() => setLeaving(null), 260)
    return () => clearTimeout(t)
  }, [design])
  return (
    <span className="orb-slot" style={{ width: size, height: size }}>
      {leaving && <span className="orb-layer orb-out" aria-hidden><ThinkingOrb state={leaving} size={size} paused={!moving} /></span>}
      <span key={shown} className="orb-layer orb-in"><ThinkingOrb state={shown} size={size} paused={!moving} aria-label={label} /></span>
    </span>
  )
}

/** An orb beside a status line whose text shimmers: the "something is happening" row. */
export function StatusLabel({ design, children, size = 20, moving = true, className = "" }: { design: OrbDesign; children: ReactNode; size?: Size; moving?: boolean; className?: string }) {
  return (
    <span className={`inline-flex items-center gap-2 ${className}`}>
      <Orb design={design} size={size} moving={moving} />
      <span className={moving ? "t-shimmer" : ""}>{children}</span>
    </span>
  )
}

/** A page or a block that is waiting for data: the orb and what it waits for. An error shows as plain text, not as a wait. */
export function Loading({ children = "Connecting…", error, design = "connecting" }: { children?: ReactNode; error?: string | null; design?: OrbDesign }) {
  if (error) return <p className="text-ink-2">{error}</p>
  return <p className="py-6 text-ink-2" role="status"><StatusLabel design={design}>{children}</StatusLabel></p>
}
