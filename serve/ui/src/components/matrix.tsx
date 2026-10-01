// Matrix dot loader (transitions.dev): a 4 x 4 matrix of dots that pulse with a delay of their own; the variant is a delay table (scan: a
// column at a time, twinkle: in no order you can see, orbit: round the ring with the centre held, pulse: the middle first and the rest
// a beat behind). Rounded: the four corners are left out. Changed from the snippet: the dot and the gap follow the size it is shown at.
const CYCLE = 1200                                  // --matrix-cycle
const CORNERS = [0, 3, 12, 15]
const RING = [1, 2, 7, 11, 14, 13, 8, 4]
const INNER = [5, 6, 9, 10]
const TWINKLE = [7, 2, 11, 5, 14, 9, 0, 12, 3, 15, 6, 10, 13, 1, 8, 4]

export type MatrixVariant = "scan" | "twinkle" | "orbit" | "pulse"

/** The delay (ms) of dot `idx`, or null for a dot that does not pulse (the centre of the orbit). */
export function matrixDelay(variant: MatrixVariant, idx: number): number | null {
  if (variant === "scan") return Math.round((idx % 4) * (CYCLE / 10))
  if (variant === "twinkle") return Math.round(TWINKLE[idx] * (CYCLE / 16))
  if (variant === "orbit") { const k = RING.indexOf(idx); return k === -1 ? null : Math.round(k * (CYCLE / 8)) }
  return Math.round((INNER.includes(idx) ? 0 : 1) * (CYCLE * 0.16))
}

export function Matrix({ variant, px, paused = false, rounded = true }: { variant: MatrixVariant; px: number; paused?: boolean; rounded?: boolean }) {
  const dot = Math.max(2, Math.round(px / 9)), gap = Math.max(2, Math.round(px / 12))
  return (
    <span className="t-matrix" data-variant={variant} data-paused={paused ? "" : undefined} aria-hidden style={{ "--matrix-dot": `${dot}px`, "--matrix-gap": `${gap}px` } as React.CSSProperties}>
      {Array.from({ length: 16 }, (_, i) => {
        if (rounded && CORNERS.includes(i)) return <i key={i} className="is-gap" />
        const d = matrixDelay(variant, i)
        return <i key={i} style={d == null ? { animation: "none" } : ({ "--d": d } as React.CSSProperties)} />
      })}
    </span>
  )
}
