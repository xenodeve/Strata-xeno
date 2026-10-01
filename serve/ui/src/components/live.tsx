import { useEffect, useRef, useState } from "react"
import { serverDesign } from "../lib/orbs"
import { Orb } from "./orb"
import type { Live } from "../lib/metrics"

const reduce = () => typeof matchMedia !== "undefined" && matchMedia("(prefers-reduced-motion: reduce)").matches

/** The server's state as a dotted orb: still while idle, breathing while a prompt is read, flowing while it writes.
 *  (vendor/thinking-orbs, MIT: it pauses itself offscreen and in a hidden tab, and is a still frame under reduced motion.) */
export function StatusOrb({ state, size = 64 }: { state: Live["state"]; size?: 64 | 32 | 20 }) {
  const { design, moving } = serverDesign(state)
  return (
    <Orb
      design={design}
      size={size}
      moving={moving}
      label={state === "reading" ? "Reading the prompt" : state === "generating" ? "Writing" : state === "unloaded" ? "Model unloaded" : "Idle"}
    />
  )
}

/** A figure that glides to its new value (cubic ease-out, 500 ms) instead of jumping. Only moves when the value changes. */
export function Num({ value, digits = 0, className }: { value: number | null | undefined; digits?: number; className?: string }) {
  const [shown, setShown] = useState<number | null>(value ?? null)
  const at = useRef<number | null>(value ?? null)
  useEffect(() => {
    if (value == null || at.current == null || reduce()) { at.current = value ?? null; setShown(value ?? null); return }
    const from = at.current, t0 = performance.now()
    let raf = 0
    const tick = (t: number) => {
      const p = Math.min(1, (t - t0) / 500), v = from + (value - from) * (1 - (1 - p) ** 3)
      at.current = v; setShown(v)
      if (p < 1) raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [value])
  return <span className={className ?? "num"}>{shown == null ? "–" : shown.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits })}</span>
}
