import { useLayoutEffect, useRef } from "react"
import { REEL_MS, popRuns, reelMove, trackPace } from "../lib/pop"

// Spinning counter (transitions.dev): every digit is a clipped reel of 0-9 that turns to its new digit, up when the figure grew and
// down when it fell, with a vertical streak while it moves. For figures that go up and down (a speed, a load, a temperature); the
// ones that only grow use Pop. The reel here is one text node of fifty lines (five turns of 0-9) so a page of figures stays light.
const REEL = Array.from({ length: 50 }, (_, i) => i % 10).join("\n")

/** One digit. It rests in the middle turn, moves by the shortest way in the direction asked, and is put back in the middle (without
 *  a transition) when it has arrived, so it never runs out of strip. What it shows at the start is not animated. */
function Reel({ digit, up, delay, dur, animate }: { digit: number; up: boolean; delay: number; dur: number; animate: boolean }) {
  const strip = useRef<HTMLSpanElement>(null)
  const at = useRef(20 + digit)
  const born = useRef(true)
  useLayoutEffect(() => {
    const el = strip.current!
    if (born.current) { born.current = false; el.style.setProperty("--pos", String(at.current)); return }
    const move = reelMove(at.current % 10, digit, up)
    if (move === 0) return
    if (!animate) {                                  // it changes too fast to follow: the digit just shows
      at.current = 20 + digit
      delete el.dataset.moving
      el.style.transition = "none"
      el.style.transitionDelay = "0ms"
      el.style.setProperty("--pos", String(at.current))
      void el.offsetWidth
      el.style.transition = ""
      return
    }
    at.current = Math.min(44, Math.max(5, at.current + move))
    el.dataset.moving = ""
    el.style.transitionDuration = `${dur}ms`
    el.style.transitionDelay = `${delay}ms`
    el.style.setProperty("--pos", String(at.current))
  }, [digit])
  const arrived = (e: React.TransitionEvent<HTMLSpanElement>) => {
    if (e.propertyName !== "transform") return
    const el = e.currentTarget
    at.current = 20 + digit
    delete el.dataset.moving
    el.style.transition = "none"
    el.style.transitionDelay = "0ms"
    el.style.setProperty("--pos", String(at.current))
    void el.offsetWidth
    el.style.transition = ""
  }
  return <span className="t-reel-col"><span ref={strip} className="t-reel-strip" onTransitionEnd={arrived}>{REEL}</span></span>
}

/** Text whose numbers are reels. */
export function Spin({ text }: { text: string }) {
  const last = useRef<Map<number, number>>(new Map())
  const clock = useRef<ReturnType<typeof trackPace> | null>(null)
  clock.current = trackPace(clock.current, text, performance.now(), REEL_MS)
  const { animate, dur } = clock.current.pace             // a figure that changes often turns for less of the time, or not at all
  const runs = popRuns(text)
  const dir: Record<number, boolean> = {}
  runs.forEach((r, i) => {
    if (!r.num) return
    const n = parseFloat(r.text.replace(/,/g, ""))
    const was = last.current.get(i)
    dir[i] = was === undefined || n >= was
    last.current.set(i, n)
  })
  return (
    <>
      <span className="sr-only">{text}</span>
      <span aria-hidden>
        {runs.map((r, i) => !r.num
          ? <span key={`t${i}`}>{r.text}</span>
          : (
            <span key={`n${i}`} className="t-reel">
              {[...r.text].map((ch, j) => /\d/.test(ch)
                ? <Reel key={`${[...r.text].length - 1 - j}`} digit={+ch} up={dir[i]} delay={Math.round(j * dur * 0.08)} dur={dur} animate={animate} />
                : <span key={`${[...r.text].length - 1 - j}c`} className="t-reel-sep">{ch}</span>)}
            </span>
          ))}
      </span>
    </>
  )
}

/** The streak filter the reels use while they move: blur along the vertical only (a CSS blur() would smear sideways). */
export function ReelFilter() {
  return <svg width="0" height="0" aria-hidden style={{ position: "absolute" }}><filter id="t-reel-blur"><feGaussianBlur stdDeviation="0 2.5" /></filter></svg>
}
