import { useRef } from "react"
import { POP_MS, popRuns, staggerOf, trackPace } from "../lib/pop"

/** Text whose numbers pop in when they change (transitions.dev "Number pop-in", see styles.css). Only the characters that
 *  changed come in again: each is its own element, keyed by its place and its digit, so the others stay as they are. What was
 *  there when the text first showed is not animated: a page does not start with every figure popping. */
export function Pop({ text }: { text: string }) {
  const still = useRef<Set<string> | null>(null)
  const clock = useRef<ReturnType<typeof trackPace> | null>(null)
  clock.current = trackPace(clock.current, text, performance.now(), POP_MS)
  const { animate, dur } = clock.current.pace             // a figure that changes often moves for less of the time, or not at all
  const runs = popRuns(text)
  const digits: { key: string; ch: string; stagger: number; run: number }[] = []
  runs.forEach((r, run) => {
    if (!r.num) return
    const chars = [...r.text]
    chars.forEach((ch, j) => {
      const fromEnd = chars.length - 1 - j
      digits.push({ key: `${run}:${fromEnd}:${ch}`, ch, stagger: staggerOf(fromEnd), run })
    })
  })
  const now = new Set(digits.map((d) => d.key))
  still.current = still.current === null ? now : animate ? new Set([...still.current].filter((k) => now.has(k))) : now     // those that stay were not just born; when it is too fast to follow, none is
  const born = still.current
  return (
    <>
      {runs.map((r, i) => !r.num
        ? <span key={`t${i}`}>{r.text}</span>
        : (
          <span key={`n${i}`} className="t-digit-group is-animating" style={{ "--digit-dur": `${dur}ms`, "--digit-stagger": `${Math.round(dur * 0.12)}ms` } as React.CSSProperties}>
            {digits.filter((d) => d.run === i).map((d) => (
              <span key={d.key} className="t-digit" data-still={born.has(d.key) ? "" : undefined} data-stagger={d.stagger || undefined}>{d.ch}</span>
            ))}
          </span>
        ))}
    </>
  )
}
