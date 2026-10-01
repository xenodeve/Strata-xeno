// Number pop-in (transitions.dev): a number that changes comes in again, character by character, with a small blur. Only the
// characters that changed come in; the last two of a number follow a step behind. This is the text cut into runs.
export interface PopRun { num: boolean; text: string }

/** The text as plain runs and number runs ("1,240.5" is one number; a full stop or comma that ends a sentence is not part of it). */
export function popRuns(text: string): PopRun[] {
  const out: PopRun[] = []
  let at = 0
  for (const m of text.matchAll(/\d(?:[\d.,]*\d)?/g)) {
    if (m.index! > at) out.push({ num: false, text: text.slice(at, m.index) })
    out.push({ num: true, text: m[0] })
    at = m.index! + m[0].length
  }
  if (at < text.length) out.push({ num: false, text: text.slice(at) })
  return out
}

/** The stagger step of a character by its place counted from the end of the number (0 = the last): the last is 2, the one before 1. */
export const staggerOf = (fromEnd: number): number => (fromEnd === 0 ? 2 : fromEnd === 1 ? 1 : 0)

/** How many cells a reel turns from digit `from` to digit `to`: up (positive) when the figure grew, down (negative) when it fell,
 *  always the short way round the ten digits in that direction. */
export function reelMove(from: number, to: number, up: boolean): number {
  return up ? (to - from + 10) % 10 : -((from - to + 10) % 10) || 0       // (|| 0: not minus zero)
}

/** How long a figure that changes should take to move, by how long it had been still before this change (`sinceMs`; null: never changed).
 *  A figure that changes often has to stay readable: it moves for a part of the time between its changes, never shorter than 90 ms
 *  (the eye cannot follow less), and when it changes faster than 150 ms it does not move at all - the digits just show. `base` is its
 *  full length, for a figure that changes rarely. */
export function pace(sinceMs: number | null, base: number): { animate: boolean; dur: number } {
  if (sinceMs === null) return { animate: true, dur: base }
  if (sinceMs < 150) return { animate: false, dur: 0 }
  return { animate: true, dur: Math.round(Math.min(base, Math.max(90, sinceMs * 0.4))) }
}
export const POP_MS = 320       // a number's pop-in at full length
export const REEL_MS = 450      // a reel's turn at full length

/** The pace of a text that changes: kept per component, it is worked out when the text becomes another one. */
export function trackPace(prev: { text: string; at: number; pace: { animate: boolean; dur: number } } | null, text: string, now: number, base: number) {
  if (prev === null) return { text, at: now, pace: { animate: true, dur: base } }
  if (prev.text === text) return prev
  return { text, at: now, pace: pace(now - prev.at, base) }
}
