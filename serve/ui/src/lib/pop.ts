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
