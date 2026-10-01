// The speed at which a prompt is read (prefill), shown under the prompt that was sent: the mean over the last second while it
// is read, then the mean of the whole read once it is done. The tokens the conversation cache already held are never part
// of a speed (no compute) and are never added to the tokens read; they are named beside them.
import { fmt } from "./format"

export interface Prefill {
  state: "reading" | "done"
  rate: number | null            // tokens per second over the last second (while reading)
  mean: number | null            // tokens per second over the whole read (when done)
  read: number | null            // tokens the engine read
  cached: number | null          // tokens the conversation cache held
}

const WINDOW_MS = 1000

/** Samples of "how far the engine has read" (a position that counts a reused prefix as read, so the first one is only a
 *  baseline) and the speeds they give. */
export class PrefillMeter {
  private t: number[] = []
  private r: number[] = []

  push(ms: number, read: number) { this.t.push(ms); this.r.push(read) }

  /** Tokens per second over the last second or a little more (from the newest sample back to the newest one that is a full
   *  second older; before a second has passed, from the first). Chunks can arrive less often than that, so when nothing
   *  moved inside the window it reaches back to the sample before the last chunk: the speed does not fall to zero
   *  between chunks. */
  rate(): number | null {
    const n = this.t.length
    if (n < 2) return null
    const t1 = this.t[n - 1], r1 = this.r[n - 1]
    let j = 0
    for (let i = n - 2; i >= 0; i--) if (this.t[i] <= t1 - WINDOW_MS) { j = i; break }
    while (j >= 0 && this.r[j] >= r1) j--
    if (j < 0) return null
    const dt = t1 - this.t[j]
    return dt > 0 ? ((r1 - this.r[j]) / dt) * 1000 : null
  }

  /** Tokens per second over everything seen after the baseline. */
  mean(): number | null {
    const n = this.t.length
    if (n < 2) return null
    const dt = this.t[n - 1] - this.t[0], dr = this.r[n - 1] - this.r[0]
    return dt > 0 && dr > 0 ? (dr / dt) * 1000 : null
  }
}

/** The line under the prompt; null when there is nothing to say. */
export function prefillText(p: Prefill): string | null {
  if (p.state === "reading") return p.rate != null ? `Reading the prompt · ${fmt(p.rate)} tok/s · last second` : "Reading the prompt…"
  const cached = p.cached ?? 0
  if (p.read === 0 && cached > 0) return `Nothing to read: all ${fmt(cached)} tokens came from the cache`
  if (p.read == null) return p.mean != null ? `Prefill ${fmt(p.mean)} tok/s mean` : null
  if (p.read <= 0) return null
  const tail = `${fmt(p.read)} tokens read${cached > 0 ? ` · ${fmt(cached)} from the cache` : ""}`
  return p.mean != null ? `Prefill ${fmt(p.mean)} tok/s mean · ${tail}` : tail
}
