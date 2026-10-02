// The tokens the model has written so far in the answer that is being made, counted live (issue #99 follow-up): an agent's answer is many rounds - think, call a tool, think again - and the
// server counts the tokens of one round at a time, so the count restarts with each round. This adds the rounds up, and between two readings of the server it moves on by the speed it reports,
// so that the number keeps running and the user can see that the agent is still at work. It is only a display: the exact total is what the answer's own line gives when it ends.

/** What the server's live status says (the part used here). */
export interface LiveReading { state?: string; generated?: number | null; tok_s?: number | null }

const AHEAD_MS = 1500          // how long the number may run on after its last reading, by the speed

export class LiveTokens {
  private key: unknown = null
  private base = 0                // tokens of the rounds that are over
  private last = 0                // tokens of the round that is running, at its last reading
  private at = 0                  // when that reading came
  private rate = 0                // tokens a second at that reading

  /** A reading of the server's live status for the answer `key` (the message being written; null when none is). */
  sample(key: unknown, live: LiveReading, now: number): void {
    if (key !== this.key) { this.key = key; this.base = 0; this.last = 0; this.rate = 0; this.at = now }
    if (key === null) return
    if (live.state !== "generating") {                       // reading the prompt, or between two rounds: the round that was running is over
      if (this.last > 0) { this.base += this.last; this.last = 0 }
      this.rate = 0
      return
    }
    const g = typeof live.generated === "number" && live.generated >= 0 ? live.generated : null
    if (g === null) return
    if (g < this.last) { this.base += this.last }            // the count started again without a reading in between: a new round
    this.last = g
    this.at = now
    this.rate = typeof live.tok_s === "number" && live.tok_s > 0 ? live.tok_s : 0
  }

  /** What to show at `now`, or null before anything has been written. */
  shown(now: number): { tokens: number; tokS: number | null } | null {
    const ahead = Math.min(Math.max(0, now - this.at), AHEAD_MS) / 1000
    const tokens = Math.floor(this.base + this.last + (this.rate > 0 ? this.rate * ahead : 0))
    return tokens > 0 ? { tokens, tokS: this.rate > 0 ? this.rate : null } : null
  }
}
