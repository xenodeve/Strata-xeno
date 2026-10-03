// The same localStorage keys as the classic app ("strata.<key>", JSON), so a conversation and the settings carry over
// between /classic/ and /next/ (same origin).
export const store = {
  get<T>(k: string, d: T): T {
    try {
      const v = localStorage.getItem("strata." + k)
      return v === null ? d : (JSON.parse(v) as T)
    } catch { return d }
  },
  /** False when the browser refused the write (its storage is full, or a private window): the caller says so, nobody is left thinking it was kept. */
  set(k: string, v: unknown): boolean {
    try { localStorage.setItem("strata." + k, JSON.stringify(v)); return true } catch { return false }
  },
  remove(k: string) {
    try { localStorage.removeItem("strata." + k) } catch { /* nothing to remove */ }
  },
}
