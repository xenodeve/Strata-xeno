// The same localStorage keys as the classic app ("strata.<key>", JSON), so a conversation and the settings carry over
// between /classic/ and /next/ (same origin).
export const store = {
  get<T>(k: string, d: T): T {
    try {
      const v = localStorage.getItem("strata." + k)
      return v === null ? d : (JSON.parse(v) as T)
    } catch { return d }
  },
  set(k: string, v: unknown) {
    try { localStorage.setItem("strata." + k, JSON.stringify(v)) } catch { /* private window: in memory only */ }
  },
}
