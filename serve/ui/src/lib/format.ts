export const fmt = (n: number | null | undefined, d = 0) =>
  n == null || Number.isNaN(n) ? "–" : Number(n).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d })
export const kfmt = (n: number | null | undefined) =>
  n == null ? "–" : n >= 1000 ? `${fmt(n / 1000, n >= 10000 ? 0 : 1)}k` : fmt(n)
export const gb = (b: number | null | undefined, d = 1) => (b == null ? "–" : fmt(b / 1073741824, d))
export const timeStr = (t: number) => new Date(t).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
