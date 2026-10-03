export const fmt = (n: number | null | undefined, d = 0) =>
  n == null || Number.isNaN(n) ? "–" : Number(n).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d })
export const kfmt = (n: number | null | undefined) =>
  n == null ? "–" : n >= 1000 ? `${fmt(n / 1000, n >= 10000 ? 0 : 1)}k` : fmt(n)
export const gb = (b: number | null | undefined, d = 1) => (b == null ? "–" : fmt(b / 1073741824, d))
export const timeStr = (t: number) => new Date(t).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })

/** A User-Agent as the name a person knows (the browser, curl, an agent), not the string. */
export function clientName(ua: string): string {
  const u = (ua || "").trim()
  if (!u) return ""
  const known: [RegExp, string][] = [
    [/claude[-_ ]?(cli|code)/i, "Claude Code"], [/open-?webui/i, "Open WebUI"], [/^curl\//i, "curl"],
    [/Edg(e|A|iOS)?\//, "Edge"], [/OPR\//, "Opera"], [/Firefox\//, "Firefox"], [/Chrome\//, "Chrome"], [/Version\/[\d.]+.*Safari\//, "Safari"],
  ]
  for (const [re, name] of known) if (re.test(u)) return name
  const first = u.split(/[\s/]/)[0]
  return first.slice(0, 24)
}
