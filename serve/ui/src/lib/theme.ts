import { useCallback, useState } from "react"

export type Theme = "system" | "light" | "dark"
const KEY = "strata-theme"

const read = (): Theme => {
  try {
    const v = localStorage.getItem(KEY)
    return v === "light" || v === "dark" ? v : "system"
  } catch { return "system" }
}

function apply(t: Theme) {
  if (t === "system") delete document.documentElement.dataset.theme
  else document.documentElement.dataset.theme = t
  try { t === "system" ? localStorage.removeItem(KEY) : localStorage.setItem(KEY, t) } catch { /* private window */ }
}

export function useTheme() {
  const [theme, set] = useState<Theme>(read)
  const cycle = useCallback(() => {
    const next: Theme = theme === "system" ? "light" : theme === "light" ? "dark" : "system"
    apply(next)
    set(next)
  }, [theme])
  return { theme, cycle }
}
