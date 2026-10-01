// The app's two languages. The English text is the key: `t("Show the thinking")` is that text in English, and its Thai is the
// entry with the same text in src/i18n/th*.ts. A missing Thai entry shows the English (never a blank), and a test fails when a
// `t(...)` string in the source has no entry (so "complete" is checked, not hoped for).
//   t("text {name}", { name })   {name} is replaced, in either language
//   tn(n, "{n} token", "{n} tokens")   the form for n (English has a plural, Thai has not: both keys get a Thai entry)
//   msg("text")   marks text kept in a constant for the test to find; show it with t(variable) when it is rendered
import { useSyncExternalStore } from "react"
import { store } from "./store"
import { th } from "../i18n/th"

export type Lang = "en" | "th"

const guess = (): Lang => (typeof navigator !== "undefined" && /^th\b/i.test(navigator.language || "") ? "th" : "en")
let lang: Lang = (() => { const v = store.get<string>("lang", ""); return v === "th" || v === "en" ? v : guess() })()
const listeners = new Set<() => void>()

export const getLang = () => lang
export function setLang(l: Lang) {
  lang = l
  store.set("lang", l)
  if (typeof document !== "undefined") document.documentElement.lang = l
  listeners.forEach((f) => f())
}
/** Sets the page's lang attribute for the language in use (at start); the app re-renders from `useLang`. */
export function initLang() { if (typeof document !== "undefined") document.documentElement.lang = lang }
export function useLang(): Lang { return useSyncExternalStore((cb) => { listeners.add(cb); return () => { listeners.delete(cb) } }, getLang) }

export function t(key: string, vars?: Record<string, string | number>): string {
  const s = lang === "th" ? th[key] ?? key : key
  return vars ? s.replace(/\{(\w+)\}/g, (m, k: string) => (k in vars ? String(vars[k]) : m)) : s
}
export const tn = (n: number, one: string, other: string, vars?: Record<string, string | number>) => t(n === 1 ? one : other, { ...vars, n })
export const msg = (s: string) => s
