import { useSyncExternalStore } from "react"
import { Orb, tint } from "./orb"
import { SLOW } from "../lib/orbs"

export type ToastKind = "info" | "success" | "warn" | "error"
interface Toast { id: number; kind: ToastKind; title: string; text: string; action?: { label: string; run: () => void } }

let items: Toast[] = []
let next = 1
const listeners = new Set<() => void>()
const emit = () => { items = [...items]; listeners.forEach((l) => l()) }

export function dismiss(id: number) {
  items = items.filter((t) => t.id !== id)
  emit()
}

export function toast(kind: ToastKind, title: string, text = "", ms = 3500, action?: Toast["action"]) {
  const id = next++
  items = [...items, { id, kind, title, text, action }]
  emit()
  setTimeout(() => dismiss(id), ms)
}

const TINT: Record<ToastKind, string | null> = { info: null, success: "--ok", warn: null, error: "--bad" }

export function ToastHost() {
  const list = useSyncExternalStore((cb) => { listeners.add(cb); return () => { listeners.delete(cb) } }, () => items)
  return (
    <div role="status" aria-live="polite" className="pointer-events-none fixed inset-x-4 bottom-4 z-50 flex flex-col items-center gap-2 sm:items-end">
      {list.map((t) => (
        <div key={t.id} className="toast-in pointer-events-auto flex w-full max-w-sm items-start gap-3 rounded-md border border-line bg-surface px-3.5 py-3 shadow-[0_8px_30px_rgb(0_0_0/0.10)]">
          <span aria-hidden className="-ml-1 -mt-0.5 shrink-0"><Orb design={t.kind === "error" ? "shaping" : "breathing"} size={20} speed={SLOW.speed} fps={SLOW.fps} color={TINT[t.kind] ? tint(TINT[t.kind]!) : undefined} /></span>
          <div className="min-w-0 flex-1">
            <div className="text-[13px] font-medium">{t.title}</div>
            {t.text && <div className="mt-0.5 text-[13px] text-ink-2 [overflow-wrap:anywhere]">{t.text}</div>}
          </div>
          {t.action && (
            <button
              type="button"
              onClick={() => { t.action!.run(); dismiss(t.id) }}
              className="shrink-0 rounded-sm px-2 py-1 text-[13px] font-medium text-accent transition-colors hover:bg-hover"
            >
              {t.action.label}
            </button>
          )}
        </div>
      ))}
    </div>
  )
}
