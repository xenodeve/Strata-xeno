import type { ReactNode } from "react"
import { cn } from "../lib/cn"

export const NOT_MEASURED = <span className="text-ink-3">not measured</span>

/** A figure, or "not measured": a tile with no instrument behind it never shows a guess. */
export function val(v: number | null | undefined, fmt: (n: number) => string, unit = ""): ReactNode {
  return v == null || Number.isNaN(v) ? NOT_MEASURED : <span className="num">{fmt(v)}{unit && <span className="text-ink-2"> {unit}</span>}</span>
}

export function Section({ title, children, aside }: { title: string; children: ReactNode; aside?: ReactNode }) {
  return (
    <section className="mt-8 first:mt-0">
      <div className="flex items-baseline justify-between gap-4">
        <h2 className="text-[15px] font-semibold">{title}</h2>
        {aside && <div className="text-[12px] text-ink-2">{aside}</div>}
      </div>
      <div className="mt-2">{children}</div>
    </section>
  )
}

export function Rows({ children }: { children: ReactNode }) {
  return <dl className="m-0">{children}</dl>
}

export function Row({ k, v, hint }: { k: ReactNode; v: ReactNode; hint?: ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-6 border-b border-line py-2 last:border-0">
      <dt className="text-ink-2">{k}{hint && <span className="ml-2 text-[12px] text-ink-3">{hint}</span>}</dt>
      <dd className="m-0 text-right [overflow-wrap:anywhere]">{v}</dd>
    </div>
  )
}

const DOT = { ok: "bg-ok", busy: "bg-accent", idle: "bg-ink-3", bad: "bg-bad" } as const
/** A status is a small glyph beside a label, never a banner. */
export function Dot({ tone, pulse }: { tone: keyof typeof DOT; pulse?: boolean }) {
  return <span aria-hidden className={cn("inline-block size-2 shrink-0 rounded-full", DOT[tone], pulse && "pulse-dot-soft")} />
}

export const ms = (n: number) => (n >= 10000 ? `${(n / 1000).toFixed(1)} s` : n >= 1000 ? `${(n / 1000).toFixed(2)} s` : `${n.toFixed(n < 10 ? 1 : 0)} ms`)
export const pct = (n: number, d = 0) => `${(n * 100).toFixed(d)}%`
export const when = (t: number) => new Date(t * 1000).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit" })
