import type { ReactNode } from "react"

export const NOT_MEASURED = <span className="text-ink-3">not measured</span>

/** A figure, or "not measured": a tile with no instrument behind it never shows a guess. */
export function val(v: number | null | undefined, fmt: (n: number) => string, unit = ""): ReactNode {
  return v == null || Number.isNaN(v) ? NOT_MEASURED : <span className="num">{fmt(v)}{unit && <span className="text-ink-2"> {unit}</span>}</span>
}

export function Section({ title, children, aside }: { title: string; children: ReactNode; aside?: ReactNode }) {
  return (
    <section className="mt-10 border-t border-line pt-4">
      <div className="flex items-baseline justify-between gap-4">
        <h2 className="text-[15px] font-semibold tracking-[-0.015em]">{title}</h2>
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
    <div className="row-wash flex items-baseline justify-between gap-6 border-b border-line py-2.5 last:border-0">
      <dt className="text-ink-2">{k}{hint && <span className="ml-2 text-[12px] text-ink-3">{hint}</span>}</dt>
      <dd className="m-0 text-right [overflow-wrap:anywhere]">{v}</dd>
    </div>
  )
}

export const ms = (n: number) => (n >= 10000 ? `${(n / 1000).toFixed(1)} s` : n >= 1000 ? `${(n / 1000).toFixed(2)} s` : `${n.toFixed(n < 10 ? 1 : 0)} ms`)
export const pct = (n: number, d = 0) => `${(n * 100).toFixed(d)}%`
export const when = (t: number) => new Date(t * 1000).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit" })

/** A page opens the same way everywhere: the title, one quiet line, space. */
export function PageHead({ title, sub, lead }: { title: ReactNode; sub?: ReactNode; lead?: ReactNode }) {
  return (
    <header className="page-head flex items-center gap-4">
      {lead}
      <div className="min-w-0">
        <h1 className="page-title">{title}</h1>
        {sub && <p className="page-sub">{sub}</p>}
      </div>
    </header>
  )
}

/** A few labelled figures on one line, instead of a table of rows. */
export function Facts({ items }: { items: [string, ReactNode][] }) {
  return (
    <dl className="m-0 flex flex-wrap gap-x-7 gap-y-2">
      {items.map(([k, v]) => (
        <div key={k} className="min-w-0">
          <dt className="text-[12px] text-ink-3">{k}</dt>
          <dd className="num m-0 text-[15px] tracking-[-0.01em]">{v}</dd>
        </div>
      ))}
    </dl>
  )
}
