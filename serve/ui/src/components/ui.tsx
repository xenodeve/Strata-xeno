import type { ReactNode } from "react"
import { cn } from "../lib/cn"

export function Switch({ checked, onChange, label, hint }: { checked: boolean; onChange: (v: boolean) => void; label: string; hint?: ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-4 py-2">
      <div className="min-w-0">
        <div className="text-[13px]">{label}</div>
        {hint && <div className="mt-0.5 text-[12px] text-ink-2">{hint}</div>}
      </div>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        aria-label={label}
        onClick={() => onChange(!checked)}
        className={cn(
          "relative mt-0.5 h-[22px] w-[38px] shrink-0 rounded-full transition-colors duration-200",
          checked ? "bg-accent" : "bg-fill-2",
        )}
      >
        <span
          aria-hidden
          className={cn("absolute left-[2px] top-[2px] size-[18px] rounded-full bg-white shadow-sm transition-transform duration-200 ease-[var(--ease)]", checked && "translate-x-4")}
        />
      </button>
    </div>
  )
}

export function Segmented<T extends string>({ value, options, onChange, label }: {
  value: T; options: { value: T; label: string }[]; onChange: (v: T) => void; label: string
}) {
  return (
    <div role="radiogroup" aria-label={label} className="grid auto-cols-fr grid-flow-col gap-0.5 rounded-md bg-fill p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={value === o.value}
          onClick={() => onChange(o.value)}
          className={cn(
            "h-7 rounded-sm px-2 text-[13px] transition-colors duration-150",
            value === o.value ? "bg-surface font-medium shadow-sm" : "text-ink-2 hover:text-ink",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

export function Field({ label, value, hint, children }: { label: string; value?: ReactNode; hint?: ReactNode; children: ReactNode }) {
  return (
    <label className="block py-2">
      <span className="flex items-baseline justify-between gap-3">
        <span className="text-[13px]">{label}</span>
        {value != null && <span className="num text-[12px] text-ink-2">{value}</span>}
      </span>
      <span className="mt-1.5 block">{children}</span>
      {hint && <span className="mt-1 block text-[12px] text-ink-2">{hint}</span>}
    </label>
  )
}

/** The small switch of a row in a list (a server, a skill, an app): on or off, no label beside it (the label is for a screen reader). */
export function MiniSwitch({ on, label, onClick, disabled }: { on: boolean; label: string; onClick: () => void; disabled?: boolean }) {
  return <button type="button" role="switch" aria-checked={on} aria-label={label} disabled={disabled} className="prompt-bar__sw" onMouseDown={(e) => e.preventDefault()} onClick={onClick}><span aria-hidden /></button>
}

export const inputCls =
  "h-8 w-full rounded-sm border border-line bg-surface px-2.5 text-[13px] outline-none transition-colors placeholder:text-ink-3 hover:border-fill-2 focus:border-accent disabled:opacity-50"

export function Button({ children, onClick, kind = "secondary", disabled, type = "button", label }: {
  children: ReactNode; onClick?: () => void; kind?: "primary" | "secondary" | "quiet"; disabled?: boolean; type?: "button" | "submit"; label?: string
}) {
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      className={cn(
        "inline-flex h-8 items-center justify-center gap-1.5 rounded-sm px-3 text-[13px] font-medium transition-[background-color,transform] duration-150 active:scale-[0.98] disabled:pointer-events-none disabled:opacity-40",
        kind === "primary" && "bg-ink text-surface hover:opacity-90",
        kind === "secondary" && "bg-fill hover:bg-fill-2",
        kind === "quiet" && "text-ink-2 hover:bg-hover hover:text-ink",
      )}
    >
      {children}
    </button>
  )
}
