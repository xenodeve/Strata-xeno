import { useRef, useState, type KeyboardEvent, type ReactNode } from "react"
import { BotTile, Orb } from "./orb"
import { LoaderMark } from "./loader"
import { Lattice } from "./thought"
import { BOT_TYPES, setAvatar, setBotChoice, useAvatar, useBotChoice, type AvatarKind } from "../lib/avatar"
import { msg, t } from "../lib/i18n"
import { cn } from "../lib/cn"
import { Collapse } from "./motion"
import type { OrbDesign } from "../lib/orbs"

// The choice of what stands for a status, as one list you can read: each way shows three of its marks (thinking, answering, an idle
// server) beside its name and one line on what it is, so nothing has to be guessed or clicked to find out. Used in a menu from the
// header and in Settings.
const OPTIONS: { kind: AvatarKind; name: string; text: string }[] = [
  { kind: "orbs", name: msg("Orbs"), text: msg("A dotted ball that takes a form for each status.") },
  { kind: "mixed", name: msg("Orbs + Loading"), text: msg("Orbs, with a lattice of dots beside the thinking. How it starts.") },
  { kind: "loading", name: msg("Loading only"), text: msg("Loaders of dots, a pattern for each status. No orbs.") },
  { kind: "bots", name: msg("Avatar"), text: msg("A small character with a shape and a mood of its own.") },
]
const SAMPLE: OrbDesign[] = ["solving", "composing", "searching"]            // thinking, answering, an idle server

const BOT_NAME: Record<(typeof BOT_TYPES)[number], string> = {
  clover: msg("Clover"), flower: msg("Flower"), triangle: msg("Triangle"), square: msg("Square"), blob: msg("Blob"), ghost: msg("Ghost"),
  circle: msg("Circle"), drop: msg("Drop"), star: msg("Star"), droid: msg("Droid"), mech: msg("Mech"), alien: msg("Alien"),
  hexagon: msg("Hexagon"), cat: msg("Cat"), cloud: msg("Cloud"), pill: msg("Pill"), pebble: msg("Pebble"), puddle: msg("Puddle"),
}

/** Three marks of one way. The avatars are not loaded until they are asked for (chosen, or the row looked at). */
function Preview({ kind, bots }: { kind: AvatarKind; bots: boolean }): ReactNode {
  if (kind === "bots" && !bots) return SAMPLE.map((d) => <span key={d} className="size-5 rounded-full border border-dashed border-line" />)
  return SAMPLE.map((d, i) => {
    if (kind === "mixed" && i === 0) return <Lattice key={d} status="working" pattern="orbit" />
    if (kind === "loading") return <LoaderMark key={d} design={d} px={20} moving />
    return <Orb key={d} as={kind === "bots" ? "bots" : "orbs"} design={d} size={20} rest={d === "searching"} />
  })
}

export function StatusMarkSettings() {
  const kind = useAvatar()
  const pick = useBotChoice()
  const [peek, setPeek] = useState(false)
  const rows = useRef<(HTMLButtonElement | null)[]>([])
  const go = (i: number) => {
    const n = (i + OPTIONS.length) % OPTIONS.length
    setAvatar(OPTIONS[n].kind)
    rows.current[n]?.focus()
  }
  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const at = OPTIONS.findIndex((o) => o.kind === kind)
    if (e.key === "ArrowDown" || e.key === "ArrowRight") { e.preventDefault(); go(at + 1) }
    else if (e.key === "ArrowUp" || e.key === "ArrowLeft") { e.preventDefault(); go(at - 1) }
  }
  return (
    <div>
      <div role="radiogroup" aria-label={t("Status marks")} className="space-y-1.5" onKeyDown={onKey}>
        {OPTIONS.map((o, i) => {
          const on = kind === o.kind
          return (
            <button
              key={o.kind}
              ref={(el) => { rows.current[i] = el }}
              type="button"
              role="radio"
              aria-checked={on}
              tabIndex={on ? 0 : -1}
              onClick={() => setAvatar(o.kind)}
              onPointerEnter={() => { if (o.kind === "bots") setPeek(true) }}
              onFocus={() => { if (o.kind === "bots") setPeek(true) }}
              className={cn("flex w-full items-center gap-3 rounded-md border px-3 py-2.5 text-left transition-colors", on ? "border-ink bg-fill" : "border-line hover:bg-hover")}
            >
              <span className="flex w-[88px] shrink-0 items-center justify-between" aria-hidden><Preview kind={o.kind} bots={on || peek} /></span>
              <span className="min-w-0 flex-1">
                <span className="block text-[13px] font-medium">{t(o.name)}{o.kind === "mixed" && <span className="ml-2 rounded-full bg-fill-2 px-1.5 py-px text-[11px] font-normal text-ink-2">{t("Default")}</span>}</span>
                <span className="block text-[12px] leading-snug text-ink-2">{t(o.text)}</span>
              </span>
              <span aria-hidden className={cn("grid size-4 shrink-0 place-items-center rounded-full border", on ? "border-ink" : "border-line")}>{on && <span className="size-2 rounded-full bg-ink" />}</span>
            </button>
          )
        })}
      </div>
      <Collapse open={kind === "bots"} instant soft>               {/* the avatars stretch open below, instead of appearing at once */}
        <div className="pt-4">
          <p className="text-[13px] text-ink-2">{t("Which avatar: one for every status, a shape for each status, or a random one for each place.")}</p>
          <div className="mt-2 flex flex-wrap gap-2" role="group" aria-label={t("Which avatar")}>
            {(["auto", "random"] as const).map((c) => (
              <button key={c} type="button" aria-pressed={pick === c} onClick={() => setBotChoice(c)} className={cn("flex h-[60px] min-w-[84px] items-center justify-center rounded-md border px-3 text-[12px] transition-colors", pick === c ? "border-ink bg-fill text-ink" : "border-line text-ink-2 hover:bg-hover")}>{c === "auto" ? t("By status") : t("Random")}</button>
            ))}
            {BOT_TYPES.map((b) => (
              <button key={b} type="button" aria-pressed={pick === b} aria-label={t(BOT_NAME[b])} title={t(BOT_NAME[b])} onClick={() => setBotChoice(b)} className={cn("grid size-[60px] place-items-center rounded-md border transition-colors", pick === b ? "border-ink bg-fill" : "border-line hover:bg-hover")}>
                <BotTile type={b} />
              </button>
            ))}
          </div>
        </div>
      </Collapse>
    </div>
  )
}
