import { lazy, Suspense, useEffect, useRef, useState, type ReactNode } from "react"
import { ThinkingOrb } from "../vendor/thinking-orbs/orb"
import { SLOW, type OrbDesign } from "../lib/orbs"
import { botFor, plainLook, randomBot, useAvatar, useBotChoice, type BotType } from "../lib/avatar"
import { t } from "../lib/i18n"

// thinking-orbs by Jakub Antalik (Libraries.dev, MIT), vendored in src/vendor/thinking-orbs. The orb pauses itself offscreen and
// in a hidden tab, and is a still frame under reduced motion. These three components only decide where and how it is shown.
type Size = 64 | 32 | 20

// The bots (Libraries.dev bot-avatars, MIT, src/vendor/bot-avatars) stand in for the orbs when chosen (About, or the header). They are
// loaded only then: with the orbs nobody downloads them.
const BotAvatar = lazy(() => import("../vendor/bot-avatars/index.es.js").then((m) => ({ default: m.BotAvatar })))

interface FaceProps { design: OrbDesign; size: Size; moving: boolean; label?: string; color?: string; speed: number; fps?: number; scale: number; rest?: boolean; drawn: BotType }

/** One avatar, as a tile in the picker (About): still looking around, not following the pointer. */
export function BotTile({ type, size = 44 }: { type: BotType; size?: number }) {
  return (
    <Suspense fallback={<span style={{ display: "block", width: size, height: size }} />}>
      <BotAvatar type={type} state="default" size={size} shading="smooth" interactive={false} aria-hidden role="presentation" />
    </Suspense>
  )
}

/** What shows for one design: its orb, or its bot (a shape of its own, working / looking around / sleeping by what it stands for). */
function Face({ design, size, moving, label, color, speed, fps, scale, rest, drawn }: FaceProps) {
  const avatar = useAvatar()
  const pick = useBotChoice()
  const hidden = label ? { "aria-label": label } : { "aria-hidden": true, role: "presentation" }
  if (avatar === "loading") {                                  // a plain ring: turns at work, breathes at rest
    const px = size * scale
    return <span {...hidden} className="t-ring" data-look={plainLook(design, { moving, rest })} style={{ width: px, height: px, borderWidth: Math.max(2, Math.round(px / 14)), ...(color ? { ["--ring" as string]: color } : null) }} />
  }
  if (avatar === "bots") {
    const bot = botFor(design, { moving, rest }, pick, drawn)
    const px = size * scale
    return (
      <Suspense fallback={<span style={{ display: "block", width: px, height: px }} />}>
        <BotAvatar type={bot.type} state={bot.state} size={px} face="eyes" color={color} speed={speed} shading={px >= 40 ? "plastic" : "smooth"} interactive={px >= 40} {...hidden} />
      </Suspense>
    )
  }
  return <ThinkingOrb state={design} size={size} paused={!moving} color={color} speed={speed} fps={fps} scale={scale} {...hidden} />
}

/** An orb in a fixed slot. When the design changes the old form dissolves (it swells a little and blurs away) while the new
 *  one resolves out of a blur (it settles from a little smaller), 450 ms, so a state change reads as one orb changing form,
 *  not two orbs swapped. */
export function Orb({ design, size = 20, moving = true, label, color, speed = 1, fps, scale = 1, rest }: { design: OrbDesign; size?: Size; moving?: boolean; label?: string; color?: string; speed?: number; fps?: number; scale?: number; rest?: boolean }) {
  const [shown, setShown] = useState(design)
  const drawn = useRef<BotType | null>(null)
  drawn.current ??= randomBot(Math.random())          // this place's avatar when "random" is chosen: it stays until the page is opened again
  const [leaving, setLeaving] = useState<OrbDesign | null>(null)
  const was = useRef(design)
  useEffect(() => {
    if (design === was.current) return
    setLeaving(was.current)
    setShown(design)
    was.current = design
    const t = setTimeout(() => setLeaving(null), 480)
    return () => clearTimeout(t)
  }, [design])
  return (
    <span className="orb-slot" style={{ width: size * scale, height: size * scale }}>
      {leaving && <span className="orb-layer orb-out" aria-hidden><Face design={leaving} size={size} moving={moving} color={color} speed={speed} fps={fps} scale={scale} rest={rest} drawn={drawn.current} /></span>}
      <span key={shown} className="orb-layer orb-in"><Face design={shown} size={size} moving={moving} label={label} color={color} speed={speed} fps={fps} scale={scale} rest={rest} drawn={drawn.current} /></span>
    </span>
  )
}

/** An orb beside a status line whose text shimmers: the "something is happening" row. */
export function StatusLabel({ design, children, size = 20, moving = true, className = "" }: { design: OrbDesign; children: ReactNode; size?: Size; moving?: boolean; className?: string }) {
  return (
    <span className={`inline-flex items-center gap-2 ${className}`}>
      <Orb design={design} size={size} moving={moving} />
      <span className={moving ? "t-shimmer" : ""}>{children}</span>
    </span>
  )
}

/** A page or a block that is waiting for data: the orb and what it waits for. An error shows as plain text, not as a wait. */
export function Loading({ children = t("Connecting…"), error, design = "connecting" }: { children?: ReactNode; error?: string | null; design?: OrbDesign }) {
  if (error) return <p className="text-ink-2">{error}</p>
  return <p className="py-6 text-ink-2" role="status"><StatusLabel design={design}>{children}</StatusLabel></p>
}

/** A line for "there is nothing here": a slowly breathing orb and the words. */
export function Empty({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <span className={`inline-flex items-center gap-2 text-ink-2 ${className}`}><Orb design="breathing" size={20} speed={SLOW.speed} fps={SLOW.fps} />{children}</span>
}

/** A token's colour as the orb's tint wants it (#rgb / rgb()): read from the page, so it follows the theme. */
export const tint = (token: string) => (typeof document === "undefined" ? undefined : getComputedStyle(document.documentElement).getPropertyValue(token).trim() || undefined)
