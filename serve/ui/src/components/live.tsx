import { phaseKind, serverDesign, type OrbDesign } from "../lib/orbs"
import { t } from "../lib/i18n"
import { Orb } from "./orb"
import { Pop } from "./pop"
import { Spin } from "./spin"
import type { Live } from "../lib/metrics"

/** What a request that is generating is doing, in words: the phase the server reports (thinking, answering, writing a tool call). */
export function generatingLabel(phase: string | null | undefined): string {
  const p = phaseKind(phase)
  return p.kind === "thinking" ? t("Model is thinking") : p.kind === "answering" ? t("Model is answering")
    : p.kind === "tool" ? t("Writing a tool call: {name}", { name: p.tool }) : p.kind === "toolDone" ? t("Tool call written") : t("Writing")
}

/** The server's state as a dotted orb: the searching globe turning slowly while idle, listening while a prompt is read, flowing while it writes.
 *  (vendor/thinking-orbs, MIT: it pauses itself offscreen and in a hidden tab, and is a still frame under reduced motion.) */
export function StatusOrb({ live, stale = false, size = 64, scale = 1, override }: {
  live: Pick<Live, "state" | "queued" | "tok_s"> & { phase?: string | null }; stale?: boolean; size?: 64 | 32 | 20; scale?: number; override?: { design: OrbDesign; label: string }
}) {
  const look = serverDesign(live, stale)
  const label = stale ? t("Not answering") : live.state === "reading" ? t("Reading the prompt") : live.state === "generating" ? generatingLabel(live.phase)
    : live.state === "unloaded" ? t("Model unloaded") : (live.queued ?? 0) > 0 ? t("Waiting in the queue") : t("Idle")
  return override
    ? <Orb design={override.design} size={size} scale={scale} moving label={override.label} />
    : <Orb design={look.design} size={size} scale={scale} moving={look.moving} speed={look.speed} fps={look.fps} label={label} />
}

/** A figure that animates when its value changes: "count" (one that only grows, a token count) pops in, only the digits that changed
 *  (components/pop.tsx); "gauge" (one that goes up and down, a speed) turns its reels up or down (components/spin.tsx). */
export function Num({ value, digits = 0, className, kind = "count" }: { value: number | null | undefined; digits?: number; className?: string; kind?: "count" | "gauge" }) {
  const text = value == null ? null : value.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits })
  return <span className={className ?? "num"}>{text == null ? "–" : kind === "gauge" ? <Spin text={text} /> : <Pop text={text} />}</span>
}
