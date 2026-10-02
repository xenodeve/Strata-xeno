import { createContext, useContext, useState, type ReactNode } from "react"
import { createPortal } from "react-dom"
import { markOf, type Command } from "../lib/slash"
import { t } from "../lib/i18n"

/** The skills in use, for showing `/name` in a message as what it is. The chat page provides them. */
export const SkillsContext = createContext<Command[]>([])

/** What a skill is: its name, what it does, and where it comes from. */
export function SkillCard({ cmd }: { cmd: Command }) {
  return (
    <>
      <span className="skill-card__name">/{cmd.name}</span>
      {cmd.description && <span className="skill-card__desc">{cmd.description}</span>}
      <span className="skill-card__from">{cmd.builtin ? t("Command of Strata") : cmd.plugin ? t("Skill from {app} · plugin {plugin}", { app: cmd.from, plugin: cmd.plugin }) : t("Skill from {app}", { app: cmd.from })}</span>
    </>
  )
}

/** `/name` in bold; pointing at it (or focusing it) shows a card with what skill it is and where it comes from. The card is drawn on the page
 *  itself (a portal, fixed), because the bubble sits in boxes that clip what hangs out of them. */
function SkillTag({ cmd, children }: { cmd: Command; children: ReactNode }) {
  const [at, setAt] = useState<{ x: number; y: number } | null>(null)
  const show = (el: HTMLElement) => { const r = el.getBoundingClientRect(); setAt({ x: Math.max(8, Math.min(r.left, innerWidth - 316)), y: r.bottom + 6 }) }
  return (
    <span
      className="skill-tag"
      tabIndex={0}
      data-skill-tag={cmd.name}
      aria-describedby={at ? "skill-card" : undefined}
      onPointerEnter={(e) => show(e.currentTarget)}
      onPointerLeave={() => setAt(null)}
      onFocus={(e) => show(e.currentTarget)}
      onBlur={() => setAt(null)}
    >
      {children}
      {at && createPortal(<span id="skill-card" role="tooltip" className="skill-tip" style={{ left: at.x, top: at.y }}><SkillCard cmd={cmd} /></span>, document.body)}
    </span>
  )
}

/** A message's text with its slash command (`/name` of a skill in use, at the start) in bold; hovering or focusing it says what skill it is
 *  and where it comes from. Any other text is shown as it is. */
export function SkillText({ text }: { text: string }) {
  const mark = markOf(text, useContext(SkillsContext))
  if (!mark) return <>{text}</>
  return (
    <>
      {text.slice(0, mark.at)}
      <SkillTag cmd={mark.cmd}>{text.slice(mark.at, mark.end)}</SkillTag>
      {text.slice(mark.end)}
    </>
  )
}
