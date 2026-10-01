import { useLayoutEffect, useRef, useState, type CSSProperties, type ReactNode } from "react"
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react"

// A menu of sections that fold open into branches, a line drawn to the one in use and a marker that glides to its section. Adapted
// from a "BranchedMenu" the developer pasted (source not named; the HugeIcons idiom is Libraries.dev's / React Bits': to be
// confirmed, see REFERENCES.md). Changes: TypeScript; controlled by the app (which topic is in use; optionally which sections are
// open, else it keeps that itself and a section opens when its topic becomes the one in use); the app's tokens for colour;
// `inert` on a folded section; one look (no size props). For the Settings page and for the chat's Recents and Projects, which
// need more of a row: an optional icon, a trailing control (the options button, a count), a custom body (the rename field), a
// disabled look, a long label cut with an ellipsis, a section with nothing in it, and the menu at the full width of its column.
export interface BranchTopic {
  value: string; label: string; icon?: IconSvgElement
  trailing?: ReactNode            // after the label, outside its button (an options button)
  custom?: ReactNode              // in place of the row (a field to rename it)
  disabled?: boolean              // looks off and says so to a screen reader; the click still reaches onSelect, which decides
  title?: string
}
export interface BranchSection {
  value: string; label: string; topics: BranchTopic[]
  trailing?: ReactNode            // after the heading (a count, an options button)
  custom?: ReactNode              // in place of the heading (a field to rename it)
  empty?: string                  // said when there is no topic
}

const PAD = 6          // the tree's padding above and below its rows
const MARK = 16        // the marker's height
const ROW = 36
const INDENT = 40
const TRUNK = 14
const RADIUS = 10
const LINE_W = 1.5

const sectionOf = (sections: BranchSection[], topic: string) => sections.find((s) => s.topics.some((x) => x.value === topic))?.value

export function BranchedMenu({ sections, active, onSelect, label, open: openProp, onToggle, fill = false }: {
  sections: BranchSection[]; active: string; onSelect: (topic: string) => void; label: string
  open?: string[]; onToggle?: (section: string) => void; fill?: boolean
}) {
  const [own, setOwn] = useState<Set<string>>(() => new Set([sectionOf(sections, active) ?? sections[0]?.value ?? ""]))
  const [seen, setSeen] = useState(active)
  if (seen !== active) {                                    // a link to a topic opens its section (set during render, not in an effect: no frame with it closed)
    setSeen(active)
    const s = sectionOf(sections, active)
    if (s && !own.has(s)) setOwn(new Set(own).add(s))
  }
  const open = openProp ? new Set(openProp) : own
  const nav = useRef<HTMLElement>(null)
  const heads = useRef<Record<string, HTMLButtonElement | null>>({})
  const marker = useRef<HTMLSpanElement>(null)
  const activeSection = sectionOf(sections, active)
  const markerShown = activeSection !== undefined && open.has(activeSection)

  useLayoutEffect(() => {
    const place = (glide: boolean) => {
      const m = marker.current
      const el = activeSection ? heads.current[activeSection] : null
      if (!m) return
      const on = markerShown && el
      if (!glide) m.style.transition = "none"
      if (on) m.style.top = `${el.offsetTop + (el.offsetHeight - MARK) / 2}px`
      m.toggleAttribute("data-on", Boolean(on))
      if (!glide) { void m.offsetHeight; m.style.transition = "" }
    }
    place(true)
    let first = true
    const ro = new ResizeObserver(() => { if (first) { first = false; return } place(false) })      // a section opening moves the heads: follow without gliding
    if (nav.current) ro.observe(nav.current)
    return () => ro.disconnect()
  }, [activeSection, markerShown])

  const toggle = (value: string) => {
    if (onToggle) onToggle(value)
    else setOwn((prev) => { const next = new Set(prev); if (!next.delete(value)) next.add(value); return next })
  }

  const r = Math.min(RADIUS, ROW / 2 - 2)
  const endX = INDENT - 8
  const rowY = (k: number) => PAD + k * ROW + ROW / 2
  const branch = (k: number) => `M ${TRUNK} ${rowY(k) - r} A ${r} ${r} 0 0 0 ${TRUNK + r} ${rowY(k)} H ${endX}`
  const reach = (k: number) => `M ${TRUNK} 0 V ${rowY(k) - r} A ${r} ${r} 0 0 0 ${TRUNK + r} ${rowY(k)} H ${endX}`
  const length = (k: number) => rowY(k) - r + (Math.PI * r) / 2 + (endX - TRUNK - r)

  if (!sections.length) return null
  return (
    <nav ref={nav} aria-label={label} className={fill ? "branched-menu branched-menu--fill" : "branched-menu"} style={{ "--bm-row": `${ROW}px`, "--bm-indent": `${INDENT}px`, "--bm-line-w": LINE_W } as CSSProperties}>
      <span ref={marker} className="branched-menu__marker" aria-hidden />
      {sections.map((section) => {
        const isOpen = open.has(section.value)
        const n = section.topics.length
        const bodyH = PAD * 2 + n * ROW
        return (
          <div key={section.value} className="branched-menu__section" data-section={section.value} data-open={isOpen ? "" : undefined}>
            <div className="branched-menu__headrow group">
              {section.custom ?? (
                <button ref={(el) => { heads.current[section.value] = el }} type="button" className="branched-menu__head" aria-expanded={isOpen} onClick={() => toggle(section.value)}>
                  {section.label}
                </button>
              )}
              {section.trailing}
            </div>
            <div className="branched-menu__body">
              <div className="branched-menu__fold" inert={!isOpen}>
                {n === 0 ? (section.empty ? <p className="branched-menu__empty">{section.empty}</p> : null) : (
                  <div className="branched-menu__tree" style={{ height: bodyH }}>
                    <svg className="branched-menu__lines" width={INDENT} height={bodyH} aria-hidden>
                      <path className="branched-menu__base" d={`M ${TRUNK} 0 V ${rowY(n - 1) - r}`} />
                      {section.topics.map((x, k) => <path key={x.value} className="branched-menu__base" d={branch(k)} />)}
                      {section.topics.map((x, k) => (
                        <path key={x.value} className="branched-menu__reach" d={reach(k)} style={{ strokeDasharray: length(k), strokeDashoffset: x.value === active ? 0 : length(k) }} />
                      ))}
                    </svg>
                    {section.topics.map((x) => (
                      <div key={x.value} className="branched-menu__row group" data-topic={x.value}>
                        {x.custom ?? (
                          <button
                            type="button"
                            className="branched-menu__item"
                            aria-current={x.value === active ? "true" : undefined}
                            aria-disabled={x.disabled ? "true" : undefined}
                            data-active={x.value === active ? "" : undefined}
                            data-disabled={x.disabled ? "" : undefined}
                            title={x.title}
                            onClick={() => onSelect(x.value)}
                          >
                            {x.icon ? <span className="branched-menu__icon" aria-hidden><HugeiconsIcon icon={x.icon} size={16} strokeWidth={1.8} /></span> : null}
                            <span className="branched-menu__label">{x.label}</span>
                          </button>
                        )}
                        {x.trailing}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        )
      })}
    </nav>
  )
}
