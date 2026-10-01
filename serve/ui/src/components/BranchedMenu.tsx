import { useLayoutEffect, useRef, useState, type CSSProperties } from "react"
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react"

// A menu of sections that fold open into branches, a line drawn to the one in use and a marker that glides to its section. Adapted
// from a "BranchedMenu" the developer pasted (source not named; the HugeIcons idiom is Libraries.dev's / React Bits': to be
// confirmed, see REFERENCES.md). Changes: TypeScript; controlled (the address says which topic is in use, and its section opens
// when a link arrives); the app's tokens for colour; `inert` on a folded section; no props for sizes (one look).
export interface BranchTopic { value: string; label: string; icon: IconSvgElement }
export interface BranchSection { value: string; label: string; topics: BranchTopic[] }

const PAD = 6          // the tree's padding above and below its rows
const MARK = 16        // the marker's height
const ROW = 36
const INDENT = 40
const TRUNK = 14
const RADIUS = 10
const LINE_W = 1.5

const sectionOf = (sections: BranchSection[], topic: string) => sections.find((s) => s.topics.some((x) => x.value === topic))?.value

export function BranchedMenu({ sections, active, onSelect, label }: { sections: BranchSection[]; active: string; onSelect: (topic: string) => void; label: string }) {
  const [open, setOpen] = useState<Set<string>>(() => new Set([sectionOf(sections, active) ?? sections[0].value]))
  const [seen, setSeen] = useState(active)
  if (seen !== active) {                                    // a link to a topic opens its section (set during render, not in an effect: no frame with it closed)
    setSeen(active)
    const s = sectionOf(sections, active)
    if (s && !open.has(s)) setOpen(new Set(open).add(s))
  }
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

  const toggle = (value: string) => setOpen((prev) => { const next = new Set(prev); if (!next.delete(value)) next.add(value); return next })

  const r = Math.min(RADIUS, ROW / 2 - 2)
  const endX = INDENT - 8
  const rowY = (k: number) => PAD + k * ROW + ROW / 2
  const branch = (k: number) => `M ${TRUNK} ${rowY(k) - r} A ${r} ${r} 0 0 0 ${TRUNK + r} ${rowY(k)} H ${endX}`
  const reach = (k: number) => `M ${TRUNK} 0 V ${rowY(k) - r} A ${r} ${r} 0 0 0 ${TRUNK + r} ${rowY(k)} H ${endX}`
  const length = (k: number) => rowY(k) - r + (Math.PI * r) / 2 + (endX - TRUNK - r)

  return (
    <nav ref={nav} aria-label={label} className="branched-menu" style={{ "--bm-row": `${ROW}px`, "--bm-indent": `${INDENT}px`, "--bm-line-w": LINE_W } as CSSProperties}>
      <span ref={marker} className="branched-menu__marker" aria-hidden />
      {sections.map((section) => {
        const isOpen = open.has(section.value)
        const bodyH = PAD * 2 + section.topics.length * ROW
        return (
          <div key={section.value} className="branched-menu__section" data-open={isOpen ? "" : undefined}>
            <button ref={(el) => { heads.current[section.value] = el }} type="button" className="branched-menu__head" aria-expanded={isOpen} onClick={() => toggle(section.value)}>
              {section.label}
            </button>
            <div className="branched-menu__body">
              <div className="branched-menu__fold" inert={!isOpen}>
                <div className="branched-menu__tree" style={{ height: bodyH }}>
                  <svg className="branched-menu__lines" width={INDENT} height={bodyH} aria-hidden>
                    <path className="branched-menu__base" d={`M ${TRUNK} 0 V ${rowY(section.topics.length - 1) - r}`} />
                    {section.topics.map((x, k) => <path key={x.value} className="branched-menu__base" d={branch(k)} />)}
                    {section.topics.map((x, k) => (
                      <path key={x.value} className="branched-menu__reach" d={reach(k)} style={{ strokeDasharray: length(k), strokeDashoffset: x.value === active ? 0 : length(k) }} />
                    ))}
                  </svg>
                  {section.topics.map((x) => (
                    <button key={x.value} type="button" className="branched-menu__item" aria-current={x.value === active ? "true" : undefined} data-active={x.value === active ? "" : undefined} onClick={() => onSelect(x.value)}>
                      <span className="branched-menu__icon" aria-hidden><HugeiconsIcon icon={x.icon} size={16} strokeWidth={1.8} /></span>
                      <span className="branched-menu__label">{x.label}</span>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )
      })}
    </nav>
  )
}
