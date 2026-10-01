import { useEffect, useLayoutEffect, useRef } from "react"
import { t } from "../lib/i18n"

const reduced = () => typeof matchMedia !== "undefined" && matchMedia("(prefers-reduced-motion: reduce)").matches

/** The agent's thinking in a window of a fixed height (transitions.dev "Reasoning stream", see styles.css), so a long thought can
 *  never push the page out of reach: the heading that closes it stays where it is. While the thought is still being written the
 *  window stays at its end, stepping down every --reason-hold; scrolling up in it lets go, and reaching the end holds again. A thought
 *  that is finished opens at its start. The edges fade only where there is more to see. */
export function ReasonStream({ text, live }: { text: string; live: boolean }) {
  const view = useRef<HTMLDivElement>(null)
  const follow = useRef(live)

  const edges = () => {
    const el = view.current
    if (!el) return
    el.toggleAttribute("data-above", el.scrollTop > 2)
    el.toggleAttribute("data-below", el.scrollHeight - el.clientHeight - el.scrollTop > 2)
  }
  useLayoutEffect(() => { if (live && view.current) view.current.scrollTop = view.current.scrollHeight; edges() }, [])    // opened while it is written: at the end
  useEffect(() => { follow.current = live }, [live])
  useEffect(edges, [text])
  useEffect(() => {
    if (!live) return
    const hold = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--reason-hold")) || 840
    const id = setInterval(() => {                                                      // a step down every hold, not a jump per token
      const el = view.current
      if (!el || !follow.current) return
      const end = el.scrollHeight - el.clientHeight
      if (end - el.scrollTop > 1) el.scrollTo({ top: end, behavior: reduced() ? "auto" : "smooth" })
    }, hold)
    return () => clearInterval(id)
  }, [live])

  const letGo = () => { follow.current = false }
  return (
    <div
      ref={view}
      className="t-reason-viewport mt-1"
      role="region"
      tabIndex={0}
      aria-label={t("The agent's thinking")}
      onWheel={(e) => { if (e.deltaY < 0) letGo() }}
      onTouchMove={letGo}
      onKeyDown={(e) => { if (["ArrowUp", "PageUp", "Home"].includes(e.key)) letGo() }}
      onScroll={() => { edges(); const el = view.current; if (el && live && el.scrollHeight - el.clientHeight - el.scrollTop < 24) follow.current = true }}
    >
      <div className="t-reason-text whitespace-pre-wrap border-l border-line pl-3 text-[13px] font-normal leading-relaxed text-ink-2 [overflow-wrap:anywhere]">{text}</div>
    </div>
  )
}
