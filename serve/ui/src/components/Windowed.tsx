import { useCallback, useEffect, useLayoutEffect, useRef, type CSSProperties, type ReactNode } from "react"

/** A list in a window of a fixed height (the look of the thinking's "Reasoning stream", styles.css), so a long list can never push the
 *  heading that closes it out of reach: the page does not grow with it, the list scrolls inside. The edges fade only where there is more
 *  to see (`data-above`, `data-below`). It is a region a keyboard can focus and scroll. */
export function Windowed({ label, height = "min(20rem, 55dvh)", children }: { label: string; height?: string; children: ReactNode }) {
  const view = useRef<HTMLDivElement>(null)
  const edges = useCallback(() => {
    const el = view.current
    if (!el) return
    el.toggleAttribute("data-above", el.scrollTop > 2)
    el.toggleAttribute("data-below", el.scrollHeight - el.clientHeight - el.scrollTop > 2)
  }, [])
  useLayoutEffect(edges)                                    // the list may have changed (a filter, a switch): look again after every draw
  useEffect(() => {
    const ro = new ResizeObserver(edges)
    if (view.current) ro.observe(view.current)
    return () => ro.disconnect()
  }, [edges])
  return (
    <div ref={view} className="t-reason-viewport" role="region" tabIndex={0} aria-label={label} style={{ "--reason-height": height } as CSSProperties} onScroll={edges}>
      {children}
    </div>
  )
}
