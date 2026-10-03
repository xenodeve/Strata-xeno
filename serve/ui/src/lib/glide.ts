// A panel that stretches where it really grew. When something inside a panel changes size - a paragraph that got longer, a row that was added -
// the part that changed glides to its new height, and the rest of the panel simply follows it in the flow: what is after the change moves with it,
// what is before it (or, in a panel anchored at its bottom, what is after it) stays where it was. The panel as a whole is not animated.
//
// `chooseStretching` is the choice, from what was seen (every element: its height before, none when it was not there, and now); `watchGrowth`
// looks at a real element tree, and stretches what the choice names. Parts that are already moving by themselves (a Collapse, a Fit, an earlier
// stretch) are left alone.

export interface Seen<T> { node: T; was: number | undefined; now: number }
const MIN_CHANGE = 3          // px: less is rounding, a border, a rule
const MS = 420

/** The parts that stretch: those that changed height by more than a little, the topmost of a part that is new (not what is inside it), and
 *  of the ones that are left only the innermost: an element that got longer makes its parents longer too, but they are not what stretches. */
export function chooseStretching<T>(all: Seen<T>[], within: (ancestor: T, node: T) => boolean): Seen<T>[] {
  const picked = all.filter((x) => (x.was === undefined ? x.now >= MIN_CHANGE : Math.abs(x.now - x.was) >= MIN_CHANGE))
  const fresh = picked.filter((x) => x.was === undefined)
  const roots = picked.filter((x) => x.was !== undefined || !fresh.some((f) => f !== x && within(f.node, x.node)))       // a new part inside a new part is part of it
  return roots.filter((x) => !roots.some((y) => y !== x && within(x.node, y.node)))
}

const reduced = () => typeof matchMedia !== "undefined" && matchMedia("(prefers-reduced-motion: reduce)").matches
const MOVING = /height|grid-template-rows|margin|padding/

/** Whether something in `root` that is part of the flow is already changing its size by itself (a CSS transition on a size): then it is doing its own stretching. */
function movingBySelf(root: HTMLElement): boolean {
  if (typeof root.getAnimations !== "function" || typeof CSSTransition === "undefined") return false
  return root.getAnimations({ subtree: true }).some((a) => {
    if (!(a instanceof CSSTransition) || a.playState !== "running" || !MOVING.test(a.transitionProperty)) return false
    const target = a.effect instanceof KeyframeEffect ? a.effect.target : null
    if (!(target instanceof HTMLElement)) return true
    const position = getComputedStyle(target).position
    return position !== "absolute" && position !== "fixed"            // a marker that slides over the panel does not move what is round it
  })
}

function stretch(el: HTMLElement, from: number, to: number, done: () => void): void {
  const keep = { height: el.style.height, overflow: el.style.overflow, transition: el.style.transition }
  el.style.transition = "none"
  el.style.overflow = "hidden"
  el.style.height = `${from}px`
  void el.offsetHeight                                              // the start is laid out before the end is set
  el.style.transition = `height ${MS}ms cubic-bezier(0.4, 0, 0.2, 1)`
  el.style.height = `${to}px`
  let over = false
  const end = () => {
    if (over) return
    over = true
    clearTimeout(timer)
    el.removeEventListener("transitionend", on)
    el.style.height = keep.height
    el.style.overflow = keep.overflow
    el.style.transition = keep.transition
    done()
  }
  const on = (e: TransitionEvent) => { if (e.target === el && e.propertyName === "height") end() }
  const timer = setTimeout(end, MS + 150)
  el.addEventListener("transitionend", on)
}

/** Watch `root` (the content of a panel): when its content changes, stretch the parts that changed size. Returns the function that stops it. */
export function watchGrowth(root: HTMLElement): () => void {
  const last = new WeakMap<Element, number>()
  const busy = new WeakSet<Element>()
  // The size of a part is its layout height (offsetHeight), not what is drawn: a menu that is still popping in is drawn a little smaller than it is,
  // and that is not a change in the content.
  const all = () => [...root.querySelectorAll<HTMLElement>("*")].filter((el) => el instanceof HTMLElement)
  const heightOf = (el: HTMLElement) => el.offsetHeight
  // "before" must be what the part measured just before the change, whenever that was: a size that changed by itself since (a font that arrived, a
  // line that wrapped differently) is not part of this change. So every size change is noted as it happens, and a change in the content is
  // compared with the last size noted.
  const sizes = new ResizeObserver((entries) => { for (const e of entries) if (e.target instanceof HTMLElement) last.set(e.target, heightOf(e.target)) })
  const watch = () => { for (const el of all()) sizes.observe(el) }
  const snapshot = () => { for (const el of all()) last.set(el, heightOf(el)); watch() }
  snapshot()
  const settle = () => {
    if (reduced() || movingBySelf(root)) { snapshot(); return }
    const seen: Seen<HTMLElement>[] = []
    for (const el of all()) {
      if (busy.has(el)) continue
      const d = getComputedStyle(el).display
      if (d === "inline" || d === "contents" || d === "none") continue            // an inline box has no height of its own to stretch
      seen.push({ node: el, was: last.get(el), now: heightOf(el) })
    }
    const go = chooseStretching(seen, (a, b) => a !== b && a.contains(b))
    for (const s of go) {
      busy.add(s.node)
      stretch(s.node, s.was ?? 0, s.now, () => { busy.delete(s.node); snapshot() })
    }
    for (const s of seen) if (!go.some((g) => g.node === s.node)) last.set(s.node, s.now)
    watch()                                                                          // the parts that came with this change are noted from now on
  }
  const mo = new MutationObserver(settle)
  mo.observe(root, { childList: true, characterData: true, subtree: true })       // not attributes: the stretch itself writes styles
  return () => { mo.disconnect(); sizes.disconnect() }
}
