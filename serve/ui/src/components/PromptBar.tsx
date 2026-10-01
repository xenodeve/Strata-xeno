import { useCallback, useEffect, useLayoutEffect, useRef, useState, type KeyboardEvent as RKeyEvent, type PointerEvent as RPointerEvent, type ReactNode, type RefObject } from "react"
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react"
import {
  Attachment01Icon, Cancel01Icon, Download01Icon, File02Icon, HelpCircleIcon, PlusSignIcon, Settings02Icon, SparklesIcon, MessageAdd01Icon,
} from "@hugeicons/core-free-icons"

// The composer: the field, and one bar of tools under it. Adapted from React Bits' PromptBar (MIT + Commons Clause: used
// inside this app only, see REFERENCES.md). Changes: our tokens instead of fixed colours; the arrow-to-stop morph is a
// small rAF tween instead of the `motion` package; the sparks at the top effort level run only while the field is
// focused, nothing is being written and the tab is visible; the menu holds this app's own actions.

const ARROW_UP = [12, 4.5, 18.5, 11, 14.25, 11, 14.25, 19.5, 9.75, 19.5, 9.75, 11, 5.5, 11]
const SQUARE = [12, 6, 18, 6, 18, 12, 18, 18, 6, 18, 6, 12, 6, 6]
const EDGE = 11
const MORPH_MS = 240

const mix = (a: number, b: number, t: number) => a + (b - a) * t
const pathAt = (t: number) => ARROW_UP.reduce((d, _, i) => (i % 2 ? d : d + `${i ? "L" : "M"}${mix(ARROW_UP[i], SQUARE[i], t).toFixed(2)} ${mix(ARROW_UP[i + 1], SQUARE[i + 1], t).toFixed(2)}`), "") + "Z"
const easeInOut = (t: number) => (t < 0.5 ? 8 * t ** 4 : 1 - (-2 * t + 2) ** 4 / 2)
const reduced = () => typeof matchMedia !== "undefined" && matchMedia("(prefers-reduced-motion: reduce)").matches

/** The send arrow that turns into a stop square while a request runs (and back), with a small squash and tilt in between. */
function SendGlyph({ busy }: { busy: boolean }) {
  const svg = useRef<SVGSVGElement>(null)
  const path = useRef<SVGPathElement>(null)
  const t = useRef(busy ? 1 : 0)
  const dir = useRef(busy ? 1 : -1)
  useEffect(() => {
    const to = busy ? 1 : 0
    dir.current = busy ? 1 : -1
    const from = t.current
    if (from === to) return
    const paint = (v: number) => {
      t.current = v
      path.current?.setAttribute("d", pathAt(v))
      const goo = reduced() ? 0 : Math.sin(v * Math.PI)
      const sx = 1 - 0.12 * goo
      if (svg.current) svg.current.style.transform = goo ? `rotate(${dir.current * 8 * goo}deg) scale(${sx}, ${1 / sx})` : ""
    }
    if (reduced()) { paint(to); return }
    const t0 = performance.now()
    let raf = 0
    const tick = (now: number) => {
      const p = Math.min(1, (now - t0) / MORPH_MS)
      paint(from + (to - from) * easeInOut(p))
      if (p < 1) raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [busy])
  return (
    <svg ref={svg} className="prompt-bar__glyph" viewBox="0 0 24 24" aria-hidden fill="currentColor" stroke="currentColor" strokeWidth="2" strokeLinejoin="round">
      <path ref={path} d={pathAt(busy ? 1 : 0)} />
    </svg>
  )
}

interface Row { key: string; name: string; description: string; icon: IconSvgElement; disabled?: boolean }

export interface PromptBarProps {
  value: string
  onChange: (v: string) => void
  inputRef: RefObject<HTMLTextAreaElement | null>
  attachments: { name: string }[]
  onRemoveAttachment: (i: number) => void
  onFiles: (files: File[]) => void
  onPasteFiles: (e: ClipboardEvent) => void
  busy: boolean
  onSend: () => void
  onStop: () => void
  onNewChat: () => void
  onSave: () => void
  onSampling: () => void
  efforts: string[]
  effort: number
  onEffort: (i: number) => void
  attachTitle: string
  dragging?: boolean
  children?: ReactNode                       // floats above the field (the "Latest" button)
}

export function PromptBar(p: PromptBarProps) {
  const root = useRef<HTMLDivElement>(null)
  const file = useRef<HTMLInputElement>(null)
  const glow = useRef<HTMLSpanElement>(null)
  const spark = useRef<HTMLCanvasElement>(null)
  const rows = useRef<(HTMLButtonElement | null)[]>([])
  const lastOpen = useRef<string | null>(null)
  const typing = useRef({ energy: 0, strokes: 0 })
  const [menu, setMenu] = useState<"plus" | "effort" | null>(null)
  const [active, setActive] = useState(0)
  const [focused, setFocused] = useState(false)
  const [visible, setVisible] = useState(() => typeof document === "undefined" || !document.hidden)
  const [pressed, setPressed] = useState(false)

  const list: Row[] = [
    { key: "attach", name: "Photos & files", description: p.attachTitle, icon: Attachment01Icon },
    { key: "new", name: "New chat", description: "Clears this chat, with undo", icon: MessageAdd01Icon, disabled: p.busy },
    { key: "save", name: "Save as Markdown", description: "Download the conversation", icon: Download01Icon },
  ]
  const cursor = Math.min(active, list.length - 1)
  const canSend = p.value.trim().length > 0 || p.attachments.length > 0
  const armed = p.busy || canSend
  const level = p.efforts[p.effort] ?? ""
  const maxed = p.efforts.length > 1 && p.effort === p.efforts.length - 1
  const sparking = maxed && focused && !p.busy && visible && !reduced()

  const closeMenu = useCallback(() => setMenu(null), [])
  const focusInput = () => p.inputRef.current?.focus({ preventScroll: true })
  // Focus first, then set the menu: giving the field focus closes menus (its onFocus), so the order matters when the field did not have it.
  const toggle = (kind: "plus" | "effort") => { const next = menu === kind ? null : kind; setActive(0); focusInput(); setMenu(next) }

  useEffect(() => {
    const on = () => setVisible(!document.hidden)
    document.addEventListener("visibilitychange", on)
    return () => document.removeEventListener("visibilitychange", on)
  }, [])

  // a menu closes when the pointer goes down anywhere else
  useEffect(() => {
    if (!menu) return
    const down = (e: PointerEvent) => { if (!root.current?.contains(e.target as Node)) closeMenu() }
    document.addEventListener("pointerdown", down)
    return () => document.removeEventListener("pointerdown", down)
  }, [menu, closeMenu])

  // the glow behind the rows slides to the row under the cursor (and is placed without a transition when the menu opens)
  useLayoutEffect(() => {
    const g = glow.current
    if (!g || menu !== "plus") return
    const row = rows.current[cursor]
    if (!row) { g.style.opacity = "0"; return }
    const fresh = lastOpen.current !== menu
    lastOpen.current = menu
    if (fresh) g.style.transition = "none"
    g.style.top = `${row.offsetTop}px`
    g.style.height = `${row.offsetHeight}px`
    g.style.opacity = "1"
    if (fresh) { void g.offsetHeight; g.style.transition = "" }
  }, [menu, cursor])
  useEffect(() => { if (menu !== "plus") lastOpen.current = null }, [menu])

  // sparks rising behind the field at the top effort level
  useEffect(() => {
    const canvas = spark.current
    if (!sparking || !canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return
    const color = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() || "#7aa7ff"
    typing.current.strokes = 0
    let raf = 0, last = performance.now(), w = 0, h = 0, due = 0, speed = 1, pulse = 0
    const parts: { x: number; y: number; r: number; vy: number; sway: number; phase: number; life: number; span: number }[] = []
    const resize = () => {
      const r = canvas.getBoundingClientRect(), dpr = Math.min(2, devicePixelRatio || 1)
      w = r.width; h = r.height
      canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr)
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    }
    const spawn = (burst: boolean) => parts.push({
      x: Math.random() * w, y: burst ? h * (0.2 + Math.random() * 0.8) : h + 3, r: 0.9 + Math.random() * 1.1, vy: -(7 + Math.random() * 9),
      sway: (Math.random() - 0.5) * 10, phase: Math.random() * Math.PI * 2, life: burst ? Math.random() * 1.2 : 0, span: 2.4 + Math.random() * 2.4,
    })
    const tick = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000)
      last = now
      const ty = typing.current
      ty.energy *= Math.exp(-dt / 0.8); pulse *= Math.exp(-dt / 0.16)
      if (ty.strokes > 0) { ty.strokes = 0; pulse = 1 }
      speed += (1 + ty.energy * 6 - speed) * (1 - Math.exp(-dt / 0.15))
      due += dt
      while (due > 0.14) { due -= 0.14; if (parts.length < 30) spawn(false) }
      ctx.clearRect(0, 0, w, h)
      ctx.fillStyle = color; ctx.shadowColor = color; ctx.shadowBlur = 6 + ty.energy * 10 + pulse * 6
      for (let i = parts.length - 1; i >= 0; i--) {
        const q = parts[i]
        q.life += dt
        if (q.life > q.span) { parts.splice(i, 1); continue }
        const k = q.life / q.span, tw = 0.7 + 0.3 * Math.sin((now / 160) * (1 + ty.energy) + q.phase)
        q.y += q.vy * dt * speed
        if (q.y < -4) { q.y = h + 3; q.x = Math.random() * w }
        const edge = Math.min(1, Math.max(0, q.y / 14), Math.max(0, (h - q.y) / 14))
        ctx.globalAlpha = Math.min(1, Math.sin(k * Math.PI) * (0.9 + ty.energy * 0.25) * tw) * edge
        ctx.beginPath()
        ctx.arc(q.x + Math.sin((now / 900) * (1 + ty.energy * 0.8) + q.phase) * q.sway, q.y, q.r * tw * (1 + ty.energy * 0.35), 0, Math.PI * 2)
        ctx.fill()
      }
      raf = requestAnimationFrame(tick)
    }
    resize()
    for (let i = 0; i < 26; i++) spawn(true)
    const ro = new ResizeObserver(resize)
    ro.observe(canvas)
    raf = requestAnimationFrame(tick)
    return () => { cancelAnimationFrame(raf); ro.disconnect(); ctx.clearRect(0, 0, w, h) }
  }, [sparking])

  // paste: pictures go to the chat, text stays in the field
  useEffect(() => {
    const el = p.inputRef.current
    if (!el) return
    el.addEventListener("paste", p.onPasteFiles)
    return () => el.removeEventListener("paste", p.onPasteFiles)
  }, [p.inputRef, p.onPasteFiles])

  const runRow = (key: string) => {
    closeMenu()
    if (key === "attach") file.current?.click()
    else if (key === "new") p.onNewChat()
    else if (key === "save") p.onSave()
  }

  const setEffort = (i: number) => { const n = Math.max(0, Math.min(p.efforts.length - 1, i)); if (n !== p.effort) p.onEffort(n) }
  const stepAt = (i: number) => `calc(${EDGE}px + (100% - ${EDGE * 2}px) * ${i / Math.max(1, p.efforts.length - 1)})`
  const fillAt = (i: number) => (i === p.efforts.length - 1 ? "100%" : `calc(${stepAt(i)} + 7px)`)
  const fromPointer = (e: RPointerEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    setEffort(Math.round(((e.clientX - r.left - EDGE) / Math.max(1, r.width - 2 * EDGE)) * (p.efforts.length - 1)))
  }
  const onEffortKey = (e: RKeyEvent<HTMLDivElement>) => {
    const step = e.key === "ArrowRight" || e.key === "ArrowUp" ? 1 : e.key === "ArrowLeft" || e.key === "ArrowDown" ? -1 : 0
    if (step) { e.preventDefault(); setEffort(p.effort + step) }
    else if (e.key === "Home") { e.preventDefault(); setEffort(0) }
    else if (e.key === "End") { e.preventDefault(); setEffort(p.efforts.length - 1) }
    else if (e.key === "Escape") { closeMenu(); focusInput() }
  }

  const onKeyDown = (e: RKeyEvent<HTMLTextAreaElement>) => {
    if (menu === "plus") {
      if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); setActive((cursor + (e.key === "ArrowDown" ? 1 : list.length - 1)) % list.length); return }
      if ((e.key === "Enter" && !e.shiftKey) || e.key === "Tab") { e.preventDefault(); if (!list[cursor].disabled) runRow(list[cursor].key); return }
    }
    if (e.key === "Escape" && menu) { e.preventDefault(); closeMenu(); return }
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); if (canSend && !p.busy) p.onSend() }   // not while an IME is composing
  }

  return (
    <div ref={root} className="prompt-bar" data-busy={p.busy ? "" : undefined}>
      {p.children}
      {menu && (
        <div className="prompt-bar__menu" role={menu === "effort" ? "dialog" : "listbox"} aria-label={menu === "effort" ? "Thinking effort" : "Actions"} data-kind={menu}>
          {menu === "effort" ? (
            <>
              <div className="prompt-bar__effort-head">
                <span className="prompt-bar__effort-title">Thinking</span>
                <span className="prompt-bar__effort-level">{level}</span>
                <span className="prompt-bar__effort-help" title="More thinking takes longer before the answer starts">
                  <HugeiconsIcon icon={HelpCircleIcon} size={14} strokeWidth={1.8} />
                </span>
              </div>
              <div className="prompt-bar__effort-ends"><span>Faster</span><span>Deeper</span></div>
              <div
                className="prompt-bar__effort-track"
                role="slider"
                tabIndex={0}
                aria-label="Thinking effort"
                aria-valuemin={0}
                aria-valuemax={p.efforts.length - 1}
                aria-valuenow={p.effort}
                aria-valuetext={level}
                style={{ "--pb-effort-x": stepAt(p.effort), "--pb-effort-fill": fillAt(p.effort) } as React.CSSProperties}
                onPointerDown={(e) => {
                  if (e.button !== 0) return
                  try { e.currentTarget.setPointerCapture(e.pointerId) } catch { /* not capturable */ }
                  e.currentTarget.focus({ preventScroll: true })
                  fromPointer(e)
                }}
                onPointerMove={(e) => { if (e.buttons & 1) fromPointer(e) }}
                onKeyDown={onEffortKey}
              >
                <span className="prompt-bar__effort-fill" />
                {p.efforts.map((label, i) => <i key={label} className="prompt-bar__effort-dot" style={{ left: stepAt(i) }} />)}
                <span className="prompt-bar__effort-thumb" />
              </div>
            </>
          ) : (
            <>
              <span ref={glow} className="prompt-bar__glow" aria-hidden />
              {list.map((row, i) => (
                <button
                  key={row.key}
                  ref={(el) => { rows.current[i] = el }}
                  type="button"
                  role="option"
                  aria-selected={i === cursor}
                  disabled={row.disabled}
                  className="prompt-bar__row"
                  onMouseDown={(e) => e.preventDefault()}
                  onPointerEnter={() => setActive(i)}
                  onClick={() => runRow(row.key)}
                >
                  <span className="prompt-bar__row-icon"><HugeiconsIcon icon={row.icon} size={15} strokeWidth={1.8} /></span>
                  <span className="prompt-bar__row-name">{row.name}</span>
                  <span className="prompt-bar__row-desc">{row.description}</span>
                </button>
              ))}
            </>
          )}
        </div>
      )}

      <div
        className="prompt-bar__field"
        role="presentation"
        data-drag={p.dragging ? "" : undefined}
        data-max={maxed ? "" : undefined}
        onFocusCapture={() => setFocused(true)}
        onBlurCapture={(e) => { if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setFocused(false) }}
        onPointerDown={(e) => { if (e.target === e.currentTarget || e.target === p.inputRef.current) closeMenu() }}
        onClick={focusInput}
      >
        <canvas ref={spark} className="prompt-bar__sparks" aria-hidden />
        {p.attachments.length > 0 && (
          <div className="prompt-bar__chips">
            {p.attachments.map((f, i) => (
              <span key={`${f.name}-${i}`} className="prompt-bar__chip">
                <HugeiconsIcon icon={File02Icon} size={12} strokeWidth={2} />
                <span className="prompt-bar__chip-name">{f.name}</span>
                <button type="button" className="prompt-bar__chip-x" aria-label={`Remove ${f.name}`} onClick={() => p.onRemoveAttachment(i)}>
                  <HugeiconsIcon icon={Cancel01Icon} size={10} strokeWidth={2.5} />
                </button>
              </span>
            ))}
          </div>
        )}

        <textarea
          ref={p.inputRef}
          className="prompt-bar__input"
          rows={1}
          value={p.value}
          placeholder="Message"
          aria-label="Message"
          onChange={(e) => {
            p.onChange(e.target.value)
            typing.current.energy = Math.min(1.6, typing.current.energy + 0.22)
            typing.current.strokes = Math.min(4, typing.current.strokes + 1)
            closeMenu()
            setActive(0)
          }}
          onFocus={closeMenu}
          onKeyDown={onKeyDown}
        />

        <div className="prompt-bar__bar">
          <button
            type="button"
            className="prompt-bar__tool"
            aria-label="Photos, files, new chat, save"
            aria-expanded={menu === "plus"}
            data-on={menu === "plus" ? "" : undefined}
            onMouseDown={(e) => e.preventDefault()}
            onClick={() => toggle("plus")}
          >
            <HugeiconsIcon icon={PlusSignIcon} size={16} strokeWidth={2} />
          </button>
          <input ref={file} type="file" multiple hidden aria-label="Attach files" onChange={(e) => { p.onFiles(Array.from(e.target.files || [])); e.target.value = "" }} />
          <button
            type="button"
            className="prompt-bar__pick"
            aria-label="Thinking effort"
            aria-expanded={menu === "effort"}
            data-on={menu === "effort" ? "" : undefined}
            data-max={maxed ? "" : undefined}
            onMouseDown={(e) => e.preventDefault()}
            onClick={() => toggle("effort")}
          >
            <HugeiconsIcon icon={SparklesIcon} size={13} strokeWidth={2} />
            <span>{level}</span>
          </button>
          <button type="button" className="prompt-bar__tool" aria-label="Sampling" title="Sampling" onMouseDown={(e) => e.preventDefault()} onClick={() => { closeMenu(); p.onSampling() }}>
            <HugeiconsIcon icon={Settings02Icon} size={15} strokeWidth={2} />
          </button>
          <span className="prompt-bar__spacer" />
          {!p.busy && <span className="prompt-bar__hint">Shift+Enter: new line</span>}
          <button
            type="button"
            className="prompt-bar__send"
            disabled={!armed}
            aria-label={p.busy ? "Stop" : "Send"}
            data-armed={armed ? "" : undefined}
            data-pressed={pressed ? "" : undefined}
            onMouseDown={(e) => e.preventDefault()}
            onPointerDown={(e) => { if (e.button === 0 && armed) setPressed(true) }}
            onPointerUp={() => setPressed(false)}
            onPointerCancel={() => setPressed(false)}
            onPointerLeave={() => setPressed(false)}
            onClick={() => { if (p.busy) p.onStop(); else if (canSend) p.onSend() }}
          >
            <SendGlyph busy={p.busy} />
          </button>
        </div>
      </div>
    </div>
  )
}
