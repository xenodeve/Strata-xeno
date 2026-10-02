import { useCallback, useEffect, useLayoutEffect, useRef, useState, type KeyboardEvent as RKeyEvent, type PointerEvent as RPointerEvent, type ReactNode, type RefObject } from "react"
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react"
import {
  Attachment01Icon, Cancel01Icon, Download01Icon, File02Icon, HelpCircleIcon, PlusSignIcon, Settings02Icon, SparklesIcon, MessageAdd01Icon, PlugSocketIcon, ComputerTerminal01Icon, Shield01Icon,
} from "@hugeicons/core-free-icons"
import { msg, t } from "../lib/i18n"
import type { McpServer } from "../lib/api"
import { markOf, matchCommands, pickCommand, slashQuery, type Command } from "../lib/slash"
import type { ContextView, PartKey } from "../lib/context"
import { fmt } from "../lib/format"
import { SkillCard } from "./SkillTip"
import { Button, MiniSwitch } from "./ui"
import { AgentControls, ModePicker, modeLabel, type AgentControlsProps } from "./AgentControls"
import { GlidePanel } from "./motion"

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

const MCP_STATE: Record<string, string> = { ready: msg("Connected"), starting: msg("Starting"), failed: msg("Failed"), stopped: msg("Stopped"), idle: msg("Waiting") }

interface Row { key: string; name: string; description: string; icon: IconSvgElement; disabled?: boolean; state?: string; checked?: boolean }

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
  agent: AgentControlsProps                  // the coding tools (issue #96): a row in the + menu that opens their switch, mode and folder
  context: { view: ContextView; canCompact: boolean; onCompact: () => void; open: number }      // the context window (issue #99): a chip with the share used, a panel with what it is made of; `open` counts the times /context asked for the panel
  skills: Command[]                          // the skills in use: "/" at the start of the message lists them, to be picked by name
  mcp: { servers: McpServer[]; tools: number; on: boolean; off: string[]; onToggleAll: () => void; onToggleServer: (name: string) => void; setupHref: string }      // the MCP tools: a row in the + menu that opens the list of servers, each with its tools and its own switch
  efforts: string[]
  effort: number
  onEffort: (i: number) => void
  attachTitle: string
  dragging?: boolean
  children?: ReactNode                       // floats above the field (the "Latest" button)
}

const PART_LABEL: Record<PartKey, string> = {
  conversation: msg("Conversation"), tools: msg("Tool calls and results"), summary: msg("Summary of earlier messages"), system: msg("Instructions, tools and memory"), free: msg("Free"),
}

/** The share of the window that is used, as a small ring. */
function ContextRing({ pct }: { pct: number }) {
  const r = 5.5, c = 2 * Math.PI * r
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden className="prompt-bar__ring">
      <circle cx="7" cy="7" r={r} fill="none" strokeWidth="2" className="prompt-bar__ring-track" />
      <circle cx="7" cy="7" r={r} fill="none" strokeWidth="2" strokeLinecap="round" strokeDasharray={`${(c * pct) / 100} ${c}`} transform="rotate(-90 7 7)" className="prompt-bar__ring-fill" />
    </svg>
  )
}

/** What the context window holds: how much is used, what it is made of, where it is compacted by itself. */
function ContextPanel({ view, canCompact, onCompact }: { view: ContextView; canCompact: boolean; onCompact: () => void }) {
  const shown = view.parts.filter((x) => x.key === "free" || x.tokens > 0)
  return (
    <div data-context-panel>
      <div className="prompt-bar__ctx-head">
        <span className="prompt-bar__ctx-title">{t("Context window")}</span>
        <span className="num prompt-bar__ctx-pct" data-level={view.level}>{view.pct}%</span>
      </div>
      <p className="num mt-0.5 text-[12px] text-ink-2" data-context-figures>{view.exact ? "" : t("About") + " "}{t("{used} of {max} tokens", { used: fmt(view.used), max: fmt(view.max) })}</p>
      <div className="prompt-bar__ctx-bar" role="img" aria-label={t("What the context window holds")}>
        {shown.map((x) => <span key={x.key} data-part={x.key} style={{ flexGrow: x.tokens }} />)}
      </div>
      <ul className="prompt-bar__ctx-list">
        {shown.map((x) => (
          <li key={x.key} data-part={x.key}>
            <i aria-hidden data-part={x.key} />
            <span>{t(PART_LABEL[x.key])}</span>
            <span className="num">{fmt(x.tokens)}</span>
          </li>
        ))}
      </ul>
      {view.autoAt !== null && <p className="mt-2 text-[12px] text-ink-2" data-context-auto>{t("Compacted by itself at {n} tokens ({pct}%).", { n: fmt(view.autoAt), pct: Math.round((view.autoAt / view.max) * 100) })}</p>}
      {!view.exact && <p className="mt-1 text-[12px] text-ink-3">{t("The server has not reported the use yet: this is a guess from the text.")}</p>}
      <div className="mt-3"><Button onClick={onCompact} disabled={!canCompact}>{t("Compact now")}</Button></div>
    </div>
  )
}

export function PromptBar(p: PromptBarProps) {
  const root = useRef<HTMLDivElement>(null)
  const file = useRef<HTMLInputElement>(null)
  const glow = useRef<HTMLSpanElement>(null)
  const spark = useRef<HTMLCanvasElement>(null)
  const rows = useRef<(HTMLButtonElement | null)[]>([])
  const lastOpen = useRef<string | null>(null)
  const typing = useRef({ energy: 0, strokes: 0 })
  const asked = useRef(p.context.open)
  const [menu, setMenu] = useState<"plus" | "effort" | "mcp" | "agent" | "mode" | "context" | null>(null)
  const [active, setActive] = useState(0)
  const [focused, setFocused] = useState(false)
  const [visible, setVisible] = useState(() => typeof document === "undefined" || !document.hidden)
  const [pressed, setPressed] = useState(false)
  const [caret, setCaret] = useState(0)
  const [slashAt, setSlashAt] = useState(0)                      // the mark in the list of skills
  const [shut, setShut] = useState<string | null>(null)          // the text at which Escape closed the list: it stays closed until the text changes
  const skillRows = useRef<(HTMLButtonElement | null)[]>([])
  const mirror = useRef<HTMLDivElement>(null)
  const [tipAt, setTipAt] = useState<number | null>(null)       // pointing at the marked command: where its card goes (from the left of the bar)

  useEffect(() => { if (p.context.open !== asked.current) { asked.current = p.context.open; setMenu("context") } }, [p.context.open])      // /context

  const usable = (x: McpServer) => x.tools.length > 0 && (x.status === "ready" || x.status === "stopped")       // a server that is up and offers tools can be switched
  const mcpOn = p.mcp.on ? p.mcp.servers.filter((x) => usable(x) && !p.mcp.off.includes(x.name)).length : 0
  const usableAgent = p.agent.info.available && p.agent.info.allowed
  const list: Row[] = [
    { key: "attach", name: t("Photos & files"), description: p.attachTitle, icon: Attachment01Icon },
    { key: "new", name: t("New chat"), description: t("Starts a new chat; this one stays in Recents"), icon: MessageAdd01Icon, disabled: p.busy },
    { key: "save", name: t("Save as Markdown"), description: t("Download the conversation"), icon: Download01Icon },
    {
      key: "mcp", name: t("MCP tools"), icon: PlugSocketIcon, checked: mcpOn > 0,
      description: p.mcp.servers.length === 0 ? t("No server is set up. Add them in Settings.") : p.mcp.tools ? t("{on} of {n} servers on · {tools} tools", { on: mcpOn, n: p.mcp.servers.length, tools: p.mcp.tools }) : t("No server is connected yet."),
      state: p.mcp.servers.length === 0 ? undefined : mcpOn === 0 ? t("off") : mcpOn === p.mcp.servers.length ? t("on") : `${mcpOn}/${p.mcp.servers.length}`,
    },
    {
      key: "agent", name: t("Coding tools"), icon: ComputerTerminal01Icon, checked: usableAgent && p.agent.on,
      description: !p.agent.info.available ? t("Not on this server") : !p.agent.info.allowed ? t("Only from the PC that runs Strata") : p.agent.folders.length ? p.agent.folders[0] + (p.agent.folders.length > 1 ? ` +${p.agent.folders.length - 1}` : "") : t("No folder yet. Set one to work in."),
      state: usableAgent ? (p.agent.on ? t("on") : t("off")) : undefined,
    },
    { key: "sampling", name: t("Sampling"), description: t("Temperature, top-p, seed and more"), icon: Settings02Icon },
  ]
  const cursor = Math.min(active, list.length - 1)
  const canSend = p.value.trim().length > 0 || p.attachments.length > 0
  const armed = p.busy || canSend
  const level = p.efforts[p.effort] ?? ""
  const maxed = p.efforts.length > 1 && p.effort === p.efforts.length - 1
  const sparking = maxed && focused && !p.busy && visible && !reduced()

  const query = slashQuery(p.value, caret)
  const found = query !== null && p.skills.length > 0 ? matchCommands(p.skills, query) : []
  const slashOpen = found.length > 0 && shut !== p.value && menu === null
  const slashMark = Math.min(slashAt, Math.max(0, found.length - 1))
  const marked = markOf(p.value, p.skills)                       // the message starts with /name of a skill in use: that word is shown in bold
  const pointAt = (e: { clientX: number; clientY: number }) => {
    const m = mirror.current?.querySelector("[data-mark]"), r = root.current?.getBoundingClientRect()
    const b = m?.getBoundingClientRect()
    if (!m || !r || !b) { setTipAt(null); return }
    const on = e.clientX >= b.left && e.clientX <= b.right && e.clientY >= b.top && e.clientY <= b.bottom
    setTipAt(on ? Math.max(0, b.left - r.left) : null)
  }
  useEffect(() => { if (slashOpen) skillRows.current[slashMark]?.scrollIntoView({ block: "nearest" }) }, [slashOpen, slashMark])
  const pickSkill = (name: string) => {
    const r = pickCommand(p.value, name)
    p.onChange(r.text)
    setCaret(r.caret)
    setSlashAt(0)
    requestAnimationFrame(() => { const el = p.inputRef.current; if (el) { el.focus({ preventScroll: true }); el.setSelectionRange(r.caret, r.caret) } })
  }
  const closeMenu = useCallback(() => setMenu(null), [])
  const focusInput = () => p.inputRef.current?.focus({ preventScroll: true })
  // Focus first, then set the menu: giving the field focus closes menus (its onFocus), so the order matters when the field did not have it.
  const toggle = (kind: "plus" | "effort" | "mcp" | "mode" | "context") => { const next = menu === kind ? null : kind; setActive(0); focusInput(); setMenu(next) }

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
    if (key === "mcp") { setMenu("mcp"); return }             // opens the list of servers, each with its tools and its own switch
    if (key === "agent") { setMenu("agent"); return }         // opens the coding tools: switch, mode, folder
    closeMenu()
    if (key === "attach") file.current?.click()
    else if (key === "new") p.onNewChat()
    else if (key === "save") p.onSave()
    else if (key === "sampling") p.onSampling()
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
    if (slashOpen && !e.nativeEvent.isComposing) {
      if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); setSlashAt((slashMark + (e.key === "ArrowDown" ? 1 : found.length - 1)) % found.length); return }
      const marked = found[slashMark]
      const typedInFull = e.key === "Enter" && marked.builtin && p.value.trim().toLowerCase() === "/" + marked.name                  // "/compact" and Enter: a command of Strata is sent, as in Claude Code, not completed
      if (((e.key === "Enter" && !e.shiftKey) || e.key === "Tab") && !typedInFull) { e.preventDefault(); pickSkill(marked.name); return }
      if (e.key === "Escape") { e.preventDefault(); setShut(p.value); return }
    }
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
      {slashOpen && (
        <div className="prompt-bar__menu" data-kind="slash" role="listbox" aria-label={t("Skills")}>
          {found.map((c, i) => (
            <button
              key={c.name}
              ref={(el) => { skillRows.current[i] = el }}
              type="button"
              role="option"
              aria-selected={i === slashMark}
              data-skill={c.name}
              className="prompt-bar__skill"
              onMouseDown={(e) => e.preventDefault()}
              onPointerEnter={() => setSlashAt(i)}
              onClick={() => pickSkill(c.name)}
            >
              <span className="prompt-bar__skill-name">/{c.name}</span>
              <span className="prompt-bar__skill-desc">{c.description}</span>
            </button>
          ))}
        </div>
      )}

      {marked && tipAt !== null && !slashOpen && (
        <div className="prompt-bar__menu" data-kind="tip" role="tooltip" style={{ left: tipAt }}>
          <SkillCard cmd={marked.cmd} />
        </div>
      )}

      {menu === "context" && (
        <div className="prompt-bar__menu" role="dialog" aria-label={t("Context window")} data-kind="context" onKeyDown={(e) => { if (e.key === "Escape") { e.preventDefault(); closeMenu(); focusInput() } }}>
          <GlidePanel cap={() => Math.min(innerHeight * 0.7, 520)} inner="px-4 pb-4 pt-3.5">
            <ContextPanel view={p.context.view} canCompact={p.context.canCompact} onCompact={() => { closeMenu(); p.context.onCompact() }} />
          </GlidePanel>
        </div>
      )}

      {menu === "mode" && (
        <div className="prompt-bar__menu" role="dialog" aria-label={t("Coding mode")} data-kind="mode" onKeyDown={(e) => { if (e.key === "Escape") { e.preventDefault(); closeMenu(); focusInput() } }}>
          <GlidePanel cap={() => Math.min(innerHeight * 0.6, 420)} inner="px-4 pb-4 pt-3.5">
            <ModePicker mode={p.agent.mode} onMode={p.agent.onMode} />
          </GlidePanel>
        </div>
      )}

      {menu === "agent" && (
        <div className="prompt-bar__menu" role="dialog" aria-label={t("Coding tools")} data-kind="agent">
          <GlidePanel cap={() => Math.min(innerHeight * 0.7, 520)} inner="px-4 pb-4 pt-3.5">      {/* it stretches where its content grows (the description of a mode, a row that comes), not as a whole */}
            <AgentControls {...p.agent} withMode={false} />
          </GlidePanel>
        </div>
      )}

      {menu === "mcp" && (
        <div className="prompt-bar__menu" role="dialog" aria-label={t("MCP tools")} data-kind="mcp">
          <div className="prompt-bar__mcp-head">
            <span className="prompt-bar__mcp-title">{t("MCP tools")}</span>
            <span className="prompt-bar__mcp-count">{p.mcp.servers.length ? t("{n} tools", { n: p.mcp.tools }) : ""}</span>
            {p.mcp.servers.length > 0 && <MiniSwitch on={p.mcp.on} label={t("All MCP tools")} onClick={p.mcp.onToggleAll} />}
          </div>
          {p.mcp.servers.length === 0 ? (
            <p className="prompt-bar__mcp-empty">{t("No server is set up. Add them in Settings.")}</p>
          ) : (
            <ul className="prompt-bar__mcp-list" aria-label={t("MCP servers")}>
              {p.mcp.servers.map((x) => {
                const can = usable(x)
                const on = p.mcp.on && can && !p.mcp.off.includes(x.name)
                return (
                  <li key={x.name} className="prompt-bar__mcp-item" data-server={x.name} data-off={on ? undefined : ""}>
                    <div className="prompt-bar__mcp-line">
                      <span className="prompt-bar__mcp-name">{x.name}</span>
                      <span className="prompt-bar__mcp-state" data-bad={x.status === "failed" ? "" : undefined}>{MCP_STATE[x.status] ? t(MCP_STATE[x.status]) : x.status}</span>
                      <MiniSwitch on={on} disabled={!can || !p.mcp.on} label={t("Use {name}", { name: x.name })} onClick={() => p.mcp.onToggleServer(x.name)} />
                    </div>
                    {x.error && <div className="prompt-bar__mcp-error">{x.error}</div>}
                    {x.tools.length > 0 && (
                      <ul className="prompt-bar__mcp-tools" aria-label={t("Tools of {name}", { name: x.name })}>
                        {x.tools.map((tl) => <li key={tl.tool} title={tl.description || undefined}>{tl.tool}</li>)}
                      </ul>
                    )}
                  </li>
                )
              })}
            </ul>
          )}
          <a className="prompt-bar__mcp-setup" href={p.mcp.setupHref} onClick={closeMenu}>{t("Set up servers")}</a>
        </div>
      )}

      {menu === "effort" && (
        <div className="prompt-bar__menu" role="dialog" aria-label={t("Thinking effort")} data-kind="effort">
          {menu === "effort" ? (
            <>
              <div className="prompt-bar__effort-head">
                <span className="prompt-bar__effort-title">{t("Thinking")}</span>
                <span className="prompt-bar__effort-level">{level}</span>
                <span className="prompt-bar__effort-help" title={t("More thinking takes longer before the answer starts")}>
                  <HugeiconsIcon icon={HelpCircleIcon} size={14} strokeWidth={1.8} />
                </span>
              </div>
              <div className="prompt-bar__effort-ends"><span>{t("Faster")}</span><span>{t("Deeper")}</span></div>
              <div
                className="prompt-bar__effort-track"
                role="slider"
                tabIndex={0}
                aria-label={t("Thinking effort")}
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
          ) : null}
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
                <button type="button" className="prompt-bar__chip-x" aria-label={t("Remove {name}", { name: f.name })} onClick={() => p.onRemoveAttachment(i)}>
                  <HugeiconsIcon icon={Cancel01Icon} size={10} strokeWidth={2.5} />
                </button>
              </span>
            ))}
          </div>
        )}

        <div className="prompt-bar__inputwrap" data-marked={marked ? "" : undefined}>
        {marked && (
          <div ref={mirror} className="prompt-bar__mirror" aria-hidden>
            {p.value.slice(0, marked.at)}<span className="prompt-bar__mark" data-mark>{p.value.slice(marked.at, marked.end)}</span>{p.value.slice(marked.end)}{"\n"}
          </div>
        )}
        <textarea
          ref={p.inputRef}
          className="prompt-bar__input"
          rows={1}
          value={p.value}
          placeholder={t("Message")}
          aria-label={t("Message")}
          onChange={(e) => {
            p.onChange(e.target.value)
            setCaret(e.target.selectionStart ?? e.target.value.length)
            setSlashAt(0)
            setShut(null)
            typing.current.energy = Math.min(1.6, typing.current.energy + 0.22)
            typing.current.strokes = Math.min(4, typing.current.strokes + 1)
            closeMenu()
            setActive(0)
          }}
          onFocus={closeMenu}
          onSelect={(e) => setCaret(e.currentTarget.selectionStart ?? 0)}
          onScroll={(e) => { if (mirror.current) mirror.current.scrollTop = e.currentTarget.scrollTop }}
          onMouseMove={pointAt}
          onMouseLeave={() => setTipAt(null)}
          onKeyDown={onKeyDown}
        />
        </div>

        <div className="prompt-bar__bar">
          {/* the button is the menu: it grows into the panel and closes back into the button (transitions.dev "Plus to menu morph") */}
          <span className="prompt-bar__anchor">
            <div className="t-morph prompt-bar__morph" data-open={menu === "plus" ? "true" : "false"} style={{ "--pb-rows": list.length } as React.CSSProperties}>
              <div className="t-morph-menu" role="listbox" aria-label={t("Actions")} inert={menu !== "plus"}>
                  <span ref={glow} className="prompt-bar__glow" aria-hidden />
                  {list.map((row, i) => (
                    <button
                      key={row.key}
                      ref={(el) => { rows.current[i] = el }}
                      type="button"
                      role="option"
                      aria-selected={i === cursor}
                  aria-checked={row.checked}
                      disabled={row.disabled}
                      className="prompt-bar__row"
                      onMouseDown={(e) => e.preventDefault()}
                      onPointerEnter={() => setActive(i)}
                      onClick={() => runRow(row.key)}
                    >
                      <span className="prompt-bar__row-icon"><HugeiconsIcon icon={row.icon} size={15} strokeWidth={1.8} /></span>
                      <span className="prompt-bar__row-name">{row.name}</span>
                      <span className="prompt-bar__row-desc">{row.description}</span>
                  {row.state && <span className="prompt-bar__row-state" data-on={row.checked ? "" : undefined}>{row.state}</span>}
                    </button>
                  ))}
              </div>
              <button
                type="button"
                className="t-morph-plus prompt-bar__tool"
                aria-label={t("Photos, files, new chat, save")}
                aria-expanded={menu === "plus"}
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => toggle("plus")}
              >
                <HugeiconsIcon icon={PlusSignIcon} size={16} strokeWidth={2} />
              </button>
            </div>
          </span>
          <input ref={file} type="file" multiple hidden aria-label={t("Attach files")} onChange={(e) => { p.onFiles(Array.from(e.target.files || [])); e.target.value = "" }} />
          {usableAgent && p.agent.on && (
            <button
              type="button"
              className="prompt-bar__pick"
              aria-label={t("Coding mode")}
              aria-expanded={menu === "mode"}
              data-on={menu === "mode" ? "" : undefined}
              data-agent-mode={p.agent.mode}
              onMouseDown={(e) => e.preventDefault()}
              onClick={() => toggle("mode")}
            >
              <HugeiconsIcon icon={Shield01Icon} size={13} strokeWidth={2} />
              <span>{modeLabel(p.agent.mode)}</span>
            </button>
          )}
          <button
            type="button"
            className="prompt-bar__pick"
            aria-label={t("Thinking effort")}
            aria-expanded={menu === "effort"}
            data-on={menu === "effort" ? "" : undefined}
            data-max={maxed ? "" : undefined}
            onMouseDown={(e) => e.preventDefault()}
            onClick={() => toggle("effort")}
          >
            <HugeiconsIcon icon={SparklesIcon} size={13} strokeWidth={2} />
            <span>{level}</span>
          </button>
          <span className="prompt-bar__spacer" />
          {p.context.view.known && (
            <button
              type="button"
              className="prompt-bar__pick prompt-bar__ctx"
              aria-label={t("Context window: {pct}% used", { pct: p.context.view.pct })}
              aria-expanded={menu === "context"}
              data-on={menu === "context" ? "" : undefined}
              data-level={p.context.view.level}
              onMouseDown={(e) => e.preventDefault()}
              onClick={() => toggle("context")}
            >
              <ContextRing pct={p.context.view.pct} />
              <span className="num">{p.context.view.pct}%</span>
            </button>
          )}
          {!p.busy && <span className="prompt-bar__hint">{t("Shift+Enter: new line")}</span>}
          <button
            type="button"
            className="prompt-bar__send"
            disabled={!armed}
            aria-label={p.busy ? t("Stop") : t("Send")}
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
