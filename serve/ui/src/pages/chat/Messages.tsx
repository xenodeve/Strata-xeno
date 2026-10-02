import { useLayoutEffect, useRef, useState, type KeyboardEvent, type MouseEvent } from "react"
import { HugeiconsIcon } from "@hugeicons/react"
import { ArrowDown01Icon, Copy01Icon, AttachmentIcon, PencilEdit01Icon, RewindIcon, Undo02Icon } from "@hugeicons/core-free-icons"
import { chat, metaText, type Message, type ToolCall } from "../../lib/chat"
import { copyText } from "../../lib/files"
import { Collapse, Fit, Handover } from "../../components/motion"
import { Orb, StatusLabel } from "../../components/orb"
import { latticePattern, phaseKind, replyDesign, toolDesign, type OrbDesign } from "../../lib/orbs"
import { createPortal } from "react-dom"
import { agentStatus } from "../../lib/status"
import { useAvatar } from "../../lib/avatar"
import { Lattice, Thought, type LatticeStatus } from "../../components/thought"
import { cn } from "../../lib/cn"
import { fmt, timeStr } from "../../lib/format"
import { markdown } from "../../lib/markdown"
import { prefillText } from "../../lib/prefill"
import { takeSend } from "../../lib/sendfx"
import { ReasonStream } from "../../components/reason"
import { Pop } from "../../components/pop"
import { Spin } from "../../components/spin"
import { SkillText } from "../../components/SkillTip"
import { CompactingStatus } from "../../components/CompactingStatus"
import { skillCall, type SkillCall } from "../../lib/panel"
import { AgentCall, HookNotes, TodoList } from "./AgentCall"
import { msg, t } from "../../lib/i18n"

const TOOL_STATE: Record<ToolCall["state"], string> = { writing: msg("Writing"), asking: msg("Waiting for you"), running: msg("Running"), done: msg("Done"), error: msg("Error"), skipped: msg("Not run") }

export function Prose({ text }: { text: string }) {
  // The answer is escaped first and formatted by lib/markdown; copy buttons inside code blocks are one delegated click.
  const onClick = (e: MouseEvent<HTMLDivElement>) => {
    const b = (e.target as HTMLElement).closest("[data-code-copy]")
    if (b) void copyText(b.closest(".code-block")!.querySelector("pre")!.textContent || "")
  }
  return <div className="prose-chat" onClick={onClick} dangerouslySetInnerHTML={{ __html: markdown(text) }} />
}

// One MCP tool call in the answer: a compact row (name, state, a one-line preview) that opens to the arguments and the
// result as the model read it. The body is built only while open: a result can be 20,000 characters.
/** What a call of the skills server says in the chat: the model loaded a skill, read a file that comes with one, or looked for one. */
function skillTitle(s: SkillCall): string {
  return s.kind === "use" ? t("Used skill: {name}", { name: s.name ?? "?" }) : s.kind === "file" ? t("Read a file of the skill {name}", { name: s.name ?? "?" }) : t("Looked for a skill")
}

function Tool({ call }: { call: ToolCall }) {
  const sk = skillCall(call)
  const args = call.arguments == null ? "" : JSON.stringify(call.arguments, null, 2)
  const preview = call.result != null ? call.result : args.replace(/\s+/g, " ")
  return (
    <div className="my-2 rounded-md border border-line" data-state={call.state}>
      <button
        type="button"
        aria-expanded={!!call.open}
        onClick={() => { call.open = !call.open; chat.notify() }}
        className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-left text-[13px] transition-colors hover:bg-hover"
      >
        {(call.state === "running" || call.state === "writing") && <Orb design={toolDesign(call.name)} size={20} />}
        <span className={cn("font-medium", (call.state === "running" || call.state === "writing") && "t-shimmer")} title={call.name} data-skill-call={sk ? sk.kind : undefined}>{sk ? skillTitle(sk) : call.tool || call.name || t("tool")}</span>
        {call.server && !sk && <span className="text-ink-2">{call.server}</span>}
        <span className="min-w-0 flex-1 truncate text-ink-3">{preview.slice(0, 200)}</span>
        <span className={cn("shrink-0", call.state === "error" ? "text-bad" : "text-ink-2")}>{t(TOOL_STATE[call.state])}</span>
        {call.ms != null && call.state !== "skipped" && <span className="num shrink-0 text-ink-3">{fmt(call.ms / 1000, 1)} s</span>}
        <HugeiconsIcon icon={ArrowDown01Icon} size={14} aria-hidden className={cn("shrink-0 text-ink-3 transition-transform duration-200", call.open && "rotate-180")} />
      </button>
      <Collapse open={!!call.open}>
        <div className="space-y-2 border-t border-line px-3 py-2.5">
          <div className="text-[12px] text-ink-2">{t("Arguments")}</div>
          <pre className="tool-pre">{args || t("(being written)")}</pre>
          {call.result != null && (
            <>
              <div className="text-[12px] text-ink-2">
                {call.ok ? t("Result") : t("Error")}{call.chars ? ` · ${t("{n} characters", { n: fmt(call.chars) })}` : ""}{call.truncated ? `, ${t("cut for the model")}` : ""}
              </div>
              <pre className="tool-pre">{call.result}</pre>
            </>
          )}
        </div>
      </Collapse>
    </div>
  )
}

// the answer's text with the tool blocks where the model called them, and the thinking of each later round below the tools of the one before it: a long run of tools would otherwise leave
// the thinking only at the top, out of sight (the first round's thinking stays above, closed once a tool has started)
function Answer({ m, streaming, show, phase }: { m: Message; streaming: boolean; show: boolean; phase: OrbDesign | null }) {
  if (!m.tools?.length) return <Prose text={m.text || ""} />
  const reasoning = m.reasoning || ""
  const parts: React.ReactNode[] = []
  let pos = 0
  let rpos: number | undefined = typeof m.tools[0].rat === "number" ? Math.min(m.tools[0].rat, reasoning.length) : undefined       // what the thought above the answer has shown
  m.tools.forEach((tc, k) => {
    const at = Math.min(Math.max(tc.at || 0, pos), m.text.length)
    const rat = typeof tc.rat === "number" ? Math.min(tc.rat, reasoning.length) : undefined
    if (rpos !== undefined && rat !== undefined && rat > rpos) {                       // a new round: what the model thought before it called this tool
      const seg = reasoning.slice(rpos, rat)
      if (seg.trim()) parts.push(<RoundThought key={`r${k}`} text={seg} working={false} show={show} phase={phase} />)
      rpos = rat
    }
    if (at > pos) parts.push(<Prose key={`p${k}`} text={m.text.slice(pos, at)} />)
    pos = at
    parts.push(tc.server === "agent" ? <AgentCall key={tc.id} call={tc} /> : <Tool key={tc.id} call={tc} />)         // the coding tools show as what they are
  })
  const tail = rpos !== undefined ? reasoning.slice(rpos) : ""
  if (tail.trim()) {                                                                  // the thinking after the last tool: live while the model thinks again
    const idle = !m.tools.some((c) => c.state === "running" || c.state === "writing" || c.state === "asking")
    parts.push(<RoundThought key="tail" text={tail} working={streaming && idle && m.text.length <= pos} show={show} phase={phase} />)
  }
  parts.push(<Prose key="rest" text={m.text.slice(pos)} />)
  return <>{parts}</>
}

// The agent's thinking: a lattice that runs the pattern of what it is doing, beside a line that shimmers while it thinks
// and settles into "Thought for 2.4s"; it opens to the reasoning text.
/** The mark beside the thought: the lattice of dots (with the orbs, or the loaders alone); with the orbs only, or the avatars, the orb of what it is doing (thinking is Solving; a tool call or the next step
 *  after one takes its own form); when it is done the orb gives way to the lattice's tick, or its cross when the reply failed. */
function ThoughtGlyph({ working, status, design }: { working: boolean; status: LatticeStatus; design: OrbDesign }) {
  const kind = useAvatar()
  // the lattice of dots (LatticeLoader) marks the thinking, as it did, whether the rest are orbs or loaders
  if (kind === "mixed" || kind === "loading") return <Lattice status={status} pattern={latticePattern(design)} />
  return (
    <span className="relative inline-block size-5">
      <span className={cn("absolute inset-0 transition-opacity duration-300", working ? "opacity-100" : "opacity-0")} aria-hidden><Orb design={design} size={20} moving={working} /></span>
      <span className={cn("absolute inset-0 grid place-items-center transition-opacity duration-300", working ? "opacity-0" : "opacity-100")}><Lattice status={status} pattern="orbit" /></span>
    </span>
  )
}

/** The thinking of one later round of an agent's work: open while it streams (if wanted), closed once it is over, as the first one is. */
function RoundThought({ text, working, show, phase }: { text: string; working: boolean; show: boolean; phase: OrbDesign | null }) {
  const [touched, setTouched] = useState<boolean | null>(null)
  const open = touched ?? (working && show)
  return (
    <div className="mb-2" data-round-thought>
      <Thought working={working} glyph={<ThoughtGlyph working={working} status={working ? "working" : "done"} design={phase ?? "solving"} />} open={open} onToggle={() => setTouched(!open)}>
        <ReasonStream text={text} live={working} />
      </Thought>
    </div>
  )
}

function Thinking({ m, text, streaming, show, phase }: { m: Message; text: string; streaming: boolean; show: boolean; phase: OrbDesign | null }) {
  const [touched, setTouched] = useState<boolean | null>(null)       // the user's own choice, once made
  const thinkingNow = streaming && !m.text && !m.tools?.length
  const open = touched ?? (thinkingNow && show)                      // open while it streams (if wanted), closed once the answer starts
  const status: LatticeStatus = thinkingNow ? "working" : m.error ? "error" : "done"
  return (
    <div className="mb-2">
      <Thought
        working={thinkingNow}
        glyph={<ThoughtGlyph working={thinkingNow} status={status} design={phase ?? "solving"} />}
        open={open}
        onToggle={() => setTouched(!open)}
        elapsed={thinkingNow ? null : m.thinkSecs}
      >
        <ReasonStream text={text} live={thinkingNow} />
      </Thought>
    </div>
  )
}

/** How far the server is in reading what it is reading, as a share beside the words; pointing at it gives the tokens read of the tokens to read. */
function ReadShare({ p }: { p?: { read: number; total: number; percent: number } | null }) {
  const [tip, setTip] = useState<{ left: number; top: number } | null>(null)
  if (!p) return null
  const show = (e: { currentTarget: HTMLElement }) => { const r = e.currentTarget.getBoundingClientRect(); setTip({ left: Math.min(Math.max(r.left + r.width / 2, 120), window.innerWidth - 120), top: r.top - 8 }) }
  const words = t("{read} of {total} tokens read", { read: fmt(p.read), total: fmt(p.total) })
  return (
    <span className="num cursor-default text-[12px] text-ink-3" data-read-share tabIndex={0} onMouseEnter={show} onFocus={show} onMouseLeave={() => setTip(null)} onBlur={() => setTip(null)} aria-label={words}>
      {p.percent}%
      {tip && createPortal(<span role="tooltip" data-read-tip className="skill-tip" style={{ left: tip.left, top: tip.top, transform: "translate(-50%, -100%)" }}>{words}</span>, document.body)}
    </span>
  )
}

// What can be done to a prompt that was sent: rewrite it (it and everything after it are replaced), or, on the last one, take
// it back (the prompt returns to the composer and its answer goes). Not offered while an answer is being written.
export interface PromptActions { canAct: boolean; last: boolean; onEdit: (text: string) => void; onUndo: () => void; onRewind?: () => void }

function PromptEditor({ text, last, onSend, onCancel }: { text: string; last: boolean; onSend: (t: string) => void; onCancel: () => void }) {
  const [value, setValue] = useState(text)
  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Escape") { e.preventDefault(); onCancel() }
    else if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); if (value.trim()) onSend(value) }
  }
  return (
    <div className="msg-in w-full max-w-[85%] space-y-2">
      <textarea
        autoFocus
        aria-label={t("Edit the prompt")}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={onKey}
        onFocus={(e) => e.currentTarget.setSelectionRange(e.currentTarget.value.length, e.currentTarget.value.length)}
        className="field-sizing-content block max-h-72 min-h-12 w-full resize-none rounded-[20px] bg-fill px-4 py-2.5 text-[15px] tracking-[-0.011em] outline-none ring-1 ring-line focus-visible:ring-2 focus-visible:ring-[var(--accent)]"
      />
      <div className="flex items-center justify-end gap-2 text-[13px]">
        {!last && <span className="mr-auto text-ink-3">{t("Everything after this prompt is replaced.")}</span>}
        <button type="button" onClick={onCancel} className="rounded-sm px-2.5 py-1 text-ink-2 transition-colors hover:bg-hover hover:text-ink">{t("Cancel")}</button>
        <button type="button" disabled={!value.trim()} onClick={() => onSend(value)} className="rounded-sm bg-ink px-3 py-1 font-medium text-surface transition-opacity disabled:opacity-40">{t("Send")}</button>
      </div>
    </div>
  )
}

/** A message that was typed while the answer was being written: it waits in line (dashed), and can be taken back to be edited, dropped, or - when the answer was stopped - sent. */
export function QueuedMessage({ text, files, answering, onEdit, onRemove, onSend }: { text: string; files: { name: string }[]; answering: boolean; onEdit: () => void; onRemove: () => void; onSend: () => void }) {
  const small = "rounded-sm px-1.5 py-0.5 transition-colors hover:bg-hover hover:text-ink"
  return (
    <div className="msg-in flex flex-col items-end gap-1" data-queued>
      {files.length > 0 && <div className="flex flex-wrap justify-end gap-1.5">{files.map((f, i) => <span key={i} className="inline-flex items-center gap-1 rounded-sm bg-fill px-2 py-1 text-[12px]"><HugeiconsIcon icon={AttachmentIcon} size={12} aria-hidden />{f.name}</span>)}</div>}
      {text && <div className="max-w-[85%] whitespace-pre-wrap rounded-[20px] rounded-br-md border border-dashed border-line px-4 py-2.5 text-[15px] tracking-[-0.011em] text-ink-2 [overflow-wrap:anywhere]">{text}</div>}
      <div className="flex items-center gap-1 px-1 text-[12px] text-ink-3">
        <span>{answering ? t("Waits for the answer to end") : t("Not sent: the answer was stopped")}</span>
        {!answering && <button type="button" onClick={onSend} className={small}>{t("Send now")}</button>}
        <button type="button" onClick={onEdit} className={small}>{t("Edit")}</button>
        <button type="button" onClick={onRemove} className={small}>{t("Remove")}</button>
      </div>
    </div>
  )
}

/** The line the conversation is summarised under: the model is writing the summary that will take the place of the earlier messages. */
export function CompactingLine() {
  return <div className="msg-in"><CompactingStatus className="text-[13px] text-ink-2" /></div>
}

/** Where the earlier messages were summarised: a line across the chat that says so, and opens to the summary the model reads in their place. */
function CompactNotice({ m }: { m: Message }) {
  const [open, setOpen] = useState(false)
  const c = m.compact!
  return (
    <div className="msg-in" data-compact>
      <div className="flex items-center gap-3 text-[12px] text-ink-3">
        <span className="h-px flex-1 bg-line" aria-hidden />
        <button type="button" aria-expanded={open} onClick={() => setOpen(!open)} className="num flex items-center gap-1.5 rounded-sm px-2 py-1 transition-colors hover:bg-hover hover:text-ink">
          <span>{c.auto ? t("Conversation compacted automatically") : t("Conversation compacted")}</span>
          <span>· {t("{before} → about {after} tokens", { before: fmt(c.before), after: fmt(c.after) })}</span>
          <HugeiconsIcon icon={ArrowDown01Icon} size={12} aria-hidden className={cn("transition-transform duration-200", open && "rotate-180")} />
        </button>
        <span className="h-px flex-1 bg-line" aria-hidden />
      </div>
      <Collapse open={open}>
        <div className="mt-2 rounded-md border border-line px-3 py-2.5 text-[13px] text-ink-2">
          {c.focus && <p className="mb-2 text-ink-3">{t("Focus: {focus}", { focus: c.focus })}</p>}
          <Prose text={m.text} />
        </div>
      </Collapse>
    </div>
  )
}

export function MessageView({ m, streaming, compacting = false, show, prefill, actions, serverPhase, serverState, reading, liveTokens }: { m: Message; streaming: boolean; compacting?: boolean; show: boolean; prefill: boolean; actions?: PromptActions; serverPhase?: string | null; serverState?: string | null; reading?: { read: number; total: number; percent: number } | null; liveTokens?: { tokens: number; tokS: number | null } | null }) {
  const ref = useRef<HTMLDivElement>(null)
  const mine = useRef<HTMLDivElement>(null)
  const [editing, setEditing] = useState(false)
  // The prompt that was just sent rises out of the composer (which it left a moment ago) to its place, fading in on the way. Its
  // own entrance is replaced by this; a message that was not just sent (a reload, a rewrite) keeps the plain one.
  useLayoutEffect(() => {
    const el = mine.current
    const from = m.role === "user" && el ? takeSend(m.time) : null
    if (!el || !from || matchMedia("(prefers-reduced-motion: reduce)").matches) return
    const rise = Math.max(24, Math.min(260, from.top - el.getBoundingClientRect().top))
    el.style.animation = "none"
    el.animate([{ transform: `translateY(${rise}px) scale(0.97)`, opacity: 0 }, { opacity: 1, offset: 0.35 }, { transform: "none", opacity: 1 }],
      { duration: 460, easing: "cubic-bezier(0.23, 1, 0.32, 1)" })
  }, [])
  if (m.compact) return <CompactNotice m={m} />
  if (m.role === "user") {
    return (
      <div ref={mine} className="msg-in group flex flex-col items-end gap-1">
        {!!m.files?.length && (
          <div className="flex flex-wrap justify-end gap-1.5">
            {m.files.map((f, i) => (
              <span key={i} className="inline-flex items-center gap-1 rounded-sm bg-fill px-2 py-1 text-[12px]">
                <HugeiconsIcon icon={AttachmentIcon} size={12} aria-hidden />{f.name}
              </span>
            ))}
          </div>
        )}
        {!!m.images?.length && (
          <div className="flex flex-wrap justify-end gap-1.5">
            {m.images.map((im, i) => im.url
              ? <img key={i} src={im.url} alt={im.name || t("image")} className="max-h-56 max-w-full rounded-md" />
              : <span key={i} className="inline-flex items-center gap-1 rounded-sm bg-fill px-2 py-1 text-[12px]">{im.name || t("image")}</span>)}
          </div>
        )}
        {(m.text || editing) && (        // a prompt of only files has no words, so no empty bubble
          <Fit className="flex w-full justify-end">
            {editing && actions
              ? <PromptEditor text={m.text} last={actions.last} onCancel={() => setEditing(false)} onSend={(t) => { setEditing(false); actions.onEdit(t) }} />
              : <div className="max-w-[85%] whitespace-pre-wrap rounded-[20px] rounded-br-md bg-fill px-4 py-2.5 text-[15px] tracking-[-0.011em] [overflow-wrap:anywhere]"><SkillText text={m.text} /></div>}
          </Fit>
        )}
        <div className="num flex items-center gap-1 px-1 text-[12px] text-ink-3">
          <span>{t("You · {time}", { time: timeStr(m.time) })}</span>
          {actions?.canAct && !editing && (
            <span className="flex items-center opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100 [@media(hover:none)]:opacity-100">
              <button type="button" aria-label={t("Edit this prompt")} title={t("Edit")} onClick={() => setEditing(true)} className="flex size-6 items-center justify-center rounded-sm transition-colors hover:bg-hover hover:text-ink"><HugeiconsIcon icon={PencilEdit01Icon} size={14} aria-hidden /></button>
              {actions.onRewind && <button type="button" aria-label={t("Rewind to before this prompt")} title={t("Rewind: go back to before this prompt")} data-rewind-button onClick={actions.onRewind} className="flex size-6 items-center justify-center rounded-sm transition-colors hover:bg-hover hover:text-ink"><HugeiconsIcon icon={RewindIcon} size={14} aria-hidden /></button>}
              {actions.last && <button type="button" aria-label={t("Take this prompt back")} title={t("Undo: take the prompt back")} onClick={actions.onUndo} className="flex size-6 items-center justify-center rounded-sm transition-colors hover:bg-hover hover:text-ink"><HugeiconsIcon icon={Undo02Icon} size={14} aria-hidden /></button>}
            </span>
          )}
        </div>
        {prefill && m.prefill && prefillText(m.prefill) && (
          <div key={m.prefill.state} className="num fade-swap px-1 text-[12px] text-ink-3" role={m.prefill.state === "reading" ? "status" : undefined}>{m.prefill.state === "reading" ? <Spin text={prefillText(m.prefill)!} /> : <Pop text={prefillText(m.prefill)!} />}</div>
        )}
      </div>
    )
  }
  const waiting = streaming && !m.text && !m.tools?.length
  const phase = replyDesign({ streaming, reasoning: m.reasoning, text: m.text, tools: m.tools })
  const status = agentStatus({ streaming, reasoning: m.reasoning, text: m.text, tools: m.tools, serverState, hookRunning: m.hookRunning })        // what is going on now, in words (the thinking has its own live block)
  const first = m.tools?.length && typeof m.tools[0].rat === "number" ? (m.reasoning || "").slice(0, m.tools[0].rat) : m.reasoning || ""        // with tools: only the thinking before the first one is shown above them
  return (
    <div ref={ref} className="msg-in max-w-[min(100%,65ch)] text-[15px] tracking-[-0.011em] lg:max-w-[72ch]">
      {first.trim() && <Thinking m={m} text={first} streaming={streaming} show={show} phase={phase} />}
      {m.error ? (
        <div className="rounded-md border border-line px-3 py-2 text-[13px] text-bad [overflow-wrap:anywhere]">{m.error}</div>
      ) : waiting ? (
        compacting ? <CompactingStatus className="text-[13px] text-ink-2" />
        : m.reasoning ? null : phaseKind(serverPhase).kind === "reading" || serverState === "reading"
          ? <span className="inline-flex items-center gap-2"><StatusLabel design="listening" className="text-[13px] text-ink-2">{t("Reading the prompt…")}</StatusLabel><ReadShare p={reading} /></span>
          : <StatusLabel design="breathing" className="text-[13px] text-ink-2">{t("Waiting for the model…")}</StatusLabel>
      ) : (
        <>
          {!!m.todos?.length && <TodoList todos={m.todos} />}
          <div className={cn(streaming && "streaming")}><Answer m={m} streaming={streaming} show={show} phase={phase} /></div>
          {!!m.hooks?.length && <HookNotes notes={m.hooks} className="mt-2 rounded-md border border-line [&>div:first-child]:border-t-0" />}
        </>
      )}
      <div className="mt-1 flex min-h-6 items-center gap-2 text-[12px] text-ink-3">
        <Handover live={status && status.kind !== "thinking" ? <span data-agent-status={status.kind} className="inline-flex items-center gap-2"><StatusLabel design={status.design} className="text-[13px] text-ink-2">{status.label}</StatusLabel>{status.kind === "reading" && <ReadShare p={reading} />}</span> : null}>
          <span className="num">{metaText(m) || (streaming ? "" : m.stopped ? t("Stopped") : "")}</span>
        </Handover>
        {streaming && !!liveTokens && <span className="num" data-live-tokens role="status" aria-label={t("Tokens written so far")}>{t("{n} tokens", { n: fmt(liveTokens.tokens) })}{liveTokens.tokS ? ` · ${fmt(liveTokens.tokS, 1)} tok/s` : ""}</span>}
        {!streaming && !!m.text && (
          <button
            type="button"
            aria-label={t("Copy the answer")}
            title={t("Copy")}
            onClick={() => void copyText(m.text)}
            className="flex size-6 items-center justify-center rounded-sm transition-colors hover:bg-hover hover:text-ink"
          >
            <HugeiconsIcon icon={Copy01Icon} size={14} aria-hidden />
          </button>
        )}
      </div>
    </div>
  )
}
