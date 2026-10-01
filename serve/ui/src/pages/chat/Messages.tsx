import { useRef, useState, type KeyboardEvent, type MouseEvent } from "react"
import { HugeiconsIcon } from "@hugeicons/react"
import { ArrowDown01Icon, Copy01Icon, AttachmentIcon, PencilEdit01Icon, Undo02Icon } from "@hugeicons/core-free-icons"
import { chat, metaText, type Message, type ToolCall } from "../../lib/chat"
import { copyText } from "../../lib/files"
import { Collapse, Fit } from "../../components/motion"
import { Orb, StatusLabel } from "../../components/orb"
import { latticePattern, replyDesign, toolDesign, type OrbDesign } from "../../lib/orbs"
import { Lattice, Thought, type LatticeStatus } from "../../components/thought"
import { cn } from "../../lib/cn"
import { fmt, timeStr } from "../../lib/format"
import { markdown } from "../../lib/markdown"
import { prefillText } from "../../lib/prefill"
import { msg, t } from "../../lib/i18n"

const TOOL_STATE: Record<ToolCall["state"], string> = { writing: msg("Writing"), running: msg("Running"), done: msg("Done"), error: msg("Error"), skipped: msg("Not run") }

function Prose({ text }: { text: string }) {
  // The answer is escaped first and formatted by lib/markdown; copy buttons inside code blocks are one delegated click.
  const onClick = (e: MouseEvent<HTMLDivElement>) => {
    const b = (e.target as HTMLElement).closest("[data-code-copy]")
    if (b) void copyText(b.closest(".code-block")!.querySelector("pre")!.textContent || "")
  }
  return <div className="prose-chat" onClick={onClick} dangerouslySetInnerHTML={{ __html: markdown(text) }} />
}

// One MCP tool call in the answer: a compact row (name, state, a one-line preview) that opens to the arguments and the
// result as the model read it. The body is built only while open: a result can be 20,000 characters.
function Tool({ call }: { call: ToolCall }) {
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
        <span className={cn("font-medium", (call.state === "running" || call.state === "writing") && "t-shimmer")} title={call.name}>{call.tool || call.name || t("tool")}</span>
        {call.server && <span className="text-ink-2">{call.server}</span>}
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

// the answer's text with the tool blocks where the model called them
function Answer({ m }: { m: Message }) {
  if (!m.tools?.length) return <Prose text={m.text || ""} />
  const parts: React.ReactNode[] = []
  let pos = 0
  m.tools.forEach((tc, k) => {
    const at = Math.min(Math.max(tc.at || 0, pos), m.text.length)
    if (at > pos) parts.push(<Prose key={`p${k}`} text={m.text.slice(pos, at)} />)
    pos = at
    parts.push(<Tool key={tc.id} call={tc} />)
  })
  parts.push(<Prose key="rest" text={m.text.slice(pos)} />)
  return <>{parts}</>
}

// The agent's thinking: a lattice that runs the pattern of what it is doing, beside a line that shimmers while it thinks
// and settles into "Thought for 2.4s"; it opens to the reasoning text.
function Thinking({ m, streaming, show, phase }: { m: Message; streaming: boolean; show: boolean; phase: OrbDesign | null }) {
  const [touched, setTouched] = useState<boolean | null>(null)       // the user's own choice, once made
  const thinkingNow = streaming && !m.text
  const open = touched ?? (thinkingNow && show)                      // open while it streams (if wanted), closed once the answer starts
  const status: LatticeStatus = thinkingNow ? "working" : m.error ? "error" : "done"
  return (
    <div className="mb-2">
      <Thought
        working={thinkingNow}
        glyph={<Lattice status={status} pattern={latticePattern(phase)} />}
        open={open}
        onToggle={() => setTouched(!open)}
        elapsed={thinkingNow ? null : m.thinkSecs}
      >
        <div className="mt-1 max-h-72 overflow-y-auto whitespace-pre-wrap border-l border-line pl-3 text-[13px] font-normal leading-relaxed text-ink-2 [overflow-wrap:anywhere]">{m.reasoning}</div>
      </Thought>
    </div>
  )
}

// What can be done to a prompt that was sent: rewrite it (it and everything after it are replaced), or, on the last one, take
// it back (the prompt returns to the composer and its answer goes). Not offered while an answer is being written.
export interface PromptActions { canAct: boolean; last: boolean; onEdit: (text: string) => void; onUndo: () => void }

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

export function MessageView({ m, streaming, show, prefill, actions }: { m: Message; streaming: boolean; show: boolean; prefill: boolean; actions?: PromptActions }) {
  const ref = useRef<HTMLDivElement>(null)
  const [editing, setEditing] = useState(false)
  if (m.role === "user") {
    return (
      <div className="msg-in group flex flex-col items-end gap-1">
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
              : <div className="max-w-[85%] whitespace-pre-wrap rounded-[20px] rounded-br-md bg-fill px-4 py-2.5 text-[15px] tracking-[-0.011em] [overflow-wrap:anywhere]">{m.text}</div>}
          </Fit>
        )}
        <div className="num flex items-center gap-1 px-1 text-[12px] text-ink-3">
          <span>{t("You · {time}", { time: timeStr(m.time) })}</span>
          {actions?.canAct && !editing && (
            <span className="flex items-center opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100 [@media(hover:none)]:opacity-100">
              <button type="button" aria-label={t("Edit this prompt")} title={t("Edit")} onClick={() => setEditing(true)} className="flex size-6 items-center justify-center rounded-sm transition-colors hover:bg-hover hover:text-ink"><HugeiconsIcon icon={PencilEdit01Icon} size={14} aria-hidden /></button>
              {actions.last && <button type="button" aria-label={t("Take this prompt back")} title={t("Undo: take the prompt back")} onClick={actions.onUndo} className="flex size-6 items-center justify-center rounded-sm transition-colors hover:bg-hover hover:text-ink"><HugeiconsIcon icon={Undo02Icon} size={14} aria-hidden /></button>}
            </span>
          )}
        </div>
        {prefill && m.prefill && prefillText(m.prefill) && (
          <div key={m.prefill.state} className="num fade-swap px-1 text-[12px] text-ink-3" role={m.prefill.state === "reading" ? "status" : undefined}>{prefillText(m.prefill)}</div>
        )}
      </div>
    )
  }
  const waiting = streaming && !m.text && !m.tools?.length
  const phase = replyDesign({ streaming, reasoning: m.reasoning, text: m.text, tools: m.tools })
  return (
    <div ref={ref} className="msg-in max-w-[min(100%,65ch)] text-[15px] tracking-[-0.011em] lg:max-w-[72ch]">
      {m.reasoning && <Thinking m={m} streaming={streaming} show={show} phase={phase} />}
      {m.error ? (
        <div className="rounded-md border border-line px-3 py-2 text-[13px] text-bad [overflow-wrap:anywhere]">{m.error}</div>
      ) : waiting ? (
        m.reasoning ? null : <StatusLabel design="breathing" className="text-[13px] text-ink-2">{t("Waiting for the model…")}</StatusLabel>
      ) : (
        <div className={cn(streaming && "streaming")}><Answer m={m} /></div>
      )}
      <div className="mt-1 flex min-h-6 items-center gap-2 text-[12px] text-ink-3">
        {phase === "composing" ? <StatusLabel design="composing" className="text-[13px] text-ink-2">{t("Writing…")}</StatusLabel>
          : phase === "weaving" ? <StatusLabel design="weaving" className="text-[13px] text-ink-2">{t("Planning the next step…")}</StatusLabel>
          : <span className="num">{metaText(m) || (streaming ? "" : m.stopped ? t("Stopped") : "")}</span>}
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
