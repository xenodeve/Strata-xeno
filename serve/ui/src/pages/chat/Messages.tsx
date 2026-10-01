import { useRef, useState, type MouseEvent } from "react"
import { HugeiconsIcon } from "@hugeicons/react"
import { ArrowDown01Icon, Copy01Icon, AttachmentIcon } from "@hugeicons/core-free-icons"
import { chat, type Message, type ToolCall } from "../../lib/chat"
import { copyText } from "../../lib/files"
import { Collapse } from "../../components/motion"
import { Orb, StatusLabel } from "../../components/orb"
import { replyDesign, toolDesign } from "../../lib/orbs"
import { cn } from "../../lib/cn"
import { fmt, timeStr } from "../../lib/format"
import { markdown } from "../../lib/markdown"

const TOOL_STATE: Record<ToolCall["state"], string> = { writing: "Writing", running: "Running", done: "Done", error: "Error", skipped: "Not run" }

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
function Tool({ t }: { t: ToolCall }) {
  const args = t.arguments == null ? "" : JSON.stringify(t.arguments, null, 2)
  const preview = t.result != null ? t.result : args.replace(/\s+/g, " ")
  return (
    <div className="my-2 rounded-md border border-line" data-state={t.state}>
      <button
        type="button"
        aria-expanded={!!t.open}
        onClick={() => { t.open = !t.open; chat.notify() }}
        className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-left text-[13px] transition-colors hover:bg-hover"
      >
        {(t.state === "running" || t.state === "writing") && <Orb design={toolDesign(t.name)} size={20} />}
        <span className={cn("font-medium", (t.state === "running" || t.state === "writing") && "t-shimmer")} title={t.name}>{t.tool || t.name || "tool"}</span>
        {t.server && <span className="text-ink-2">{t.server}</span>}
        <span className="min-w-0 flex-1 truncate text-ink-3">{preview.slice(0, 200)}</span>
        <span className={cn("shrink-0", t.state === "error" ? "text-bad" : "text-ink-2")}>{TOOL_STATE[t.state]}</span>
        {t.ms != null && t.state !== "skipped" && <span className="num shrink-0 text-ink-3">{fmt(t.ms / 1000, 1)} s</span>}
        <HugeiconsIcon icon={ArrowDown01Icon} size={14} aria-hidden className={cn("shrink-0 text-ink-3 transition-transform duration-200", t.open && "rotate-180")} />
      </button>
      <Collapse open={!!t.open}>
        <div className="space-y-2 border-t border-line px-3 py-2.5">
          <div className="text-[12px] text-ink-2">Arguments</div>
          <pre className="tool-pre">{args || "(being written)"}</pre>
          {t.result != null && (
            <>
              <div className="text-[12px] text-ink-2">
                {t.ok ? "Result" : "Error"}{t.chars ? ` · ${fmt(t.chars)} characters` : ""}{t.truncated ? ", cut for the model" : ""}
              </div>
              <pre className="tool-pre">{t.result}</pre>
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
  m.tools.forEach((t, k) => {
    const at = Math.min(Math.max(t.at || 0, pos), m.text.length)
    if (at > pos) parts.push(<Prose key={`p${k}`} text={m.text.slice(pos, at)} />)
    pos = at
    parts.push(<Tool key={t.id} t={t} />)
  })
  parts.push(<Prose key="rest" text={m.text.slice(pos)} />)
  return <>{parts}</>
}

function Thinking({ m, streaming, show }: { m: Message; streaming: boolean; show: boolean }) {
  const [touched, setTouched] = useState<boolean | null>(null)       // the user's own choice, once made
  const thinkingNow = streaming && !m.text
  const open = touched ?? (thinkingNow && show)                      // open while it streams (if wanted), closed once the answer starts
  const title = thinkingNow ? "Thinking…" : m.thinkSecs != null ? `Thought for ${fmt(m.thinkSecs, 1)} s` : "Thoughts"
  return (
    <div className="mb-2">
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setTouched(!open)}
        className="flex items-center gap-1.5 rounded-sm py-0.5 text-[13px] text-ink-2 transition-colors hover:text-ink"
      >
        {thinkingNow && <Orb design="solving" size={20} />}
        <span className={cn(thinkingNow && "t-shimmer")}>{title}</span>
        <HugeiconsIcon icon={ArrowDown01Icon} size={14} aria-hidden className={cn("transition-transform duration-200", open && "rotate-180")} />
      </button>
      <Collapse open={open}>
        <div className="mt-1 max-h-72 overflow-y-auto whitespace-pre-wrap border-l border-line pl-3 text-[13px] leading-relaxed text-ink-2 [overflow-wrap:anywhere]">{m.reasoning}</div>
      </Collapse>
    </div>
  )
}

export function MessageView({ m, streaming, show }: { m: Message; streaming: boolean; show: boolean }) {
  const ref = useRef<HTMLDivElement>(null)
  if (m.role === "user") {
    return (
      <div className="msg-in flex flex-col items-end gap-1">
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
              ? <img key={i} src={im.url} alt={im.name || "image"} className="max-h-56 max-w-full rounded-md" />
              : <span key={i} className="inline-flex items-center gap-1 rounded-sm bg-fill px-2 py-1 text-[12px]">{im.name || "image"}</span>)}
          </div>
        )}
        <div className="max-w-[85%] whitespace-pre-wrap rounded-[20px] rounded-br-md bg-fill px-4 py-2.5 text-[15px] tracking-[-0.011em] [overflow-wrap:anywhere]">{m.text}</div>
        <div className="num px-1 text-[12px] text-ink-3">You · {timeStr(m.time)}</div>
      </div>
    )
  }
  const waiting = streaming && !m.text && !m.tools?.length
  const phase = replyDesign({ streaming, reasoning: m.reasoning, text: m.text, tools: m.tools })
  return (
    <div ref={ref} className="msg-in max-w-[min(100%,65ch)] text-[15px] tracking-[-0.011em] lg:max-w-[72ch]">
      {m.reasoning && <Thinking m={m} streaming={streaming} show={show} />}
      {m.error ? (
        <div className="rounded-md border border-line px-3 py-2 text-[13px] text-bad [overflow-wrap:anywhere]">{m.error}</div>
      ) : waiting ? (
        m.reasoning ? null : <StatusLabel design="breathing" className="text-[13px] text-ink-2">Waiting for the model…</StatusLabel>
      ) : (
        <div className={cn(streaming && "streaming")}><Answer m={m} /></div>
      )}
      <div className="mt-1 flex min-h-6 items-center gap-2 text-[12px] text-ink-3">
        {phase === "composing" ? <StatusLabel design="composing" className="text-[13px] text-ink-2">Writing…</StatusLabel>
          : phase === "weaving" ? <StatusLabel design="weaving" className="text-[13px] text-ink-2">Planning the next step…</StatusLabel>
          : <span className="num">{m.meta || (streaming ? "" : m.stopped ? "Stopped" : "")}</span>}
        {!streaming && !!m.text && (
          <button
            type="button"
            aria-label="Copy the answer"
            title="Copy"
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
