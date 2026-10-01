import { useCallback, useEffect, useRef, useState, type DragEvent } from "react"
import { HugeiconsIcon } from "@hugeicons/react"
import { ArrowDown01Icon } from "@hugeicons/core-free-icons"
import { apiHeaders, getHealth, getMcp, NO_HEALTH, url, type Health, type McpInfo } from "../../lib/api"
import { chat, exportMarkdown, useChatVersion, type Attachment, type Settings } from "../../lib/chat"
import { readFiles } from "../../lib/files"
import { toast } from "../../components/toast"
import { StatusOrb } from "../../components/live"
import { PromptBar } from "../../components/PromptBar"
import { useMetrics } from "../../lib/metrics"
import { MessageView } from "./Messages"
import { SettingsSheet } from "./SettingsSheet"

const NO_MCP: McpInfo = { servers: [], tools: 0 }

// The thinking levels the chat offers, as the server's reasoning_effort names them.
const EFFORTS: { label: string; value: Settings["thinking"] }[] = [
  { label: "Off", value: "none" }, { label: "Low", value: "low" }, { label: "Medium", value: "medium" }, { label: "High", value: "high" },
]

export function Chat() {
  useChatVersion()
  const [health, setHealth] = useState<Health>(NO_HEALTH)
  const [mcp, setMcp] = useState<McpInfo>(NO_MCP)
  const [projection, setProjection] = useState(false)
  const [text, setText] = useState("")
  const [files, setFiles] = useState<Attachment[]>([])
  const [sheet, setSheet] = useState(false)
  const [dragging, setDragging] = useState(false)
  const list = useRef<HTMLDivElement>(null)
  const [away, setAway] = useState(false)                      // the reader scrolled up: the answer no longer drags the page down
  const input = useRef<HTMLTextAreaElement>(null)
  const pinned = useRef(true)                                   // follow the answer while the reader is at the bottom
  const busy = chat.busy
  const { data: metrics, stale } = useMetrics()
  const live = metrics?.live ?? { state: "idle" as const, queued: 0, tok_s: null }
  const closeSheet = useCallback(() => setSheet(false), [])
  useEffect(() => { if (metrics) chat.samplePrefill(metrics.live, performance.now()) }, [metrics])      // each /metrics reply while a prompt is read

  useEffect(() => {
    chat.onError = (title, t) => toast("error", title, t, 6000)
    let cancelled = false
    const load = async () => {
      for (;;) {                                                // the server may still be starting
        try { const h = await getHealth(); if (!cancelled) setHealth(h); break } catch { await new Promise((r) => setTimeout(r, 2000)) }
        if (cancelled) return
      }
      for (let i = 0; i < 20 && !cancelled; i++) {              // right after the start, MCP servers may still be starting
        const m = await getMcp()
        if (m && !cancelled) setMcp(m)
        if (!m?.servers.some((s) => s.status === "starting")) break
        await new Promise((r) => setTimeout(r, 3000))
      }
      try {
        const r = await fetch(url("metrics"), { headers: apiHeaders() })
        const c = r.ok ? ((await r.json()) as { engine?: { cvec?: number | string } }).engine?.cvec : 0
        if (!cancelled) setProjection(!!c && c !== "0")
      } catch { /* an older server */ }
      const q = new URLSearchParams(location.search).get("q")  // ?q=... starts a chat (a shortcut)
      if (q && !cancelled && !chat.messages.length) setText(q)
    }
    void load()
    input.current?.focus()
    return () => { cancelled = true }
  }, [])

  // The page (the window) scrolls, so that is what follows the answer. The list's size changes for many reasons - text
  // streaming in, the thinking opening, a tool block - so it is watched, not the renders. The reader's own scroll up
  // (wheel, touch, keys) lets go; reaching the bottom again takes hold again.
  useEffect(() => {
    const bottom = () => document.documentElement.scrollHeight - (scrollY + innerHeight)
    const follow = () => { if (pinned.current) scrollTo({ top: document.documentElement.scrollHeight, behavior: "instant" as ScrollBehavior }) }
    const release = () => { pinned.current = false; setAway(true) }
    const onScroll = () => { if (bottom() < 40) { pinned.current = true; setAway(false) } }
    const onWheel = (e: WheelEvent) => { if (e.deltaY < 0) release() }
    let touchY = 0
    const onTouchStart = (e: TouchEvent) => { touchY = e.touches[0].clientY }
    const onTouchMove = (e: TouchEvent) => { if (e.touches[0].clientY > touchY + 6) release() }        // a finger moving down scrolls up
    const onKey = (e: globalThis.KeyboardEvent) => { if (["ArrowUp", "PageUp", "Home"].includes(e.key) && !(e.target as HTMLElement)?.closest("textarea")) release() }
    const ro = new ResizeObserver(follow)
    if (list.current) ro.observe(list.current)
    addEventListener("scroll", onScroll, { passive: true })
    addEventListener("wheel", onWheel, { passive: true })
    addEventListener("touchstart", onTouchStart, { passive: true })
    addEventListener("touchmove", onTouchMove, { passive: true })
    addEventListener("keydown", onKey)
    follow()                                                      // opening a chat that already has messages: the end of it
    return () => {
      ro.disconnect()
      removeEventListener("scroll", onScroll); removeEventListener("wheel", onWheel)
      removeEventListener("touchstart", onTouchStart); removeEventListener("touchmove", onTouchMove); removeEventListener("keydown", onKey)
    }
  }, [])
  const toLatest = () => { pinned.current = true; setAway(false); scrollTo({ top: document.documentElement.scrollHeight, behavior: "smooth" }) }

  const send = () => {
    if (busy || (!text.trim() && !files.length)) return
    const t = text, f = files
    setText(""); setFiles([])
    pinned.current = true
    setAway(false)
    void chat.send(t, f, { health, mcp, projectionLoaded: projection })
  }
  const add = async (list: Iterable<File>) => { const a = await readFiles(list, health); if (a.length) setFiles((x) => [...x, ...a]) }
  const onDrop = (e: DragEvent) => {
    setDragging(false)
    if (!e.dataTransfer.files.length) return
    e.preventDefault()
    void add(Array.from(e.dataTransfer.files))
    input.current?.focus()
  }
  const onDragOver = (e: DragEvent) => {
    if (!Array.from(e.dataTransfer.types).includes("Files")) return
    e.preventDefault()
    setDragging(true)
  }

  const newChat = () => {
    if (busy) { toast("warn", "Still writing", "Stop the answer first."); return }
    const undo = chat.clear()
    if (undo) toast("info", "New chat", "The last one was cleared.", 6000, { label: "Undo", run: undo })
  }
  const download = () => {
    if (!chat.messages.length) { toast("info", "Nothing to save yet"); return }
    const a = document.createElement("a")
    a.href = URL.createObjectURL(new Blob([exportMarkdown(chat.messages, health.model)], { type: "text/markdown" }))
    a.download = `strata-chat-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-")}.md`
    a.click()
    setTimeout(() => URL.revokeObjectURL(a.href), 5000)
  }

  return (
    <section className="flex min-h-[calc(100dvh-9rem)] flex-col" onDrop={onDrop} onDragOver={onDragOver} onDragLeave={() => setDragging(false)}>
      <div ref={list} className="flex-1 space-y-6 pb-6" aria-live="off">
        {chat.messages.length === 0 && (
          <div className="mx-auto mt-[11vh] flex max-w-[44ch] flex-col items-center text-center">
            <StatusOrb live={live} stale={stale} size={64} override={text.trim() ? { design: "listening", label: "Listening" } : undefined} />
            <h1 className="display mt-6" style={{ fontSize: "clamp(28px, 4vw, 40px)" }}>What can I help with?</h1>
            <p className="lede mt-3">{health.model} runs on this PC. Nothing leaves it.</p>
          </div>
        )}
        {chat.messages.map((m, i) => (
          <MessageView key={i} m={m} streaming={busy?.msg === m} show={chat.settings.show} prefill={chat.settings.prefill} />
        ))}
      </div>

      <div className="sticky bottom-3 z-10">
        <PromptBar
          value={text}
          onChange={setText}
          inputRef={input}
          attachments={files}
          onRemoveAttachment={(i) => setFiles((x) => x.filter((_, j) => j !== i))}
          onFiles={(f) => void add(f)}
          onPasteFiles={(e) => {
            const imgs = Array.from(e.clipboardData?.files || []).filter((f) => f.type.startsWith("image/"))
            if (health.images && imgs.length) { e.preventDefault(); void add(imgs) }
          }}
          busy={!!busy}
          onSend={send}
          onStop={() => chat.stop()}
          onNewChat={newChat}
          onSave={download}
          onSampling={() => setSheet(true)}
          efforts={EFFORTS.map((e) => e.label)}
          effort={Math.max(0, EFFORTS.findIndex((e) => e.value === chat.settings.thinking))}
          onEffort={(i) => chat.setSettings({ ...chat.settings, thinking: EFFORTS[i].value })}
          attachTitle={health.images ? "Text files and pictures" : "Text files"}
          dragging={dragging}
        >
          {away && busy && (
            <button
              type="button"
              onClick={toLatest}
              aria-label="Jump to the latest"
              className="msg-in absolute -top-11 left-1/2 z-[2] flex h-8 -translate-x-1/2 items-center gap-1.5 rounded-full border border-line bg-surface px-3 text-[13px] shadow-[0_4px_16px_rgb(0_0_0/0.10)] transition-colors hover:bg-hover"
            >
              <HugeiconsIcon icon={ArrowDown01Icon} size={14} aria-hidden />Latest
            </button>
          )}
        </PromptBar>
      </div>

      <SettingsSheet open={sheet} onClose={closeSheet} mcp={mcp} projectionLoaded={projection} />
    </section>
  )
}
