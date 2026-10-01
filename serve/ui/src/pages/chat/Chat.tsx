import { useCallback, useEffect, useRef, useState, type DragEvent, type FormEvent, type KeyboardEvent } from "react"
import { HugeiconsIcon } from "@hugeicons/react"
import { ArrowDown01Icon, ArrowUp02Icon, AttachmentIcon, Cancel01Icon, Download01Icon, PlusSignIcon, Settings02Icon, StopIcon } from "@hugeicons/core-free-icons"
import { apiHeaders, getHealth, getMcp, NO_HEALTH, url, type Health, type McpInfo } from "../../lib/api"
import { chat, exportMarkdown, useChatVersion, type Attachment } from "../../lib/chat"
import { readFiles } from "../../lib/files"
import { cn } from "../../lib/cn"
import { toast } from "../../components/toast"
import { MessageView } from "./Messages"
import { SettingsSheet } from "./SettingsSheet"

const NO_MCP: McpInfo = { servers: [], tools: 0 }

function IconButton({ icon, label, onClick, disabled }: { icon: typeof PlusSignIcon; label: string; onClick: () => void; disabled?: boolean }) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      disabled={disabled}
      onClick={onClick}
      className="flex size-8 items-center justify-center rounded-sm text-ink-2 transition-colors duration-150 hover:bg-hover hover:text-ink disabled:pointer-events-none disabled:opacity-40"
    >
      <HugeiconsIcon icon={icon} size={17} strokeWidth={1.6} aria-hidden />
    </button>
  )
}

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
  const closeSheet = useCallback(() => setSheet(false), [])

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
  const onSubmit = (e: FormEvent) => { e.preventDefault(); send() }
  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); send() }   // not while a Thai/CJK IME is composing
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
      <div className="mb-2 flex items-center justify-end gap-0.5">
        <IconButton icon={PlusSignIcon} label="New chat" onClick={newChat} />
        <IconButton icon={Download01Icon} label="Save the chat as Markdown" onClick={download} />
        <IconButton icon={Settings02Icon} label="Sampling" onClick={() => setSheet(true)} />
      </div>

      <div ref={list} className="flex-1 space-y-6 pb-6" aria-live="off">
        {chat.messages.length === 0 && (
          <div className="mx-auto mt-[12vh] max-w-[40ch] text-center">
            <h1 className="text-xl font-semibold">What can I help with?</h1>
            <p className="mt-2 text-ink-2">{health.model} runs on this PC. Nothing leaves it.</p>
          </div>
        )}
        {chat.messages.map((m, i) => (
          <MessageView key={i} m={m} streaming={busy?.msg === m} show={chat.settings.show} />
        ))}
      </div>

      <form
        onSubmit={onSubmit}
        className={cn(
          "sticky bottom-3 rounded-lg border bg-surface p-2 shadow-[0_4px_24px_rgb(0_0_0/0.06)] transition-colors duration-150",
          dragging ? "border-accent" : "border-line focus-within:border-fill-2",
        )}
      >
        {away && busy && (
          <button
            type="button"
            onClick={toLatest}
            aria-label="Jump to the latest"
            className="msg-in absolute -top-11 left-1/2 flex h-8 -translate-x-1/2 items-center gap-1.5 rounded-full border border-line bg-surface px-3 text-[13px] shadow-[0_4px_16px_rgb(0_0_0/0.10)] transition-colors hover:bg-hover"
          >
            <HugeiconsIcon icon={ArrowDown01Icon} size={14} aria-hidden />Latest
          </button>
        )}
        {files.length > 0 && (
          <div className="mb-1.5 flex flex-wrap gap-1.5 px-1">
            {files.map((f, i) => (
              <span key={i} className="inline-flex items-center gap-1 rounded-sm bg-fill py-0.5 pl-2 pr-0.5 text-[12px]">
                <HugeiconsIcon icon={AttachmentIcon} size={12} aria-hidden />{f.name}
                <button type="button" aria-label={`Remove ${f.name}`} onClick={() => setFiles((x) => x.filter((_, j) => j !== i))} className="flex size-5 items-center justify-center rounded-sm hover:bg-hover">
                  <HugeiconsIcon icon={Cancel01Icon} size={12} aria-hidden />
                </button>
              </span>
            ))}
          </div>
        )}
        <textarea
          ref={input}
          value={text}
          rows={1}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKey}
          onPaste={(e) => {
            const imgs = Array.from(e.clipboardData.files).filter((f) => f.type.startsWith("image/"))
            if (health.images && imgs.length) { e.preventDefault(); void add(imgs) }
          }}
          placeholder="Message"
          aria-label="Message"
          className="block max-h-[40dvh] min-h-9 w-full resize-none bg-transparent px-2 py-1.5 text-[14px] outline-none placeholder:text-ink-3 [field-sizing:content]"
        />
        <div className="flex items-center justify-between">
          <label className="flex size-8 cursor-pointer items-center justify-center rounded-sm text-ink-2 transition-colors hover:bg-hover hover:text-ink" title={health.images ? "Attach a text file or a picture (or drop it here)" : "Attach a text file (or drop it here)"}>
            <HugeiconsIcon icon={AttachmentIcon} size={17} strokeWidth={1.6} aria-hidden />
            <input type="file" multiple hidden aria-label="Attach files" onChange={(e) => { void add(Array.from(e.target.files || [])); e.target.value = "" }} />
          </label>
          <div className="flex items-center gap-2">
            {!busy && <span className="hidden text-[12px] text-ink-3 sm:inline">Shift+Enter: new line</span>}
            {busy ? (
              <button type="button" onClick={() => chat.stop()} aria-label="Stop" className="flex h-8 items-center gap-1.5 rounded-full bg-fill px-3 text-[13px] font-medium transition-colors hover:bg-fill-2">
                <HugeiconsIcon icon={StopIcon} size={14} aria-hidden />Stop
              </button>
            ) : (
              <button type="submit" aria-label="Send" disabled={!text.trim() && !files.length} className="flex size-8 items-center justify-center rounded-full bg-ink text-surface transition-[opacity,transform] duration-150 active:scale-95 disabled:opacity-25">
                <HugeiconsIcon icon={ArrowUp02Icon} size={16} strokeWidth={2} aria-hidden />
              </button>
            )}
          </div>
        </div>
      </form>

      <SettingsSheet open={sheet} onClose={closeSheet} mcp={mcp} projectionLoaded={projection} />
    </section>
  )
}
