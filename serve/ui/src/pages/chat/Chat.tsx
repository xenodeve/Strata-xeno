import { useCallback, useEffect, useRef, useState, type DragEvent } from "react"
import { HugeiconsIcon } from "@hugeicons/react"
import { ArrowDown01Icon, Menu01Icon, SidebarLeftIcon } from "@hugeicons/core-free-icons"
import { apiHeaders, getAgent, getHealth, getImport, getMcp, NO_HEALTH, url, type Health, type McpInfo } from "../../lib/api"
import { chat, exportMarkdown, isPrompt, useChatVersion, type Attachment, type Message } from "../../lib/chat"
import { readFiles } from "../../lib/files"
import { cn } from "../../lib/cn"
import { toast } from "../../components/toast"
import { StatusOrb } from "../../components/live"
import { PromptBar } from "../../components/PromptBar"
import { Sidebar } from "../../components/Sidebar"
import { ConfirmDialog } from "../../components/ConfirmDialog"
import { useMetrics } from "../../lib/metrics"
import { readProgress } from "../../lib/status"
import { href } from "../../lib/router"
import { effortChoices, settleEffort } from "../../lib/effort"
import { noteSend } from "../../lib/sendfx"
import { Collapse, useMounted } from "../../components/motion"
import { nextDesign, type OrbDesign } from "../../lib/orbs"
import { builtinCommands, commandsOf, type Command } from "../../lib/slash"
import { compactCommand } from "../../lib/compact"
import { contextView, isContextCommand } from "../../lib/context"
import { INIT_PROMPT, isInitCommand, isMemoryCommand } from "../../lib/memory"
import { forgetRules, NO_AGENT, rulesOf, type AgentInfo, type AgentMode } from "../../lib/agent"
import { store } from "../../lib/store"
import { SkillsContext } from "../../components/SkillTip"
import { msg, t } from "../../lib/i18n"
import { CompactingLine, MessageView, QueuedMessage } from "./Messages"
import { resolveMentions } from "../../lib/mention"
import { RewindDialog } from "../../components/RewindDialog"
import { isRewindCommand } from "../../lib/rewind"
import { SettingsSheet } from "./SettingsSheet"
import { PanelDock } from "../../components/SidePanel"
import { ProjectPicker } from "../../components/ProjectPicker"
import { panelState, type PanelState, type PanelTab } from "../../lib/panel"

const NO_MCP: McpInfo = { servers: [], tools: 0 }

// The orb's accessible name in the empty chat: one of the nine forms, as words.
const ORB_NAMES: Record<OrbDesign, string> = {
  working: msg("Working"), searching: msg("Searching"), solving: msg("Solving"), listening: msg("Listening"), connecting: msg("Connecting"),
  weaving: msg("Weaving"), composing: msg("Composing"), breathing: msg("Breathing"), shaping: msg("Shaping"),
}

/** Messages an undo took away, still drawn for the length of their closing: the box shrinks to nothing while it fades. A
 *  negative top margin cancels the gap the message before it leaves (the gap is part of what closes), so nothing jumps
 *  when they are gone. Not interactive, and read by nobody. */
function Leaving({ messages, onGone }: { messages: Message[]; onGone: () => void }) {
  const [open, setOpen] = useState(true)
  useEffect(() => {
    const raf = requestAnimationFrame(() => requestAnimationFrame(() => setOpen(false)))
    const done = setTimeout(onGone, 420)
    return () => { cancelAnimationFrame(raf); clearTimeout(done) }
  }, [onGone])
  return (
    <div className="ghost -mt-6" aria-hidden inert>
      <div className={cn("collapse-grid", open && "is-open")}>
        <div className="min-h-0 overflow-hidden"><div className="space-y-6 pt-6">
          {messages.map((m, i) => <MessageView key={i} m={m} streaming={false} show={false} prefill={false} />)}
        </div></div>
      </div>
    </div>
  )
}

/** The empty chat's orb shows a different one of the nine forms every 5 s while the server has nothing to do (`on`); when it
 *  is busy, or the reader types, the orb goes back to saying what is really happening. */
function useAmbientOrb(on: boolean, every = 5000): OrbDesign | null {
  const [design, setDesign] = useState<OrbDesign | null>(null)
  useEffect(() => {
    if (!on) { setDesign(null); return }
    setDesign((d) => nextDesign(d, Math.random()))
    const id = setInterval(() => setDesign((d) => nextDesign(d, Math.random())), every)
    return () => clearInterval(id)
  }, [on, every])
  return on ? design : null
}

export function Chat({ id }: { id?: string }) {
  useChatVersion()
  const [drawer, setDrawer] = useState(false)                  // the list as a drawer on a phone
  const drawerMounted = useMounted(drawer, 320)                // it slides out and is taken away when it is gone
  const [drawerShown, setDrawerShown] = useState(false)
  useEffect(() => {
    if (!drawer) { setDrawerShown(false); return }
    const id = requestAnimationFrame(() => requestAnimationFrame(() => setDrawerShown(true)))
    return () => cancelAnimationFrame(id)
  }, [drawer])
  const [health, setHealth] = useState<Health>(NO_HEALTH)
  const [mcp, setMcp] = useState<McpInfo>(NO_MCP)
  const [projection, setProjection] = useState(false)
  const [skills, setSkills] = useState<Command[]>([])            // the skills in use, for "/" in the prompt
  const [agentInfo, setAgentInfo] = useState<AgentInfo>(NO_AGENT)  // the coding tools: does the server have them, and may this page use them
  useEffect(() => {
    let gone = false
    void getAgent().then((v) => { if (!gone && v) setAgentInfo(v) })
    return () => { gone = true }
  }, [])
  useEffect(() => {
    let gone = false
    void getImport().then((v) => { if (!gone && v?.available && v.skills?.settings.enabled) setSkills(commandsOf(v.skills.items, Object.fromEntries((v.harnesses ?? []).map((h) => [h.id, h.label])))) })
    return () => { gone = true }
  }, [])
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
  useEffect(() => { void chat.resumeRuns() }, [])                // an answer that was being written when the page was refreshed went on in the server: it is read again
  const closeSheet = useCallback(() => setSheet(false), [])
  // The thinking levels are the model's: what its template accepts, as the server lists them. A level saved for another model
  // (or "high" where this one calls it "xhigh") is moved onto one this model has.
  const offered = (metrics?.engine.efforts as string[] | undefined) ?? []
  const choices = effortChoices(offered)
  const offeredKey = offered.join(",")
  useEffect(() => {
    const now = settleEffort(chat.settings.thinking, offered, metrics?.engine.effort_default as string | null | undefined)
    if (now !== chat.settings.thinking) chat.setSettings({ ...chat.settings, thinking: now })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [offeredKey])
  const typing = text.trim() !== ""
  const ambient = useAmbientOrb(chat.messages.length === 0 && !typing && !stale && live.state === "idle" && !(live.queued > 0))
  useEffect(() => { if (metrics) chat.samplePrefill(metrics.live, performance.now()) }, [metrics])      // each /metrics reply while a prompt is read

  useEffect(() => {
    chat.onError = (title, text) => toast("error", title, text, 6000)
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
    // Opening or closing a part of the conversation (the thinking, a tool call) changes the list's height: that must not be
    // followed like new text, or the page scrolls with it and the part seems to open upward. The reader chose to look at it,
    // so the follow lets go for the length of the animation, then holds again only if the reader is still at the end.
    let settle = 0
    const onToggle = (e: MouseEvent) => {
      if (!(e.target as HTMLElement)?.closest?.("[aria-expanded]") || !list.current?.contains(e.target as Node)) return
      pinned.current = false
      clearTimeout(settle)
      settle = window.setTimeout(() => { if (bottom() < 40) pinned.current = true; else setAway(true) }, 450)
    }
    list.current?.addEventListener("click", onToggle, true)
    const listEl = list.current
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
      clearTimeout(settle)
      listEl?.removeEventListener("click", onToggle, true)
      removeEventListener("scroll", onScroll); removeEventListener("wheel", onWheel)
      removeEventListener("touchstart", onTouchStart); removeEventListener("touchmove", onTouchMove); removeEventListener("keydown", onKey)
    }
  }, [])
  // The address and the open conversation agree. A link to a conversation opens it (one that does not exist shows a new chat); what the
  // page does (a first prompt adds a conversation, New chat, a choice in the list) puts its id in the address. The address is
  // replaced, not added to, so switching is not a trail of history entries.
  const active = chat.index.active
  useEffect(() => {
    if (id === undefined || id === chat.index.active) return
    if (!chat.open(id) && !chat.index.items.some((i) => i.id === id)) chat.newSession()
  }, [id])
  useEffect(() => {
    const want = chat.index.active ? href("chat", chat.index.active) : href("chat")
    if (location.hash !== want) location.replace(want)
  }, [active, id])
  useEffect(() => { pinned.current = true; setAway(false) }, [active])                   // another conversation opens at its end
  useEffect(() => {
    if (!drawer) return
    const key = (e: globalThis.KeyboardEvent) => { if (e.key === "Escape") setDrawer(false) }
    addEventListener("keydown", key)
    return () => removeEventListener("keydown", key)
  }, [drawer])
  const toLatest = () => { pinned.current = true; setAway(false); scrollTo({ top: document.documentElement.scrollHeight, behavior: "smooth" }) }

  const [panel, setPanel] = useState<PanelState>(() => panelState(store.get("panel", null)))      // the right panel: open or not, and which tab (kept in the browser)
  const keepPanel = (next: PanelState) => { setPanel(next); store.set("panel", next) }
  const [rewindAt, setRewindAt] = useState<number | null>(null)       // the rewind dialog: which prompt (an index into the chat's prompts) it opens on
  const [ctxOpen, setCtxOpen] = useState(0)                      // how many times /context was typed: the panel opens at each
  const rewindPrompts = () => chat.messages.flatMap((m, index) => (isPrompt(m) ? [{ index, text: m.text || m.files?.map((f) => f.name).join(", ") || "", checkpoint: String(m.time) }] : []))
  const commands = [...builtinCommands(), ...skills.filter((c) => !builtinCommands().some((b) => b.name === c.name))]       // what "/" lists: Strata's own, then the skills
  const send = () => {
    if (!text.trim() && !files.length) return
    const word = files.length ? null : /^\s*\/(compact|context|memory|init|clear|permissions)\s*$/i.exec(text)?.[1]?.toLowerCase()
    if (busy && (word === "compact" || word === "init" || (!word && /^\s*\/(compact|init)\b/i.test(text)))) { toast("warn", t("Still writing"), t("Stop the answer first.")); return }
    if (!files.length && isContextCommand(text)) { setText(""); setCtxOpen((n) => n + 1); return }                // "/context": the panel with the context window opens
    if (!files.length && isRewindCommand(text)) {                                                                                       // "/rewind": go back to before a prompt
      setText("")
      if (busy) toast("warn", t("Still writing"), t("Stop the answer first."))
      else if (!rewindPrompts().length) toast("info", t("Nothing to rewind yet"), t("Send a prompt first."))
      else setRewindAt(rewindPrompts().length - 1)
      return
    }
    if (!files.length && isMemoryCommand(text)) { setText(""); keepPanel({ open: true, tab: "memory" }); return }                      // "/memory": the panel opens on the notes
    if (!files.length && /^\s*\/permissions\s*$/i.test(text)) { setText(""); location.hash = href("settings", "permissions"); return }                // "/permissions": the page of the rules
    if (!files.length && /^\s*\/clear\s*$/i.test(text)) { setText(""); newChat(); return }                                              // "/clear": a new chat, this one stays in Recents
    const asked = files.length ? null : compactCommand(text)           // "/compact" (and what to focus on) is not a message: the conversation is summarised
    if (asked) {
      setText("")
      pinned.current = true
      if (!chat.canCompact()) toast("info", t("Nothing to compact yet"), t("Send a prompt and get its answer first."))
      else void chat.compact(ctx(), asked.focus)
      return
    }
    const typed = !files.length && isInitCommand(text) ? INIT_PROMPT : text, f = files                                    // "/init": the prompt that has the model write the project's CLAUDE.md
    const folders = chat.folders()
    const go = async () => {
      const found = agentInfo.allowed ? await resolveMentions(folders, typed) : []                                    // the files the prompt mentions with @ go with it
      const all = [...f, ...found.filter((x) => !f.some((y) => y.name === x.name))]
      const context = { health, mcp, projectionLoaded: projection, skills: skills.map((c) => c.name), agent: agentInfo, folder: folders }
      if (!busy) void chat.send(typed, all, context)
      else void chat.steer(typed, all, context).then((taken) => { if (!taken && !chat.queue(typed, all, context)) void chat.send(typed, all, context) })        // an agent is working: it reads this at its next step; else it waits in line and goes when the answer ends
    }
    if (!busy) noteSend(input.current?.getBoundingClientRect())          // where the prompt rises from
    setText(""); setFiles([])
    pinned.current = true
    setAway(false)
    void go()
  }
  const ctx = () => ({ health, mcp, projectionLoaded: projection, skills: skills.map((c) => c.name), agent: agentInfo, folder: chat.folders() })
  const project = chat.index.projects.find((p) => p.id === chat.currentProject())       // the open conversation's project (or the one a new conversation was started in): its folders are where the coding tools work
  const chatKey = chat.index.active ?? "new"
  const editPrompt = (i: number, t: string) => { pinned.current = true; setAway(false); void chat.edit(i, t, ctx()) }
  const [leaving, setLeaving] = useState<Message[]>([])        // what an undo took away: it closes up (height and fade) before it is gone
  const [asking, setAsking] = useState(false)               // taking back the first prompt deletes the conversation: asked first
  const undoPrompt = () => {
    if (chat.undoWouldEmpty()) { setAsking(true); return }
    takeBack()
  }
  const cancelAsk = useCallback(() => setAsking(false), [])
  const takeBack = () => {
    const back = chat.undoLast()
    if (!back) return
    setLeaving(back.removed)
    setText((cur) => (cur.trim() ? back.text + "\n\n" + cur : back.text))      // a draft already in the composer stays
    setFiles((f) => [...back.attachments, ...f])
    input.current?.focus()
  }
  const lastPrompt = chat.messages.reduce((at, m, i) => (isPrompt(m) ? i : at), -1)
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
    chat.newSession()                                         // the one that was open stays in Recents (and goes on answering, if it was)
    input.current?.focus()
  }
  const download = () => {
    if (!chat.messages.length) { toast("info", t("Nothing to save yet")); return }
    const a = document.createElement("a")
    a.href = URL.createObjectURL(new Blob([exportMarkdown(chat.messages, health.model)], { type: "text/markdown" }))
    a.download = `strata-chat-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-")}.md`
    a.click()
    setTimeout(() => URL.revokeObjectURL(a.href), 5000)
  }

  return (
    <SkillsContext.Provider value={commands}>
    <div className="md:max-[1599px]:flex md:gap-6 min-[1600px]:grid min-[1600px]:grid-cols-[minmax(380px,1fr)_minmax(0,56rem)_minmax(380px,1fr)]">      {/* with room to spare the conversation sits in the middle of the window and the sidebars open into the space at its sides, so opening or closing one does not move it */}
    <div className="max-md:hidden"><Sidebar /></div>
    {drawerMounted && (
      <div className="fixed inset-0 z-40 md:hidden">
        <button type="button" aria-label={t("Close")} onClick={() => setDrawer(false)} className={cn("absolute inset-0 bg-black/40 transition-opacity duration-300", drawerShown ? "opacity-100" : "opacity-0")} />
        <div className={cn("absolute inset-y-0 left-0 w-[min(86vw,320px)] overflow-y-auto bg-surface p-3 shadow-xl transition-transform duration-300 ease-[var(--ease)]", drawerShown ? "translate-x-0" : "-translate-x-full")}><Sidebar drawer onClose={() => setDrawer(false)} /></div>
      </div>
    )}
    <section className="mx-auto flex min-h-[calc(100dvh-9rem)] w-full min-w-0 max-w-[56rem] flex-1 flex-col" onDrop={onDrop} onDragOver={onDragOver} onDragLeave={() => setDragging(false)}>
      <div className="pointer-events-none sticky top-[4.5rem] z-10 mb-2 flex items-center justify-between">      {/* stays in reach when the conversation is scrolled: only its buttons take clicks, the rest lets them through to the messages */}
        <button type="button" onClick={() => setDrawer(true)} className="pointer-events-auto flex h-8 items-center gap-1.5 rounded-sm bg-bg/85 px-2 text-[13px] text-ink-2 backdrop-blur-sm transition-colors hover:bg-hover hover:text-ink md:hidden">
          <HugeiconsIcon icon={Menu01Icon} size={16} strokeWidth={1.6} aria-hidden />{t("Recents")}
        </button>
        <button type="button" aria-label={t("Session panel")} aria-pressed={panel.open} title={t("Git, plan, skills and context")} onClick={() => keepPanel({ ...panel, open: !panel.open })} data-panel-toggle className={cn("pointer-events-auto ml-auto flex h-8 items-center gap-1.5 rounded-sm px-2 text-[13px] backdrop-blur-sm transition-colors hover:bg-hover hover:text-ink", panel.open ? "bg-fill text-ink" : "bg-bg/85 text-ink-2")}>
          <HugeiconsIcon icon={SidebarLeftIcon} size={16} strokeWidth={1.6} aria-hidden className="-scale-x-100" />
        </button>
      </div>
      <div ref={list} className="flex-1 space-y-6 pb-6" aria-live="off">
        <Collapse open={chat.messages.length === 0} instant className="-mb-6">
          <div className="mx-auto mt-[11vh] flex max-w-[44ch] flex-col items-center text-center">
            <StatusOrb live={live} stale={stale} size={64} scale={2.5} override={typing ? { design: "listening", label: t(ORB_NAMES.listening) } : ambient ? { design: ambient, label: t(ORB_NAMES[ambient]) } : undefined} />
            <h1 className="display mt-6" style={{ fontSize: "clamp(28px, 4vw, 40px)" }}>{t("What can I help with?")}</h1>
            <div className="mt-3"><ProjectPicker projects={chat.index.projects} value={project?.id ?? null} onPick={(id) => { chat.setPendingProject(id) }} /></div>
            <p className="lede mt-3">{t("{name} runs on this PC. Nothing leaves it.", { name: metrics?.model_info?.name ? [metrics.model_info.name, metrics.model_info.variant].filter(Boolean).join(" · ") : health.model })}</p>
          </div>
        </Collapse>
        {chat.messages.map((m, i) => (
          <MessageView key={i} m={m} streaming={busy?.msg === m} compacting={chat.compacting && busy?.msg === m} show={chat.settings.show} prefill={chat.settings.prefill} serverPhase={busy?.msg === m ? (live as { phase?: string | null }).phase : undefined} serverState={busy?.msg === m ? live.state : undefined} reading={busy?.msg === m && live.state === "reading" ? readProgress((live as { prompt_read?: number | null }).prompt_read, (live as { prompt_total?: number | null }).prompt_total) : null}
            actions={isPrompt(m) ? { canAct: !busy, last: i === lastPrompt, onEdit: (t) => editPrompt(i, t), onUndo: undoPrompt, onRewind: () => setRewindAt(Math.max(0, rewindPrompts().findIndex((p) => p.index === i))) } : undefined} />
        ))}
        {chat.compacting && !!busy && !chat.messages.includes(busy.msg) && <CompactingLine />}
        {chat.queuedOf().map((q) => (
          <QueuedMessage key={q.id} text={q.text} files={q.files} answering={!!busy} steered={q.steered}
            onEdit={() => { const x = chat.unqueue(q.id); if (x) { setText((cur) => (cur.trim() ? x.text + "\n\n" + cur : x.text)); setFiles((fs) => [...x.files, ...fs]); input.current?.focus() } }}
            onRemove={() => { chat.unqueue(q.id) }}
            onSend={() => { void chat.sendQueued(q.id) }} />
        ))}
        {leaving.length > 0 && <Leaving messages={leaving} onGone={() => setLeaving([])} />}
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
          context={{ view: contextView(chat.messages, chat.contextReported(), health.max_context, chat.settings.autoCompact !== false), canCompact: chat.canCompact(), onCompact: () => { void chat.compact(ctx()) }, open: ctxOpen }}
          mention={{ folders: chat.folders(), enabled: agentInfo.allowed }}
          skills={commands}
          agent={{
            info: agentInfo, on: chat.settings.agent !== false, mode: (chat.settings.agentMode === "plan" || chat.settings.agentMode === "auto" ? chat.settings.agentMode : "ask") as AgentMode,
            folders: project ? project.folders ?? [] : chat.folders(), folderOf: project ? { kind: "project", name: project.name } : { kind: "default" }, rules: rulesOf(store, chatKey).length,
            onToggle: () => chat.setSettings({ ...chat.settings, agent: chat.settings.agent === false }),
            onMode: (m) => chat.setSettings({ ...chat.settings, agentMode: m }),
            onFolders: (list) => (project ? chat.setProjectFolders(project.id, list) : chat.setSettings({ ...chat.settings, agentFolder: list[0]?.trim() ?? "" })),
            onForget: () => { forgetRules(store, chatKey); chat.notify() }, permsHref: href("settings", "permissions"),
          }}
          mcp={{
            servers: mcp.servers, tools: mcp.tools, on: chat.settings.mcp !== false, off: Array.isArray(chat.settings.mcpOff) ? chat.settings.mcpOff : [], setupHref: href("settings", "mcp-servers"),
            onToggleAll: () => chat.setSettings({ ...chat.settings, mcp: chat.settings.mcp === false }),
            onToggleServer: (name) => { const off = Array.isArray(chat.settings.mcpOff) ? chat.settings.mcpOff : []; chat.setSettings({ ...chat.settings, mcpOff: off.includes(name) ? off.filter((n) => n !== name) : [...off, name] }) },
          }}
          onSampling={() => setSheet(true)}
          efforts={choices.map((c) => t(c.label))}
          effort={Math.max(0, choices.findIndex((c) => c.value === chat.settings.thinking))}
          onEffort={(i) => chat.setSettings({ ...chat.settings, thinking: choices[i].value })}
          attachTitle={health.images ? t("Text files and pictures") : t("Text files")}
          dragging={dragging}
        >
          {away && busy && (
            <button
              type="button"
              onClick={toLatest}
              aria-label={t("Jump to the latest")}
              className="msg-in absolute -top-11 left-1/2 z-[2] flex h-8 -translate-x-1/2 items-center gap-1.5 rounded-full border border-line bg-surface px-3 text-[13px] shadow-[0_4px_16px_rgb(0_0_0/0.10)] transition-colors hover:bg-hover"
            >
              <HugeiconsIcon icon={ArrowDown01Icon} size={14} aria-hidden />{t("Latest")}
            </button>
          )}
        </PromptBar>
      </div>

      <ConfirmDialog
        open={asking}
        title={t("Delete this conversation?")}
        text={t("Taking back the first prompt leaves nothing in this conversation, so it is deleted from Recents. The prompt goes back to the composer.")}
        confirmLabel={t("Delete conversation")}
        onCancel={cancelAsk}
        onConfirm={() => { setAsking(false); takeBack() }}
      />
      <SettingsSheet open={sheet} onClose={closeSheet} efforts={choices} mcp={mcp} projectionLoaded={projection} />
    </section>
    <RewindDialog
      open={rewindAt !== null} prompts={rewindPrompts()} start={rewindAt ?? 0} session={chat.index.active}
      onCancel={() => setRewindAt(null)}
      onDone={(cut) => {
        setRewindAt(null)
        if (!cut) return
        const back = chat.rewindTo(cut.index)                                         // the conversation is cut at the prompt, and the prompt goes back to the composer
        if (!back) return
        setLeaving(back.removed)
        setText((cur) => (cur.trim() ? back.text + "\n\n" + cur : back.text))
        setFiles((f) => [...back.attachments, ...f])
        input.current?.focus()
      }}
    />
    <PanelDock
      open={panel.open} tab={panel.tab} onTab={(tab: PanelTab) => keepPanel({ ...panel, tab })} onClose={() => keepPanel({ ...panel, open: false })}
      data={{
        folders: chat.folders(), allowed: agentInfo.allowed, busy: !!busy, messages: chat.messages, commands, onAsk: (text) => { setText(text); input.current?.focus() },
        context: contextView(chat.messages, chat.contextReported(), health.max_context, chat.settings.autoCompact !== false), canCompact: chat.canCompact(), onCompact: () => { void chat.compact(ctx()) },
      }}
    />
    </div>
    </SkillsContext.Provider>
  )
}
