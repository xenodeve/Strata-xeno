import { useCallback, useEffect, useRef, useState, type ReactNode } from "react"
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react"
import { ArrowDown01Icon, ArrowReloadHorizontalIcon, BookOpenTextIcon, CheckmarkCircle01Icon, GitBranchIcon, GitCommitIcon, Layers01Icon, Cancel01Icon } from "@hugeicons/core-free-icons"
import type { Message } from "../lib/chat"
import type { Command } from "../lib/slash"
import type { ContextView } from "../lib/context"
import { baseName, getGit, getGitDiff, parseDiff, STATUS_WORD, type DiffLine, type GitFile, type GitInfo, type GitResult } from "../lib/git"
import { planOf, skillsUsed, type PanelTab } from "../lib/panel"
import { getMemory, getMemoryFile, saveMemorySwitches, sizeText, withSwitch, type MemoryList, type MemorySource } from "../lib/memory"
import { toast } from "./toast"
import { MiniSwitch } from "./ui"
import { cn } from "../lib/cn"
import { t } from "../lib/i18n"
import { ContextPanel } from "./ContextPanel"
import { Collapse, GlidePanel, useMounted } from "./motion"
import { Loading } from "./orb"
import { TodoList } from "../pages/chat/AgentCall"
import { Prose } from "../pages/chat/Messages"

// The Chat's right panel (issue #99): what the session is doing, beside the conversation. Git (the branch, the other branches, what changed, the diff of a file, the last commits,
// the worktrees: read only, from the server), the plan (the model's to-do list and the plan it sent for approval), the skills used, and the context window. It is shown beside the chat
// on a wide screen and over it, as a sheet, on a narrow one; whether it is open and which tab is kept in the browser.

const TAB_META: Record<PanelTab, { icon: IconSvgElement; label: () => string }> = {
  git: { icon: GitBranchIcon, label: () => t("Git") },
  plan: { icon: CheckmarkCircle01Icon, label: () => t("Plan") },
  skills: { icon: BookOpenTextIcon, label: () => t("Skills") },
  memory: { icon: BookOpenTextIcon, label: () => t("Memory") },
  context: { icon: Layers01Icon, label: () => t("Context") },
}
export const SHOWN_TABS: PanelTab[] = ["git", "plan", "skills", "memory", "context"]

export interface PanelData {
  folders: string[]                  // the project's folders (or the default one): one Git view each
  allowed: boolean                   // this page may use the coding tools (and so read the repositories): only from this PC
  busy: boolean                      // an answer is being written: the Git view refreshes when it ends
  messages: Message[]
  commands: Command[]
  onAsk: (text: string) => void      // puts text in the composer (to ask the model to change a memory file)
  context: ContextView
  canCompact: boolean
  onCompact: () => void
}

const STATUS_COLOR: Record<string, string> = { M: "text-[#b7791f]", A: "text-[#2f9e6b]", D: "text-bad", R: "text-accent", C: "text-accent", T: "text-ink-2", U: "text-bad", "?": "text-ink-3" }

function StatusBadge({ status }: { status: string }) {
  return <span className={cn("w-4 shrink-0 text-center font-mono text-[11px] font-semibold", STATUS_COLOR[status] ?? "text-ink-2")} title={t(STATUS_WORD[status] ?? status)} aria-label={t(STATUS_WORD[status] ?? status)}>{status}</span>
}

// ------------------------------------------------------------------------------------------------ Git
function DiffView({ lines }: { lines: DiffLine[] }) {
  return (
    <div className="overflow-x-auto rounded-sm border border-line bg-surface font-mono text-[11.5px] leading-[1.45]" data-diff>
      {lines.map((l, i) => l.kind === "hunk"
        ? <div key={i} className="bg-fill px-2 py-0.5 text-ink-3">{l.text}</div>
        : l.kind === "note"
          ? <div key={i} className="px-2 text-ink-3 italic">{l.text}</div>
          : (
            <div key={i} data-line={l.kind} className={cn("flex min-w-max", l.kind === "add" && "bg-[color-mix(in_srgb,#2f9e6b_16%,transparent)]", l.kind === "del" && "bg-[color-mix(in_srgb,#d33_14%,transparent)]")}>
              <span className="num w-8 shrink-0 select-none pr-1 text-right text-ink-3">{l.old ?? ""}</span>
              <span className="num w-8 shrink-0 select-none pr-1 text-right text-ink-3">{l.now ?? ""}</span>
              <span className="w-3 shrink-0 select-none text-center text-ink-3">{l.kind === "add" ? "+" : l.kind === "del" ? "−" : ""}</span>
              <span className="whitespace-pre pr-2">{l.text}</span>
            </div>
          ))}
    </div>
  )
}

/** One changed file: its status and path, and the diff that opens under it. */
function FileRow({ folder, file, staged, untracked }: { folder: string; file: GitFile; staged: boolean; untracked: boolean }) {
  const [open, setOpen] = useState(false)
  const [diff, setDiff] = useState<{ lines: DiffLine[]; binary: boolean; truncated: boolean } | "loading" | "error" | null>(null)
  const toggle = () => {
    const next = !open
    setOpen(next)
    if (next && diff === null) {
      setDiff("loading")
      void getGitDiff(folder, file.path, staged, untracked).then((r) => setDiff(r === "unavailable" || !r.ok ? "error" : { lines: parseDiff(r.diff), binary: r.binary, truncated: r.truncated }))
    }
  }
  return (
    <li data-file={file.path} data-group={staged ? "staged" : untracked ? "untracked" : "unstaged"}>
      <button type="button" aria-expanded={open} onClick={toggle} className="flex w-full items-center gap-2 rounded-sm px-1.5 py-1 text-left text-[12.5px] transition-colors hover:bg-hover">
        <StatusBadge status={file.status} />
        <span className="min-w-0 flex-1 truncate" title={file.from ? `${file.from} → ${file.path}` : file.path}>{file.from ? `${baseName(file.from)} → ` : ""}{file.path}</span>
        <HugeiconsIcon icon={ArrowDown01Icon} size={12} aria-hidden className={cn("shrink-0 text-ink-3 transition-transform duration-200", open && "rotate-180")} />
      </button>
      <Collapse open={open}>
        <div className="px-1 pb-1.5 pt-0.5">
          {diff === "loading" || diff === null ? <p className="text-[12px] text-ink-3">{t("Reading the change…")}</p>
            : diff === "error" ? <p className="text-[12px] text-bad">{t("The change could not be read.")}</p>
            : diff.binary ? <p className="text-[12px] text-ink-3">{t("A binary file: no lines to show.")}</p>
            : diff.lines.length === 0 ? <p className="text-[12px] text-ink-3">{t("No lines changed.")}</p>
            : <><DiffView lines={diff.lines} />{diff.truncated && <p className="mt-1 text-[12px] text-ink-3">{t("The diff is long: only the start is shown.")}</p>}</>}
        </div>
      </Collapse>
    </li>
  )
}

function Group({ title, files, folder, staged, untracked, count }: { title: string; files: GitFile[]; folder: string; staged: boolean; untracked: boolean; count: number }) {
  if (!files.length) return null
  return (
    <section data-group-title={title}>
      <h3 className="mb-0.5 flex items-center justify-between px-1.5 text-[11.5px] font-medium text-ink-3"><span>{title}</span><span className="num">{count}</span></h3>
      <ul>{files.map((f) => <FileRow key={f.path + (staged ? ":s" : "")} folder={folder} file={f} staged={staged} untracked={untracked} />)}</ul>
    </section>
  )
}

function Section({ title, count, children, open: start = false }: { title: string; count?: number; children: ReactNode; open?: boolean }) {
  const [open, setOpen] = useState(start)
  return (
    <section className="border-t border-line pt-2" data-section-title={title}>
      <button type="button" aria-expanded={open} onClick={() => setOpen(!open)} className="flex w-full items-center gap-2 rounded-sm px-1.5 py-1 text-left text-[12.5px] font-medium transition-colors hover:bg-hover">
        <span className="flex-1">{title}</span>
        {count !== undefined && <span className="num text-[11.5px] text-ink-3">{count}</span>}
        <HugeiconsIcon icon={ArrowDown01Icon} size={12} aria-hidden className={cn("text-ink-3 transition-transform duration-200", open && "rotate-180")} />
      </button>
      <Collapse open={open}><div className="pb-1 pt-0.5">{children}</div></Collapse>
    </section>
  )
}

function GitView({ info, folder }: { info: GitInfo; folder: string }) {
  const c = info.counts ?? { staged: 0, unstaged: 0, untracked: 0, conflicted: 0 }
  const total = c.staged + c.unstaged + c.untracked + c.conflicted
  return (
    <div className="space-y-3" data-git-view>
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <span className="inline-flex items-center gap-1.5 rounded-sm bg-fill px-2 py-1 text-[12.5px] font-medium" data-branch>
          <HugeiconsIcon icon={GitBranchIcon} size={13} aria-hidden />
          {info.detached ? t("detached at {sha}", { sha: info.oid ?? "" }) : info.branch ?? t("no commits yet")}
        </span>
        {!!info.ahead && <span className="num text-[12px] text-ink-2" title={t("Commits not pushed to {upstream}", { upstream: info.upstream ?? "" })} data-ahead>↑{info.ahead}</span>}
        {!!info.behind && <span className="num text-[12px] text-ink-2" title={t("Commits on {upstream} not here", { upstream: info.upstream ?? "" })} data-behind>↓{info.behind}</span>}
        {info.upstream && <span className="truncate text-[12px] text-ink-3">{info.upstream}</span>}
      </div>

      {total === 0
        ? <p className="px-1.5 text-[12.5px] text-ink-2" data-clean>{t("Nothing has changed: the working tree is clean.")}</p>
        : (
          <div className="space-y-2.5">
            <Group title={t("Conflicts")} files={info.conflicted ?? []} folder={folder} staged={false} untracked={false} count={c.conflicted} />
            <Group title={t("Staged")} files={info.staged ?? []} folder={folder} staged untracked={false} count={c.staged} />
            <Group title={t("Not staged")} files={info.unstaged ?? []} folder={folder} staged={false} untracked={false} count={c.unstaged} />
            <Group title={t("New files")} files={info.untracked ?? []} folder={folder} staged={false} untracked count={c.untracked} />
            {info.truncated && <p className="px-1.5 text-[12px] text-ink-3">{t("There are more changes than are listed.")}</p>}
          </div>
        )}

      <Section title={t("Branches")} count={info.branches?.length}>
        <ul className="space-y-0.5">
          {(info.branches ?? []).map((b) => (
            <li key={b.name} data-branch-row={b.name} data-current={b.current ? "" : undefined} className="flex items-center gap-2 px-1.5 py-0.5 text-[12.5px]">
              <span className={cn("size-1.5 shrink-0 rounded-full", b.current ? "bg-accent" : "bg-transparent")} aria-hidden />
              <span className={cn("min-w-0 flex-1 truncate", b.current && "font-medium")}>{b.name}</span>
              {b.track && <span className="shrink-0 text-[11.5px] text-ink-3">{b.track.replace(/[[\]]/g, "")}</span>}
            </li>
          ))}
          {info.moreBranches && <li className="px-1.5 text-[12px] text-ink-3">{t("There are more branches than are listed.")}</li>}
        </ul>
      </Section>

      {(info.worktrees?.length ?? 0) > 1 && (
        <Section title={t("Worktrees")} count={info.worktrees!.length}>
          <ul className="space-y-0.5">
            {info.worktrees!.map((w) => (
              <li key={w.path} data-worktree data-current={w.current ? "" : undefined} className="px-1.5 py-0.5 text-[12.5px]">
                <div className="flex items-center gap-2"><span className={cn("size-1.5 shrink-0 rounded-full", w.current ? "bg-accent" : "bg-transparent")} aria-hidden /><span className="truncate font-medium">{w.detached ? t("detached at {sha}", { sha: w.head ?? "" }) : w.branch ?? t("bare")}</span></div>
                <div className="truncate pl-3.5 font-mono text-[11.5px] text-ink-3" title={w.path}>{w.path}</div>
              </li>
            ))}
          </ul>
        </Section>
      )}

      <Section title={t("Recent commits")} count={info.commits?.length} open>
        {(info.commits ?? []).length === 0 ? <p className="px-1.5 text-[12.5px] text-ink-3">{t("No commits yet.")}</p> : (
          <ul className="space-y-1">
            {info.commits!.map((k) => (
              <li key={k.sha} data-commit className="flex items-start gap-2 px-1.5 text-[12.5px]">
                <HugeiconsIcon icon={GitCommitIcon} size={13} aria-hidden className="mt-0.5 shrink-0 text-ink-3" />
                <span className="min-w-0 flex-1"><span className="block truncate" title={k.subject}>{k.subject}</span><span className="num block truncate text-[11.5px] text-ink-3">{k.sha} · {k.author} · {new Date(k.time * 1000).toLocaleDateString(undefined, { month: "short", day: "numeric" })}</span></span>
              </li>
            ))}
          </ul>
        )}
      </Section>
    </div>
  )
}

function GitTab({ folders, allowed, busy }: Pick<PanelData, "folders" | "allowed" | "busy">) {
  const [at, setAt] = useState(0)
  const [seen, setSeen] = useState<Record<string, GitResult | "loading">>({})
  const folder = folders[Math.min(at, folders.length - 1)] ?? null
  const load = useCallback((f: string) => {
    setSeen((s) => ({ ...s, [f]: s[f] && s[f] !== "loading" ? s[f] : "loading" }))        // what was shown stays while it is read again
    void getGit(f).then((r) => setSeen((s) => ({ ...s, [f]: r })))
  }, [])
  useEffect(() => { if (folder && allowed) load(folder) }, [folder, allowed, load])
  const was = useRef(busy)
  useEffect(() => { if (was.current && !busy && folder && allowed) load(folder); was.current = busy }, [busy, folder, allowed, load])      // an answer ended: the files may have changed

  if (!allowed) return <p className="text-[13px] text-ink-2">{t("Git is read from this PC's files, so it is shown only on the PC that runs Strata (or with the API key).")}</p>
  if (!folder) return <p className="text-[13px] text-ink-2">{t("This chat has no project folder, so there is no repository to show. Make a project with a folder, or set the default folder in Settings.")}</p>
  const r = seen[folder]
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-1.5">
        {folders.length > 1 ? (
          <div role="tablist" aria-label={t("Folders of the project")} className="flex min-w-0 flex-1 gap-1 overflow-x-auto">
            {folders.map((f, i) => (
              <button key={f} type="button" role="tab" aria-selected={i === at} title={f} onClick={() => setAt(i)} className={cn("shrink-0 rounded-sm px-2 py-1 text-[12px] transition-colors", i === at ? "bg-fill font-medium" : "text-ink-2 hover:bg-hover")}>{baseName(f)}</button>
            ))}
          </div>
        ) : <span className="min-w-0 flex-1 truncate font-mono text-[11.5px] text-ink-3" title={folder}>{folder}</span>}
        <button type="button" aria-label={t("Read the repository again")} title={t("Refresh")} onClick={() => load(folder)} className="flex size-7 shrink-0 items-center justify-center rounded-sm text-ink-2 transition-colors hover:bg-hover hover:text-ink">
          <HugeiconsIcon icon={ArrowReloadHorizontalIcon} size={14} aria-hidden />
        </button>
      </div>
      {r === undefined || r === "loading" ? <Loading>{t("Reading the repository…")}</Loading>
        : r === "unavailable" ? <p className="text-[13px] text-ink-2">{t("The server could not be asked about this folder.")}</p>
        : !r.ok ? <p className="text-[13px] text-bad">{r.error === "not a folder" ? t("This folder is not on this PC.") : r.error}</p>
        : !r.git ? <p className="text-[13px] text-ink-2">{t("Git is not installed on this PC.")}</p>
        : !r.repo ? <p className="text-[13px] text-ink-2" data-not-repo>{t("This folder is not in a Git repository.")}</p>
        : r.error ? <p className="text-[13px] text-bad">{t("Git could not read this repository.")}</p>
        : <GitView info={r} folder={folder} />}
    </div>
  )
}

// ------------------------------------------------------------------------------------------------ plan, skills
function PlanTab({ messages }: { messages: Message[] }) {
  const p = planOf(messages)
  if (!p.total && !p.approved) return <p className="text-[13px] text-ink-2" data-empty-plan>{t("No plan yet. When the model keeps a to-do list or sends a plan for approval, it is here.")}</p>
  return (
    <div className="space-y-3">
      {p.total > 0 && (
        <div>
          <div className="mb-1.5 flex items-center justify-between text-[12px] text-ink-2"><span>{t("To-do list")}</span><span className="num" data-progress>{t("{done} of {total} done", { done: p.done, total: p.total })}</span></div>
          <div className="mb-2 h-1 overflow-hidden rounded-full bg-fill" aria-hidden><div className="h-full rounded-full bg-accent transition-[width] duration-300" style={{ width: `${(p.done / p.total) * 100}%` }} /></div>
          <TodoList todos={p.todos} />
        </div>
      )}
      {p.approved && (
        <Section title={t("The plan that was sent for approval")} open>
          <div className="px-1.5 text-[13px]" data-plan-text><Prose text={p.approved} /></div>
        </Section>
      )}
    </div>
  )
}

function SkillsTab({ messages, commands }: { messages: Message[]; commands: Command[] }) {
  const used = skillsUsed(messages, commands.filter((c) => !c.builtin).map((c) => c.name))
  if (!used.length) return <p className="text-[13px] text-ink-2" data-empty-skills>{t("No skill has been used in this chat. Type / to pick one, or the model may load one when it fits.")}</p>
  return (
    <ul className="space-y-1.5">
      {used.map((u) => {
        const cmd = commands.find((c) => c.name === u.name)
        return (
          <li key={u.name} data-skill-used={u.name} className="rounded-sm border border-line px-2.5 py-2">
            <div className="flex items-center gap-2 text-[13px]"><span className="font-medium">/{u.name}</span><span className="rounded-sm bg-fill px-1.5 text-[11px] text-ink-2">{u.by === "you" ? t("you asked") : t("the model loaded")}</span><span className="num ml-auto text-[11.5px] text-ink-3">×{u.uses + u.files}</span></div>
            {cmd && <div className="mt-0.5 text-[12px] text-ink-3">{cmd.plugin ? t("Skill from {app} · plugin {plugin}", { app: cmd.from, plugin: cmd.plugin }) : t("Skill from {app}", { app: cmd.from })}</div>}
            {cmd?.description && <div className="mt-0.5 line-clamp-2 text-[12px] text-ink-2">{cmd.description}</div>}
          </li>
        )
      })}
    </ul>
  )
}

// ------------------------------------------------------------------------------------------------ memory
/** One file of a source: its name, and its text when opened (read from the server when asked). */
function MemoryFile({ folders, source, n, onAsk }: { folders: string[]; source: MemorySource; n: number; onAsk: (text: string) => void }) {
  const [open, setOpen] = useState(false)
  const [got, setGot] = useState<{ name: string; text: string; cut: boolean } | "loading" | "error" | null>(null)
  const name = source.shown[n]
  const toggle = () => {
    const next = !open
    setOpen(next)
    if (next && got === null) {
      setGot("loading")
      void getMemoryFile(folders, source.id, n).then((r) => setGot(r ?? "error"))
    }
  }
  return (
    <li data-memory-file={name}>
      <button type="button" aria-expanded={open} onClick={toggle} className="flex w-full items-center gap-2 rounded-sm px-1.5 py-1 text-left text-[12.5px] transition-colors hover:bg-hover">
        <span className="min-w-0 flex-1 truncate font-mono" title={name}>{name}</span>
        <HugeiconsIcon icon={ArrowDown01Icon} size={12} aria-hidden className={cn("shrink-0 text-ink-3 transition-transform duration-200", open && "rotate-180")} />
      </button>
      <Collapse open={open}>
        <div className="px-1.5 pb-2 pt-0.5">
          {got === "loading" || got === null ? <p className="text-[12px] text-ink-3">{t("Reading the file…")}</p>
            : got === "error" ? <p className="text-[12px] text-bad">{t("The file could not be read.")}</p>
            : <>
              <div className="max-h-64 overflow-y-auto rounded-sm border border-line px-2.5 py-2 text-[12.5px]" data-memory-text><Prose text={got.text} /></div>
              {got.cut && <p className="mt-1 text-[12px] text-ink-3">{t("The file is long: only the start is shown here.")}</p>}
              <button type="button" onClick={() => onAsk(t("Please update {file}: ", { file: name }))} className="mt-1.5 rounded-sm px-1.5 py-1 text-[12px] text-ink-2 transition-colors hover:bg-hover hover:text-ink">{t("Ask the model to change it")}</button>
            </>}
        </div>
      </Collapse>
    </li>
  )
}

function MemoryTab({ folders, allowed, onAsk }: { folders: string[]; allowed: boolean; onAsk: (text: string) => void }) {
  const [list, setList] = useState<MemoryList | null | "loading">("loading")
  const key = folders.join("|")
  const load = useCallback(() => { void getMemory(folders).then(setList) }, [key])       // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { if (allowed) load() }, [allowed, load])
  if (!allowed) return <p className="text-[13px] text-ink-2">{t("Notes are read from this PC's files, so they are shown only on the PC that runs Strata (or with the API key).")}</p>
  if (list === "loading") return <Loading>{t("Looking for notes…")}</Loading>
  if (!list) return <p className="text-[13px] text-ink-2">{t("The server could not be asked for the notes.")}</p>
  const others = list.sources.filter((s) => s.kind !== "project")
  const flip = async (s: MemorySource, value: boolean) => {
    const on = withSwitch(others.filter((x) => x.on).map((x) => x.id), s.id, value)
    setList({ ...list, sources: list.sources.map((x) => (x.id === s.id ? { ...x, on: value } : x)) })
    const err = await saveMemorySwitches(on)
    if (err) toast("error", t("The switch was not saved"), err)
    load()
  }
  const inUse = list.sources.filter((s) => s.on)
  return (
    <div className="space-y-3" data-memory-tab>
      <p className="text-[12.5px] text-ink-2">{inUse.length ? t("The chat is handed these notes after its rules. They never change what it may do.") : t("No notes are in use for this chat: no instruction file was found in the project's folders.")}</p>
      {list.sources.map((s) => (
        <section key={s.id} data-memory-source={s.id} data-on={s.on ? "" : undefined} className="rounded-md border border-line">
          <div className="flex items-center gap-2 px-2.5 py-2">
            <div className="min-w-0 flex-1">
              <div className="truncate text-[13px] font-medium">{s.kind === "project" ? t("The project's own: {name}", { name: s.label }) : `${s.label} · ${s.kind === "memory" ? t("what it remembers about this project") : t("its instructions for you")}`}</div>
              <div className="num text-[11.5px] text-ink-3">{sizeText(s.bytes)}{s.kind === "project" ? ` · ${t("always on")}` : ""}</div>
            </div>
            {s.kind !== "project" && <MiniSwitch on={s.on} label={t("Read {name}", { name: `${s.label}: ${s.kind === "memory" ? t("memory") : t("instructions")}` })} onClick={() => void flip(s, !s.on)} />}
          </div>
          <ul className={cn("border-t border-line px-1 py-1", !s.on && "opacity-60")}>
            {s.shown.map((_, n) => <MemoryFile key={s.id + n} folders={folders} source={s} n={n} onAsk={onAsk} />)}
          </ul>
        </section>
      ))}
      {list.sources.length === 0 && <p className="text-[12.5px] text-ink-3">{t("Type /init and the model will look at the project and write a CLAUDE.md for it.")}</p>}
      {others.length === 0 && list.sources.length > 0 && <p className="text-[12px] text-ink-3">{t("Notes your other coding apps keep would be listed here, off until you switch them on.")}</p>}
    </div>
  )
}

// ------------------------------------------------------------------------------------------------ the panel
const WIDE = "(min-width: 1280px)"
const WIDTH = 380
/** Whether the screen is wide enough for the panel to sit beside the chat (else it is a sheet over it). */
function useWide(): boolean {
  const [wide, setWide] = useState(() => typeof matchMedia !== "undefined" && matchMedia(WIDE).matches)
  useEffect(() => {
    const m = matchMedia(WIDE)
    const on = () => setWide(m.matches)
    on()
    m.addEventListener("change", on)
    return () => m.removeEventListener("change", on)
  }, [])
  return wide
}

/** The panel beside the chat on a wide screen (its width stretches open and shrinks to nothing) and as a sheet from the right on a narrow one. */
export function PanelDock({ open, tab, onTab, onClose, data }: { open: boolean; tab: PanelTab; onTab: (t: PanelTab) => void; onClose: () => void; data: PanelData }) {
  const wide = useWide()
  const listed = useMounted(open, 320)                                  // the body stays while the width shrinks around it
  const [shown, setShown] = useState(false)
  useEffect(() => {
    if (!open) { setShown(false); return }
    const id = requestAnimationFrame(() => requestAnimationFrame(() => setShown(true)))
    return () => cancelAnimationFrame(id)
  }, [open])
  useEffect(() => {
    if (!open || wide) return
    const key = (e: globalThis.KeyboardEvent) => { if (e.key === "Escape") onClose() }
    addEventListener("keydown", key)
    return () => removeEventListener("keydown", key)
  }, [open, wide, onClose])
  if (wide) {
    return (
      <aside aria-label={t("Session panel")} aria-hidden={!open} data-panel={open ? "open" : "closed"} className={cn("sticky top-[4.5rem] shrink-0 self-start justify-self-end overflow-hidden transition-[width,margin] duration-300 ease-[var(--ease)]", open ? "ml-6 w-[380px] min-[1600px]:ml-0" : "ml-0 w-0")}>
        {listed && <div inert={!open} className={cn("side-in flex max-h-[calc(100dvh-5.5rem)] flex-col transition-opacity duration-200", !open && "pointer-events-none opacity-0")} style={{ width: WIDTH }}><PanelBody tab={tab} onTab={onTab} data={data} /></div>}
      </aside>
    )
  }
  if (!listed) return null
  return (
    <div className="fixed inset-0 z-40" data-panel="sheet">
      <button type="button" aria-label={t("Close the panel")} onClick={onClose} className={cn("absolute inset-0 bg-black/40 transition-opacity duration-300", shown && open ? "opacity-100" : "opacity-0")} />
      <aside aria-label={t("Session panel")} className={cn("absolute inset-y-0 right-0 w-[min(92vw,380px)] overflow-y-auto bg-surface p-3 shadow-xl transition-transform duration-300 ease-[var(--ease)]", shown && open ? "translate-x-0" : "translate-x-full")}>
        <PanelBody tab={tab} onTab={onTab} onClose={onClose} data={data} />
      </aside>
    </div>
  )
}

export function PanelBody({ tab, onTab, onClose, data }: { tab: PanelTab; onTab: (t: PanelTab) => void; onClose?: () => void; data: PanelData }) {
  const shown = SHOWN_TABS.includes(tab) ? tab : "git"
  return (
    <div className="flex min-h-0 flex-col gap-3">
      <div className="flex items-center gap-1">
        <div role="tablist" aria-label={t("Session panel")} className="flex min-w-0 flex-1 gap-0.5">
          {SHOWN_TABS.map((k) => (
            <button key={k} type="button" role="tab" aria-selected={k === shown} data-tab={k} onClick={() => onTab(k)} className={cn("flex h-8 items-center gap-1.5 rounded-sm px-2 text-[12.5px] transition-colors", k === shown ? "bg-fill font-medium" : "text-ink-2 hover:bg-hover hover:text-ink")}>
              <HugeiconsIcon icon={TAB_META[k].icon} size={14} aria-hidden /><span>{TAB_META[k].label()}</span>
            </button>
          ))}
        </div>
        {onClose && <button type="button" aria-label={t("Close the panel")} onClick={onClose} className="flex size-8 shrink-0 items-center justify-center rounded-sm text-ink-2 transition-colors hover:bg-hover hover:text-ink"><HugeiconsIcon icon={Cancel01Icon} size={15} aria-hidden /></button>}
      </div>
      <GlidePanel role="tabpanel" aria-label={TAB_META[shown].label()} cap={() => Math.max(240, innerHeight - 190)} inner="pb-2" className="-mx-1 px-1">
        {shown === "git" && <GitTab folders={data.folders} allowed={data.allowed} busy={data.busy} />}
        {shown === "plan" && <PlanTab messages={data.messages} />}
        {shown === "skills" && <SkillsTab messages={data.messages} commands={data.commands} />}
        {shown === "memory" && <MemoryTab folders={data.folders} allowed={data.allowed} onAsk={data.onAsk} />}
        {shown === "context" && <ContextPanel view={data.context} canCompact={data.canCompact} onCompact={data.onCompact} />}
      </GlidePanel>
    </div>
  )
}
