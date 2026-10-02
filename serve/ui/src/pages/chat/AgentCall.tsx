import { useState } from "react"
import { HugeiconsIcon } from "@hugeicons/react"
import { ArrowDown01Icon, CheckmarkCircle02Icon, CircleIcon, Alert02Icon } from "@hugeicons/core-free-icons"
import { chat, type AskedQuestion, type Todo, type ToolCall } from "../../lib/chat"
import { diffRows, toolSummary } from "../../lib/agent"
import type { Effect, Scope } from "../../lib/perms"
import { Collapse } from "../../components/motion"
import { Orb } from "../../components/orb"
import { toolDesign } from "../../lib/orbs"
import { cn } from "../../lib/cn"
import { fmt } from "../../lib/format"
import { markdown } from "../../lib/markdown"
import { msg, t } from "../../lib/i18n"
import { eventText, noteText, type HookNote } from "../../lib/hooks"

// A call of the chat's coding tools (issue #96: Read, Write, Edit, Glob, Grep, Bash, ...), shown by what it is: the command and its output,
// an edit as a diff, a file's lines. When the server asks the user about it, the question is a card on the call: Allow, Allow for this chat,
// or Deny. The server runs the call only after "allow"; the card is only a way to say so.

const STATE: Record<ToolCall["state"], string> = {
  writing: msg("Writing"), asking: msg("Waiting for you"), running: msg("Running"), done: msg("Done"), error: msg("Error"), skipped: msg("Not run"),
}

// why the server asks (its words, in the language in use when it is one of the usual ones)
const WHY: Record<string, string> = {
  "this chat has no project folder": msg("This chat has no project folder."),
  "it is outside the project folder": msg("It is outside the project folder."),
  "it reads outside the project folder": msg("It reads outside the project folder."),
  "it looks like a secret (a key, a token, an .env file)": msg("It looks like a secret (a key, a token, an .env file)."),
  "it reads a file that looks like a secret": msg("It reads a file that looks like a secret."),
  "a change in .git can run programs (hooks) later": msg("A change in .git can run programs (hooks) later."),
  "a command asks every time": msg("A command asks every time."),
  "it is not a plain read-only command": msg("It is not a plain read-only command."),
  "this command can do harm that is hard to undo": msg("This command can do harm that is hard to undo."),
  "it writes to a file or hides what runs (a substitution, a redirection)": msg("It writes to a file or hides what runs (a substitution, a redirection)."),
  "this git command can change something": msg("This git command can change something."),
  "approve the plan to leave plan mode": msg("Approve the plan to leave plan mode."),
}
const why = (s: string) => (WHY[s] ? t(WHY[s]) : s)

const QUESTION: Record<string, string> = {
  Bash: msg("Run this command?"), Read: msg("Read this file?"), Glob: msg("Look for files here?"), Grep: msg("Search here?"),
  Write: msg("Write this file?"), Edit: msg("Change this file?"), NotebookEdit: msg("Change this notebook?"), ExitPlanMode: msg("Approve this plan?"),
}

const str = (x: unknown) => (typeof x === "string" ? x : "")
const arg = (call: ToolCall, k: string) => (call.arguments && typeof call.arguments === "object" ? (call.arguments as Record<string, unknown>)[k] : undefined)
const Out = ({ text }: { text: string }) => <pre className="tool-pre">{text}</pre>

/** What a card says after the user kept the answer as a rule. */
function keptText(k: { scope: "project" | "everywhere"; effect: "allow" | "deny" }): string {
  return k.effect === "allow" ? (k.scope === "project" ? t("You allowed it in this project from now on.") : t("You allowed it everywhere from now on.")) : (k.scope === "project" ? t("You refused it in this project from now on.") : t("You refused it everywhere from now on."))
}

/** The model asks the user (AskUserQuestion): each question with its choices, one or several, and a line for something else; Send, or Skip. Answered, it is the answers. */
function Ask({ call }: { call: ToolCall }) {
  const asked = call.question!
  const [picked, setPicked] = useState<Record<string, string[]>>({})
  const [other, setOther] = useState<Record<string, string>>({})
  const [busy, setBusy] = useState(false)
  const done = asked.answers !== undefined
  const chosen = (q: string) => [...(picked[q] ?? []), ...(other[q]?.trim() ? [other[q].trim()] : [])]
  const complete = asked.questions.every((q) => chosen(q.question).length > 0)
  const toggle = (q: AskedQuestion["questions"][number], label: string) => setPicked((p) => {
    const now = p[q.question] ?? []
    return { ...p, [q.question]: q.multiSelect ? (now.includes(label) ? now.filter((x) => x !== label) : [...now, label]) : now[0] === label ? [] : [label] }
  })
  const send = async (answers: Record<string, string[]> | null) => { setBusy(true); await chat.answerQuestion(call.id, answers); setBusy(false) }
  if (done) {
    return (
      <div className="space-y-1 border-t border-line px-3 py-2 text-[12.5px]" data-asked-answers>
        {asked.answers === null ? <div className="text-ink-2">{t("You skipped the questions.")}</div> : asked.questions.map((q) => (
          <div key={q.question}><span className="text-ink-2">{q.header}: </span><span className="font-medium">{(asked.answers?.[q.question] ?? []).join(", ") || t("(no answer)")}</span></div>
        ))}
      </div>
    )
  }
  return (
    <div role="group" aria-label={t("The model asks you")} data-agent-question className="space-y-4 border-t border-line bg-hover/40 px-3 py-3">
      {asked.questions.map((q) => (
        <fieldset key={q.question} data-question={q.header} className="space-y-1.5">
          <legend className="flex items-center gap-2 text-[13px] font-medium"><span className="rounded-sm bg-fill px-1.5 text-[11px] text-ink-2">{q.header}</span>{q.question}</legend>
          {q.multiSelect && <div className="text-[11.5px] text-ink-3">{t("You can pick more than one.")}</div>}
          <div className="space-y-1">
            {q.options.map((o) => {
              const on = (picked[q.question] ?? []).includes(o.label)
              return (
                <label key={o.label} data-option={o.label} className={cn("flex cursor-pointer items-start gap-2 rounded-sm border px-2.5 py-1.5 text-[13px] transition-colors", on ? "border-accent bg-fill" : "border-line hover:bg-hover")}>
                  <input type={q.multiSelect ? "checkbox" : "radio"} name={call.id + q.question} checked={on} onChange={() => toggle(q, o.label)} className="mt-0.5" />
                  <span><span className="block font-medium">{o.label}</span>{o.description && <span className="block text-[12px] text-ink-2">{o.description}</span>}</span>
                </label>
              )
            })}
            <input className="h-8 w-full rounded-sm border border-line bg-surface px-2.5 text-[13px] outline-none transition-colors placeholder:text-ink-3 hover:border-fill-2 focus:border-accent" aria-label={t("Something else: {question}", { question: q.header })} placeholder={t("Something else")} value={other[q.question] ?? ""} onChange={(e) => setOther((o) => ({ ...o, [q.question]: e.target.value }))} />
          </div>
        </fieldset>
      ))}
      <div className="flex flex-wrap items-center gap-2">
        <button type="button" disabled={busy || !complete} onClick={() => void send(Object.fromEntries(asked.questions.map((q) => [q.question, chosen(q.question)])))} className="rounded-sm bg-ink px-3 py-1 text-[13px] font-medium text-surface transition-opacity disabled:opacity-40">{t("Send")}</button>
        <button type="button" disabled={busy} onClick={() => void send(null)} className="rounded-sm border border-line px-3 py-1 text-[13px] transition-colors hover:bg-hover disabled:opacity-40">{t("Skip")}</button>
      </div>
    </div>
  )
}

function Question({ call }: { call: ToolCall }) {
  const ask = call.ask!
  const [busy, setBusy] = useState(false)
  const [more, setMore] = useState(false)
  const project = chat.currentProject()
  const go = async (d: "allow" | "allow_chat" | "deny", keep?: { scope: Scope; effect: Effect }) => { setBusy(true); await chat.answer(call.id, d, keep); setBusy(false) }
  const here: Scope | null = project ? { kind: "project", id: project } : null
  const a = ask.arguments ?? call.arguments
  const get = (k: string) => (a && typeof a === "object" ? str((a as Record<string, unknown>)[k]) : "")
  return (
    <div role="group" aria-label={t("Permission needed")} data-agent-ask className={cn("space-y-2 border-t px-3 py-3", ask.danger ? "border-bad/40 bg-bad/5" : "border-line bg-hover/40")}>
      <div className="flex items-center gap-2 text-[13px] font-medium">
        {ask.danger && <HugeiconsIcon icon={Alert02Icon} size={15} aria-hidden className="text-bad" />}
        {t(QUESTION[ask.tool] ?? "Allow this?")}
      </div>
      {ask.tool === "Bash" && <Out text={get("command")} />}
      {(ask.tool === "Read" || ask.tool === "Write" || ask.tool === "Edit") && <div className="font-mono text-[12px] [overflow-wrap:anywhere]">{get("file_path")}</div>}
      {ask.tool === "NotebookEdit" && <div className="font-mono text-[12px] [overflow-wrap:anywhere]">{get("notebook_path")}{get("cell_id") ? ` — ${get("cell_id")}` : ""}</div>}
      {(ask.tool === "Glob" || ask.tool === "Grep") && <div className="font-mono text-[12px] [overflow-wrap:anywhere]">{get("pattern")}{get("path") ? ` — ${get("path")}` : ""}</div>}
      {ask.tool === "ExitPlanMode" && <div className="prose-chat max-h-72 overflow-auto text-[13px]" dangerouslySetInnerHTML={{ __html: markdown(get("plan")) }} />}
      <div className="text-[12px] text-ink-2">{why(ask.why)}</div>
      {ask.danger && <div className="text-[12px] font-medium text-bad">{t("It can do harm that is hard to undo. Only allow it if you meant it.")}</div>}
      <div className="flex flex-wrap items-center gap-2 pt-1">
        <button type="button" disabled={busy} onClick={() => void go("allow")} className="rounded-sm bg-ink px-3 py-1 text-[13px] font-medium text-surface transition-opacity disabled:opacity-40">{t("Allow")}</button>
        {ask.rule && !ask.danger && (
          <button type="button" disabled={busy} onClick={() => void go("allow_chat")} title={ask.rule} className="rounded-sm border border-line px-3 py-1 text-[13px] transition-colors hover:bg-hover disabled:opacity-40">{t("Allow for this chat")}</button>
        )}
        <button type="button" disabled={busy} onClick={() => void go("deny")} className="rounded-sm border border-line px-3 py-1 text-[13px] transition-colors hover:bg-hover disabled:opacity-40">{t("Deny")}</button>
        {ask.rule && (
          <button type="button" aria-expanded={more} data-ask-more onClick={() => setMore(!more)} className="rounded-sm px-2 py-1 text-[12px] text-ink-2 transition-colors hover:bg-hover hover:text-ink">{more ? t("Fewer choices") : t("More choices")}</button>
        )}
      </div>
      {ask.rule && more && (
        <div className="flex flex-wrap items-center gap-2" data-ask-keep>
          {!ask.danger && here && <button type="button" disabled={busy} onClick={() => void go("allow_chat", { scope: here, effect: "allow" })} className="rounded-sm border border-line px-2.5 py-1 text-[12px] transition-colors hover:bg-hover disabled:opacity-40">{t("Always allow in this project")}</button>}
          {!ask.danger && <button type="button" disabled={busy} onClick={() => void go("allow_chat", { scope: { kind: "everywhere" }, effect: "allow" })} className="rounded-sm border border-line px-2.5 py-1 text-[12px] transition-colors hover:bg-hover disabled:opacity-40">{t("Always allow everywhere")}</button>}
          {here && <button type="button" disabled={busy} onClick={() => void go("deny", { scope: here, effect: "deny" })} className="rounded-sm border border-line px-2.5 py-1 text-[12px] transition-colors hover:bg-hover disabled:opacity-40">{t("Never in this project")}</button>}
          <button type="button" disabled={busy} onClick={() => void go("deny", { scope: { kind: "everywhere" }, effect: "deny" })} className="rounded-sm border border-line px-2.5 py-1 text-[12px] transition-colors hover:bg-hover disabled:opacity-40">{t("Never anywhere")}</button>
        </div>
      )}
      {ask.rule && !ask.danger && <div className="font-mono text-[11px] text-ink-3 [overflow-wrap:anywhere]">{ask.rule}</div>}
    </div>
  )
}

function Judge({ call }: { call: ToolCall }) {
  const j = call.judge
  if (!j) return null
  const n = j.severity ?? "?"
  const text = j.verdict === "allow" ? t("Auto mode checked it: safe ({n}/5)", { n }) : j.verdict === "block" ? t("Auto mode blocked it: too risky ({n}/5)", { n }) : t("Auto mode was unsure ({n}/5), so it asked you", { n })
  return <div data-agent-judge={j.verdict} className={cn("border-t border-line px-3 py-1.5 text-[12px]", j.verdict === "block" ? "text-bad" : "text-ink-2")}>{text}</div>
}

/** What the user's own hooks did (Settings > Hooks): a line each, with the command and what it printed. A hook that stopped a call is shown as a refusal; one that failed or ran out of time, as a note. */
export function HookNotes({ notes, className }: { notes?: HookNote[]; className?: string }) {
  if (!notes?.length) return null
  return (
    <div className={className} data-hook-notes>
      {notes.map((n, i) => (
        <div key={i} data-hook={n.on} data-hook-state={n.blocked ? "blocked" : n.ok ? "ok" : "failed"} className={cn("border-t border-line px-3 py-1.5 text-[12px]", n.blocked ? "text-bad" : "text-ink-2")}>
          <div><span className="font-medium">{noteText(n)}</span><span className="text-ink-3"> · {eventText(n.on)} · </span><span className="font-mono text-ink-3 [overflow-wrap:anywhere]">{n.command}</span></div>
          {(n.text || n.error) && <div className="mt-0.5 whitespace-pre-wrap font-mono text-[11.5px] [overflow-wrap:anywhere]">{n.error || n.text}</div>}
        </div>
      ))}
    </div>
  )
}

function Body({ call }: { call: ToolCall }) {
  const name = call.tool || call.name
  const result = call.result != null ? <Out text={call.result} /> : null
  if (name === "Bash") return <div className="space-y-2"><Out text={`$ ${str(arg(call, "command"))}`} />{result}</div>
  if (name === "Edit") {
    const rows = diffRows(str(arg(call, "old_string")), str(arg(call, "new_string")))
    return (
      <div className="space-y-2">
        <div className="tool-pre p-0" role="group" aria-label={t("Change")}>
          {rows.map((r, i) => (
            <div key={i} data-diff={r.kind} className={cn("whitespace-pre-wrap px-3 [overflow-wrap:anywhere]", r.kind === "del" && "bg-bad/10 text-bad", r.kind === "add" && "bg-ok/10 text-ok", r.kind === "more" && "text-ink-3")}>
              {r.kind === "del" ? "- " : r.kind === "add" ? "+ " : "  "}{r.text}
            </div>
          ))}
        </div>
        {result}
      </div>
    )
  }
  if (name === "Write") return <div className="space-y-2"><Out text={str(arg(call, "content")).split("\n").slice(0, 60).join("\n") + (str(arg(call, "content")).split("\n").length > 60 ? "\n…" : "")} />{result}</div>
  if (name === "NotebookEdit") return <div className="space-y-2"><Out text={str(arg(call, "new_source"))} />{result}</div>
  if (name === "ExitPlanMode") return <div className="prose-chat text-[13px]" dangerouslySetInnerHTML={{ __html: markdown(str(arg(call, "plan"))) }} />
  if (name === "TodoWrite") return <TodoList todos={Array.isArray(arg(call, "todos")) ? (arg(call, "todos") as Todo[]) : []} />
  return result ?? <Out text={JSON.stringify(call.arguments ?? {}, null, 2)} />
}

export function AgentCall({ call }: { call: ToolCall }) {
  const name = call.tool || call.name
  const waiting = !!call.question && call.question.answers === undefined
  const pending = (!!call.ask && !call.ask.answer) || waiting
  const open = pending || !!call.open
  const busy = call.state === "running" || call.state === "writing"
  const summary = toolSummary(name, call.arguments)
  return (
    <div className="my-2 rounded-md border border-line" data-state={call.state} data-agent-call={name}>
      <button
        type="button"
        aria-expanded={open}
        onClick={() => { call.open = !call.open; chat.notify() }}
        className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-left text-[13px] transition-colors hover:bg-hover"
      >
        {busy && <Orb design={toolDesign(call.name)} size={20} />}
        <span className={cn("font-medium", busy && "t-shimmer")}>{name}</span>
        <span className="min-w-0 flex-1 truncate font-mono text-[12px] text-ink-3">{summary}</span>
        <span className={cn("shrink-0", call.state === "error" ? "text-bad" : call.state === "asking" ? "font-medium text-ink" : "text-ink-2")}>{t(STATE[call.state])}</span>
        {call.ms != null && call.state !== "skipped" && <span className="num shrink-0 text-ink-3">{fmt(call.ms / 1000, 1)} s</span>}
        <HugeiconsIcon icon={ArrowDown01Icon} size={14} aria-hidden className={cn("shrink-0 text-ink-3 transition-transform duration-200", open && "rotate-180")} />
      </button>
      <Judge call={call} />
      <HookNotes notes={call.hooks} />
      {call.question && <Ask call={call} />}
      {pending && !waiting && <Question call={call} />}
      {call.ask?.answer && <div className="border-t border-line px-3 py-1.5 text-[12px] text-ink-2">{call.ask.kept ? keptText(call.ask.kept) : call.ask.answer === "deny" ? t("You did not allow it.") : call.ask.answer === "allow_chat" ? t("You allowed it for this chat.") : t("You allowed it.")}</div>}
      <Collapse open={open && !pending}>
        <div className="space-y-2 border-t border-line px-3 py-2.5"><Body call={call} /></div>
      </Collapse>
    </div>
  )
}

/** The steps the model keeps for a longer task (TodoWrite): done, the one in progress, and what is left. */
export function TodoList({ todos }: { todos: Todo[] }) {
  if (!todos.length) return null
  const done = todos.filter((x) => x.status === "completed").length
  return (
    <div className="my-2 rounded-md border border-line px-3 py-2.5" role="group" aria-label={t("Steps")} data-todos>
      <div className="mb-1.5 text-[12px] text-ink-2">{t("{done} of {n} steps done", { done: fmt(done), n: fmt(todos.length) })}</div>
      <ul className="m-0 list-none space-y-1 p-0">
        {todos.map((x, i) => (
          <li key={i} data-todo={x.status} className={cn("flex items-start gap-2 text-[13px]", x.status === "completed" && "text-ink-3 line-through")}>
            <HugeiconsIcon icon={x.status === "completed" ? CheckmarkCircle02Icon : CircleIcon} size={15} aria-hidden className={cn("mt-0.5 shrink-0", x.status === "in_progress" ? "text-ink" : "text-ink-3")} />
            <span className={cn(x.status === "in_progress" && "font-medium")}>{x.status === "in_progress" ? x.activeForm : x.content}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}
