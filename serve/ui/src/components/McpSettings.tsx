import { useCallback, useEffect, useRef, useState } from "react"
import { getMcpConfig, saveMcpConfig } from "../lib/api"
import { fmt } from "../lib/format"
import { msg, t, tn } from "../lib/i18n"
import { MASK, bodyFor, emptyDraft, entryFromRow, entryOf, limitsOf, parsePaste, stateOf, toDraft, type Change, type Draft, type McpConfigView, type McpLimits, type McpRow } from "../lib/mcpconfig"
import { cn } from "../lib/cn"
import { Reveal } from "./motion"
import { StatusLabel } from "./orb"
import { toast } from "./toast"
import { Button, Field, Segmented, inputCls } from "./ui"

// Setting up MCP servers (issue #79; on the Settings page): the list with each one's state and tools, add / edit / disable / delete, a box for a
// block pasted from Claude Desktop, and the limits. Each change is saved at once (the server restarts its servers from the
// file; no restart of Strata). Without the right to change them (not this PC, no API key) it is a list and says so.
const STATE: Record<string, string> = {
  ready: msg("Connected"), starting: msg("Starting"), failed: msg("Failed"), stopped: msg("Stopped"), idle: msg("Waiting"), disabled: msg("Disabled"),
}
const areaCls = "w-full resize-y rounded-sm border border-line bg-surface px-2.5 py-1.5 font-mono text-[12px] leading-snug outline-none transition-colors placeholder:text-ink-3 hover:border-fill-2 focus:border-accent disabled:opacity-50"

type Form = { mode: "new" | "edit"; draft: Draft; problems: Record<string, string>; failure: string | null }

export function McpSettings() {
  const [view, setView] = useState<McpConfigView | null>(null)
  const [busy, setBusy] = useState(false)
  const [form, setForm] = useState<Form | null>(null)
  const [paste, setPaste] = useState<{ text: string; problem: string | null } | null>(null)
  const [sure, setSure] = useState<string | null>(null)                       // the server whose Delete was asked once
  const [limits, setLimits] = useState<Record<keyof McpLimits, string> | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const sureTimer = useRef<ReturnType<typeof setTimeout>>(undefined)

  const show = useCallback((v: McpConfigView) => {
    setView(v)
    setLimits((old) => old ?? { timeout_s: String(v.settings.timeout_s), max_result_chars: String(v.settings.max_result_chars), max_rounds: String(v.settings.max_rounds) })
  }, [])
  useEffect(() => { void getMcpConfig().then((v) => v && show(v)) }, [show])
  const starting = view?.servers.some((s) => s.status === "starting") ?? false
  useEffect(() => {                                                           // while a server comes up, look again: its tools appear when it is ready
    if (!starting) return
    const id = setInterval(() => { void getMcpConfig().then((v) => v && setView(v)) }, 1500)
    return () => clearInterval(id)
  }, [starting])

  if (!view) return null
  const rows = view.servers
  const can = view.editable
  const taken = rows.filter((r) => r.source === "config").map((r) => r.name)

  /** Save the list with `change` (and `settings`): null when it was saved, else why not. */
  const commit = async (change?: Change | Change[], settings?: Partial<McpLimits>): Promise<string | null> => {
    setBusy(true)
    setNotice(null)
    const r = await saveMcpConfig(bodyFor(rows, change, settings))
    setBusy(false)
    if ("error" in r) return r.error
    show(r.view)
    return null
  }

  const openForm = (mode: Form["mode"], draft: Draft) => { setPaste(null); setForm({ mode, draft, problems: {}, failure: null }) }
  const setField = (k: keyof Draft, v: string) => setForm((f) => f && { ...f, draft: { ...f.draft, [k]: v } as Draft, problems: { ...f.problems, [k]: "" } })

  const submit = async () => {
    if (!form) return
    const { entry, problems } = entryOf(form.draft, form.mode === "new" ? taken : [])
    if (problems.length) { setForm({ ...form, problems: Object.fromEntries(problems.map((p) => [p.field, p.message])), failure: null }); return }
    const old = rows.find((r) => r.source === "config" && r.name === form.draft.name)
    const err = await commit({ name: form.draft.name, entry: { ...entry, ...(old?.disabled ? { disabled: true } : {}) } })
    if (err) setForm({ ...form, failure: err })
    else { setForm(null); toast("success", t("Saved"), t("The servers start again; no restart is needed.")) }
  }

  const add = async () => {
    if (!paste) return
    const { entries, problem } = parsePaste(paste.text, taken)
    if (problem) { setPaste({ ...paste, problem }); return }
    const count = Object.keys(entries).length
    const err = await commit(Object.entries(entries).map(([name, entry]) => ({ name, entry })))
    if (err) setPaste({ ...paste, problem: err })
    else { setPaste(null); toast("success", t("Saved"), tn(count, "{n} server added.", "{n} servers added.")) }
  }

  const flip = async (r: McpRow) => {
    const { disabled: _was, ...rest } = entryFromRow(r)
    const err = await commit({ name: r.name, entry: r.disabled ? rest : { ...rest, disabled: true } })
    if (err) setNotice(err)
  }

  const remove = async (r: McpRow) => {
    if (sure !== r.name) {
      setSure(r.name)
      clearTimeout(sureTimer.current)
      sureTimer.current = setTimeout(() => setSure(null), 4000)
      return
    }
    setSure(null)
    const err = await commit({ name: r.name, entry: null })
    if (err) setNotice(err)
  }

  const saveLimits = async () => {
    if (!limits) return
    const { limits: n, bad } = limitsOf(limits)
    if (!n) { setNotice(t("{name} must be a number.", { name: bad ? t(LIMIT_NAME[bad]) : "" })); return }
    const err = await commit(undefined, n)
    if (err) setNotice(err)
    else toast("success", t("Saved"), t("The limits apply from the next tool call."))
  }

  const nTools = view.tools
  return (
    <section>
      <h2 className="text-[15px] font-semibold">{t("MCP servers")}</h2>
      <p className="mt-1 text-[13px] text-ink-2">
        {rows.length ? t("{n} tools; the model calls them when it decides to.", { n: fmt(nTools) }) : t("No server is set up yet. A server gives the model tools: files, a browser, a database.")}
      </p>
      {!can && (
        <p role="note" className="mt-2 rounded-md bg-fill px-3 py-2 text-[12px] text-ink-2">
          {view.config_file === null
            ? t("This server was started without a run config file, so there is nowhere to save the servers. Start it with --config to set them up here.")
            : t("MCP servers are programs Strata starts, so they can be changed only from this PC itself, or when Strata has an API key and it is entered under API key above.")}
        </p>
      )}

      <ul className="m-0 mt-2 list-none p-0" aria-label={t("MCP servers")}>
        {rows.map((s) => {
          const state = stateOf(s)
          const mine = can && s.editable
          return (
            <li key={`${s.source}:${s.name}`} className={cn("border-b border-line py-2.5 last:border-0", s.overridden && "opacity-60")} data-server={s.name}>
              <div className="flex items-baseline justify-between gap-4">
                <span className="min-w-0 font-medium [overflow-wrap:anywhere]">{s.name}</span>
                {state === "starting"
                  ? <StatusLabel design="connecting" className="text-ink-2">{t(STATE[state])}</StatusLabel>
                  : <span className={state === "failed" ? "text-bad" : "text-ink-2"}>{STATE[state] ? t(STATE[state]) : state}</span>}
              </div>
              <div className="text-[12px] text-ink-2 [overflow-wrap:anywhere]">
                {s.kind === "address" ? t("Address") : t("Program")} · {t("{n} tools", { n: fmt(s.tools.length) })}
                {s.url && <> · {s.url}</>}
                {s.command && <> · <span className="font-mono">{[s.command, ...(s.args ?? [])].join(" ")}</span></>}
              </div>
              {s.tools.length > 0 && <div className="mt-0.5 text-[12px] text-ink-3 [overflow-wrap:anywhere]">{s.tools.slice(0, 8).map((x) => x.tool).join(", ")}{s.tools.length > 8 ? ", …" : ""}</div>}
              {s.source === "file" && <div className="mt-0.5 text-[12px] text-ink-3">{t("From {file}: edit that file to change it.", { file: s.file || "--mcp-config" })}</div>}
              {s.overridden && <div className="mt-0.5 text-[12px] text-ink-3">{t("A server of the same name in --mcp-config is used instead.")}</div>}
              {s.error && <div className="mt-1 text-[12px] text-bad [overflow-wrap:anywhere]">{s.error}</div>}
              {mine && (
                <div className="mt-1.5 flex flex-wrap gap-1.5">
                  <Button kind="quiet" disabled={busy} onClick={() => openForm("edit", toDraft(s))} label={`${t("Edit")} ${s.name}`}>{t("Edit")}</Button>
                  <Button kind="quiet" disabled={busy} onClick={() => void flip(s)} label={`${s.disabled ? t("Turn on") : t("Turn off")} ${s.name}`}>{s.disabled ? t("Turn on") : t("Turn off")}</Button>
                  <Button kind="quiet" disabled={busy} onClick={() => void remove(s)} label={`${t("Delete")} ${s.name}`}>{sure === s.name ? t("Sure? Delete") : t("Delete")}</Button>
                </div>
              )}
            </li>
          )
        })}
      </ul>
      {notice && <p role="alert" className="mt-2 text-[12px] text-bad [overflow-wrap:anywhere]">{notice}</p>}

      {can && (
        <div className="mt-3 flex flex-wrap gap-2">
          <Button disabled={busy} onClick={() => openForm("new", emptyDraft())}>{t("Add a server")}</Button>
          <Button disabled={busy} onClick={() => { setForm(null); setPaste({ text: "", problem: null }) }}>{t("Paste from Claude Desktop")}</Button>
        </div>
      )}

      <Reveal>
        {form ? (
          <form className="mt-3 rounded-md border border-line p-3" aria-label={form.mode === "new" ? t("Add a server") : t("Edit")} onSubmit={(e) => { e.preventDefault(); void submit() }}>
            <Field label={t("Name")} hint={form.problems.name ? <span className="text-bad">{form.problems.name}</span> : t("Letters, digits, dots, dashes and underscores. The model sees it in front of each tool.")}>
              <input className={inputCls} value={form.draft.name} disabled={form.mode === "edit"} aria-label={t("Name")} onChange={(e) => setField("name", e.target.value)} placeholder="files" autoComplete="off" spellCheck={false} />
            </Field>
            <div className="py-2">
              <Segmented label={t("Kind")} value={form.draft.kind} onChange={(v) => setField("kind", v)} options={[{ value: "program", label: t("A program on this PC") }, { value: "address", label: t("An address (URL)") }]} />
            </div>
            {form.draft.kind === "program" ? (
              <>
                <Field label={t("Program")} hint={form.problems.command ? <span className="text-bad">{form.problems.command}</span> : t("What Strata starts, e.g. npx or uvx or the path of a program.")}>
                  <input className={`${inputCls} font-mono`} value={form.draft.command} aria-label={t("Program")} onChange={(e) => setField("command", e.target.value)} placeholder="npx" autoComplete="off" spellCheck={false} />
                </Field>
                <Field label={t("Arguments")} hint={form.problems.args ? <span className="text-bad">{form.problems.args}</span> : t("One argument per line.")}>
                  <textarea className={areaCls} rows={3} value={form.draft.args} aria-label={t("Arguments")} onChange={(e) => setField("args", e.target.value)} placeholder={"-y\n@modelcontextprotocol/server-filesystem\nC:/work"} spellCheck={false} />
                </Field>
                <Field label={t("Environment")} hint={form.problems.env ? <span className="text-bad">{form.problems.env}</span> : t("One per line: NAME=value. After saving the values are hidden; leave {mask} to keep one.", { mask: MASK })}>
                  <textarea className={areaCls} rows={2} value={form.draft.env} aria-label={t("Environment")} onChange={(e) => setField("env", e.target.value)} placeholder="TOKEN=…" spellCheck={false} />
                </Field>
                <Field label={t("Folder")} hint={t("Where it runs. Empty: where Strata runs.")}>
                  <input className={`${inputCls} font-mono`} value={form.draft.cwd} aria-label={t("Folder")} onChange={(e) => setField("cwd", e.target.value)} autoComplete="off" spellCheck={false} />
                </Field>
              </>
            ) : (
              <>
                <Field label={t("Address")} hint={form.problems.url ? <span className="text-bad">{form.problems.url}</span> : t("A server that speaks Streamable HTTP, often at /mcp.")}>
                  <input className={`${inputCls} font-mono`} value={form.draft.url} aria-label={t("Address")} onChange={(e) => setField("url", e.target.value)} placeholder="http://127.0.0.1:3000/mcp" autoComplete="off" spellCheck={false} />
                </Field>
                <Field label={t("Headers")} hint={form.problems.headers ? <span className="text-bad">{form.problems.headers}</span> : t("One per line: Name=value, e.g. Authorization=Bearer …. After saving the values are hidden; leave {mask} to keep one.", { mask: MASK })}>
                  <textarea className={areaCls} rows={2} value={form.draft.headers} aria-label={t("Headers")} onChange={(e) => setField("headers", e.target.value)} spellCheck={false} />
                </Field>
              </>
            )}
            {form.failure && <p role="alert" className="mt-1 text-[12px] text-bad [overflow-wrap:anywhere]">{form.failure}</p>}
            <div className="mt-2 flex gap-2">
              <Button type="submit" kind="primary" disabled={busy}>{t("Save")}</Button>
              <Button kind="quiet" disabled={busy} onClick={() => setForm(null)}>{t("Cancel")}</Button>
            </div>
          </form>
        ) : null}
      </Reveal>

      <Reveal>
        {paste ? (
          <div className="mt-3 rounded-md border border-line p-3">
            <Field label={t("Paste from Claude Desktop")} hint={t("The mcpServers block of claude_desktop_config.json, as it is.")}>
              <textarea className={areaCls} rows={7} value={paste.text} aria-label={t("The block to paste")} onChange={(e) => setPaste({ text: e.target.value, problem: null })} placeholder={'{\n  "mcpServers": {\n    "files": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:/work"] }\n  }\n}'} spellCheck={false} />
            </Field>
            {paste.problem && <p role="alert" className="text-[12px] text-bad [overflow-wrap:anywhere]">{paste.problem}</p>}
            <div className="mt-2 flex gap-2">
              <Button kind="primary" disabled={busy || !paste.text.trim()} onClick={() => void add()}>{t("Add")}</Button>
              <Button kind="quiet" disabled={busy} onClick={() => setPaste(null)}>{t("Cancel")}</Button>
            </div>
          </div>
        ) : null}
      </Reveal>

      {limits && (
        <div className="mt-5">
          <h3 className="text-[13px] font-medium">{t("Limits")}</h3>
          <div className="grid gap-x-4 sm:grid-cols-3">
            {(["timeout_s", "max_result_chars", "max_rounds"] as const).map((k) => (
              <Field key={k} label={t(LIMIT_NAME[k])} hint={t(LIMIT_HINT[k])}>
                <input className={`${inputCls} num`} inputMode="numeric" value={limits[k]} disabled={!can} aria-label={t(LIMIT_NAME[k])} onChange={(e) => setLimits({ ...limits, [k]: e.target.value })} />
              </Field>
            ))}
          </div>
          {can && <Button disabled={busy} onClick={() => void saveLimits()}>{t("Save the limits")}</Button>}
        </div>
      )}
    </section>
  )
}

const LIMIT_NAME: Record<keyof McpLimits, string> = {
  timeout_s: msg("Time for one tool call (seconds)"), max_result_chars: msg("Longest tool result (characters)"), max_rounds: msg("Tool rounds in one answer"),
}
const LIMIT_HINT: Record<keyof McpLimits, string> = {
  timeout_s: msg("From 1 to 600."), max_result_chars: msg("From 1,000 to 500,000. The model reads only this much."), max_rounds: msg("From 1 to 30. Then it must answer."),
}
