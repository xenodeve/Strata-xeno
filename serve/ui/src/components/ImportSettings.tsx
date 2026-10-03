import { useCallback, useEffect, useState } from "react"
import { HugeiconsIcon } from "@hugeicons/react"
import { ArrowDown01Icon, ArrowRight01Icon } from "@hugeicons/core-free-icons"
import { getImport, postImport } from "../lib/api"
import { fmt } from "../lib/format"
import { harnessOn, filterItems, itemOn, setHarness, setItem, setMaster, skillSummary, type Candidate, type ImportView, type SkillSettings } from "../lib/importer"
import { sizeText, withSwitch } from "../lib/memory"
import { msg, t } from "../lib/i18n"
import { cn } from "../lib/cn"
import { Collapse } from "./motion"
import { toast } from "./toast"
import { Button, MiniSwitch, inputCls } from "./ui"
import { Windowed } from "./Windowed"

// Skills and MCP servers of the other coding apps on this PC (issue #94). Skills are imported automatically and are on until switched off
// (the whole import, an app, a skill); MCP servers are only listed, by app, and imported by a click: nothing is started until then.
// The server reads the other apps' files and decides; secrets never come here. Without the right to change things it is a list.
const REASON: Record<string, string> = {
  "it speaks SSE only; Strata speaks Streamable HTTP": msg("It speaks SSE only; Strata speaks Streamable HTTP."),
  "no command or address to run": msg("It has no command or address to run."),
  "not a server entry": msg("It is not a server entry."),
}

function useImport() {
  const [view, setView] = useState<ImportView | null>(null)
  const [busy, setBusy] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)
  useEffect(() => { void getImport().then((v) => v && setView(v)) }, [])
  /** Send a change; the answer is the new state. On a refusal the page keeps what it had and says why. */
  const send = useCallback(async (body: unknown): Promise<boolean> => {
    setBusy(true)
    setNotice(null)
    const r = await postImport(body)
    setBusy(false)
    if ("error" in r) { setNotice(r.error); return false }
    setView(r.view)
    return true
  }, [])
  const rescan = useCallback(async () => {
    setBusy(true)
    setNotice(null)
    const v = await getImport(true)
    setBusy(false)
    if (v) setView(v)
    else setNotice(t("The folders could not be read again."))
  }, [])
  return { view, setView, busy, notice, send, rescan }
}

function Note({ view }: { view: ImportView }) {
  if (view.editable) return null
  return (
    <p role="note" className="mt-2 rounded-md bg-fill px-3 py-2 text-[12px] text-ink-2">
      {view.config_file === null
        ? t("This server was started without a run config file, so there is nowhere to save the import settings. Start it with --config.")
        : t("These can be changed only from this PC itself, or when Strata has an API key and it is entered in Settings, under API key.")}
    </p>
  )
}

export function ImportSettings({ part }: { part: "skills" | "mcp" | "memory" }) {
  const { view, setView, busy, notice, send, rescan } = useImport()
  const [filter, setFilter] = useState("")
  const [open, setOpen] = useState<string[]>([])               // the apps whose skills are shown

  if (!view) return null
  if (!view.available) return <section><h2 className="text-[15px] font-semibold">{part === "skills" ? t("Skills from other apps") : part === "memory" ? t("Memory from other apps") : t("MCP servers from other apps")}</h2><p className="mt-1 text-[13px] text-ink-2">{t("Importing from other apps is not available on this server.")}</p></section>
  const can = !!view.editable
  const labels = Object.fromEntries((view.harnesses ?? []).map((h) => [h.id, h.label]))
  const rescanButton = <Button disabled={busy} onClick={() => void rescan()}>{busy ? t("Rescanning…") : t("Rescan")}</Button>

  if (part === "memory") {
    const mem = view.memory ?? { settings: { on: [] }, sources: [] }
    const flip = async (id: string, value: boolean) => {
      const next = { on: withSwitch(mem.settings.on, id, value) }
      setView({ ...view, memory: { settings: next, sources: mem.sources.map((x) => (x.id === id ? { ...x, on: value } : x)) } })      // at once; the server's answer replaces it, or it is put back
      if (!(await send({ memory: next }))) setView(view)
    }
    return (
      <section>
        <h2 className="text-[15px] font-semibold">{t("Memory from other apps")}</h2>
        <p className="mt-1 text-[13px] text-ink-2">{t("Your other coding apps keep notes about how you like to work, and about your projects. The chat can read them, as those apps do. They are your own words, but written for another app, so each is off until you switch it on. They never change what the chat may do: asking you first still applies.")}</p>
        <Note view={view} />
        {mem.sources.length === 0 && <p className="py-3 text-[13px] text-ink-2">{t("No notes were found in your other coding apps.")}</p>}
        {mem.sources.map((x) => (
          <div key={x.id} data-memory-source={x.id} className="flex items-start gap-3 border-t border-line py-2.5">
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-baseline gap-x-2"><span className="text-[13px] font-medium">{x.label}</span><span className="text-[12px] text-ink-2">{x.kind === "memory" ? t("what it remembers about each project") : t("its instructions for you")}</span></div>
              <div className="font-mono text-[12px] text-ink-3 [overflow-wrap:anywhere]">{x.shown.join(", ")}</div>
              {x.bytes != null && <div className="num text-[12px] text-ink-3">{sizeText(x.bytes)}</div>}
            </div>
            <MiniSwitch on={x.on} disabled={!can || busy} label={t("Read {name}", { name: `${x.label}: ${x.kind === "memory" ? t("memory") : t("instructions")}` })} onClick={() => void flip(x.id, !x.on)} />
          </div>
        ))}
        <div className="flex justify-end pt-2">{rescanButton}</div>
        {notice && <p role="alert" className="mt-2 text-[12px] text-bad [overflow-wrap:anywhere]">{notice}</p>}
      </section>
    )
  }

  if (part === "skills") {
    const sk = view.skills!
    const s = sk.settings
    const sum = skillSummary(sk.items)
    const change = async (next: SkillSettings) => {
      setView({ ...view, skills: { ...sk, settings: next } })        // at once; the server's answer (the counts) replaces it, or it is put back
      if (!(await send({ skills: next }))) setView(view)
    }
    const matching = filterItems(sk.items, filter, labels)
    return (
      <section>
        <h2 className="text-[15px] font-semibold">{t("Skills from other apps")}</h2>
        <p className="mt-1 text-[13px] text-ink-2">{t("A skill is a set of instructions the model can load when a task fits. The ones found in your other coding apps are imported automatically and stay on until you switch them off here.")}</p>
        <Note view={view} />
        <div className="mt-3 flex items-center justify-between gap-4 border-t border-line py-2.5">
          <div className="min-w-0">
            <div className="text-[13px] font-medium">{t("Import skills")}</div>
            <div className="num text-[12px] text-ink-2">{t("{used} skills in use ({total} found in {apps} apps)", { used: fmt(sum.used), total: fmt(sum.total), apps: sum.apps })}</div>
          </div>
          <MiniSwitch on={s.enabled} disabled={!can || busy} label={t("Import skills")} onClick={() => void change(setMaster(s, !s.enabled))} />
        </div>
        <div className="flex items-center gap-2 pb-2">
          <input className={cn(inputCls, "flex-1")} value={filter} placeholder={t("Filter by name or what it does")} aria-label={t("Filter by name or what it does")} onChange={(e) => setFilter(e.target.value)} />
          {rescanButton}
        </div>
        {filter.trim() && matching.length === 0 && <p className="py-2 text-[13px] text-ink-2">{t("No skill matches.")}</p>}
        {(view.harnesses ?? []).filter((h) => h.skills > 0).map((h) => {                  // an app that has no skills (it may have MCP servers) is not in this list
          const mine = matching.filter((i) => i.harness === h.id)
          if (filter.trim() && mine.length === 0) return null
          const expanded = open.includes(h.id) || !!filter.trim()
          const on = harnessOn(s, h.id)
          return (
            <div key={h.id} data-harness={h.id} className="border-t border-line py-2.5">
              <div className="flex items-center gap-3">
                <button type="button" aria-expanded={expanded} onClick={() => setOpen(open.includes(h.id) ? open.filter((x) => x !== h.id) : [...open, h.id])} className="flex min-w-0 flex-1 items-baseline gap-2 text-left">
                  <HugeiconsIcon icon={expanded ? ArrowDown01Icon : ArrowRight01Icon} size={12} aria-hidden className="shrink-0 self-center text-ink-3" />
                  <span className="text-[13px] font-medium">{h.label}</span>
                  <span className="num text-[12px] text-ink-2">{h.skills ? t("{used} of {n} in use", { used: fmt(h.skills_used), n: fmt(h.skills) }) : t("no skills")}</span>
                </button>
                <MiniSwitch on={on} disabled={!can || busy || !s.enabled || h.skills === 0} label={t("Import from {name}", { name: h.label })} onClick={() => void change(setHarness(s, h.id, !on))} />
              </div>
              {h.files.length > 0 && <div className="mt-0.5 pl-5 text-[12px] text-ink-3 [overflow-wrap:anywhere]">{t("Read from {files}", { files: h.files.join(", ") })}</div>}
              {h.errors.map((e) => <div key={e} className="mt-0.5 pl-5 text-[12px] text-bad [overflow-wrap:anywhere]">{e}</div>)}
              <Collapse open={expanded && h.skills > 0}>
                <Windowed label={t("Skills of {name}", { name: h.label })}>
                <ul className={cn("m-0 mt-1 list-none p-0 pl-5", (!on || !s.enabled) && "opacity-60")}>
                  {mine.map((i) => {
                    const own = itemOn(s, i.harness, i.name)
                    return (
                      <li key={i.id} data-skill={i.id} className={cn("flex items-start gap-3 py-1.5", !i.used && "opacity-60")}>
                        <div className="min-w-0 flex-1">
                          <div className="flex flex-wrap items-baseline gap-x-2 text-[13px]">
                            <span className="font-medium [overflow-wrap:anywhere]">{i.name}</span>
                            {i.origin.startsWith("plugin:") && <span className="rounded-full bg-fill px-1.5 text-[11px] text-ink-2">{i.origin.replace("plugin:", "")}</span>}
                            {i.same_as && <span className="rounded-full bg-fill px-1.5 text-[11px] text-ink-2">{t("Same as {name}", { name: labels[i.same_as.split(":")[0]] ?? i.same_as })}</span>}
                          </div>
                          {i.description && <div className="line-clamp-2 text-[12px] text-ink-2">{i.description}</div>}
                        </div>
                        <MiniSwitch on={own} disabled={!can || busy || !on || !s.enabled} label={t("Use {name}", { name: i.name })} onClick={() => void change(setItem(s, i.harness, i.name, !own))} />
                      </li>
                    )
                  })}
                </ul>
                </Windowed>
              </Collapse>
            </div>
          )
        })}
        {notice && <p role="alert" className="mt-2 text-[12px] text-bad [overflow-wrap:anywhere]">{notice}</p>}
      </section>
    )
  }

  const importOne = async (c: Candidate) => {
    if (await send({ import_mcp: { harness: c.harness, name: c.name } })) toast("success", t("Imported"), t("{name} is now in MCP tools > Servers.", { name: c.name }))
  }
  const line = (c: Candidate) => [c.command, ...(c.args ?? [])].filter(Boolean).join(" ") || c.url || ""
  return (
    <section>
      <h2 className="text-[15px] font-semibold">{t("MCP servers from other apps")}</h2>
      <p className="mt-1 text-[13px] text-ink-2">{t("Nothing here runs until you import it. Importing copies a server into Strata's own list (MCP tools > Servers), where you can edit it, turn it off or delete it.")}</p>
      <Note view={view} />
      <div className="mt-3 flex justify-end">{rescanButton}</div>
      {(view.mcp?.harnesses ?? []).length === 0 && <p className="py-3 text-[13px] text-ink-2">{t("No MCP server was found in your other coding apps.")}</p>}
      {(view.mcp?.harnesses ?? []).map((h) => {
        const info = (view.harnesses ?? []).find((x) => x.id === h.id)
        return (
          <div key={h.id} data-harness={h.id} className="border-t border-line py-2.5">
            <div className="flex items-baseline gap-2"><span className="text-[13px] font-medium">{h.label}</span><span className="num text-[12px] text-ink-2">{t("{n} servers", { n: fmt(h.servers.length) })}</span></div>
            {info && info.files.length > 0 && <div className="text-[12px] text-ink-3 [overflow-wrap:anywhere]">{t("Read from {files}", { files: info.files.join(", ") })}</div>}
            <ul className="m-0 mt-1 list-none p-0">
              {h.servers.map((c) => (
                <li key={c.id} data-candidate={c.id} data-state={c.state} className="flex items-start gap-3 py-1.5">
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-baseline gap-x-2 text-[13px]">
                      <span className="font-medium [overflow-wrap:anywhere]">{c.name}</span>
                      {c.kind && <span className="text-[12px] text-ink-2">{c.kind === "address" ? t("Address") : t("Program")}</span>}
                      {c.disabled_in_source && c.state === "available" && <span className="rounded-full bg-fill px-1.5 text-[11px] text-ink-2">{t("Off in {name}", { name: h.label })}</span>}
                    </div>
                    {can && c.state !== "unsupported" && line(c) && <div className="font-mono text-[12px] text-ink-2 [overflow-wrap:anywhere]">{line(c)}</div>}
                    {c.state === "unsupported" && <div className="text-[12px] text-ink-2">{c.reason ? (REASON[c.reason] ? t(REASON[c.reason]) : c.reason) : t("Not supported")}</div>}
                    {c.note && <div className="text-[12px] text-ink-3">{c.note}</div>}
                  </div>
                  {c.state === "available" && <Button disabled={!can || busy} label={`${t("Import")} ${c.name}`} onClick={() => void importOne(c)}>{t("Import")}</Button>}
                  {c.state === "already" && <span className="shrink-0 text-[12px] text-ink-2">{t("Already in Strata as {name}", { name: c.already_as ?? c.name })}</span>}
                  {c.state === "unsupported" && <span className="shrink-0 text-[12px] text-ink-3">{t("Not supported")}</span>}
                </li>
              ))}
            </ul>
          </div>
        )
      })}
      {notice && <p role="alert" className="mt-2 text-[12px] text-bad [overflow-wrap:anywhere]">{notice}</p>}
    </section>
  )
}
