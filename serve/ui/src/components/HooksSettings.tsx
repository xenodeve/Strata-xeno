import { useEffect, useState } from "react"
import { getHooks, postHooksOff } from "../lib/api"
import { eventText, type HooksView } from "../lib/hooks"
import { t } from "../lib/i18n"
import { MiniSwitch } from "./ui"

// The user's hooks (issue #99, serve/hooks.py): commands of their own that run before and after a tool call, when a prompt is sent and when the model has finished. They are written in the
// server's run config, by hand, from this PC - never here, so no page can make Strata run a command. This page lists them and switches each one off or on. None are defined by default.

const EXAMPLE = `"hooks": [
  { "event": "before_tool", "matcher": "Bash", "command": "./check-command.sh", "timeout": 10 },
  { "event": "after_tool", "matcher": "Edit|Write", "command": "npm run lint --silent" }
]`

export function HooksSettings() {
  const [view, setView] = useState<HooksView | null | undefined>(undefined)
  const [busy, setBusy] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)
  useEffect(() => { void getHooks().then(setView) }, [])
  const flip = async (id: string, on: boolean) => {
    if (!view) return
    const off = view.hooks.filter((h) => (h.id === id ? !on : !h.on)).map((h) => h.id)
    setBusy(true)
    setNotice(null)
    const r = await postHooksOff(off)
    setBusy(false)
    if ("error" in r) setNotice(r.error)
    else setView(r.view)
  }
  return (
    <section data-hooks>
      <h2 className="text-[15px] font-semibold">{t("Hooks")}</h2>
      <p className="mt-1 text-[13px] text-ink-2">{t("Commands of your own that run around the chat's tool calls: before a call (exit code 2 stops it, and the model is told why), after it (what they print goes to the model, for example a failed lint), when you send a prompt, and when the model has finished.")}</p>
      <p className="mt-2 text-[13px] text-ink-2">{t("They are written in the server's run config, by hand, so a web page cannot make Strata run a command; none are set by default. A hook cannot allow what a rule or the safe defaults refuse, and one that fails or takes too long is shown in the chat and does not stop the answer.")}</p>
      {view === undefined ? null : view === null ? (
        <p role="note" className="mt-3 rounded-md bg-fill px-3 py-2 text-[12px] text-ink-2">{t("The hooks are shown only on this PC itself, or when Strata has an API key and it is entered in Settings, under API key.")}</p>
      ) : (
        <>
          {!view.shell && <p role="note" className="mt-3 rounded-md bg-fill px-3 py-2 text-[12px] text-ink-2">{t("This PC has no shell for the coding tools, so no hook can run.")}</p>}
          {!view.editable && <p role="note" className="mt-3 rounded-md bg-fill px-3 py-2 text-[12px] text-ink-2">{t("This server was started without a run config file, so there is nowhere to save the switches. Start it with --config.")}</p>}
          {view.hooks.length === 0 ? (
            <div className="mt-3">
              <p className="text-[13px] text-ink-3">{t("No hooks are set.")}</p>
              <p className="mt-2 text-[12.5px] text-ink-2">{t("To add one, write it in the run config ({file}):", { file: view.config_file ?? "--config" })}</p>
              <pre className="mt-1.5 overflow-x-auto rounded-md bg-fill px-3 py-2 font-mono text-[12px]">{EXAMPLE}</pre>
            </div>
          ) : (
            <ul className="mt-3 space-y-1.5" data-hook-list>
              {view.hooks.map((h) => (
                <li key={h.id} data-hook-id={h.id} data-hook-on={h.on} className="flex items-start gap-3 rounded-md border border-line px-3 py-2">
                  <div className="min-w-0 flex-1">
                    <div className="text-[13px] font-medium">{eventText(h.event)}{h.matcher && <span className="font-normal text-ink-3"> · {t("tools")} <span className="font-mono">{h.matcher}</span></span>}</div>
                    <div className="mt-0.5 font-mono text-[12px] text-ink-2 [overflow-wrap:anywhere]">{h.command}</div>
                    <div className="mt-0.5 text-[11.5px] text-ink-3">{t("Time limit: {n} s", { n: h.timeout })}</div>
                  </div>
                  <MiniSwitch on={h.on} disabled={busy || !view.editable} label={t("Run the hook {command}", { command: h.command })} onClick={() => void flip(h.id, !h.on)} />
                </li>
              ))}
            </ul>
          )}
          {view.problems.length > 0 && (
            <div role="alert" className="mt-3 text-[12.5px] text-bad" data-hook-problems>
              <div className="font-medium">{t("Left out, because of what is written in the run config:")}</div>
              <ul className="mt-1 list-disc pl-5">{view.problems.map((p) => <li key={p}>{p}</li>)}</ul>
            </div>
          )}
        </>
      )}
      {notice && <p role="alert" className="mt-2 text-[12px] text-bad">{notice}</p>}
    </section>
  )
}
