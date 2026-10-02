import { useEffect, useState } from "react"
import { getHelpers, postHelpers, type HelpersView } from "../lib/api"
import { t } from "../lib/i18n"
import { Switch } from "./ui"

// Sub-agents for the coding tools (issue #99, serve/subagent.py): the Task tool lets the model hand a side task to a helper that works on its own and returns a short report. OFF by default: most
// people run a local model with one engine slot, and a helper they did not ask for would stall the chat. Kept in the server's run config, from this PC only.

export function HelpersSettings() {
  const [view, setView] = useState<HelpersView | null | undefined>(undefined)
  const [busy, setBusy] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)
  useEffect(() => { void getHelpers().then(setView) }, [])
  const flip = async (on: boolean) => {
    setBusy(true)
    setNotice(null)
    const r = await postHelpers(on)
    setBusy(false)
    if ("error" in r) setNotice(r.error)
    else setView(r.view)
  }
  return (
    <section data-helpers>
      <h2 className="text-[15px] font-semibold">{t("Sub-agents")}</h2>
      <p className="mt-1 text-[13px] text-ink-2">{t("Lets the model hand a side task, such as looking through the code, to a helper that works on it with its own conversation and answers with a short report. That keeps long searches out of the chat.")}</p>
      <p className="mt-2 text-[13px] text-ink-2">{t("It is off by default: a helper uses the same model, and a model on one engine slot works on one thing at a time, so the main answer waits and the conversation may be read again afterwards. A helper that only reads and searches is the usual kind; one that can change files asks you like any other, and neither can start a helper or ask you questions.")}</p>
      {view === undefined ? null : view === null ? (
        <p role="note" className="mt-3 rounded-md bg-fill px-3 py-2 text-[12px] text-ink-2">{t("Sub-agents are shown only on this PC itself, or when Strata has an API key and it is entered in Settings, under API key.")}</p>
      ) : (
        <>
          {!view.available && <p role="note" className="mt-3 rounded-md bg-fill px-3 py-2 text-[12px] text-ink-2">{t("This server has no coding tools, so there is nothing to switch on.")}</p>}
          {!view.editable && <p role="note" className="mt-3 rounded-md bg-fill px-3 py-2 text-[12px] text-ink-2">{t("This server was started without a run config file, so there is nowhere to save these settings. Start it with --config.")}</p>}
          <div className="mt-2">
            <Switch checked={view.on} onChange={(v) => void (view.editable && !busy && flip(v))} label={t("Let the model use helpers")} hint={view.on ? t("On: the model can start a helper, one at a time.") : t("Off: the model does everything itself.")} />
          </div>
        </>
      )}
      {notice && <p role="alert" className="mt-2 text-[12px] text-bad">{notice}</p>}
    </section>
  )
}
