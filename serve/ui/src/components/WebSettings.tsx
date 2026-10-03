import { useEffect, useState } from "react"
import { getWeb, postWeb, type WebView } from "../lib/api"
import { t } from "../lib/i18n"
import { cn } from "../lib/cn"
import { Button, Segmented, Switch, inputCls } from "./ui"

// Web access for the chat's coding tools (issue #99, serve/web.py): WebFetch and WebSearch. They are OFF until the user switches them on here, because they are the one thing in Strata that sends
// something off this PC; the model is not even offered them while off, and every call then asks first, with the address or the search words. Kept in the server's run config, from this PC only.

export function WebSettings() {
  const [view, setView] = useState<WebView | null | undefined>(undefined)
  const [busy, setBusy] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)
  const [url, setUrl] = useState("")
  const [key, setKey] = useState("")
  useEffect(() => { void getWeb().then((v) => { setView(v); if (v) setUrl(v.searxng_url) }) }, [])
  const send = async (body: Record<string, unknown>): Promise<boolean> => {
    setBusy(true)
    setNotice(null)
    const r = await postWeb(body)
    setBusy(false)
    if ("error" in r) { setNotice(r.error); return false }
    setView(r.view)
    setUrl(r.view.searxng_url)
    return true
  }
  const can = !!view && view.editable && !busy
  return (
    <section data-web>
      <h2 className="text-[15px] font-semibold">{t("Web access")}</h2>
      <p className="mt-1 text-[13px] text-ink-2">{t("Lets the model fetch a web page and search the web (the tools WebFetch and WebSearch). It is off by default, because it is the one thing here that sends something off this PC: the address of a page, or your search words, goes to the internet.")}</p>
      <p className="mt-2 text-[13px] text-ink-2">{t("Even when it is on, every fetch and every search asks you first and shows what would be sent; \"Always allow this site\" is a rule you can add. Addresses on this PC or on your network are refused, and what a page says is treated as text, never as an order.")}</p>
      {view === undefined ? null : view === null ? (
        <p role="note" className="mt-3 rounded-md bg-fill px-3 py-2 text-[12px] text-ink-2">{t("Web access is shown only on this PC itself, or when Strata has an API key and it is entered in Settings, under API key.")}</p>
      ) : (
        <>
          {!view.available && <p role="note" className="mt-3 rounded-md bg-fill px-3 py-2 text-[12px] text-ink-2">{t("This server has no coding tools, so there is nothing to switch on.")}</p>}
          {!view.editable && <p role="note" className="mt-3 rounded-md bg-fill px-3 py-2 text-[12px] text-ink-2">{t("This server was started without a run config file, so there is nowhere to save these settings. Start it with --config.")}</p>}
          <div className="mt-2">
            <Switch checked={view.on} onChange={(v) => void (can && send({ on: v }))} label={t("Let the model use the web")} hint={view.on ? t("On: the model can ask to fetch pages and search. You are asked every time.") : t("Off: nothing leaves this PC.")} />
          </div>
          <div className={cn("mt-3 space-y-3", !view.on && "opacity-60")} data-web-provider>
            <div>
              <div className="mb-1.5 text-[13px] font-medium">{t("Who sees your searches")}</div>
              <Segmented label={t("Search provider")} value={view.provider} onChange={(v) => void (can && send({ provider: v }))} options={[{ value: "searxng", label: "SearXNG" }, { value: "brave", label: "Brave Search" }]} />
            </div>
            {view.provider === "searxng" ? (
              <form className="space-y-1.5" onSubmit={(e) => { e.preventDefault(); if (can) void send({ searxng_url: url.trim() }) }}>
                <label className="block text-[12.5px] text-ink-2" htmlFor="web-searxng">{t("The address of your SearXNG (it can be on your own network; it must offer the JSON format)")}</label>
                <div className="flex gap-2">
                  <input id="web-searxng" className={cn(inputCls, "min-w-0 flex-1 font-mono")} value={url} placeholder="http://localhost:8080" autoComplete="off" spellCheck={false} disabled={!can} onChange={(e) => setUrl(e.target.value)} />
                  <Button type="submit" disabled={!can || url.trim() === view.searxng_url}>{t("Save")}</Button>
                </div>
              </form>
            ) : (
              <form className="space-y-1.5" onSubmit={(e) => { e.preventDefault(); if (can && key.trim()) void send({ brave_key: key.trim() }).then((ok) => ok && setKey("")) }}>
                <label className="block text-[12.5px] text-ink-2" htmlFor="web-brave">{view.brave_key_set ? t("A Brave Search key is saved on the server. Enter another to replace it.") : t("Your Brave Search API key (kept on the server, never shown again)")}</label>
                <div className="flex gap-2">
                  <input id="web-brave" type="password" className={cn(inputCls, "min-w-0 flex-1 font-mono")} value={key} autoComplete="off" spellCheck={false} disabled={!can} onChange={(e) => setKey(e.target.value)} />
                  <Button type="submit" disabled={!can || !key.trim()}>{t("Save")}</Button>
                  {view.brave_key_set && <Button disabled={!can} onClick={() => void send({ brave_key: "" })}>{t("Remove the key")}</Button>}
                </div>
              </form>
            )}
            <p className="text-[12px] text-ink-3">{t("Searching needs one of these: DuckDuckGo is not offered because it turns away programs. Fetching a page needs neither.")}</p>
          </div>
        </>
      )}
      {notice && <p role="alert" className="mt-2 text-[12px] text-bad">{notice}</p>}
    </section>
  )
}
