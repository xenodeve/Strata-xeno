import { dismissFreshKey, useFreshKey } from "../lib/keys"
import { t } from "../lib/i18n"
import { toast } from "./toast"
import { Button } from "./ui"

// A new key's text, this once (#310). Drawn by App above the waiting screen, so it stays until the user is done with it even when this
// device has to wait for approval the moment the first key turns team mode on.
export function FreshKeyCard() {
  const k = useFreshKey()
  if (!k) return null
  const copy = async () => {
    try { await navigator.clipboard.writeText(k.secret); toast("success", t("Key copied")) } catch { toast("error", t("The browser did not copy it: select the key and copy it by hand")) }
  }
  return (
    <div role="alert" data-new-key className="toast-in fixed inset-x-3 top-3 z-40 mx-auto max-w-[540px] rounded-lg border border-accent bg-surface px-4 py-3 shadow-[0_14px_40px_-12px_rgb(0_0_0/0.35)]">
      <div className="text-[13px] font-medium">{t("The key for {name}", { name: k.name })}</div>
      <div className="mt-2 flex items-center gap-2">
        <code className="min-w-0 flex-1 select-all rounded-sm bg-fill px-2.5 py-1.5 font-mono text-[12.5px] [overflow-wrap:anywhere]">{k.secret}</code>
        <Button onClick={() => void copy()}>{t("Copy")}</Button>
      </div>
      <p className="mt-2 text-[12px] text-ink-2">{t("Shown once: copy it now. Strata keeps only a fingerprint of it, so it cannot show it again.")}</p>
      {k.browser === "stored" && <p className="mt-1 text-[12px] text-ink-2">{t("This browser now uses this key (Settings, API key), so this page keeps working.")}</p>}
      {k.browser === "uncertain" && <p className="mt-1 text-[12px] text-bad">{t("This browser kept the API key it had, and the server could not be asked whether that key is still valid. If the page now says it needs an API key, copy this key and enter it under Settings, API key.")}</p>}
      {k.browser === "failed" && <p className="mt-1 text-[12px] text-bad">{t("This browser could not keep the key, and from now on this page needs it: copy it, then enter it under Settings, API key.")}</p>}
      <div className="mt-2"><Button kind="quiet" onClick={dismissFreshKey}>{t("I have copied it")}</Button></div>
    </div>
  )
}
