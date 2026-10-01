import { useState } from "react"
import { t } from "../lib/i18n"
import { store } from "../lib/store"
import { toast } from "../components/toast"
import { inputCls } from "../components/ui"
import { StatusMarkSettings } from "../components/StatusMarks"
import { McpSettings } from "../components/McpSettings"

// What can be set in the app, apart from what About tells: the status marks and the API key (kept in this browser), and the MCP
// servers (saved in the run config, from this PC).
export function Settings() {
  const [key, setKey] = useState(() => store.get("apikey", ""))
  return (
    <div className="max-w-[65ch] space-y-8">
      <section>
        <h1 className="page-title">{t("Settings")}</h1>
        <p className="page-sub">{t("How the app looks and what it connects to. The tools the model may use are set here too.")}</p>
      </section>

      <section>
        <h2 className="text-[15px] font-semibold">{t("Status marks")}</h2>
        <p className="mt-1 mb-3 text-[13px] text-ink-2">{t("What shows that something is happening: thinking, answering, reading. Pick one; the same list opens from the button at the top.")}</p>
        <StatusMarkSettings />
      </section>

      <section>
        <h2 className="text-[15px] font-semibold">{t("API key")}</h2>
        <p className="mt-1 text-[13px] text-ink-2">{t("Only needed when the server was started with one. Kept in this browser.")}</p>
        <input
          className={`${inputCls} mt-2`}
          type="password"
          autoComplete="off"
          value={key}
          placeholder={t("No key")}
          aria-label={t("API key")}
          onChange={(e) => setKey(e.target.value)}
          onBlur={() => { store.set("apikey", key.trim()); toast("success", t("API key saved"), t("Kept in this browser only.")) }}
        />
      </section>

      <McpSettings />
    </div>
  )
}
