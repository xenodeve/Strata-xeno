import { useState } from "react"
import { Key01Icon, PaintBoardIcon, PlugSocketIcon, ServerStack01Icon } from "@hugeicons/core-free-icons"
import { href } from "../lib/router"
import { msg, t, useLang } from "../lib/i18n"
import { store } from "../lib/store"
import { toast } from "../components/toast"
import { inputCls } from "../components/ui"
import { BranchedMenu, type BranchSection } from "../components/BranchedMenu"
import { StatusMarkSettings } from "../components/StatusMarks"
import { McpSettings } from "../components/McpSettings"

// What can be set in the app, apart from what About tells. A menu of sections and topics; the address says which topic shows
// (#/settings/mcp-servers), so a link can point at one. The status marks and the API key are kept in this browser; the MCP
// servers and their limits are saved in the run config, from this PC.
const TOPICS = ["status-marks", "api-key", "mcp-servers", "mcp-limits"] as const
type Topic = (typeof TOPICS)[number]
const LABEL: Record<Topic, string> = { "status-marks": msg("Status marks"), "api-key": msg("API key"), "mcp-servers": msg("Servers"), "mcp-limits": msg("Limits") }
const SECTIONS: { value: string; label: string; topics: Topic[] }[] = [
  { value: "general", label: msg("General"), topics: ["status-marks", "api-key"] },
  { value: "mcp", label: msg("MCP tools"), topics: ["mcp-servers", "mcp-limits"] },
]
const ICON = { "status-marks": PaintBoardIcon, "api-key": Key01Icon, "mcp-servers": ServerStack01Icon, "mcp-limits": PlugSocketIcon }

function ApiKey() {
  const [key, setKey] = useState(() => store.get("apikey", ""))
  return (
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
  )
}

function Marks() {
  return (
    <section>
      <h2 className="text-[15px] font-semibold">{t("Status marks")}</h2>
      <p className="mt-1 mb-3 text-[13px] text-ink-2">{t("What shows that something is happening: thinking, answering, reading. Pick one; the same list opens from the button at the top.")}</p>
      <StatusMarkSettings />
    </section>
  )
}

export function Settings({ topic }: { topic?: string }) {
  useLang()                                                  // the labels are translated when the menu is built
  const active: Topic = (TOPICS as readonly string[]).includes(topic ?? "") ? (topic as Topic) : "status-marks"
  const sections: BranchSection[] = SECTIONS.map((s) => ({ value: s.value, label: t(s.label), topics: s.topics.map((x) => ({ value: x, label: t(LABEL[x]), icon: ICON[x] })) }))
  return (
    <div className="space-y-6">
      <section>
        <h1 className="page-title">{t("Settings")}</h1>
        <p className="page-sub">{t("How the app looks and what it connects to. The tools the model may use are set here too.")}</p>
      </section>
      <div className="flex gap-10 max-sm:flex-col max-sm:gap-4">
        <aside className="shrink-0 sm:w-[200px]">
          <BranchedMenu sections={sections} active={active} label={t("Settings sections")} onSelect={(x) => { location.hash = href("settings", x) }} />
        </aside>
        <div key={active} className="panel-in min-w-0 max-w-[65ch] flex-1">
          {active === "status-marks" ? <Marks /> : active === "api-key" ? <ApiKey /> : <McpSettings part={active === "mcp-servers" ? "servers" : "limits"} />}
        </div>
      </div>
    </div>
  )
}
