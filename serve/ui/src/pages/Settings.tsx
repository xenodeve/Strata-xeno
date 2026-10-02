import { useEffect, useState, useSyncExternalStore } from "react"
import { BookOpen01Icon, ComputerTerminal01Icon, Download04Icon, Key01Icon, PaintBoardIcon, PlugSocketIcon, ServerStack01Icon } from "@hugeicons/core-free-icons"
import { href } from "../lib/router"
import { msg, t, useLang } from "../lib/i18n"
import { store } from "../lib/store"
import { toast } from "../components/toast"
import { inputCls } from "../components/ui"
import { BranchedMenu, type BranchSection } from "../components/BranchedMenu"
import { StatusMarkSettings } from "../components/StatusMarks"
import { McpSettings } from "../components/McpSettings"
import { ImportSettings } from "../components/ImportSettings"
import { AgentControls } from "../components/AgentControls"
import { getAgent } from "../lib/api"
import { chat } from "../lib/chat"
import { NO_AGENT, type AgentInfo } from "../lib/agent"

// What can be set in the app, apart from what About tells. A menu of sections and topics; the address says which topic shows
// (#/settings/mcp-servers), so a link can point at one. The status marks and the API key are kept in this browser; the MCP
// servers and their limits are saved in the run config, from this PC.
const TOPICS = ["status-marks", "api-key", "coding-tools", "mcp-servers", "mcp-limits", "import-skills", "import-mcp"] as const
type Topic = (typeof TOPICS)[number]
const LABEL: Record<Topic, string> = { "status-marks": msg("Status marks"), "api-key": msg("API key"), "coding-tools": msg("Coding tools"), "mcp-servers": msg("Servers"), "mcp-limits": msg("Limits"), "import-skills": msg("Skills"), "import-mcp": msg("MCP servers") }
const SECTIONS: { value: string; label: string; topics: Topic[] }[] = [
  { value: "general", label: msg("General"), topics: ["status-marks", "api-key", "coding-tools"] },
  { value: "mcp", label: msg("MCP tools"), topics: ["mcp-servers", "mcp-limits"] },
  { value: "import", label: msg("Import"), topics: ["import-skills", "import-mcp"] },
]
const ICON = { "status-marks": PaintBoardIcon, "api-key": Key01Icon, "coding-tools": ComputerTerminal01Icon, "mcp-servers": ServerStack01Icon, "mcp-limits": PlugSocketIcon, "import-skills": BookOpen01Icon, "import-mcp": Download04Icon }

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

/** The coding tools' settings (the same controls as the panel in the prompt bar): on or off, the mode, the folder for chats that are in no project. */
function CodingTools() {
  const [info, setInfo] = useState<AgentInfo>(NO_AGENT)
  useEffect(() => { void getAgent().then((v) => v && setInfo(v)) }, [])
  useSyncExternalStore(chat.subscribe, chat.getVersion)
  const s = chat.settings
  return (
    <section>
      <h2 className="text-[15px] font-semibold">{t("Coding tools")}</h2>
      <p className="mt-1 mb-3 text-[13px] text-ink-2">{t("The chat can read, search and change files and run commands on this PC with Claude Code's own tools, and asks you before anything that is not plainly safe. Here are the defaults; a project's own folder is set from inside one of its chats.")}</p>
      <AgentControls
        info={info} on={s.agent !== false} mode={s.agentMode === "plan" || s.agentMode === "auto" ? s.agentMode : "ask"} folder={s.agentFolder?.trim() || null} folderOf={{ kind: "default" }} rules={0}
        onToggle={() => chat.setSettings({ ...s, agent: s.agent === false })} onMode={(m) => chat.setSettings({ ...s, agentMode: m })}
        onFolder={(path) => chat.setSettings({ ...s, agentFolder: path.trim() })} onForget={() => {}}
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
          <BranchedMenu startOpen sections={sections} active={active} label={t("Settings sections")} onSelect={(x) => { location.hash = href("settings", x) }} />
        </aside>
        <div key={active} className="panel-in min-w-0 max-w-[65ch] flex-1">
          {active === "status-marks" ? <Marks /> : active === "api-key" ? <ApiKey /> : active === "coding-tools" ? <CodingTools /> : active === "import-skills" ? <ImportSettings part="skills" /> : active === "import-mcp" ? <ImportSettings part="mcp" /> : <McpSettings part={active === "mcp-servers" ? "servers" : "limits"} />}
        </div>
      </div>
    </div>
  )
}
