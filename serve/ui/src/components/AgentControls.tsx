import { useEffect, useState } from "react"
import type { AgentInfo, AgentMode } from "../lib/agent"
import { msg, t } from "../lib/i18n"
import { Button, MiniSwitch, Segmented, inputCls } from "./ui"

// The settings of the chat's coding tools (issue #96): on or off, the mode, the folder they work in, and the rules the user allowed for the
// chat. One component for the panel in the prompt bar and for the Settings page. The server decides what may run; these are the user's choices.

const MODE_LABEL: Record<AgentMode, string> = { ask: msg("Ask"), plan: msg("Plan"), auto: msg("Auto") }
const MODE_TEXT: Record<AgentMode, string> = {
  ask: msg("Files inside the folder are free to read and change. Everything else asks you first."),
  plan: msg("Nothing is changed. The model reads, then sends a plan for you to approve."),
  auto: msg("A second check decides what would ask you. What is risky is blocked or asked; dangerous commands and secrets always ask you."),
}
export const modeLabel = (m: AgentMode) => t(MODE_LABEL[m])

export interface AgentControlsProps {
  info: AgentInfo
  on: boolean
  mode: AgentMode
  folder: string | null
  folderOf: { kind: "project"; name: string } | { kind: "default" }
  rules: number
  onToggle: () => void
  onMode: (m: AgentMode) => void
  onFolder: (path: string) => void
  onForget: () => void
}

export function AgentControls(p: AgentControlsProps) {
  const [draft, setDraft] = useState(p.folder ?? "")
  useEffect(() => setDraft(p.folder ?? ""), [p.folder])
  if (!p.info.available) return <p className="text-[13px] text-ink-2">{t("The coding tools are switched off on this server (\"agent\": false in the run config).")}</p>
  if (!p.info.allowed) return <p role="note" className="text-[13px] text-ink-2">{t("The coding tools work only from the PC that runs Strata (or with the API key), because they change files and run commands there.")}</p>
  const changed = draft.trim() !== (p.folder ?? "")
  return (
    <div className="space-y-3" data-agent-controls>
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0">
          <div className="text-[13px] font-medium">{t("Coding tools")}</div>
          <div className="text-[12px] text-ink-2">{t("Read, search and change files, and run commands, like Claude Code.")}</div>
        </div>
        <MiniSwitch on={p.on} label={t("Coding tools")} onClick={p.onToggle} />
      </div>
      <div className={p.on ? "" : "pointer-events-none opacity-50"} aria-disabled={!p.on}>
        <Segmented label={t("Mode")} value={p.mode} onChange={p.onMode} options={(["ask", "plan", "auto"] as AgentMode[]).map((m) => ({ value: m, label: modeLabel(m) }))} />
        <p className="mt-1.5 text-[12px] text-ink-2" data-agent-mode-text>{t(MODE_TEXT[p.mode])}</p>
        <label className="mt-3 block">
          <span className="text-[13px]">{t("Folder")}</span>
          <span className="mt-1.5 flex items-center gap-2">
            <input
              className={`${inputCls} flex-1 font-mono`}
              value={draft}
              aria-label={t("Folder the tools work in")}
              placeholder="C:/work/my-project"
              autoComplete="off"
              spellCheck={false}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && changed) p.onFolder(draft) }}
            />
            <Button disabled={!changed} onClick={() => p.onFolder(draft)}>{t("Save")}</Button>
          </span>
          <span className="mt-1 block text-[12px] text-ink-2">
            {p.folderOf.kind === "project" ? t("The folder of the project \"{name}\". Inside it files are free; outside it everything asks.", { name: p.folderOf.name }) : t("For chats that are in no project. Give a project its own folder from inside one of its chats.")}
          </span>
        </label>
        <p className="mt-2 text-[12px] text-ink-2">{p.info.shell ? t("Commands run in {shell}, in the folder, and ask you first unless they only read.", { shell: p.info.shell }) : t("No shell was found on this PC, so commands cannot run.")}</p>
        {p.rules > 0 && (
          <div className="mt-2 flex items-center justify-between gap-3 text-[12px] text-ink-2">
            <span>{t("{n} rules allowed for this chat", { n: p.rules })}</span>
            <Button kind="quiet" onClick={p.onForget}>{t("Forget them")}</Button>
          </div>
        )}
      </div>
    </div>
  )
}
