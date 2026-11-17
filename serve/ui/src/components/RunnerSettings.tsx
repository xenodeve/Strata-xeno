import { useCallback, useEffect, useState } from "react"
import { getMcp, getRunnerAgent, type McpInfo } from "../lib/api"
import { fmt } from "../lib/format"
import { msg, t } from "../lib/i18n"
import { MACHINE_LABEL, runnerId, setRunnerId, type Machine, type RunnerState } from "../lib/runner"
import { Button, Segmented, inputCls } from "./ui"
import { placeHeading } from "../lib/place"

// This browser's runner (#310 part 2). The runner is the part of Strata Desktop on the engineer's own PC: it runs the coding tools, the MCP servers
// and the skills of a project on that PC for the server. The page knows it by its id (`strata.runner`): Strata Desktop sets it, and here it can be
// pasted (the runner prints it when it connects). The server answers for a runner of this key only, so an id grants nothing by itself.

export type { RunnerState }

/** GET /agent?runner= for `runner`, asked again every `every` ms (it costs the server no job: the answer is from the runner's hello). */
export function useRunnerAgent(runner: string, every = 15000): [RunnerState, () => void] {
  const [state, setState] = useState<RunnerState>(undefined)
  const ask = useCallback(() => { if (runner) void getRunnerAgent(runner).then(setState); else setState(undefined) }, [runner])
  useEffect(() => {
    ask()
    if (!runner || !every) return
    const id = setInterval(ask, every)
    return () => clearInterval(id)
  }, [runner, every, ask])
  return [state, ask]
}

/** What the runner is now, in words. */
export function runnerLine(state: RunnerState): string {
  if (state === undefined) return t("Asking the server…")
  if (state === null) return t("The server could not be asked.")
  if (state === "unknown") return t("The server knows no runner with this id for your key. Start the runner on your PC with your key; it prints the id to paste here.")
  return state.online ? t("{name} is online.", { name: state.machine || t(MACHINE_LABEL.my) }) : t("{name} is offline: start the runner on it.", { name: state.machine || t(MACHINE_LABEL.my) })
}

/** The machine a project works on: "My PC" (through this browser's runner) or "Server PC" (the workstation; offered only when its coding tools are this page's). */
export function MachinePicker({ value, server, onPick, state, locked }: { value: Machine; server: boolean; onPick: (m: Machine) => void; state?: RunnerState; locked?: boolean }) {
  return (
    <div data-machine-picker={value} className={locked ? "pointer-events-none opacity-60" : undefined} aria-disabled={locked || undefined}>
      {server || value === "server"
        ? <Segmented label={t("Where the project is")} value={value} onChange={onPick} options={(["my", "server"] as Machine[]).map((m) => ({ value: m, label: t(MACHINE_LABEL[m]) }))} />
        : <div className="text-[13px]"><span className="text-ink-2">{t("Where the project is")}: </span><span className="font-medium">{t(MACHINE_LABEL.my)}</span></div>}
      <p className="mt-1.5 text-[12px] text-ink-2" data-machine-text>
        {value === "my" ? t("Its files are on your PC: the tools run there, through this browser's runner.") : t("Its files are on the workstation that runs Strata: the tools run there.")}
        {value === "my" && state !== undefined && <> {runnerLine(state)}</>}
      </p>
    </div>
  )
}

const STATE: Record<string, string> = { ready: msg("Connected"), starting: msg("Starting"), failed: msg("Failed"), stopped: msg("Stopped"), idle: msg("Waiting"), disabled: msg("Disabled") }

function useMyMcp(runner: string): McpInfo | null | undefined {
  const [info, setInfo] = useState<McpInfo | null | undefined>(undefined)
  useEffect(() => { let gone = false; void getMcp(runner).then((v) => { if (!gone) setInfo(v) }); return () => { gone = true } }, [runner])
  return info
}

const myNote = (info: McpInfo | null | undefined) => (
  <p role="note" className="mt-2 rounded-md bg-fill px-3 py-2 text-[12px] text-ink-2">
    {info === null ? t("The server could not tell what your PC has: is this browser's runner known for your key?")
      : t("From the runner on your PC ({name}), as it said when it connected. They are set in the runner's own file (runner.json on your PC), so they are read-only here.", { name: info?.machine || t(MACHINE_LABEL.my) })}
    {info && info.online === false && <> {t("Your PC is offline now.")}</>}
  </p>
)

/** Settings > MCP servers, above the Server PC's, when the editor of My PC's servers (McpSettings, #338) could not be had - `why`: that PC
 *  is offline, its Strata Desktop is older, or this page is not on that PC. The MCP servers of the engineer's own PC as its runner said when
 *  it connected (GET /mcp?runner=). */
export function MyPcMcp({ runner, why }: { runner: string; why: string }) {
  const info = useMyMcp(runner)
  const servers = (info?.servers ?? []).filter((s) => !/^my_skills(_\d+)?$/.test(s.name))
  return (
    <section data-my-pc="mcp" className="mb-8">
      <h2 className="text-[15px] font-semibold" data-place-heading="my">{`${t("MCP servers")} · ${placeHeading("my", info?.machine)}`}</h2>
      <p role="note" className="mt-2 rounded-md bg-fill px-3 py-2 text-[12px] text-ink-2 [overflow-wrap:anywhere]">
        {info === null ? t("The server could not tell what your PC has: is this browser's runner known for your key?")
          : t("From the runner on your PC ({name}), as it said when it connected. They cannot be changed from here now: {why}", { name: info?.machine || t(MACHINE_LABEL.my), why })}
      </p>
      {info && !servers.length && <p className="py-3 text-[13px] text-ink-2">{t("Your PC's runner has no MCP server.")}</p>}
      <ul className="m-0 mt-2 list-none p-0" aria-label={t("MCP servers on My PC")}>
        {servers.map((s) => (
          <li key={s.name} className="border-b border-line py-2.5 last:border-0" data-my-server={s.name}>
            <div className="flex items-baseline justify-between gap-4">
              <span className="min-w-0 font-medium [overflow-wrap:anywhere]">{s.name}</span>
              <span className={s.status === "failed" ? "text-bad" : "text-ink-2"}>{STATE[s.status] ? t(STATE[s.status]) : s.status}</span>
            </div>
            <div className="text-[12px] text-ink-2">{t("{n} tools", { n: fmt(s.tools.length) })}</div>
            {s.tools.length > 0 && <div className="mt-0.5 text-[12px] text-ink-3 [overflow-wrap:anywhere]">{s.tools.slice(0, 8).map((x) => x.tool).join(", ")}{s.tools.length > 8 ? ", …" : ""}</div>}
            {s.error && <div className="mt-1 text-[12px] text-bad [overflow-wrap:anywhere]">{s.error}</div>}
          </li>
        ))}
      </ul>
    </section>
  )
}

/** Settings > Skills, above the Server PC's: the skills the engineer's own PC has in use (GET /mcp?runner=), read-only. */
export function MyPcSkills({ runner }: { runner: string }) {
  const info = useMyMcp(runner)
  const skills = info?.skills ?? []
  return (
    <section data-my-pc="skills" className="mb-8">
      <h2 className="text-[15px] font-semibold">{t("Skills on My PC")}</h2>
      {myNote(info)}
      {info && !skills.length && <p className="py-3 text-[13px] text-ink-2">{t("No skill is in use on your PC.")}</p>}
      <ul className="m-0 mt-2 list-none p-0" aria-label={t("Skills on My PC")}>
        {skills.slice(0, 500).map((s) => (
          <li key={`${s.app ?? ""}:${s.name}`} className="border-b border-line py-2 last:border-0" data-my-skill={s.name}>
            <div className="flex items-baseline gap-2"><span className="font-medium">{s.name}</span>{s.app && <span className="text-[12px] text-ink-3">{s.app}</span>}</div>
            {s.description && <div className="text-[12px] text-ink-2 [overflow-wrap:anywhere]">{s.description}</div>}
          </li>
        ))}
      </ul>
      {skills.length > 500 && <p className="text-[12px] text-ink-3">{t("{n} more", { n: fmt(skills.length - 500) })}</p>}
    </section>
  )
}

/** Settings > This browser's runner: paste the id the runner printed; it shows which PC that is and whether it is online. */
export function RunnerSettings() {
  const [kept, setKept] = useState(runnerId)
  const [draft, setDraft] = useState(kept)
  const [problem, setProblem] = useState<string | null>(null)
  const [state, ask] = useRunnerAgent(kept, 0)
  const save = () => {
    if (!setRunnerId(draft)) { setProblem(t("A runner id is r and 16 letters or digits (0-9, a-f), as the runner prints it.")); return }
    setProblem(null)
    setKept(runnerId())
    setDraft(runnerId())
  }
  const forget = () => { setRunnerId(""); setKept(""); setDraft(""); setProblem(null) }
  return (
    <section data-runner-settings>
      <h2 className="text-[15px] font-semibold">{t("This browser's runner")}</h2>
      <p className="mt-1 text-[13px] text-ink-2">{t("The runner is the part of Strata Desktop on your own PC. With it, a project can be on your PC: the coding tools, MCP servers and skills of its chats run there, and the model runs here. Strata Desktop sets this for you; in another browser, paste the id the runner printed when it connected.")}</p>
      <div className="mt-3 flex items-center gap-2">
        <input
          className={`${inputCls} flex-1 font-mono`}
          value={draft}
          placeholder="r0123456789abcdef"
          aria-label={t("Runner id")}
          autoComplete="off"
          spellCheck={false}
          onChange={(e) => { setDraft(e.target.value); setProblem(null) }}
          onKeyDown={(e) => { if (e.key === "Enter") save() }}
        />
        <Button disabled={draft.trim() === kept} onClick={save}>{t("Save")}</Button>
        {kept && <Button kind="quiet" onClick={forget}>{t("Forget")}</Button>}
      </div>
      {problem && <p role="alert" className="mt-2 text-[12px] text-bad">{problem}</p>}
      {kept ? (
        <div className="mt-3 flex items-center justify-between gap-3 border-t border-line pt-2.5 text-[13px]" data-runner-state={state === undefined ? "asking" : state === null ? "error" : state === "unknown" ? "unknown" : state.online ? "online" : "offline"}>
          <span className={state === "unknown" ? "text-bad" : "text-ink-2"}>{runnerLine(state)}</span>
          <Button kind="quiet" onClick={ask}>{t("Check again")}</Button>
        </div>
      ) : <p className="mt-3 text-[12px] text-ink-2">{t("No runner is set: this browser works with the Server PC only.")}</p>}
    </section>
  )
}
