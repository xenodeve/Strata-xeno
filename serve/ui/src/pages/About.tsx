import { useEffect, useState } from "react"
import { apiHeaders, getHealth, getMcp, NO_HEALTH, root, url, type Health, type McpInfo } from "../lib/api"
import { parseArch } from "../lib/arch"
import { fmt } from "../lib/format"
import { msg, t } from "../lib/i18n"
import { store } from "../lib/store"
import { ModelFacts } from "../components/ModelBlock"
import { SLOT, Sentence } from "../components/bits"
import type { ModelInfo } from "../lib/metrics"
import { StatusLabel } from "../components/orb"
import { toast } from "../components/toast"
import { inputCls, Segmented } from "../components/ui"
import { BOT_TYPES, setAvatar, setBotChoice, useAvatar, useBotChoice, type AvatarKind } from "../lib/avatar"
import { BotTile } from "../components/orb"

interface Engine { engine?: string; version?: string; context?: number; kv?: string; spec?: number; mtp_max?: number; cpu_isa?: string; gpu_arch?: string }

function Row({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-6 border-b border-line py-2 last:border-0">
      <dt className="text-ink-2">{k}</dt>
      <dd className="num m-0 text-right [overflow-wrap:anywhere]">{v}</dd>
    </div>
  )
}

const MCP_STATE: Record<string, string> = { ready: msg("Connected"), starting: msg("Starting"), failed: msg("Failed"), stopped: msg("Stopped"), idle: msg("Waiting") }

const BOT_NAME: Record<(typeof BOT_TYPES)[number], string> = {
  clover: msg("Clover"), flower: msg("Flower"), triangle: msg("Triangle"), square: msg("Square"), blob: msg("Blob"), ghost: msg("Ghost"),
  circle: msg("Circle"), drop: msg("Drop"), star: msg("Star"), droid: msg("Droid"), mech: msg("Mech"), alien: msg("Alien"),
  hexagon: msg("Hexagon"), cat: msg("Cat"), cloud: msg("Cloud"), pill: msg("Pill"), pebble: msg("Pebble"), puddle: msg("Puddle"),
}

export function About() {
  const avatar = useAvatar()
  const pick = useBotChoice()
  const [health, setHealth] = useState<Health>(NO_HEALTH)
  const [engine, setEngine] = useState<Engine>({})
  const [model, setModel] = useState<ModelInfo | null>(null)
  const [mcp, setMcp] = useState<McpInfo>({ servers: [], tools: 0 })
  const [key, setKey] = useState(() => store.get("apikey", ""))

  useEffect(() => {
    void getHealth().then(setHealth).catch(() => {})
    void getMcp().then((m) => m && setMcp(m))
    void fetch(url("metrics"), { headers: apiHeaders() })
      .then((r) => (r.ok ? r.json() : null))
      .then((m: { engine?: Engine; model_info?: ModelInfo | null } | null) => { if (m?.engine) setEngine(m.engine); setModel(m?.model_info ?? null) })
      .catch(() => {})
  }, [])

  return (
    <div className="max-w-[65ch] space-y-8">
      <section>
        <h1 className="page-title">{t("About")}</h1>
        <p className="page-sub">{t("Strata runs a large mixture-of-experts model on this PC. Nothing leaves it.")}</p>
      </section>

      <section>
        <h2 className="text-[15px] font-semibold">{t("The model")}</h2>
        <div className="mt-2"><ModelFacts info={model} kv={engine.kv} /></div>
      </section>

      <section>
        <h2 className="text-[15px] font-semibold">{t("This server")}</h2>
        <dl className="mt-2 m-0">
          <Row k={t("Model")} v={health.model} />
          <Row k="Context" v={health.max_context ? t("{n} tokens", { n: fmt(health.max_context) }) : "–"} />
          <Row k={t("Pictures")} v={health.images ? t("on") : t("off")} />
          <Row k="Engine" v={engine.engine || engine.version || "–"} />
          {engine.cpu_isa && <Row k={t("CPU kernels")} v={engine.cpu_isa} />}
          {engine.gpu_arch && <Row k={t("GPU architecture")} v={parseArch(engine.gpu_arch).map((a) => `${a.sm} ${a.name.replace("NVIDIA GeForce ", "")}`).join(", ")} />}
        </dl>
      </section>

      <section>
        <h2 className="text-[15px] font-semibold">{t("Status avatars")}</h2>
        <p className="mt-1 text-[13px] text-ink-2">{t("What stands for a status: a dotted orb, a plain ring that turns, or a small avatar with a shape and a mood of its own.")}</p>
        <div className="mt-2">
          <Segmented label={t("Status avatars")} value={avatar} onChange={(v) => setAvatar(v as AvatarKind)} options={[{ value: "orbs", label: t("Orbs") }, { value: "loading", label: t("Loading") }, { value: "bots", label: t("Avatar") }]} />
        </div>
        {avatar === "bots" && (
          <div className="mt-3">
            <p className="text-[13px] text-ink-2">{t("Which avatar: one for every status, a shape for each status, or a random one for each place.")}</p>
            <div className="mt-2 flex flex-wrap gap-2" role="group" aria-label={t("Which avatar")}>
              <button type="button" aria-pressed={pick === "auto"} onClick={() => setBotChoice("auto")} className={`flex h-[60px] min-w-[84px] items-center justify-center rounded-md border px-3 text-[12px] transition-colors ${pick === "auto" ? "border-ink bg-fill text-ink" : "border-line text-ink-2 hover:bg-hover"}`}>{t("By status")}</button>
              <button type="button" aria-pressed={pick === "random"} onClick={() => setBotChoice("random")} className={`flex h-[60px] min-w-[84px] items-center justify-center rounded-md border px-3 text-[12px] transition-colors ${pick === "random" ? "border-ink bg-fill text-ink" : "border-line text-ink-2 hover:bg-hover"}`}>{t("Random")}</button>
              {BOT_TYPES.map((b) => (
                <button key={b} type="button" aria-pressed={pick === b} aria-label={t(BOT_NAME[b])} title={t(BOT_NAME[b])} onClick={() => setBotChoice(b)} className={`grid size-[60px] place-items-center rounded-md border transition-colors ${pick === b ? "border-ink bg-fill" : "border-line hover:bg-hover"}`}>
                  <BotTile type={b} />
                </button>
              ))}
            </div>
          </div>
        )}
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

      {mcp.servers.length > 0 && (
        <section>
          <h2 className="text-[15px] font-semibold">{t("MCP servers")}</h2>
          <p className="mt-1 text-[13px] text-ink-2">{t("{n} tools; the model calls them when it decides to.", { n: fmt(mcp.tools) })}</p>
          <ul className="mt-2 m-0 list-none p-0">
            {mcp.servers.map((s) => (
              <li key={s.name} className="border-b border-line py-2 last:border-0">
                <div className="flex items-baseline justify-between gap-4">
                  <span className="font-medium">{s.name}</span>
                  {s.status === "starting"
                    ? <StatusLabel design="connecting" className="text-ink-2">{t(MCP_STATE[s.status])}</StatusLabel>
                    : <span className={s.status === "failed" ? "text-bad" : "text-ink-2"}>{MCP_STATE[s.status] ? t(MCP_STATE[s.status]) : s.status}</span>}
                </div>
                <div className="text-[12px] text-ink-2">{s.transport} · {t("{n} tools", { n: fmt(s.tools.length) })}</div>
                {s.error && <div className="mt-1 text-[12px] text-bad [overflow-wrap:anywhere]">{s.error}</div>}
              </li>
            ))}
          </ul>
        </section>
      )}

      <section>
        <h2 className="text-[15px] font-semibold">{t("The classic app")}</h2>
        <p className="mt-1 text-[13px] text-ink-2">
          <Sentence text={t("Still here, with its Monitor, until the Live and Hardware pages replace it: {link}.", { link: SLOT })} node={<a href={`${root()}classic/`}>{t("open the classic app")}</a>} />
        </p>
      </section>
    </div>
  )
}
