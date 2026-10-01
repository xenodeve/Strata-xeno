import { useEffect, useState } from "react"
import { apiHeaders, getHealth, NO_HEALTH, root, url, type Health } from "../lib/api"
import { parseArch } from "../lib/arch"
import { fmt } from "../lib/format"
import { t } from "../lib/i18n"
import { store } from "../lib/store"
import { ModelFacts } from "../components/ModelBlock"
import { SLOT, Sentence } from "../components/bits"
import type { ModelInfo } from "../lib/metrics"
import { toast } from "../components/toast"
import { inputCls } from "../components/ui"
import { StatusMarkSettings } from "../components/StatusMarks"
import { McpSettings } from "../components/McpSettings"

interface Engine { engine?: string; version?: string; context?: number; kv?: string; spec?: number; mtp_max?: number; cpu_isa?: string; gpu_arch?: string }

function Row({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-6 border-b border-line py-2 last:border-0">
      <dt className="text-ink-2">{k}</dt>
      <dd className="num m-0 text-right [overflow-wrap:anywhere]">{v}</dd>
    </div>
  )
}

export function About() {
  const [health, setHealth] = useState<Health>(NO_HEALTH)
  const [engine, setEngine] = useState<Engine>({})
  const [model, setModel] = useState<ModelInfo | null>(null)
  const [key, setKey] = useState(() => store.get("apikey", ""))

  useEffect(() => {
    void getHealth().then(setHealth).catch(() => {})
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

      <section>
        <h2 className="text-[15px] font-semibold">{t("The classic app")}</h2>
        <p className="mt-1 text-[13px] text-ink-2">
          <Sentence text={t("Still here, with its Monitor, until the Live and Hardware pages replace it: {link}.", { link: SLOT })} node={<a href={`${root()}classic/`}>{t("open the classic app")}</a>} />
        </p>
      </section>
    </div>
  )
}
