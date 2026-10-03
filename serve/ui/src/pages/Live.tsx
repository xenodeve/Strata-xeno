import { Loading } from "../components/orb"
import { Chart } from "../components/Chart"
import { Swap } from "../components/motion"
import { Pop } from "../components/pop"
import { StatusOrb } from "../components/live"
import { Row, Rows, Section, val } from "../components/bits"
import { fmt, gb } from "../lib/format"
import { t } from "../lib/i18n"
import { href } from "../lib/router"
import { useMetrics, type Live as LiveT } from "../lib/metrics"
import { RequestList } from "./Requests"
import { modelLine } from "../components/ModelBlock"
import { phaseKind } from "../lib/orbs"

function headline(l: LiveT): { tone: "ok" | "busy" | "idle"; text: string } {
  switch (l.state) {
    case "reading": return { tone: "busy", text: l.prompt_total ? t("Reading the prompt · {read} of {total} tokens", { read: fmt(l.prompt_read), total: fmt(l.prompt_total) }) : t("Reading the prompt") }
    case "generating": {
      const p = phaseKind(l.phase), n = fmt(l.generated)
      return { tone: "busy", text: p.kind === "thinking" ? t("Thinking · {n} tokens", { n }) : p.kind === "answering" ? t("Answering · {n} tokens", { n })
        : p.kind === "tool" ? t("Writing a tool call: {name} · {n} tokens", { name: p.tool, n }) : p.kind === "toolDone" ? t("Tool call written · {n} tokens", { n })
        : t("Writing · {n} tokens", { n }) }
    }
    case "unloaded": return { tone: "idle", text: t("Model unloaded") }
    default: return { tone: "idle", text: t("Idle") }
  }
}

export function Live() {
  const { data: m, error, stale } = useMetrics()
  if (!m) return <Loading error={error} />
  const l = m.live
  const running = l.state === "reading" || l.state === "generating"       // no request: idle, which is not the same as not measured
  const IDLE = <span className="text-ink-3">{t("idle")}</span>
  const h = headline(l)
  const gpus = m.hardware.gpus ?? (m.hardware.gpu_mem_total != null ? [{ index: 0, util: m.hardware.gpu_util ?? null, mem_used: m.hardware.gpu_mem_used ?? null, mem_total: m.hardware.gpu_mem_total, temp: m.hardware.gpu_temp ?? null, power: m.hardware.gpu_power ?? null }] : [])
  const tokS = m.history.tok_s || []
  const prefill = m.history.prefill_tok_s_mean || []

  return (
    <div>
      <header className="page-head flex items-center gap-4">
        <StatusOrb live={l} stale={stale} size={32} />
        <div className="min-w-0">
          <h1 className="page-title"><Swap k={m.live.state}><Pop text={h.text} /></Swap></h1>
          <p className="page-sub !mt-1">
            {stale ? t("The server is not answering. Showing the last numbers it gave.") : l.queued ? t("{n} waiting in the queue", { n: l.queued }) : modelLine(m)}
          </p>
        </div>
      </header>

      <Section title={t("Speed")} aside={l.tok_s_window_s ? t("windowed over {n} s", { n: l.tok_s_window_s }) : undefined}>
        <Rows>
          <Row k="Decode" hint={t("now")} v={running ? val(l.tok_s, (n) => n.toFixed(1), "tok/s") : IDLE} />
          <Row k="Decode" hint={t("mean of this request")} v={running ? val(l.tok_s_mean, (n) => n.toFixed(1), "tok/s") : IDLE} />
          <Row k="Prefill" hint={t("mean of this request")} v={running ? val(l.prefill_tok_s_mean, (n) => n.toFixed(0), "tok/s") : IDLE} />
          <Row k={t("Elapsed")} v={running ? val(l.elapsed_s, (n) => n.toFixed(1), "s", "count") : IDLE} />
        </Rows>
        <div className="mt-3">
          <div className="mb-1 text-[12px] text-ink-2">{t("Decode, last minute")}</div>
          <Chart series={[{ label: "tok/s", values: tokS }]} unit="" area height={110} />
          {prefill.some((v) => v) && (
            <>
              <div className="mb-1 mt-4 text-[12px] text-ink-2">{t("Prefill, last minute")}</div>
              <Chart series={[{ label: "tok/s", values: prefill }]} area height={90} />
            </>
          )}
        </div>
      </Section>

      <Section title={t("Cards")} aside={<a href={href("hardware")}>{t("Hardware")}</a>}>
        {gpus.length === 0 ? <p className="text-ink-2">{t("No GPU is reporting.")}</p> : (
          <Rows>
            {gpus.map((g) => (
              <Row key={g.index} k={t("GPU {n}", { n: g.index })} v={
                <span className="num">
                  {g.util != null ? `${g.util}%` : "–"} · {gb(g.mem_used)} / {gb(g.mem_total)} GB · {g.temp != null ? `${g.temp}°C` : "–"}
                </span>
              } />
            ))}
          </Rows>
        )}
      </Section>

      <Section title={t("Recent requests")} aside={<a href={href("requests")}>{t("All")}</a>}>
        <RequestList rows={m.requests.slice(0, 5)} empty={t("No request since the server started.")} />
      </Section>
    </div>
  )
}
