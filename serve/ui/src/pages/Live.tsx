import { Chart } from "../components/Chart"
import { Swap } from "../components/motion"
import { Dot, Row, Rows, Section, val } from "../components/bits"
import { fmt, gb } from "../lib/format"
import { href } from "../lib/router"
import { useMetrics, type Live as LiveT } from "../lib/metrics"
import { RequestList } from "./Requests"

function headline(l: LiveT): { tone: "ok" | "busy" | "idle"; text: string } {
  switch (l.state) {
    case "reading": return { tone: "busy", text: l.prompt_total ? `Reading the prompt · ${fmt(l.prompt_read)} of ${fmt(l.prompt_total)} tokens` : "Reading the prompt" }
    case "generating": return { tone: "busy", text: `Writing · ${fmt(l.generated)} tokens` }
    case "unloaded": return { tone: "idle", text: "Model unloaded" }
    default: return { tone: "idle", text: "Idle" }
  }
}

export function Live() {
  const { data: m, error, stale } = useMetrics()
  if (!m) return <p className="text-ink-2">{error || "Connecting…"}</p>
  const l = m.live
  const h = headline(l)
  const busy = l.state === "reading" || l.state === "generating"
  const gpus = m.hardware.gpus ?? (m.hardware.gpu_mem_total != null ? [{ index: 0, util: m.hardware.gpu_util ?? null, mem_used: m.hardware.gpu_mem_used ?? null, mem_total: m.hardware.gpu_mem_total, temp: m.hardware.gpu_temp ?? null, power: m.hardware.gpu_power ?? null }] : [])
  const tokS = m.history.tok_s || []
  const prefill = m.history.prefill_tok_s_mean || []

  return (
    <div>
      <div className="flex items-center gap-2.5">
        <Dot tone={h.tone} pulse={busy} />
        <h1 className="text-xl font-semibold"><Swap k={m.live.state}>{h.text}</Swap></h1>
      </div>
      <p className="mt-1 text-[13px] text-ink-2">
        {stale ? "The server is not answering. Showing the last numbers it gave." : l.queued ? `${l.queued} waiting in the queue` : m.engine.model as string}
      </p>

      <Section title="Speed" aside={l.tok_s_window_s ? `windowed over ${l.tok_s_window_s} s` : undefined}>
        <Rows>
          <Row k="Decode" hint="now" v={val(l.tok_s, (n) => n.toFixed(1), "tok/s")} />
          <Row k="Decode" hint="mean of this request" v={val(l.tok_s_mean, (n) => n.toFixed(1), "tok/s")} />
          <Row k="Prefill" hint="mean of this request" v={val(l.prefill_tok_s_mean, (n) => n.toFixed(0), "tok/s")} />
          <Row k="Elapsed" v={val(l.elapsed_s, (n) => n.toFixed(1), "s")} />
        </Rows>
        <div className="mt-3">
          <div className="mb-1 text-[12px] text-ink-2">Decode, last minute</div>
          <Chart series={[{ label: "tok/s", values: tokS }]} unit="" area height={110} />
          {prefill.some((v) => v) && (
            <>
              <div className="mb-1 mt-4 text-[12px] text-ink-2">Prefill, last minute</div>
              <Chart series={[{ label: "tok/s", values: prefill }]} area height={90} />
            </>
          )}
        </div>
      </Section>

      <Section title="Cards" aside={<a href={href("hardware")}>Hardware</a>}>
        {gpus.length === 0 ? <p className="text-ink-2">No GPU is reporting.</p> : (
          <Rows>
            {gpus.map((g) => (
              <Row key={g.index} k={`GPU ${g.index}`} v={
                <span className="num">
                  {g.util != null ? `${g.util}%` : "–"} · {gb(g.mem_used)} / {gb(g.mem_total)} GB · {g.temp != null ? `${g.temp}°C` : "–"}
                </span>
              } />
            ))}
          </Rows>
        )}
      </Section>

      <Section title="Recent requests" aside={<a href={href("requests")}>All</a>}>
        <RequestList rows={m.requests.slice(0, 5)} empty="No request since the server started." />
      </Section>
    </div>
  )
}
