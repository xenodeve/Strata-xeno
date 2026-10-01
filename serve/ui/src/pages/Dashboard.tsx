import { useEffect, useState, type ReactNode } from "react"
import { Chart } from "../components/Chart"
import { Reveal, Swap } from "../components/motion"
import { Dot, NOT_MEASURED, Row, Rows, ms, pct, val } from "../components/bits"
import { getMcp, type McpInfo } from "../lib/api"
import { archOf, parseArch } from "../lib/arch"
import { fmt, gb, kfmt } from "../lib/format"
import { useMetrics, type Live, type Metrics } from "../lib/metrics"
import { href } from "../lib/router"
import { ModelFacts, modelLine } from "../components/ModelBlock"
import { Bar, cards, disksOf, link, modelDisk } from "./Hardware"
import { RequestList } from "./Requests"

const TIERS: [string, string][] = [
  ["tier_primary", "Primary GPU"], ["tier_secondary", "Secondary GPU"], ["tier_pcie", "RAM over PCIe"], ["tier_cpu", "RAM on the CPU"],
]

function headline(l: Live): { tone: "busy" | "idle"; text: string } {
  if (l.state === "reading") return { tone: "busy", text: l.prompt_total ? `Reading the prompt · ${fmt(l.prompt_read)} of ${fmt(l.prompt_total)}` : "Reading the prompt" }
  if (l.state === "generating") return { tone: "busy", text: `Writing · ${fmt(l.generated)} tokens` }
  return { tone: "idle", text: l.state === "unloaded" ? "Model unloaded" : "Idle" }
}

/** One part of the overview: a heading that opens its page, and the few rows that answer "is it fine?". */
function Block({ title, to, children }: { title: string; to: string; children: ReactNode }) {
  return (
    <section className="border-t border-line pt-3">
      <div className="flex items-baseline justify-between gap-4">
        <h2 className="text-[15px] font-semibold"><a href={to} className="no-underline hover:underline">{title}</a></h2>
        <a href={to} className="text-[12px] text-ink-2" aria-label={`Open ${title}`}>Open →</a>
      </div>
      <div className="mt-1.5">{children}</div>
    </section>
  )
}

const mean = (xs: (number | null | undefined)[]) => {
  const v = xs.filter((x): x is number => x != null)
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null
}

function Totals({ m }: { m: Metrics }) {
  const t = m.totals
  const decode = t.decode_ms > 0 ? t.output_tokens / (t.decode_ms / 1000) : null
  const read = t.prompt_tokens - (t.reused || 0)
  const prefill = t.prompt_ms > 0 && read > 0 ? read / (t.prompt_ms / 1000) : null
  return (
    <Rows>
      <Row k="Requests since the server started" v={<span className="num">{fmt(t.requests)}</span>} />
      <Row k="Prefill" hint="prompt tokens read" v={<span className="num">{kfmt(read)}</span>} />
      <Row k="Conversation cache" hint="prompt tokens already held" v={<span className="num">{kfmt(t.reused || 0)}</span>} />
      <Row k="Answer" hint="tokens written" v={<span className="num">{kfmt(t.output_tokens)}</span>} />
      <Row k="Average decode" v={val(decode, (n) => n.toFixed(1), "tok/s")} />
      <Row k="Average prefill" v={val(prefill, (n) => n.toFixed(0), "tok/s")} />
    </Rows>
  )
}

export function Dashboard() {
  const { data: m, error, stale } = useMetrics()
  const [mcp, setMcp] = useState<McpInfo | null>(null)
  useEffect(() => { void getMcp().then(setMcp) }, [])
  if (!m) return <p className="text-ink-2">{error || "Connecting…"}</p>

  const h = headline(m.live)
  const hw = m.hardware
  const gpus = cards(m)
  const disks = disksOf(m)
  const holds = modelDisk(m)
  const arch = parseArch(String(m.engine.gpu_arch ?? ""))
  const last = m.requests[0]
  const recent = m.requests.slice(0, 12).reverse()
  const recentSpeeds = recent.map((r) => r.decode?.tok_s_mean ?? r.decode_tok_s ?? null)
  const stats = last?.stats
  const tierSum = stats ? TIERS.reduce((a, [k]) => a + (stats[k] ?? 0), 0) : 0
  const held = gpus.flatMap((g) => (g.throttle && g.throttle.length ? [`GPU ${g.index}: ${g.throttle.join(", ")}`] : []))
  const mcpReady = mcp?.servers.filter((s) => s.status === "ready" || s.status === "stopped").length ?? 0

  return (
    <div>
      <div className="flex items-center gap-2.5">
        <Dot tone={h.tone} pulse={h.tone === "busy"} />
        <h1 className="text-xl font-semibold"><Swap k={m.live.state}>{h.text}</Swap></h1>
      </div>
      <p className="mt-1 text-[13px] text-ink-2">
        {stale ? "The server is not answering. Showing the last numbers it gave." : modelLine(m)}
      </p>

      <div className="mt-6 grid gap-x-10 gap-y-6 lg:grid-cols-2">
        <div className="stagger space-y-6">
          <Block title="Speed" to={href("live")}>
            <Rows>
              <Row k="Decode" hint="now" v={val(m.live.tok_s, (n) => n.toFixed(1), "tok/s")} />
              <Row k="Decode" hint="last request" v={last ? val(last.decode?.tok_s_mean ?? last.decode_tok_s, (n) => n.toFixed(1), "tok/s") : NOT_MEASURED} />
              <Row k="Prefill" hint="last request" v={last ? val(last.prefill?.tok_s_mean ?? last.prefill_tok_s, (n) => n.toFixed(0), "tok/s") : NOT_MEASURED} />
            </Rows>
            {recentSpeeds.some((v) => v != null) && (
              <div className="mt-2">
                <div className="mb-1 text-[12px] text-ink-2">Decode of the last {recentSpeeds.length} requests, tok/s <span className="text-ink-3">· mean {val(mean(recentSpeeds), (n) => n.toFixed(1))}</span></div>
                <Chart series={[{ label: "tok/s", values: recentSpeeds }]} area height={80} />
              </div>
            )}
          </Block>

          <Block title="Requests" to={href("requests")}>
            <Totals m={m} />
            <div className="mt-2"><RequestList rows={m.requests.slice(0, 3)} empty="No request since the server started." /></div>
          </Block>

          <Reveal>
            {stats && tierSum > 0 ? (
              <Block title="Where the last request's experts ran" to={last ? href("requests", last.id) : href("requests")}>
                <div className="flex h-2 w-full overflow-hidden rounded-full bg-fill" role="img" aria-label={TIERS.map(([k, l]) => `${l} ${pct((stats[k] ?? 0) / tierSum)}`).join(", ")}>
                  {TIERS.map(([k], i) => <div key={k} className="bg-ink" style={{ width: `${((stats[k] ?? 0) / tierSum) * 100}%`, opacity: 1 - i * 0.22 }} />)}
                </div>
                <Rows>
                  {TIERS.map(([k, l]) => <Row key={k} k={l} v={<span className="num">{pct((stats[k] ?? 0) / tierSum)}</span>} />)}
                  {last.decode_ms ? <Row k="Decode time" v={<span className="num">{ms(last.decode_ms)}</span>} /> : null}
                </Rows>
              </Block>
            ) : null}
          </Reveal>
        </div>

        <div className="stagger space-y-6">
          <Block title="Hardware" to={href("hardware")}>
            {gpus.length === 0 ? <p className="text-ink-2">No GPU is reporting.</p> : gpus.map((g) => (
              <div key={g.index} className="py-1.5">
                <div className="flex items-baseline justify-between gap-3 text-[13px]">
                  <a href={href("hardware", "gpu", String(g.index))} className="min-w-0 truncate no-underline hover:underline">{(g.name || `GPU ${g.index}`).replace("NVIDIA GeForce ", "")}<span className="text-ink-3"> {archOf(arch, g.name)}</span></a>
                  <span className="num shrink-0 text-ink-2">{g.util != null ? `${g.util}%` : "–"} · {g.temp != null ? `${g.temp}°C` : "–"}</span>
                </div>
                <div className="mt-1"><Bar used={g.mem_used} total={g.mem_total} label={`GPU ${g.index} memory`} /></div>
                <div className="num mt-0.5 text-[12px] text-ink-2">{gb(g.mem_used)} / {gb(g.mem_total)} GB · {link(g.pcie_gen, g.pcie_width, g.pcie_width_max, g.pcie_gen_max)}</div>
              </div>
            ))}
            <Reveal>{held.length > 0 ? <p className="mt-1 text-[13px]">Held back: {held.join("; ")}</p> : null}</Reveal>
            <Rows>
              <Row k="CPU" hint={m.engine.cpu_isa ? String(m.engine.cpu_isa) : undefined} v={val(hw.cpu, (n) => n.toFixed(0), "%")} />
              <Row k="RAM" v={hw.ram_total ? <span className="num">{gb(hw.ram_used)} / {gb(hw.ram_total)} GB</span> : NOT_MEASURED} />
              {disks.filter((d) => holds === d.index).map((d) => <Row key={d.index} k="Model on" v={<span className="text-right">{d.model} <span className="text-ink-2">{d.bus} {d.media}</span></span>} />)}
            </Rows>
          </Block>

          <Block title="Model" to={href("about")}>
            <ModelFacts info={m.model_info} kv={m.engine.kv as string | undefined} />
          </Block>

          <Block title="This server" to={href("about")}>
            <Rows>
              <Row k="Engine" v={m.engine.engine ? String(m.engine.engine) : NOT_MEASURED} />
              <Row k="Context" v={m.engine.max_context ? <span className="num">{fmt(Number(m.engine.max_context))} tokens</span> : NOT_MEASURED} />
              <Row k="Drafting" hint="MTP / suffix" v={m.engine.spec != null ? <span className="num">{String(m.engine.mtp_max ?? "–")} / {String(m.engine.lookup ?? "–")}</span> : NOT_MEASURED} />
              <Row k="MCP" v={mcp && mcp.servers.length ? <span className="num">{mcpReady} of {mcp.servers.length} servers · {fmt(mcp.tools)} tools</span> : <span className="text-ink-3">none configured</span>} />
              <Row k="History kept" v={<span className="num">{fmt(m.requests_kept)} in memory</span>} />
            </Rows>
          </Block>
        </div>
      </div>
    </div>
  )
}
