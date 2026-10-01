import { Loading } from "../components/orb"
import { useEffect, useState, type ReactNode } from "react"
import { Chart } from "../components/Chart"
import { Facts, NOT_MEASURED, Row, Rows, ms, pct, val } from "../components/bits"
import { Num, StatusOrb } from "../components/live"
import { ModelFacts, modelLine } from "../components/ModelBlock"
import { Disclosure, Reveal, Swap } from "../components/motion"
import { getMcp, type McpInfo } from "../lib/api"
import { archOf, parseArch } from "../lib/arch"
import { fmt, gb, kfmt } from "../lib/format"
import { promptSplit, useMetrics, type Live } from "../lib/metrics"
import { href } from "../lib/router"
import { Bar, GpuOrb, cards, disksOf, link, modelDisk } from "./Hardware"
import { RequestList } from "./Requests"

const TIERS: [string, string][] = [
  ["tier_primary", "Primary GPU"], ["tier_secondary", "Secondary GPU"], ["tier_pcie", "RAM over PCIe"], ["tier_cpu", "RAM on the CPU"],
]

/** The state as a sentence at display size: the numbers inside it glide, the sentence sharpens in when the state changes. */
function Headline({ l }: { l: Live }): ReactNode {
  switch (l.state) {
    case "reading": return l.prompt_total
      ? <>Reading <Num value={l.prompt_read} /> of {fmt(l.prompt_total)} tokens</>
      : <>Reading the prompt</>
    case "generating": return l.tok_s ? <>Writing at <Num value={l.tok_s} digits={0} /> tok/s</> : <>Writing</>
    case "unloaded": return <>Model unloaded</>
    default: return <>Ready</>
  }
}

/** One part of the overview: a heading that opens its page, and the few rows that answer "is it fine?". */
function Block({ title, to, children }: { title: string; to: string; children: ReactNode }) {
  return (
    <section className="border-t border-line pt-4">
      <a href={to} className="block-head flex items-baseline justify-between gap-4 no-underline" aria-label={`Open ${title}`}>
        <h2 className="text-[15px] font-semibold tracking-[-0.015em]">{title}</h2>
        <span className="open-link text-[12px]">Open <span className="arrow">→</span></span>
      </a>
      <div className="mt-2">{children}</div>
    </section>
  )
}

const mean = (xs: (number | null | undefined)[]) => {
  const v = xs.filter((x): x is number => x != null)
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null
}

export function Dashboard() {
  const { data: m, error, stale } = useMetrics()
  const [mcp, setMcp] = useState<McpInfo | null>(null)
  useEffect(() => { void getMcp().then(setMcp) }, [])
  if (!m) return <Loading error={error} />

  const l = m.live
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
  const t = m.totals
  const totalRead = t.prompt_tokens - (t.reused || 0)
  const split = last ? promptSplit(last.prompt_tokens, last.reused) : null
  const info = m.model_info
  const lastDecode = last ? (last.decode?.tok_s_mean ?? last.decode_tok_s ?? null) : null

  return (
    <div>
      <section className="grid items-center gap-x-6 gap-y-4 pb-10 pt-2 sm:grid-cols-[auto_1fr]">
        <StatusOrb live={l} stale={stale} size={64} />
        <div className="min-w-0">
          <h1 className="display"><Swap k={l.state}><Headline l={l} /></Swap></h1>
          <p className="lede mt-2.5">
            {stale ? "The server is not answering. Showing the last numbers it gave." : modelLine(m)}
          </p>
          {last && (
            <p className="num mt-1 text-[13px] text-ink-3">
              Last request · {lastDecode != null ? `${lastDecode.toFixed(1)} tok/s decode` : "no decode"}
              {split && ` · ${fmt(split.read)} tokens read${split.cached ? `, ${fmt(split.cached)} cached` : ""}`} · {last.duration_s.toFixed(1)} s
            </p>
          )}
        </div>
      </section>

      <div className="grid gap-x-12 gap-y-10 lg:grid-cols-12">
        <div className="stagger space-y-10 lg:col-span-7">
          <Block title="Speed" to={href("live")}>
            <Facts items={[
              ["Decode now", l.state === "generating" ? val(l.tok_s, (n) => n.toFixed(1), "tok/s") : <span className="text-ink-3">idle</span>],
              ["Last request", last ? val(lastDecode, (n) => n.toFixed(1), "tok/s") : NOT_MEASURED],
              ["Prefill, last", last ? val(last.prefill?.tok_s_mean ?? last.prefill_tok_s, (n) => n.toFixed(0), "tok/s") : NOT_MEASURED],
            ]} />
            {recentSpeeds.some((v) => v != null) && (
              <div className="mt-4">
                <div className="mb-1 flex items-baseline justify-between text-[12px] text-ink-3">
                  <span>Decode of the last {recentSpeeds.length} requests</span>
                  <span className="num">mean {val(mean(recentSpeeds), (n) => n.toFixed(1))}</span>
                </div>
                <Chart series={[{ label: "tok/s", values: recentSpeeds }]} area height={132} />
              </div>
            )}
          </Block>

          <Block title="Requests" to={href("requests")}>
            <Facts items={[
              ["Since start", fmt(t.requests)], ["Read", kfmt(totalRead)], ["Cached", kfmt(t.reused || 0)], ["Written", kfmt(t.output_tokens)],
              ["Avg decode", val(t.decode_ms > 0 ? t.output_tokens / (t.decode_ms / 1000) : null, (n) => n.toFixed(1), "tok/s")],
            ]} />
            <div className="mt-3"><RequestList rows={m.requests.slice(0, 3)} empty="No request since the server started." /></div>
          </Block>
        </div>

        <div className="stagger space-y-10 lg:col-span-5">
          <Block title="Hardware" to={href("hardware")}>
            {gpus.length === 0 ? <p className="text-ink-2">No GPU is reporting.</p> : gpus.map((g) => (
              <div key={g.index} className="row-wash py-2">
                <div className="flex items-baseline justify-between gap-3 text-[13px]">
                  <a href={href("hardware", "gpu", String(g.index))} className="flex min-w-0 items-center gap-2 truncate no-underline hover:underline"><GpuOrb g={g} />{(g.name || `GPU ${g.index}`).replace("NVIDIA GeForce ", "")}<span className="text-ink-3"> {archOf(arch, g.name)}</span></a>
                  <span className="num shrink-0 text-ink-2">{g.util != null ? `${g.util}%` : "–"} · {g.temp != null ? `${g.temp}°C` : "–"}</span>
                </div>
                <div className="mt-1.5"><Bar used={g.mem_used} total={g.mem_total} label={`GPU ${g.index} memory`} /></div>
                <div className="num mt-1 text-[12px] text-ink-3">{gb(g.mem_used)} / {gb(g.mem_total)} GB · {link(g.pcie_gen, g.pcie_width, g.pcie_width_max, g.pcie_gen_max)}</div>
              </div>
            ))}
            <Reveal>{held.length > 0 ? <p className="mt-1 text-[13px]">Held back: {held.join("; ")}</p> : null}</Reveal>
            <div className="mt-3 grid grid-cols-2 gap-x-6">
              <div>
                <div className="mb-1.5 flex items-baseline justify-between text-[12px]"><span className="text-ink-3">CPU{m.engine.cpu_isa ? ` · ${m.engine.cpu_isa}` : ""}</span><span className="num text-ink-2">{hw.cpu != null ? `${hw.cpu.toFixed(0)}%` : "–"}</span></div>
                <Bar used={hw.cpu} total={100} label="CPU busy" />
              </div>
              <div>
                <div className="mb-1.5 flex items-baseline justify-between text-[12px]"><span className="text-ink-3">RAM</span><span className="num text-ink-2">{hw.ram_total ? `${gb(hw.ram_used)} / ${gb(hw.ram_total)} GB` : "–"}</span></div>
                <Bar used={hw.ram_used} total={hw.ram_total} label="RAM used" />
              </div>
            </div>
            {disks.filter((d) => holds === d.index).map((d) => <p key={d.index} className="mt-3 text-[12px] text-ink-3">Model on {d.model} · {d.bus} {d.media}</p>)}
          </Block>

          <Reveal>
            {stats && tierSum > 0 ? (
              <Block title="Where the last request's experts ran" to={last ? href("requests", last.id) : href("requests")}>
                <div className="flex h-2 w-full overflow-hidden rounded-full bg-fill" role="img" aria-label={TIERS.map(([k, n]) => `${n} ${pct((stats[k] ?? 0) / tierSum)}`).join(", ")}>
                  {TIERS.map(([k], i) => <div key={k} className="bg-ink transition-[width] duration-500 ease-[var(--ease)]" style={{ width: `${((stats[k] ?? 0) / tierSum) * 100}%`, opacity: 1 - i * 0.24 }} />)}
                </div>
                <dl className="m-0 mt-3 flex flex-wrap gap-x-6 gap-y-1.5 text-[13px]">
                  {TIERS.map(([k, n], i) => (
                    <div key={k} className="flex items-center gap-2">
                      <span aria-hidden className="size-2 rounded-full bg-ink" style={{ opacity: 1 - i * 0.24 }} />
                      <dt className="text-ink-2">{n}</dt><dd className="num m-0">{pct((stats[k] ?? 0) / tierSum)}</dd>
                    </div>
                  ))}
                </dl>
                {last?.decode_ms ? <p className="num mt-2 text-[12px] text-ink-3">Over {ms(last.decode_ms)} of decode</p> : null}
              </Block>
            ) : null}
          </Reveal>

          <div>
          <section className="border-t border-line py-2">
            <Disclosure block title="Model" hint={info ? [info.variant, info.bpw != null ? `${info.bpw.toFixed(2)} bpw` : null, `${(info.bytes / 1e9).toFixed(1)} GB`].filter(Boolean).join(" · ") : undefined}>
              <div className="pb-2 pt-1"><ModelFacts info={info} kv={m.engine.kv as string | undefined} /></div>
            </Disclosure>
          </section>
          <section className="border-t border-line py-2">
            <Disclosure block title="This server" hint={[m.engine.engine, m.engine.max_context ? `${fmt(Number(m.engine.max_context) / 1024)}K context` : null, mcp?.servers.length ? `${mcpReady}/${mcp.servers.length} MCP` : null].filter(Boolean).join(" · ")}>
              <div className="pb-2 pt-1">
                <Rows>
                  <Row k="Engine" v={m.engine.engine ? String(m.engine.engine) : NOT_MEASURED} />
                  <Row k="Context" v={m.engine.max_context ? <span className="num">{fmt(Number(m.engine.max_context))} tokens</span> : NOT_MEASURED} />
                  <Row k="Drafting" hint="MTP / suffix" v={m.engine.spec != null ? <span className="num">{String(m.engine.mtp_max ?? "–")} / {String(m.engine.lookup ?? "–")}</span> : NOT_MEASURED} />
                  <Row k="MCP" v={mcp && mcp.servers.length ? <span className="num">{mcpReady} of {mcp.servers.length} servers · {fmt(mcp.tools)} tools</span> : <span className="text-ink-3">none configured</span>} />
                  <Row k="History kept" v={<span className="num">{fmt(m.requests_kept)} in memory</span>} />
                </Rows>
                <p className="mt-2 text-[13px]"><a href={href("about")}>Open About</a></p>
              </div>
            </Disclosure>
          </section>
          </div>
        </div>
      </div>
    </div>
  )
}
