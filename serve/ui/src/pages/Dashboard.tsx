import { Loading } from "../components/orb"
import { useEffect, useState, type ReactNode } from "react"
import { Chart } from "../components/Chart"
import { Facts, NOT_MEASURED, Row, Rows, SLOT, Sentence, ms, pct, val } from "../components/bits"
import { Num, StatusOrb } from "../components/live"
import { ModelFacts, modelLine } from "../components/ModelBlock"
import { Disclosure, Reveal, Swap } from "../components/motion"
import { getMcp, type McpInfo } from "../lib/api"
import { archOf, parseArch } from "../lib/arch"
import { fmt, gb, kfmt } from "../lib/format"
import { msg, t } from "../lib/i18n"
import { promptSplit, useMetrics, type Live } from "../lib/metrics"
import { phaseKind } from "../lib/orbs"
import { href } from "../lib/router"
import { Bar, GpuOrb, cards, disksOf, link, modelDisk } from "./Hardware"
import { RequestList } from "./Requests"

const TIERS: [string, string][] = [
  ["tier_primary", msg("Primary GPU")], ["tier_secondary", msg("Secondary GPU")], ["tier_pcie", msg("RAM over PCIe")], ["tier_cpu", msg("RAM on the CPU")],
]

/** The state as a sentence at display size: the numbers inside it glide, the sentence sharpens in when the state changes. */
function Headline({ l }: { l: Live }): ReactNode {
  switch (l.state) {
    case "reading": return l.prompt_total
      ? <Sentence text={t("Reading {n} of {total} tokens", { n: SLOT, total: fmt(l.prompt_total) })} node={<Num value={l.prompt_read} />} />
      : <>{t("Reading the prompt")}</>
    case "generating": {
      const p = phaseKind(l.phase)
      if (p.kind === "tool") return <>{t("Writing a tool call: {name}", { name: p.tool })}</>
      if (p.kind === "toolDone") return <>{t("Tool call written")}</>
      if (!l.tok_s) return <>{p.kind === "thinking" ? t("Model is thinking") : p.kind === "answering" ? t("Model is answering") : t("Writing")}</>
      const text = p.kind === "thinking" ? t("Thinking at {n} tok/s", { n: SLOT }) : p.kind === "answering" ? t("Answering at {n} tok/s", { n: SLOT }) : t("Writing at {n} tok/s", { n: SLOT })
      return <Sentence text={text} node={<Num value={l.tok_s} digits={0} kind="gauge" />} />
    }
    case "unloaded": return <>{t("Model unloaded")}</>
    default: return <>{t("Ready")}</>
  }
}

/** One part of the overview: a heading that opens its page, and the few rows that answer "is it fine?". */
function Block({ title, to, children }: { title: string; to: string; children: ReactNode }) {
  return (
    <section className="border-t border-line pt-4">
      <a href={to} className="block-head flex items-baseline justify-between gap-4 no-underline" aria-label={t("Open {title}", { title })}>
        <h2 className="text-[15px] font-semibold tracking-[-0.015em]">{title}</h2>
        <span className="open-link text-[12px]">{t("Open")} <span className="arrow">→</span></span>
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
  const held = gpus.flatMap((g) => (g.throttle && g.throttle.length ? [t("GPU {n}: {why}", { n: g.index, why: g.throttle.join(", ") })] : []))
  const mcpReady = mcp?.servers.filter((s) => s.status === "ready" || s.status === "stopped").length ?? 0
  const tot = m.totals
  const totalRead = tot.prompt_tokens - (tot.reused || 0)
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
            {stale ? t("The server is not answering. Showing the last numbers it gave.") : modelLine(m)}
          </p>
          {last && (
            <p className="num mt-1 text-[13px] text-ink-3">
              {t("Last request")} · {lastDecode != null ? t("{v} tok/s decode", { v: lastDecode.toFixed(1) }) : t("no decode")}
              {split && ` · ${split.cached ? t("{read} tokens read, {cached} cached", { read: fmt(split.read), cached: fmt(split.cached) }) : t("{read} tokens read", { read: fmt(split.read) })}`} · {last.duration_s.toFixed(1)} s
            </p>
          )}
        </div>
      </section>

      <div className="grid gap-x-12 gap-y-10 lg:grid-cols-12">
        <div className="stagger space-y-10 lg:col-span-7">
          <Block title={t("Speed")} to={href("live")}>
            <Facts items={[
              [t("Decode now"), l.state === "generating" ? val(l.tok_s, (n) => n.toFixed(1), "tok/s") : <span className="text-ink-3">{t("idle")}</span>],
              [t("Last request"), last ? val(lastDecode, (n) => n.toFixed(1), "tok/s") : NOT_MEASURED],
              [t("Prefill, last"), last ? val(last.prefill?.tok_s_mean ?? last.prefill_tok_s, (n) => n.toFixed(0), "tok/s") : NOT_MEASURED],
            ]} />
            {recentSpeeds.some((v) => v != null) && (
              <div className="mt-4">
                <div className="mb-1 flex items-baseline justify-between text-[12px] text-ink-3">
                  <span>{t("Decode of the last {n} requests", { n: recentSpeeds.length })}</span>
                  <span className="num">{t("mean")} {val(mean(recentSpeeds), (n) => n.toFixed(1))}</span>
                </div>
                <Chart series={[{ label: "tok/s", values: recentSpeeds }]} area height={132} />
              </div>
            )}
          </Block>

          <Block title={t("Requests")} to={href("requests")}>
            <Facts items={[
              [t("Since start"), fmt(tot.requests)], [t("Read"), kfmt(totalRead)], [t("Cached"), kfmt(tot.reused || 0)], [t("Written"), kfmt(tot.output_tokens)],
              [t("Avg decode"), val(tot.decode_ms > 0 ? tot.output_tokens / (tot.decode_ms / 1000) : null, (n) => n.toFixed(1), "tok/s")],
            ]} />
            <div className="mt-3"><RequestList rows={m.requests.slice(0, 3)} empty={t("No request since the server started.")} /></div>
          </Block>
        </div>

        <div className="stagger space-y-10 lg:col-span-5">
          <Block title={t("Hardware")} to={href("hardware")}>
            {gpus.length === 0 ? <p className="text-ink-2">{t("No GPU is reporting.")}</p> : gpus.map((g) => (
              <div key={g.index} className="row-wash py-2">
                <div className="flex items-baseline justify-between gap-3 text-[13px]">
                  <a href={href("hardware", "gpu", String(g.index))} className="flex min-w-0 items-center gap-2 truncate no-underline hover:underline"><GpuOrb g={g} />{(g.name || t("GPU {n}", { n: g.index })).replace("NVIDIA GeForce ", "")}<span className="text-ink-3"> {archOf(arch, g.name)}</span></a>
                  <span className="num shrink-0 text-ink-2">{g.util != null ? `${g.util}%` : "–"} · {g.temp != null ? `${g.temp}°C` : "–"}</span>
                </div>
                <div className="mt-1.5"><Bar used={g.mem_used} total={g.mem_total} label={t("GPU {n} memory", { n: g.index })} /></div>
                <div className="num mt-1 text-[12px] text-ink-3">{gb(g.mem_used)} / {gb(g.mem_total)} GB · {link(g.pcie_gen, g.pcie_width, g.pcie_width_max, g.pcie_gen_max)}</div>
              </div>
            ))}
            <Reveal>{held.length > 0 ? <p className="mt-1 text-[13px]">{t("Held back: {list}", { list: held.join("; ") })}</p> : null}</Reveal>
            <div className="mt-3 grid grid-cols-2 gap-x-6">
              <div>
                <div className="mb-1.5 flex items-baseline justify-between text-[12px]"><span className="text-ink-3">CPU{m.engine.cpu_isa ? ` · ${m.engine.cpu_isa}` : ""}</span><span className="num text-ink-2">{hw.cpu != null ? `${hw.cpu.toFixed(0)}%` : "–"}</span></div>
                <Bar used={hw.cpu} total={100} label={t("CPU busy")} />
              </div>
              <div>
                <div className="mb-1.5 flex items-baseline justify-between text-[12px]"><span className="text-ink-3">RAM</span><span className="num text-ink-2">{hw.ram_total ? `${gb(hw.ram_used)} / ${gb(hw.ram_total)} GB` : "–"}</span></div>
                <Bar used={hw.ram_used} total={hw.ram_total} label={t("RAM used")} />
              </div>
            </div>
            {disks.filter((d) => holds === d.index).map((d) => <p key={d.index} className="mt-3 text-[12px] text-ink-3">{t("Model on {model}", { model: d.model ?? "" })} · {d.bus} {d.media}</p>)}
          </Block>

          <Reveal>
            {stats && tierSum > 0 ? (
              <Block title={t("Where the last request's experts ran")} to={last ? href("requests", last.id) : href("requests")}>
                <div className="flex h-2 w-full overflow-hidden rounded-full bg-fill" role="img" aria-label={TIERS.map(([k, n]) => `${t(n)} ${pct((stats[k] ?? 0) / tierSum)}`).join(", ")}>
                  {TIERS.map(([k], i) => <div key={k} className="bg-ink transition-[width] duration-500 ease-[var(--ease)]" style={{ width: `${((stats[k] ?? 0) / tierSum) * 100}%`, opacity: 1 - i * 0.24 }} />)}
                </div>
                <dl className="m-0 mt-3 flex flex-wrap gap-x-6 gap-y-1.5 text-[13px]">
                  {TIERS.map(([k, n], i) => (
                    <div key={k} className="flex items-center gap-2">
                      <span aria-hidden className="size-2 rounded-full bg-ink" style={{ opacity: 1 - i * 0.24 }} />
                      <dt className="text-ink-2">{t(n)}</dt><dd className="num m-0">{pct((stats[k] ?? 0) / tierSum)}</dd>
                    </div>
                  ))}
                </dl>
                {last?.decode_ms ? <p className="num mt-2 text-[12px] text-ink-3">{t("Over {time} of decode", { time: ms(last.decode_ms) })}</p> : null}
              </Block>
            ) : null}
          </Reveal>

          <div>
          <section className="border-t border-line py-2">
            <Disclosure block title={t("Model")} hint={info ? [info.variant, info.bpw != null ? `${info.bpw.toFixed(2)} bpw` : null, `${(info.bytes / 1e9).toFixed(1)} GB`].filter(Boolean).join(" · ") : undefined}>
              <div className="pb-2 pt-1"><ModelFacts info={info} kv={m.engine.kv as string | undefined} /></div>
            </Disclosure>
          </section>
          <section className="border-t border-line py-2">
            <Disclosure block title={t("This server")} hint={[m.engine.engine, m.engine.max_context ? `${fmt(Number(m.engine.max_context) / 1024)}K context` : null, mcp?.servers.length ? `${mcpReady}/${mcp.servers.length} MCP` : null].filter(Boolean).join(" · ")}>
              <div className="pb-2 pt-1">
                <Rows>
                  <Row k="Engine" v={m.engine.engine ? String(m.engine.engine) : NOT_MEASURED} />
                  <Row k="Context" v={m.engine.max_context ? <span className="num">{t("{n} tokens", { n: fmt(Number(m.engine.max_context)) })}</span> : NOT_MEASURED} />
                  <Row k={t("Drafting")} hint="MTP / suffix" v={m.engine.spec != null ? <span className="num">{String(m.engine.mtp_max ?? "–")} / {String(m.engine.lookup ?? "–")}</span> : NOT_MEASURED} />
                  <Row k="MCP" v={mcp && mcp.servers.length ? <span className="num">{t("{ready} of {total} servers · {tools} tools", { ready: mcpReady, total: mcp.servers.length, tools: fmt(mcp.tools) })}</span> : <span className="text-ink-3">{t("none configured")}</span>} />
                  <Row k={t("History kept")} v={<span className="num">{t("{n} in memory", { n: fmt(m.requests_kept) })}</span>} />
                </Rows>
                <p className="mt-2 text-[13px]"><a href={href("about")}>{t("Open About")}</a></p>
              </div>
            </Disclosure>
          </section>
          </div>
        </div>
      </div>
    </div>
  )
}
