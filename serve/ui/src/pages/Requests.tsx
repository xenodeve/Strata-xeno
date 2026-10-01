import { Fragment, useEffect, useState, type ReactNode } from "react"
import { Chart } from "../components/Chart"
import { Empty, Loading } from "../components/orb"
import { Disclosure } from "../components/motion"
import { Facts, NOT_MEASURED, Row, Rows, Section, ms, pct, val, when } from "../components/bits"
import { clientName, fmt } from "../lib/format"
import { attribute, overlapOpportunityMs } from "../lib/stall"
import { getKeepLeft, getRequest, getRequestPage, promptSplit, setKeep, type RequestDetail, type RequestRow } from "../lib/metrics"
import { href } from "../lib/router"
import { msg, t } from "../lib/i18n"

/** Puts elements (a bold figure, a link) into a translated sentence at its {name} places, so the sentence stays one piece for the translator. */
function rich(text: string, parts: Record<string, ReactNode>): ReactNode {
  return text.split(/(\{\w+\})/g).map((p, i) => (/^\{\w+\}$/.test(p) && p.slice(1, -1) in parts ? <Fragment key={i}>{parts[p.slice(1, -1)]}</Fragment> : p))
}

/** A figure with its spread on hover or focus: the first depth of detail. */
function Tip({ children, tip }: { children: ReactNode; tip: ReactNode }) {
  return (
    <span className="group relative inline-block outline-none" tabIndex={0}>
      <span className="cursor-help decoration-dotted decoration-ink-3 underline-offset-4 group-hover:underline group-focus:underline">{children}</span>
      <span role="tooltip" className="pointer-events-none absolute bottom-full right-0 z-20 mb-1.5 w-max max-w-64 origin-bottom-right scale-[0.98] rounded-md border border-line bg-surface px-2.5 py-1.5 text-left text-[12px] leading-snug opacity-0 shadow-[0_6px_24px_rgb(0_0_0/0.10)] transition-[opacity,transform] duration-150 group-hover:scale-100 group-hover:opacity-100 group-focus:scale-100 group-focus:opacity-100">
        {tip}
      </span>
    </span>
  )
}

const r1 = (n: number) => n.toFixed(1)

function decodeCell(r: RequestRow): ReactNode {
  const d = r.decode
  const mean = d?.tok_s_mean ?? r.decode_tok_s
  if (mean == null) return NOT_MEASURED
  const tip = d && d.windows > 0
    ? <>{rich(t("min {min} · max {max} · mean {mean} tok/s"), { min: <b className="num">{r1(d.tok_s_min!)}</b>, max: <b className="num">{r1(d.tok_s_max!)}</b>, mean: <b className="num">{r1(mean)}</b> })}<br /><span className="text-ink-2">{t("over {n} windows of 16 tokens", { n: d.windows })}</span></>
    : <>{rich(t("mean {mean} tok/s"), { mean: <b className="num">{r1(mean)}</b> })}<br /><span className="text-ink-2">{t("fewer than 16 tokens: no window to compare")}</span></>
  return <Tip tip={tip}><span className="num">{r1(mean)}</span><span className="text-ink-2"> tok/s</span></Tip>
}

function prefillCell(r: RequestRow): ReactNode {
  const p = r.prefill
  if (!p) return r.prefill_tok_s != null ? <span className="num">{r.prefill_tok_s.toFixed(0)}<span className="text-ink-2"> tok/s</span></span> : NOT_MEASURED
  const tip = p.chunks === 1
    ? <>{rich(t("1 chunk · {mean} tok/s"), { mean: <b className="num">{p.tok_s_mean!.toFixed(0)}</b> })}</>
    : <>{rich(t("{n} chunks: min {min} · max {max} · mean {mean} tok/s", { n: p.chunks }), { min: <b className="num">{p.tok_s_min!.toFixed(0)}</b>, max: <b className="num">{p.tok_s_max!.toFixed(0)}</b>, mean: <b className="num">{p.tok_s_mean!.toFixed(0)}</b> })}</>
  return <Tip tip={tip}><span className="num">{p.tok_s_mean!.toFixed(0)}</span><span className="text-ink-2"> tok/s</span></Tip>
}

export function RequestList({ rows, empty }: { rows: RequestRow[]; empty: ReactNode }) {
  if (!rows.length) return typeof empty === "string" ? <p><Empty>{empty}</Empty></p> : <>{empty}</>
  return (
    <ul className="m-0 list-none p-0">
      {rows.map((r) => (
        <li key={r.id} className="row-in border-b border-line last:border-0">
          <div className="row-wash flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1 py-2.5">
            <a href={href("requests", r.id)} className="min-w-0 flex-1 basis-56 no-underline">
              <div className="truncate">{r.preview || <span className="text-ink-3">{t("(no text)")}</span>}</div>
              <div className="mt-0.5 truncate text-[12px] text-ink-2">
                <span className="num">{when(r.time)}</span> · {r.dialect === "anthropic" ? "Anthropic" : r.dialect === "openai" ? "OpenAI" : t("unknown API")}
                {r.client && ` · ${clientName(r.client)}`}{r.tools.length > 0 && ` · ${t("{n} tools", { n: r.tools.length })}`}
                {r.finish !== "stop" && r.finish !== "length" && ` · ${r.finish}`}
              </div>
            </a>
            <div className="flex shrink-0 items-baseline gap-5 text-[13px]">
              {(() => {
                const sp = promptSplit(r.prompt_tokens, r.reused)
                return (
                  <span className="num text-ink-2" title={t("Prompt tokens the engine read (prefill) and tokens it already held in the conversation cache, then the answer")}>
                    {t("{n} read", { n: fmt(sp.read) })}{sp.cached > 0 && <> · {t("{n} cached", { n: fmt(sp.cached) })}</>} → {fmt(r.output_tokens)}
                  </span>
                )
              })()}
              <span title={t("Prefill")}>{prefillCell(r)}</span>
              <span title={t("Decode")}>{decodeCell(r)}</span>
            </div>
          </div>
        </li>
      ))}
    </ul>
  )
}

export function Requests() {
  const [rows, setRows] = useState<RequestRow[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const SIZE = 50

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    getRequestPage(page, SIZE)
      .then((p) => { if (!cancelled) { setRows((x) => (page === 0 ? p.items : [...x, ...p.items])); setTotal(p.total); setError(null) } })
      .catch((e: Error) => !cancelled && setError(e.message))
      .finally(() => !cancelled && setLoading(false))
    return () => { cancelled = true }
  }, [page])

  return (
    <div>
      <h1 className="page-title">{t("Requests")}</h1>
      <p className="page-sub">
        {total ? t("{n} kept on this PC.", { n: fmt(total) }) : t("Every finished request is kept here.")} {t("Hover a speed for its spread.")}
      </p>
      {error && <p className="mt-4 text-bad">{error}</p>}
      <KeepPrompts />
      <p className="mt-2 text-[13px] text-ink-2">{rich(t("Have an engine timeline ({code})? {link}."), { code: <code>STRATA_TIMELINE</code>, link: <a href={href("requests", "trace")}>{t("Open it in the viewer")}</a> })}</p>
      <div className="mt-4">
        <RequestList rows={rows} empty={loading ? <Loading design="working">{t("Loading…")}</Loading> : t("No request yet.")} />
      </div>
      {rows.length < total && (
        <button type="button" disabled={loading} onClick={() => setPage((p) => p + 1)} className="mt-4 h-8 rounded-sm bg-fill px-3 text-[13px] font-medium transition-colors hover:bg-fill-2 disabled:opacity-40">
          {t("Show more")}
        </button>
      )}
    </div>
  )
}

const TIERS: [string, string][] = [
  ["tier_primary", msg("Primary GPU's VRAM")], ["tier_secondary", msg("Secondary GPU's VRAM")],
  ["tier_pcie", msg("RAM, copied over PCIe to a GPU")], ["tier_cpu", msg("RAM, computed on the CPU")],
]

const STAGES: [string, string, string?][] = [
  ["ms_gpu_wait", msg("Waiting for the GPU to ring a layer")], ["ms_pool", msg("Inside the CPU pool (per layer)")],
  ["ms_plan", msg("Plan"), msg("of the pool time")], ["ms_actq", msg("Quantize activations"), msg("of the pool time")],
  ["ms_jobs", msg("Dispatch jobs"), msg("of the pool time")], ["ms_cpu", msg("CPU expert compute"), msg("of the pool time")],
  ["ms_stage", msg("Staging the window")], ["ms_commit", msg("Commit and emit")], ["ms_draft", msg("MTP draft")],
]

export function RequestDetailPage({ id }: { id: string }) {
  const [d, setD] = useState<RequestDetail | null | undefined>(undefined)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => { getRequest(id).then(setD).catch((e: Error) => setError(e.message)) }, [id])

  if (error) return <p className="text-bad">{error}</p>
  if (d === undefined) return <Loading design="working">{t("Loading…")}</Loading>
  if (d === null) return <div><p>{t("This request is not in the history.")}</p><p><a href={href("requests")}>{t("Back to Requests")}</a></p></div>

  const r = d.summary
  const det = d.detail
  const stats = det?.stats ?? r.stats
  const split = promptSplit(r.prompt_tokens, r.reused)
  const tierTotal = stats ? TIERS.reduce((a, [k]) => a + (stats[k] ?? 0), 0) : 0

  return (
    <div>
      <p className="mb-3 text-[13px]"><a href={href("requests")}>← {t("Requests")}</a></p>
      <h1 className="page-title line-clamp-2 !text-[clamp(22px,2.8vw,30px)] [overflow-wrap:anywhere]">{r.preview || t("(no text)")}</h1>
      <p className="page-sub"><span className="num">{when(r.time)}</span> · {r.dialect}{r.client && ` · ${clientName(r.client)}`}</p>

      <Section title={t("Overview")}>
        <Facts items={[
          [t("Prefill, read"), <>{t("{n} tokens", { n: fmt(split.read) })}</>],
          [t("From the cache"), <>{t("{n} tokens", { n: fmt(split.cached) })}{split.cached > 0 && <span className="text-ink-2"> · {pct(split.cachedShare)}</span>}</>],
          [t("Answer"), <>{t("{n} tokens", { n: fmt(r.output_tokens) })}</>],
          [t("Took"), val(r.duration_s, (n) => n.toFixed(1), "s")],
          [t("Ended"), r.finish],
          [t("Expert cache hit"), val(r.hit_rate, (n) => pct(n, 1))],
        ]} />
        {(det?.prompt || r.tools.length > 0) && (
          <Rows>
            {det?.prompt && <Row k={t("Prompt")} hint={t("kept for replay")} v={<span className="num">{t("{n} messages", { n: det.prompt.length })}</span>} />}
            {r.tools.length > 0 && <Row k={t("Tools offered")} v={<span className="text-ink-2">{r.tools.join(", ")}</span>} />}
          </Rows>
        )}
      </Section>

      <Section title={t("Prefill")} aside={t("the {n} tokens read, per chunk", { n: fmt(split.read) })}>
        {r.prefill ? (
          <>
            <Facts items={[
              [t("Chunks"), r.prefill.chunks === 1 ? t("1 chunk") : String(r.prefill.chunks)],
              ...(r.prefill.chunks > 1 ? [[t("Slowest"), val(r.prefill.tok_s_min, (n) => n.toFixed(0), "tok/s")], [t("Fastest"), val(r.prefill.tok_s_max, (n) => n.toFixed(0), "tok/s")]] as [string, ReactNode][] : []),
              [t("Mean"), val(r.prefill.tok_s_mean, (n) => n.toFixed(0), "tok/s")],
            ]} />
            {det?.prefill_chunks && det.prefill_chunks.length > 1 && (
              <div className="mt-4">
                <div className="mb-1 text-[12px] text-ink-3">{t("Speed of each chunk, tok/s")}</div>
                <Chart series={[{ label: "tok/s", values: det.prefill_chunks.map(([tk, m]) => (m > 0 ? tk / (m / 1000) : null)) }]} height={96} />
              </div>
            )}
          </>
        ) : <p className="text-ink-2">{split.read === 0 && r.prompt_tokens ? t("Nothing was read: the whole prompt came from the conversation cache.") : <>{t("Not measured.")}</>}</p>}
      </Section>

      <Section title={t("Decode")} aside={t("over a sliding 16-token window")}>
        {r.decode ? (
          <>
            <Facts items={[
              [t("Mean"), val(r.decode.tok_s_mean, r1, "tok/s")],
              ...(r.decode.windows > 0 ? [[t("Slowest window"), val(r.decode.tok_s_min, r1, "tok/s")], [t("Fastest window"), val(r.decode.tok_s_max, r1, "tok/s")]] as [string, ReactNode][] : []),
            ]} />
            {det?.decode_series && det.decode_series.length > 1 && (
              <div className="mt-4">
                <div className="mb-1 text-[12px] text-ink-3">{t("Trend over the answer, tok/s")}</div>
                <Chart series={[{ label: "tok/s", values: det.decode_series }]} area height={120} />
              </div>
            )}
          </>
        ) : <p className="text-ink-2">{t("No tokens were written.")}</p>}
        {d.detail_state === "deleted" && <p className="mt-3 text-[13px] text-ink-2">{t("Detail deleted: the history keeps 2 GB of detail and this one was the oldest. The summary stays.")}</p>}
      </Section>

      <Section title={t("Where the decode time went")} aside={stats ? undefined : t("needs the engine's STATS line")}>
        {!stats ? <p className="text-ink-2">{t("Not measured: this engine does not report its per-request counters yet.")}</p> : (
          <>
            <div className="text-[12px] text-ink-2">{t("Routed expert entries by where they ran")}</div>
            <Rows>
              {TIERS.map(([k, label]) => (
                <Row key={k} k={t(label)} v={<span className="num">{fmt(stats[k] ?? 0)}{tierTotal > 0 && <span className="text-ink-2"> · {pct((stats[k] ?? 0) / tierTotal, 0)}</span>}</span>} />
              ))}
              <Row k={t("Loaded from the SSD into RAM")} v={<span className="num">{fmt(stats.nvme_loads ?? 0)}{(stats.nvme_loads ?? 0) > 0 && <span className="text-ink-2"> · {ms(stats.nvme_ms ?? 0)}</span>}</span>} />
            </Rows>
            <div className="mt-4">
              <Disclosure title={t("Host time per stage")} hint={r.decode_ms ? t("of {total}", { total: ms(r.decode_ms) }) : undefined}>
              <Rows>
              {STAGES.map(([k, label, hint]) => stats[k] != null && (
                <Row key={k} k={t(label)} hint={hint && t(hint)} v={<span className="num">{ms(stats[k])}{r.decode_ms ? <span className="text-ink-2"> · {pct(stats[k] / r.decode_ms, 0)}</span> : null}</span>} />
              ))}
              </Rows>
              </Disclosure>
              <Disclosure title={t("Who waited for whom")} hint={t("a dependency definition")}>
                <Stall stats={stats} decodeMs={r.decode_ms} />
              </Disclosure>
            </div>
            <p className="mt-3 text-[12px] text-ink-3">{t("Counted by the engine over {n} verify windows. Stages overlap, so they do not add up to the total.", { n: fmt(stats.windows ?? 0) })}</p>
          </>
        )}
      </Section>
    </div>
  )
}

const KIND_TONE = { busy: "bg-ink", wait: "bg-ink-2", idle: "bg-fill-2" } as const

/** Who waited for whom: each resource's decode time split into busy, waiting (named) and idle. */
function Stall({ stats, decodeMs }: { stats: Record<string, number>; decodeMs: number | null }) {
  const a = attribute(stats, decodeMs)
  if (!a) return null
  const opp = overlapOpportunityMs(stats)
  return (
    <div className="pb-2 pt-1">
      <p className="text-[12px] text-ink-3">{t("A dependency definition from the engine's counters: idle time is put down to what the host was waiting on. It is not a hardware measurement; cycle-level stalls need the Nsight capture.")}</p>
      {a.map((res) => (
        <div key={res.resource} className="mt-3">
          <div className="mb-1 text-[13px] font-medium">{t(res.resource)}</div>
          <div className="flex h-2 w-full overflow-hidden rounded-full bg-fill" role="img" aria-label={t("{label}: {value}", { label: t(res.resource), value: res.buckets.map((b) => t("{label} {share}", { label: t(b.label), share: pct(b.ms / res.total) })).join(", ") })}>
            {res.buckets.map((b) => <div key={b.key} className={KIND_TONE[b.kind]} style={{ width: `${(b.ms / res.total) * 100}%`, opacity: b.kind === "busy" ? 1 : b.kind === "wait" ? 0.55 : 0.35 }} title={t("{label}: {value}", { label: t(b.label), value: ms(b.ms) })} />)}
          </div>
          <Rows>
            {res.buckets.map((b) => <Row key={b.key} k={t(b.label)} v={<span className="num">{ms(b.ms)}<span className="text-ink-2"> · {pct(b.ms / res.total)}</span></span>} />)}
            {res.resource === "CPU pool" && <Row k={t("Wake, park and repark of the pool")} v={NOT_MEASURED} />}
          </Rows>
        </div>
      ))}
      <Rows>
        <Row k={t("Overlap opportunity")} hint={t("upper bound: SSD time only")} v={opp == null ? NOT_MEASURED : <span className="num">{ms(opp)}</span>} />
        <Row k={t("PCIe copy waits")} hint={t("not in the bound")} v={NOT_MEASURED} />
      </Rows>
    </div>
  )
}

/** Q8: only the numbers, the first 200 characters, the tool names and the client are kept. The whole prompt is kept
 *  only when asked for, for the next few requests, and goes to that request's detail file. */
function KeepPrompts() {
  const [left, setLeft] = useState<number | null>(null)
  const [err, setErr] = useState<string | null>(null)
  useEffect(() => { getKeepLeft().then(setLeft).catch(() => {}) }, [])
  const set = (n: number) => setKeep(n).then(setLeft).then(() => setErr(null)).catch((e: Error) => setErr(e.message))
  return (
    <div className="mt-4 flex flex-wrap items-center gap-x-3 gap-y-1.5 text-[13px] text-ink-2">
      <span>
        {left ? rich(left > 1 ? t("The full prompt of the next {n} requests is being kept, for replay.") : t("The full prompt of the next {n} request is being kept, for replay."), { n: <b className="num text-ink">{left}</b> }) : t("Prompts are kept as 200 characters. Keep the whole prompt of the next:")}
      </span>
      {[1, 5, 10].map((n) => (
        <button key={n} type="button" onClick={() => void set(n)} className="h-7 rounded-sm bg-fill px-2.5 font-medium text-ink transition-colors hover:bg-fill-2">{n}</button>
      ))}
      {left ? <button type="button" onClick={() => void set(0)} className="h-7 rounded-sm px-2.5 transition-colors hover:bg-hover">{t("Stop keeping")}</button> : null}
      {err && <span className="text-bad">{err}</span>}
    </div>
  )
}
