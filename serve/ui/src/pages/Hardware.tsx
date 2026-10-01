import { Empty, Loading, Orb } from "../components/orb"
import { gpuDesign } from "../lib/orbs"
import { useEffect, useRef, type ReactNode } from "react"
import { Chart } from "../components/Chart"
import { NOT_MEASURED, Row, Rows, Section, val } from "../components/bits"
import { fmt, gb } from "../lib/format"
import { archOf, parseArch } from "../lib/arch"
import { useMetrics, type Gpu, type Metrics } from "../lib/metrics"
import { href } from "../lib/router"
import { msg, t } from "../lib/i18n"

interface Disk { index: number; model: string | null; bus: string | null; media: string | null; size_gb: number | null }
interface DiskRate { index: number; read_mb: number | null; write_mb: number | null; read_ms_op: number | null }
type Card = Gpu & {
  name?: string; power_limit?: number; pcie_gen?: number; pcie_gen_max?: number; pcie_width?: number; pcie_width_max?: number
  pcie_rx_mb?: number; pcie_tx_mb?: number; sm_clock?: number; mem_clock?: number; throttle?: string[] | null
}

export const cards = (m: Metrics): Card[] => (m.hardware.gpus as Card[] | undefined) ?? []
export const disksOf = (m: Metrics): Disk[] => ((m.hardware_static as { storage?: { disks?: Disk[] } }).storage?.disks) ?? []
export const modelDisk = (m: Metrics): number | null => ((m.hardware_static as { storage?: { model_disk?: number | null } }).storage?.model_disk) ?? null
export const rates = (m: Metrics): DiskRate[] | null => ((m.hardware as { disks?: DiskRate[] | null }).disks) ?? null

export const link = (gen?: number | null, w?: number | null, wmax?: number | null, genMax?: number | null) =>
  gen == null || w == null ? NOT_MEASURED
    : <span className="num">Gen {gen} ×{w}{(wmax != null && wmax !== w) || (genMax != null && genMax !== gen) ? <span className="text-ink-2"> {t("of Gen {gen} ×{w}", { gen: genMax ?? gen, w: wmax ?? w })}</span> : ""}</span>

/** A card's orb says what the card is doing: at rest, at work, working hard, or held back by a limit. */
export function GpuOrb({ g }: { g: Card }) {
  const look = gpuDesign(g)
  const held = !!g.throttle?.length
  return <Orb design={look.design} size={20} moving={look.moving} speed={look.speed} fps={look.fps} label={held ? t("Held back") : look.design === "weaving" ? t("Working hard") : look.design === "searching" ? t("Idle") : t("Working")} />
}

export function Bar({ used, total, label }: { used?: number | null; total?: number | null; label: string }) {
  const f = used != null && total ? Math.min(1, used / total) : 0
  return (
    <div role="meter" aria-label={label} aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(f * 100)} className="h-1 w-full overflow-hidden rounded-full bg-fill">
      <div className="h-full rounded-full bg-ink-2 transition-[width] duration-500 ease-[var(--ease)]" style={{ width: `${f * 100}%` }} />
    </div>
  )
}

function Device({ title, to, children, sub, lead }: { title: ReactNode; to?: string; children: ReactNode; sub?: ReactNode; lead?: ReactNode }) {
  return (
    <div className="border-b border-line py-3 last:border-0">
      <div className="flex items-center gap-3">
        {lead}
        <div className="min-w-0">
          <div className="font-medium">{to ? <a href={to} className="no-underline hover:underline">{title}</a> : title}</div>
          {sub && <div className="text-[12px] text-ink-2">{sub}</div>}
        </div>
      </div>
      <div className="mt-2">{children}</div>
    </div>
  )
}

export function Hardware() {
  const { data: m, error } = useMetrics()
  if (!m) return <Loading error={error} />
  const hw = m.hardware
  const arch = parseArch(String(m.engine.gpu_arch ?? ""))
  const ds = disksOf(m)
  const dr = rates(m)
  const holds = modelDisk(m)
  const st = m.hardware_static

  return (
    <div>
      <h1 className="page-title">{t("Hardware")}</h1>
      <p className="page-sub">{t("What each part is doing for the model. A figure the engine does not measure says so.")}</p>

      <Section title={t("GPUs")}>
        {cards(m).length === 0 ? <p><Empty>{t("No GPU is reporting.")}</Empty></p> : cards(m).map((g) => (
          <Device key={g.index} lead={<GpuOrb g={g} />} title={g.name || t("GPU {n}", { n: g.index })} to={href("hardware", "gpu", String(g.index))}
            sub={<>{archOf(arch, g.name) ?? t("architecture unknown")}{g.throttle && g.throttle.length > 0 && <span className="text-ink"> · {t("held back by {what}", { what: g.throttle.join(", ") })}</span>}</>}>
            <Bar used={g.mem_used} total={g.mem_total} label={t("GPU {n} memory", { n: g.index })} />
            <div className="num mt-1.5 flex flex-wrap justify-between gap-x-4 text-[13px] text-ink-2">
              <span>{g.util != null ? t("{n}% busy", { n: g.util }) : "–"}</span>
              <span>{gb(g.mem_used)} / {gb(g.mem_total)} GB</span>
              <span>{g.temp != null ? `${g.temp}°C` : "–"}</span>
              <span>{g.power != null ? `${g.power.toFixed(0)} W` : "–"}</span>
              <span>{link(g.pcie_gen, g.pcie_width, g.pcie_width_max, g.pcie_gen_max)}</span>
            </div>
          </Device>
        ))}
      </Section>

      <Section title="CPU">
        <Device title={st.cpu_name || "CPU"} sub={st.threads ? (st.cores ? t("{cores} cores, {threads} threads", { cores: st.cores, threads: st.threads }) : t("{threads} threads", { threads: st.threads })) : undefined}>
          <Rows>
            <Row k={t("Expert kernels in use")} hint={t("Q2_0 rows")} v={m.engine.cpu_isa ? String(m.engine.cpu_isa) : NOT_MEASURED} />
            <Row k={t("Busy")} v={val(hw.cpu, (n) => n.toFixed(0), "%")} />
            <Row k={t("Pool workers")} v={m.engine.pool_workers != null ? <span className="num">{String(m.engine.pool_workers)}</span> : NOT_MEASURED} />
          </Rows>
        </Device>
      </Section>

      <Section title={t("Memory")}>
        <Device title="RAM">
          <Bar used={hw.ram_used} total={hw.ram_total} label={t("RAM used")} />
          <Rows>
            <Row k={t("Used")} v={<span className="num">{gb(hw.ram_used)} / {gb(hw.ram_total)} GB</span>} />
            <Row k={t("Bandwidth")} hint={t("the CPU pool's own, per request")} v={NOT_MEASURED} />
          </Rows>
        </Device>
      </Section>

      <Section title={t("Storage")}>
        {ds.length === 0 ? <p className="text-ink-2">{t("The disks could not be listed on this system.")}</p> : ds.map((d) => {
          const r = dr?.find((x) => x.index === d.index)
          return (
            <Device key={d.index} title={d.model || t("Disk {n}", { n: d.index })} to={href("hardware", "ssd", String(d.index))}
              sub={<>{[d.bus, d.media, d.size_gb ? `${fmt(d.size_gb, 0)} GB` : null].filter(Boolean).join(" · ")}{holds === d.index && <span className="text-ink"> · {t("holds the model")}</span>}</>}>
              <div className="num text-[13px] text-ink-2">
                {r ? <>{t("read {r} MB/s · write {w} MB/s", { r: fmt(r.read_mb, 1), w: fmt(r.write_mb, 1) })}</> : <>{t("throughput")} <span className="text-ink-3">{dr === null ? t("not measured: psutil is not installed here") : t("not measured")}</span></>}
              </div>
            </Device>
          )
        })}
      </Section>
    </div>
  )
}

const HISTORY: [string, string, string][] = [
  ["util", msg("Busy, %"), ""], ["mem_used", msg("VRAM, GB"), "gb"], ["temp", msg("Temperature, °C"), ""], ["power", msg("Power, W"), ""],
  ["pcie_rx_mb", msg("PCIe into the card (NVML), MB/s"), ""], ["pcie_tx_mb", msg("PCIe out of the card (NVML), MB/s"), ""],
]

export function GpuPage({ n }: { n: string }) {
  const { data: m, error } = useMetrics()
  if (!m) return <Loading error={error} />
  const g = cards(m).find((c) => String(c.index) === n)
  if (!g) return <div><p>{t("This card is not reporting.")}</p><p><a href={href("hardware")}>{t("Back to Hardware")}</a></p></div>
  const arch = archOf(parseArch(String(m.engine.gpu_arch ?? "")), g.name)

  return (
    <div>
      <p className="mb-3 text-[13px]"><a href={href("hardware")}>← Hardware</a></p>
      <h1 className="page-title">{g.name || t("GPU {n}", { n: g.index })}</h1>
      <p className="page-sub">{t("NVML index {n}", { n: g.index })}{arch && ` · ${arch}`}</p>

      <Section title={t("Now")}>
        <Rows>
          <Row k={t("Busy")} v={val(g.util, (v) => v.toFixed(0), "%")} />
          <Row k="VRAM" v={g.mem_total ? <span className="num">{gb(g.mem_used)} / {gb(g.mem_total)} GB</span> : NOT_MEASURED} />
          <Row k={t("Temperature")} v={val(g.temp, (v) => v.toFixed(0), "°C")} />
          <Row k={t("Power")} v={g.power != null ? <span className="num">{g.power.toFixed(0)}{g.power_limit ? ` ${t("of {n}", { n: g.power_limit.toFixed(0) })}` : ""} <span className="text-ink-2">W</span></span> : NOT_MEASURED} />
          <Row k={t("Clocks")} v={g.sm_clock != null ? <span className="num">{t("{core} MHz core · {mem} MHz memory", { core: g.sm_clock, mem: g.mem_clock ?? "" })}</span> : NOT_MEASURED} />
          <Row k={t("Held back by")} v={g.throttle == null ? NOT_MEASURED : g.throttle.length ? g.throttle.join(", ") : <span className="text-ink-2">{t("nothing")}</span>} />
          <Row k={t("PCIe link")} hint={t("drops when idle")} v={link(g.pcie_gen, g.pcie_width, g.pcie_width_max, g.pcie_gen_max)} />
          <Row k={t("PCIe traffic")} hint={t("NVML, all uses of the link")} v={g.pcie_rx_mb != null ? <span className="num">{t("{rx} in · {tx} out", { rx: fmt(g.pcie_rx_mb, 0), tx: fmt(g.pcie_tx_mb, 0) })} <span className="text-ink-2">MB/s</span></span> : NOT_MEASURED} />
        </Rows>
      </Section>

      <Section title={t("Last minute")}>
        <div className="space-y-4">
          {HISTORY.map(([k, label, unit]) => {
            const v = m.history[`gpu${g.index}_${k}`]
            if (!v || !v.some((x) => x != null)) return null
            return (
              <div key={k}>
                <div className="mb-1 text-[12px] text-ink-2">{t(label)}</div>
                <Chart series={[{ label: t(label), values: unit === "gb" ? v.map((x) => (x == null ? null : x / 1073741824)) : v }]} height={80} area />
              </div>
            )
          })}
        </div>
      </Section>

      <Section title={t("Not measured yet")}>
        <Rows>
          <Row k={t("Copies over PCIe by the engine")} hint={t("count, bytes, ms, GB/s per request")} v={NOT_MEASURED} />
          <Row k={t("Kernel latency")} hint={t("opt-in")} v={NOT_MEASURED} />
        </Rows>
      </Section>
    </div>
  )
}

export function SsdPage({ n }: { n: string }) {
  const { data: m, error } = useMetrics()
  const trail = useRef<{ read: (number | null)[]; write: (number | null)[]; lat: (number | null)[] }>({ read: [], write: [], lat: [] })
  const idx = Number(n)
  const r = m ? rates(m)?.find((x) => x.index === idx) : undefined
  useEffect(() => {
    if (!m) return
    const tr = trail.current
    const push = (a: (number | null)[], v: number | null) => { a.push(v); if (a.length > 120) a.shift() }
    push(tr.read, r?.read_mb ?? null); push(tr.write, r?.write_mb ?? null); push(tr.lat, r?.read_ms_op ?? null)
  }, [m?.time]) // eslint-disable-line react-hooks/exhaustive-deps
  if (!m) return <Loading error={error} />
  const d = disksOf(m).find((x) => x.index === idx)
  if (!d) return <div><p>{t("This disk is not listed.")}</p><p><a href={href("hardware")}>Back to Hardware</a></p></div>
  const dr = rates(m)

  return (
    <div>
      <p className="mb-3 text-[13px]"><a href={href("hardware")}>← Hardware</a></p>
      <h1 className="page-title">{d.model || t("Disk {n}", { n: d.index })}</h1>
      <p className="page-sub">{t("Disk {n}", { n: d.index })}{modelDisk(m) === d.index && ` · ${t("holds the model")}`}</p>

      <Section title={t("The disk")}>
        <Rows>
          <Row k={t("Bus")} v={d.bus ?? NOT_MEASURED} />
          <Row k={t("Kind")} v={d.media ?? NOT_MEASURED} />
          <Row k={t("Size")} v={d.size_gb ? <span className="num">{fmt(d.size_gb, 0)} GB</span> : NOT_MEASURED} />
          <Row k={t("PCIe link")} hint={t("needs the device's own link register")} v={NOT_MEASURED} />
          <Row k={t("Device latency")} hint={t("needs a boot microbench")} v={NOT_MEASURED} />
        </Rows>
      </Section>

      <Section title={t("What the OS counted")} aside={t("queue included")}>
        {dr === null ? <p className="text-ink-2">{t("Not measured: psutil is not installed on this server, so the OS's per-disk counters are not read.")}</p> : !r ? <p className="text-ink-2">{t("No counter for this disk.")}</p> : (
          <>
            <Rows>
              <Row k={t("Read")} v={val(r.read_mb, (v) => v.toFixed(1), "MB/s")} />
              <Row k={t("Write")} v={val(r.write_mb, (v) => v.toFixed(1), "MB/s")} />
              <Row k={t("Time per read")} v={val(r.read_ms_op, (v) => v.toFixed(2), "ms")} />
            </Rows>
            <div className="mt-3 space-y-4">
              <div><div className="mb-1 text-[12px] text-ink-2">{t("Read, MB/s")}</div><Chart series={[{ label: t("read"), values: trail.current.read }]} height={80} area /></div>
              <div><div className="mb-1 text-[12px] text-ink-2">{t("Time per read, ms")}</div><Chart series={[{ label: "ms", values: trail.current.lat }]} height={80} /></div>
            </div>
          </>
        )}
      </Section>
    </div>
  )
}
