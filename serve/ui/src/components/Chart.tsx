import { useEffect, useRef, useState } from "react"
import uPlot from "uplot"
import "uplot/dist/uPlot.min.css"

// A line chart on uPlot, coloured from the page's own tokens. It redraws only when its data changes, and it is
// destroyed with the page: nothing keeps running behind it.
export interface Series { label: string; values: (number | null)[]; color?: string }

const css = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim()

export function Chart({ series, x, height = 120, unit = "", area = false, zero = true }: {
  series: Series[]; x?: number[]; height?: number; unit?: string; area?: boolean; zero?: boolean
}) {
  const host = useRef<HTMLDivElement>(null)
  const plot = useRef<uPlot | null>(null)
  const [tip, setTip] = useState<{ v: number; x: number } | null>(null)
  const xs = x ?? series[0]?.values.map((_, i) => i) ?? []

  useEffect(() => {
    const el = host.current
    if (!el) return
    const ink = css("--ink-2") || "#666"
    const strong = css("--ink") || "#111"
    const grid = css("--line") || "#ddd"
    const palette = [strong, css("--ink-2") || "#666"]            // ink, not a hue: the accent is for focus and emphasis only
    const opts: uPlot.Options = {
      width: el.clientWidth || 300,
      height,
      padding: [8, 8, 6, 0],
      legend: { show: false },
      cursor: { show: true, points: { size: 7, fill: css("--surface") || "#fff", stroke: strong, width: 1.5 }, drag: { x: false, y: false }, x: true, y: false },
      hooks: { setCursor: [(u) => { const i = u.cursor.idx; setTip(i == null || u.data[1][i] == null ? null : { v: u.data[1][i] as number, x: u.cursor.left ?? 0 }) }] },
      // from zero unless asked otherwise: a tiny range zoomed to the frame would make noise look like swings
      scales: { x: { time: false }, y: zero ? { range: (_u: uPlot, _lo: number, hi: number) => [0, Math.max(1, hi * 1.12)] as [number, number] } : {} },
      axes: [
        { show: false },
        { stroke: ink, grid: { stroke: grid, width: 1 }, ticks: { show: false }, size: 44, font: "11px var(--font-sans)", values: (_u, v) => v.map((n) => `${+n.toFixed(1)}${unit}`) },
      ],
      series: [
        {},
        ...series.map((s, i) => ({
          label: s.label, stroke: s.color || palette[i % palette.length], width: 1.5, points: { show: false },
          // a soft wash that fades to nothing at the baseline
          fill: area && i === 0 ? (u: uPlot) => {
            const g = u.ctx.createLinearGradient(0, u.bbox.top, 0, u.bbox.top + u.bbox.height)
            g.addColorStop(0, `color-mix(in srgb, ${s.color || palette[0]} 16%, transparent)`)
            g.addColorStop(1, `color-mix(in srgb, ${s.color || palette[0]} 0%, transparent)`)
            return g
          } : undefined,
        })),
      ],
    }
    const u = new uPlot(opts, [xs, ...series.map((s) => s.values)] as uPlot.AlignedData, el)
    plot.current = u
    const ro = new ResizeObserver(() => u.setSize({ width: el.clientWidth, height }))
    ro.observe(el)
    return () => { ro.disconnect(); u.destroy(); plot.current = null }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [series.length, height, unit, area, zero])

  useEffect(() => {
    plot.current?.setData([xs, ...series.map((s) => s.values)] as uPlot.AlignedData)
  })

  return (
    <div className="relative">
      <div ref={host} className="w-full" role="img" data-points={xs.length} aria-label={series.map((s) => s.label).join(", ")} />
      {tip && (
        <div className="num pointer-events-none absolute top-0 rounded-sm bg-surface/90 px-1.5 py-0.5 text-[12px] font-medium shadow-[0_1px_8px_rgb(0_0_0/0.08)]" style={{ left: Math.min(tip.x + 52, 9999), transform: "translateX(-50%)" }}>
          {+tip.v.toFixed(1)}{unit}
        </div>
      )}
    </div>
  )
}
