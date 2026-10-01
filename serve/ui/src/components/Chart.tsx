import { useEffect, useRef } from "react"
import uPlot from "uplot"
import "uplot/dist/uPlot.min.css"

// A line chart on uPlot, coloured from the page's own tokens. It redraws only when its data changes, and it is
// destroyed with the page: nothing keeps running behind it.
export interface Series { label: string; values: (number | null)[]; color?: string }

const css = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim()

export function Chart({ series, x, height = 120, unit = "", area = false }: {
  series: Series[]; x?: number[]; height?: number; unit?: string; area?: boolean
}) {
  const host = useRef<HTMLDivElement>(null)
  const plot = useRef<uPlot | null>(null)
  const xs = x ?? series[0]?.values.map((_, i) => i) ?? []

  useEffect(() => {
    const el = host.current
    if (!el) return
    const ink = css("--ink-2") || "#666"
    const grid = css("--line") || "#ddd"
    const palette = [css("--accent") || "#06f", css("--ink-2") || "#666"]
    const opts: uPlot.Options = {
      width: el.clientWidth || 300,
      height,
      padding: [8, 8, 6, 0],
      legend: { show: false },
      cursor: { show: true, points: { size: 6 }, drag: { x: false, y: false } },
      scales: { x: { time: false } },
      axes: [
        { show: false },
        { stroke: ink, grid: { stroke: grid, width: 1 }, ticks: { show: false }, size: 44, font: "11px var(--font-sans)", values: (_u, v) => v.map((n) => `${+n.toFixed(1)}${unit}`) },
      ],
      series: [
        {},
        ...series.map((s, i) => ({
          label: s.label, stroke: s.color || palette[i % palette.length], width: 1.5, points: { show: false },
          fill: area && i === 0 ? (s.color || palette[0]) + "22" : undefined,
        })),
      ],
    }
    const u = new uPlot(opts, [xs, ...series.map((s) => s.values)] as uPlot.AlignedData, el)
    plot.current = u
    const ro = new ResizeObserver(() => u.setSize({ width: el.clientWidth, height }))
    ro.observe(el)
    return () => { ro.disconnect(); u.destroy(); plot.current = null }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [series.length, height, unit, area])

  useEffect(() => {
    plot.current?.setData([xs, ...series.map((s) => s.values)] as uPlot.AlignedData)
  })

  return <div ref={host} className="w-full" role="img" aria-label={series.map((s) => s.label).join(", ")} />
}
