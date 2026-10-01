import { useEffect, useMemo, useRef, useState, type DragEvent, type PointerEvent as RPointerEvent } from "react"
import { Row, Rows, Section } from "../components/bits"
import { fmt } from "../lib/format"
import { href } from "../lib/router"
import { closeTrace, parseTrace, spansIn, topNames, type Trace } from "../lib/trace"

const LABEL_W = 150, ROW = 20, AXIS = 22, BAR = 14

// A colour per span name, evenly spread: names must be told apart in a lane of thousands of spans. Charts may use hue.
const hue = (id: number) => (id * 47) % 360
const us = (n: number) => (n >= 1e6 ? `${(n / 1e6).toFixed(2)} s` : n >= 1e3 ? `${(n / 1e3).toFixed(n >= 1e5 ? 0 : 2)} ms` : `${n.toFixed(n < 10 ? 2 : 0)} µs`)

interface Hover { x: number; y: number; name: string; lane: string; dur: number; at: number; args?: Record<string, unknown> }

/** The engine's pipeline timeline (STRATA_TIMELINE, #33): lanes of spans, zoom and pan, what the window spent its time on. */
export function TracePage() {
  const [trace, setTrace] = useState<Trace | null>(null)
  const [source, setSource] = useState<{ name: string; text: string } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [view, setView] = useState<[number, number]>([0, 1])
  const [query, setQuery] = useState("")
  const [hover, setHover] = useState<Hover | null>(null)
  const [width, setWidth] = useState(900)
  const [busy, setBusy] = useState(false)
  const box = useRef<HTMLDivElement>(null)
  const canvas = useRef<HTMLCanvasElement>(null)
  const drag = useRef<{ x: number; a: number; b: number } | null>(null)

  const open = async (file: File) => {
    setBusy(true); setError(null)
    try {
      const text = await file.text()
      const t = parseTrace(text)
      if (!t.count) throw new Error("There is no span in this file.")
      setTrace(t); setSource({ name: file.name, text }); setView([t.t0, t.t1]); setHover(null)
    } catch (e) { setError((e as Error).message) } finally { setBusy(false) }
  }
  const onDrop = (e: DragEvent) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) void open(f) }

  useEffect(() => {
    const el = box.current
    if (!el) return
    const ro = new ResizeObserver(() => setWidth(Math.max(320, el.clientWidth)))
    ro.observe(el); setWidth(Math.max(320, el.clientWidth))
    return () => ro.disconnect()
  }, [trace])

  const height = trace ? AXIS + trace.lanes.length * ROW : 0
  const needle = query.trim().toLowerCase()
  const matches = useMemo(() => (trace && needle ? new Set(trace.names.flatMap((n, i) => (n.toLowerCase().includes(needle) ? [i] : []))) : null), [trace, needle])

  useEffect(() => {
    const c = canvas.current
    if (!c || !trace) return
    const dpr = devicePixelRatio || 1
    c.width = width * dpr; c.height = height * dpr
    const g = c.getContext("2d")!
    g.setTransform(dpr, 0, 0, dpr, 0, 0)
    const css = getComputedStyle(document.documentElement)
    const ink2 = css.getPropertyValue("--ink-2").trim() || "#666", line = css.getPropertyValue("--line").trim() || "#ddd"
    const dark = css.colorScheme.includes("dark") && !css.colorScheme.includes("light dark")
    const [a, b] = view, plotW = width - LABEL_W, k = plotW / (b - a)
    g.clearRect(0, 0, width, height)
    g.font = "11px ui-sans-serif, system-ui, sans-serif"; g.textBaseline = "middle"
    // the time axis: ticks at a round step
    const step = 10 ** Math.floor(Math.log10((b - a) / 6)) * ([1, 2, 5].find((m) => ((b - a) / 6) / 10 ** Math.floor(Math.log10((b - a) / 6)) <= m) ?? 10)
    g.fillStyle = ink2; g.strokeStyle = line; g.lineWidth = 1
    for (let t = Math.ceil((a - trace.t0) / step) * step + trace.t0; t < b; t += step) {
      const x = LABEL_W + (t - a) * k
      g.beginPath(); g.moveTo(x + 0.5, AXIS - 4); g.lineTo(x + 0.5, height); g.stroke()
      g.fillText(us(t - trace.t0), x + 3, AXIS / 2 - 1)
    }
    trace.lanes.forEach((lane, li) => {
      const y = AXIS + li * ROW
      g.fillStyle = ink2; g.fillText(lane.name.slice(0, 22), 6, y + ROW / 2)
      const [lo, hi] = spansIn(lane, a, b)
      let lastX = -1, lastId = -1
      for (let i = lo; i < hi; i++) {
        const s = lane.ts[i], e = s + lane.dur[i]
        if (e < a || s > b) continue
        const x0 = LABEL_W + Math.max(0, (s - a) * k), x1 = LABEL_W + Math.min(plotW, (e - a) * k)
        const w = Math.max(1, x1 - x0)
        const id = lane.nameIdx[i]
        if (w <= 1.5 && Math.floor(x0) === lastX && id === lastId) continue          // many tiny spans in one pixel: one mark
        lastX = Math.floor(x0); lastId = id
        const on = !matches || matches.has(id)
        g.fillStyle = `hsl(${hue(id)} ${dark ? 45 : 55}% ${dark ? 58 : 50}% / ${on ? 0.9 : 0.15})`
        g.fillRect(x0, y + (ROW - BAR) / 2, w, BAR)
        if (w > 60 && on) { g.fillStyle = dark ? "#000" : "#fff"; g.fillText(trace.names[id], x0 + 3, y + ROW / 2, w - 6) }
      }
    })
  }, [trace, view, width, height, matches])

  const timeAt = (clientX: number) => {
    const r = canvas.current!.getBoundingClientRect()
    return view[0] + ((clientX - r.left - LABEL_W) / (width - LABEL_W)) * (view[1] - view[0])
  }
  const zoom = (factor: number, cx: number) => {
    if (!trace) return
    const t = timeAt(cx), [a, b] = view
    const span = Math.min(trace.t1 - trace.t0, Math.max(10, (b - a) * factor))
    const na = Math.max(trace.t0, Math.min(t - ((t - a) / (b - a)) * span, trace.t1 - span))
    setView([na, na + span])
  }
  const onDown = (e: RPointerEvent) => { drag.current = { x: e.clientX, a: view[0], b: view[1] }; (e.target as Element).setPointerCapture(e.pointerId) }
  const onUp = () => { drag.current = null }
  const onMove = (e: RPointerEvent) => {
    if (!trace) return
    const d = drag.current
    if (d) {
      const dt = ((e.clientX - d.x) / (width - LABEL_W)) * (d.b - d.a), span = d.b - d.a
      const na = Math.max(trace.t0, Math.min(d.a - dt, trace.t1 - span))
      setView([na, na + span]); return
    }
    const r = canvas.current!.getBoundingClientRect()
    const li = Math.floor((e.clientY - r.top - AXIS) / ROW)
    const lane = trace.lanes[li]
    if (!lane || e.clientX - r.left < LABEL_W) { setHover(null); return }
    const t = timeAt(e.clientX), px = (view[1] - view[0]) / (width - LABEL_W)
    const [lo, hi] = spansIn(lane, t - px, t + px)
    let best = -1
    for (let i = lo; i < hi; i++) if (lane.ts[i] <= t + px && lane.ts[i] + Math.max(lane.dur[i], px) >= t - px) best = i      // the latest started: the innermost
    if (best < 0) { setHover(null); return }
    setHover({ x: e.clientX - r.left, y: e.clientY - r.top, name: trace.names[lane.nameIdx[best]], lane: lane.name, dur: lane.dur[best], at: lane.ts[best] - trace.t0, args: lane.args[best] })
  }

  const zoomRef = useRef(zoom)
  zoomRef.current = zoom
  useEffect(() => {
    const c = canvas.current
    if (!c) return
    const on = (e: WheelEvent) => { e.preventDefault(); zoomRef.current(Math.exp(e.deltaY * 0.0015), e.clientX) }
    c.addEventListener("wheel", on, { passive: false })
    return () => c.removeEventListener("wheel", on)
  }, [trace])

  const top = useMemo(() => (trace ? topNames(trace, view[0], view[1], 12) : []), [trace, view])
  const download = () => {
    if (!source) return
    const a = document.createElement("a")
    a.href = URL.createObjectURL(new Blob([closeTrace(source.text)], { type: "application/json" }))
    a.download = source.name.replace(/\.json$/i, "") + ".closed.json"
    a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 5000)
  }

  return (
    <div onDragOver={(e) => e.preventDefault()} onDrop={onDrop}>
      <p className="mb-3 text-[13px]"><a href={href("requests")}>← Requests</a></p>
      <h1 className="text-xl font-semibold">Timeline</h1>
      <p className="mt-1 max-w-[65ch] text-[13px] text-ink-2">
        The engine's pipeline timeline: start the engine with <code>STRATA_TIMELINE=run.json</code>, then open the file here. It is read in this
        browser and goes nowhere. Scroll to zoom, drag to move, double-click to see it all.
      </p>
      <div className="mt-4 flex flex-wrap items-center gap-2 text-[13px]">
        <label className="inline-flex h-8 cursor-pointer items-center rounded-sm bg-fill px-3 font-medium transition-colors hover:bg-fill-2">
          {busy ? "Reading…" : "Open a timeline file"}
          <input type="file" hidden accept=".json,application/json" onChange={(e) => { const f = e.target.files?.[0]; if (f) void open(f); e.target.value = "" }} />
        </label>
        {trace && source && (
          <>
            <span className="text-ink-2">{source.name} · <span className="num">{fmt(trace.count)}</span> spans · <span className="num">{trace.lanes.length}</span> lanes · <span className="num">{us(trace.t1 - trace.t0)}</span></span>
            <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Find a span name" aria-label="Find a span name" className="h-8 w-44 rounded-sm border border-line bg-surface px-2.5 outline-none focus:border-accent" />
            <button type="button" onClick={download} className="h-8 rounded-sm px-3 text-ink-2 transition-colors hover:bg-hover hover:text-ink">Download for Perfetto</button>
          </>
        )}
      </div>
      {error && <p className="mt-3 text-bad">{error}</p>}

      {!trace ? (
        <p className="mt-8 text-ink-2">{busy ? "Reading the file…" : "Drop a timeline file here, or open one."}</p>
      ) : (
        <>
          <div ref={box} className="relative mt-4 overflow-hidden rounded-md border border-line">
            <canvas
              ref={canvas}
              style={{ width, height, touchAction: "none", cursor: drag.current ? "grabbing" : "grab", display: "block" }}
              role="img"
              aria-label={`Timeline of ${trace.lanes.length} lanes; the table below lists what the visible window spent its time on`}
              onPointerDown={onDown} onPointerUp={onUp} onPointerMove={onMove} onPointerLeave={() => setHover(null)}
              onDoubleClick={() => setView([trace.t0, trace.t1])}
            />
            {hover && (
              <div className="pointer-events-none absolute z-10 max-w-xs rounded-md border border-line bg-surface px-2.5 py-1.5 text-[12px] shadow-[0_6px_24px_rgb(0_0_0/0.12)]" style={{ left: Math.min(hover.x + 12, width - 240), top: hover.y + 14 }}>
                <div className="font-medium">{hover.name}</div>
                <div className="num text-ink-2">{hover.lane} · {us(hover.dur)} · at {us(hover.at)}</div>
                {hover.args && <div className="num text-ink-3">{Object.entries(hover.args).map(([k, v]) => `${k}=${String(v)}`).join(" ")}</div>}
              </div>
            )}
          </div>
          <Section title="In the window" aside={<span className="num">{us(view[1] - view[0])}</span>}>
            <p className="mb-1 text-[12px] text-ink-3">Lanes run in parallel, so these totals can add up to more than the window.</p>
            <Rows>
              {top.map((r) => <Row key={r.name} k={r.name} hint={`${fmt(r.count)} span${r.count === 1 ? "" : "s"}`} v={<span className="num">{us(r.total)}</span>} />)}
            </Rows>
          </Section>
        </>
      )}
    </div>
  )
}
