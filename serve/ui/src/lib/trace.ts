// The engine's STRATA_TIMELINE file (#33) in the browser. It is a Chrome trace: a JSON array of events, one per line,
// left OPEN (no closing bracket) because the engine appends after every request. "X" events are spans on a thread (a
// lane), "M" events name the threads. Times are microseconds on the engine's own clock.
export interface Lane {
  pid: number; tid: number; name: string
  ts: Float64Array; dur: Float64Array; nameIdx: Uint32Array; args: (Record<string, unknown> | undefined)[]
  maxDur: number
}
export interface Trace { lanes: Lane[]; names: string[]; t0: number; t1: number; count: number }

interface Ev { ph?: string; pid?: number; tid?: number; ts?: number; dur?: number; name?: string; args?: Record<string, unknown> }

function events(text: string): { evs: Ev[]; lines: string[] } {
  const s = text.trim()
  if (!s) return { evs: [], lines: [] }
  try {                                                         // a closed array, or an object with traceEvents
    const j = JSON.parse(s)
    const evs: Ev[] = Array.isArray(j) ? j : j?.traceEvents
    if (Array.isArray(evs)) return { evs, lines: evs.map((e) => JSON.stringify(e)) }
  } catch { /* the engine's open array: line by line */ }
  const evs: Ev[] = []
  const lines: string[] = []
  for (const raw of s.split("\n")) {
    let l = raw.trim()
    if (l.startsWith("[")) l = l.slice(1).trim()
    if (l.endsWith("]")) l = l.slice(0, -1).trim()
    if (l.endsWith(",")) l = l.slice(0, -1).trim()
    if (!l.startsWith("{") || !l.endsWith("}")) continue         // blank, the bracket, or a line the engine did not finish
    try { evs.push(JSON.parse(l)); lines.push(l) } catch { /* a torn line */ }
  }
  if (!evs.length) throw new Error("This is not a trace file (a Chrome trace array, as STRATA_TIMELINE writes).")
  return { evs, lines }
}

export function parseTrace(text: string): Trace {
  const { evs } = events(text)
  const threadNames = new Map<string, string>()
  const byLane = new Map<string, { pid: number; tid: number; spans: Ev[] }>()
  for (const e of evs) {
    const pid = e.pid ?? 0, tid = e.tid ?? 0, key = `${pid}:${tid}`
    if (e.ph === "M" && e.name === "thread_name" && typeof e.args?.name === "string") threadNames.set(key, e.args.name)
    if (e.ph === "X" && typeof e.ts === "number") {
      let l = byLane.get(key)
      if (!l) byLane.set(key, (l = { pid, tid, spans: [] }))
      l.spans.push(e)
    }
  }
  const names: string[] = []
  const nameId = new Map<string, number>()
  const id = (n: string) => { let i = nameId.get(n); if (i === undefined) { i = names.length; names.push(n); nameId.set(n, i) } return i }
  let t0 = Infinity, t1 = -Infinity, count = 0
  const lanes: Lane[] = [...byLane.entries()].sort((a, b) => a[1].pid - b[1].pid || a[1].tid - b[1].tid).map(([key, l]) => {
    l.spans.sort((a, b) => a.ts! - b.ts!)
    const n = l.spans.length
    const lane: Lane = { pid: l.pid, tid: l.tid, name: threadNames.get(key) ?? `thread ${l.tid}`, ts: new Float64Array(n), dur: new Float64Array(n),
      nameIdx: new Uint32Array(n), args: new Array(n), maxDur: 0 }
    l.spans.forEach((e, i) => {
      const d = e.dur ?? 0
      lane.ts[i] = e.ts!; lane.dur[i] = d; lane.nameIdx[i] = id(e.name ?? ""); lane.args[i] = e.args
      if (d > lane.maxDur) lane.maxDur = d
      if (e.ts! < t0) t0 = e.ts!
      if (e.ts! + d > t1) t1 = e.ts! + d
    })
    count += n
    return lane
  })
  if (!count) { t0 = 0; t1 = 0 }
  return { lanes, names, t0, t1, count }
}

/** The same events as one valid JSON array, which ui.perfetto.dev and chrome://tracing open. */
export function closeTrace(text: string): string {
  const { lines } = events(text)
  return "[\n" + lines.join(",\n") + "\n]"
}

/** [lo, hi): the spans of a lane that may overlap [a, b] (callers still clip each to the window). */
export function spansIn(lane: Lane, a: number, b: number): [number, number] {
  const n = lane.ts.length
  let lo = 0, hi = n, from = a - lane.maxDur
  while (lo < hi) { const m = (lo + hi) >> 1; if (lane.ts[m] < from) lo = m + 1; else hi = m }
  while (lo < n && lane.ts[lo] + lane.dur[lo] < a) lo++          // leading spans that ended before the window
  let h0 = lo, h1 = n
  while (h0 < h1) { const m = (h0 + h1) >> 1; if (lane.ts[m] <= b) h0 = m + 1; else h1 = m }
  return [lo, Math.max(lo, h0)]
}

/** What the window spent its time on: per span name, how many spans and how much of the window they covered. */
export function topNames(t: Trace, a: number, b: number, limit: number): { name: string; count: number; total: number }[] {
  const acc = new Map<number, { count: number; total: number }>()
  for (const lane of t.lanes) {
    const [lo, hi] = spansIn(lane, a, b)
    for (let i = lo; i < hi; i++) {
      const s = Math.max(lane.ts[i], a), e = Math.min(lane.ts[i] + lane.dur[i], b)
      if (e < s || (e === s && lane.dur[i] > 0)) continue
      const r = acc.get(lane.nameIdx[i]) ?? { count: 0, total: 0 }
      r.count++; r.total += e - s
      acc.set(lane.nameIdx[i], r)
    }
  }
  return [...acc.entries()].map(([i, r]) => ({ name: t.names[i], ...r })).sort((x, y) => y.total - x.total).slice(0, limit)
}
