// Stall attribution (UI S7): where each resource's time went during a request's decode, from the engine's STATS line.
//
// This is a DEPENDENCY DEFINITION, not a hardware measurement: a worker's idle time is attributed to what the host was
// waiting on at that moment. Cycle-level stalls (CPU PMU, GPU warp stalls) come only from the Nsight capture.
// Stages overlap in places, so a bucket's share is of the decode wall time and the "other" bucket is what the
// counters do not name.
export type Stats = Record<string, number>

export interface Bucket { key: string; label: string; ms: number; kind: "busy" | "wait" | "idle" }
export interface Attribution {
  resource: string
  total: number
  buckets: Bucket[]           // in the order they are drawn; the last is "other", never negative
  overhead: null              // wake / park / repark of the pool: no instrument yet, so not a number
}

const n = (s: Stats, k: string) => (Number.isFinite(s[k]) && s[k] > 0 ? s[k] : 0)

function build(resource: string, total: number, parts: Omit<Bucket, "ms">[], s: Stats): Attribution {
  const buckets: Bucket[] = parts.map((p) => ({ ...p, ms: n(s, p.key) })).filter((b) => b.ms > 0)
  const named = buckets.reduce((a, b) => a + b.ms, 0)
  const scale = named > total && total > 0 ? total / named : 1                 // overlapping stages never exceed the wall time
  const scaled = buckets.map((b) => ({ ...b, ms: b.ms * scale }))
  const other = Math.max(0, total - scaled.reduce((a, b) => a + b.ms, 0))
  if (other > total * 0.005) scaled.push({ key: "other", label: "Not named by the counters", ms: other, kind: "idle" })
  return { resource, total, buckets: scaled, overhead: null }
}

/** The CPU pool and the GPU side of one request's decode. Null when the engine sent no counters or no time. */
export function attribute(stats: Stats | null | undefined, decodeMs: number | null | undefined): Attribution[] | null {
  if (!stats || !decodeMs || decodeMs <= 0) return null
  return [
    build("CPU pool", decodeMs, [
      { key: "ms_cpu", label: "Computing experts", kind: "busy" },
      { key: "ms_gpu_wait", label: "Waiting for the GPU's output", kind: "wait" },
      { key: "ms_plan", label: "Waiting for the dispatch plan", kind: "wait" },
      { key: "ms_actq", label: "Waiting for the activation quantize", kind: "wait" },
      { key: "ms_jobs", label: "Waiting for jobs to be handed out", kind: "wait" },
      { key: "ms_stage", label: "Waiting for the host to stage the window", kind: "wait" },
      { key: "ms_commit", label: "Idle: the host commits and emits", kind: "idle" },
      { key: "ms_draft", label: "Idle: the host drafts the next window", kind: "idle" },
    ], stats),
    build("GPUs", decodeMs, [
      { key: "ms_gpu_wait", label: "Computing a layer (the host waits for it)", kind: "busy" },
      { key: "ms_pool", label: "Waiting for the CPU's experts", kind: "wait" },
      { key: "ms_stage", label: "Waiting for the host to launch the window", kind: "wait" },
      { key: "ms_commit", label: "Idle: the host commits and emits", kind: "idle" },
      { key: "ms_draft", label: "Idle: the host drafts the next window", kind: "idle" },
    ], stats),
  ]
}

/** An upper bound on what overlapping reads with compute could win back: the time the pipeline sat on the SSD.
 *  PCIe copy waits are not measured yet, so they are not in it. */
export function overlapOpportunityMs(stats: Stats | null | undefined): number | null {
  if (!stats || !Number.isFinite(stats.nvme_ms)) return null
  return Math.max(0, stats.nvme_ms)
}
