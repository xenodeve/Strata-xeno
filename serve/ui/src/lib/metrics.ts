// GET /metrics, polled while the tab is visible and not at all while it is hidden (nothing loops in the background).
import { useEffect, useRef, useState } from "react"
import { apiHeaders, url } from "./api"

export interface Gpu { index: number; util: number | null; mem_used: number | null; mem_total: number | null; temp: number | null; power: number | null }
export interface Hardware {
  gpu_util?: number | null; gpu_mem_used?: number | null; gpu_mem_total?: number | null; gpu_temp?: number | null
  gpu_power?: number | null; gpu_power_limit?: number | null; gpu_pcie_gen?: number | null; gpu_pcie_gen_max?: number | null
  gpu_pcie_width?: number | null; gpu_pcie_rx_mb?: number | null; gpu_pcie_tx_mb?: number | null
  cpu?: number | null; ram_used?: number | null; ram_total?: number | null; disk_read_mb?: number | null; disk_write_mb?: number | null
  gpus?: Gpu[]
}
export interface Live {
  state: "reading" | "generating" | "idle" | "unloaded"; queued: number; phase: string | null
  prompt_tokens: number | null; prompt_read: number | null; prompt_total: number | null; generated: number | null
  max_tokens: number | null; elapsed_s: number | null; tok_s: number | null; tok_s_mean: number | null
  prefill_tok_s_mean: number | null; tok_s_window_s: number | null
}
export interface Rate { tok_s_min: number | null; tok_s_max: number | null; tok_s_mean: number | null }
export interface RequestRow {
  id: string; dialect: string; client: string; tools: string[]; preview: string; time: number; duration_s: number
  finish: string; prompt_tokens: number; reused: number | null; output_tokens: number
  prompt_ms: number | null; decode_ms: number | null; decode_tok_s: number | null; hit_rate: number | null
  prefill_tok_s: number | null
  prefill: (Rate & { chunks: number }) | null
  decode: (Rate & { windows: number }) | null
  stats: Record<string, number> | null
}
export interface Metrics {
  engine: Record<string, string | number | null>
  live: Live
  requests: RequestRow[]
  requests_kept: number
  totals: Record<string, number>
  hardware: Hardware
  hardware_static: { gpu_name?: string | null; gpu_count?: number; cpu_name?: string | null; cores?: number | null; threads?: number | null; psutil?: boolean }
  history: Record<string, (number | null)[]>
  time: number
}

export type MetricsState = { data: Metrics | null; error: string | null; stale: boolean }

export function useMetrics(): MetricsState {
  const [state, set] = useState<MetricsState>({ data: null, error: null, stale: false })
  const fails = useRef(0)
  useEffect(() => {
    let timer = 0
    let stop = false
    const tick = async () => {
      if (stop) return
      if (document.hidden) { timer = window.setTimeout(tick, 1000); return }
      let busy = false
      try {
        const r = await fetch(url("metrics"), { headers: apiHeaders() })
        if (!r.ok) throw new Error(r.status === 401 ? "This server needs an API key (About)." : `HTTP ${r.status}`)
        const data = (await r.json()) as Metrics
        fails.current = 0
        busy = data.live.state === "reading" || data.live.state === "generating"
        if (!stop) set({ data, error: null, stale: false })
      } catch (e) {
        fails.current++
        if (!stop) set((s) => ({ data: s.data, error: (e as Error).message, stale: fails.current >= 3 }))
      }
      timer = window.setTimeout(tick, busy ? 500 : 1000)       // twice as often while a request runs
    }
    void tick()
    const visible = () => { if (!document.hidden) { clearTimeout(timer); void tick() } }
    document.addEventListener("visibilitychange", visible)
    return () => { stop = true; clearTimeout(timer); document.removeEventListener("visibilitychange", visible) }
  }, [])
  return state
}

export interface RequestPage { items: RequestRow[]; total: number; page: number; size: number }
export interface RequestDetail {
  summary: RequestRow
  detail: { prefill_chunks?: [number, number][]; decode_series?: number[]; stats?: Record<string, number> | null } | null
  detail_state: "kept" | "deleted"
}

export async function getRequestPage(page: number, size: number): Promise<RequestPage> {
  const r = await fetch(url(`metrics/requests?page=${page}&size=${size}`), { headers: apiHeaders() })
  if (!r.ok) throw new Error(r.status === 401 ? "This server needs an API key (About)." : `HTTP ${r.status}`)
  return (await r.json()) as RequestPage
}

export async function getRequest(id: string): Promise<RequestDetail | null> {
  const r = await fetch(url(`metrics/requests/${encodeURIComponent(id)}`), { headers: apiHeaders() })
  if (r.status === 404) return null
  if (!r.ok) throw new Error(`HTTP ${r.status}`)
  return (await r.json()) as RequestDetail
}
