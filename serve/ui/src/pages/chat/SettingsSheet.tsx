import { useEffect, useState } from "react"
import { chat, DEFAULTS, type Settings } from "../../lib/chat"
import { apiHeaders, url, type McpInfo } from "../../lib/api"
import { toast } from "../../components/toast"
import { Button, Field, inputCls, Segmented, Switch } from "../../components/ui"

const THINKING_TEXT: Record<Settings["thinking"], string> = { none: "answers right away", low: "short", medium: "medium", high: "thorough (default)" }

async function savedShared(): Promise<boolean> {
  try {
    const r = await fetch(url("settings"), { headers: apiHeaders() })
    return r.ok ? !!((await r.json()) as { shared?: boolean }).shared : false
  } catch { return false }                       // an older server: the switch just stays off
}

function sharedDefaults(s: Settings, projection: boolean) {
  const d: Record<string, unknown> = { reasoning_effort: s.thinking, temperature: +s.temperature }
  if (+s.temperature > 0) Object.assign(d, { top_p: +s.top_p, top_k: +s.top_k })
  if (s.seed) d.seed = +s.seed
  if (s.max) d.max_tokens = +s.max
  if (projection) d.experimental_speed_projection = s.esp !== false
  return d
}

export function SettingsSheet({ open, onClose, mcp, projectionLoaded }: { open: boolean; onClose: () => void; mcp: McpInfo; projectionLoaded: boolean }) {
  const [s, setS] = useState<Settings>(chat.settings)
  const [shared, setShared] = useState(false)
  const [wasShared, setWasShared] = useState(false)
  const patch = (p: Partial<Settings>) => setS((x) => ({ ...x, ...p }))

  useEffect(() => {
    if (!open) return
    setS(chat.settings)
    void savedShared().then((v) => { setShared(v); setWasShared(v) })
    const esc = (e: KeyboardEvent) => { if (e.key === "Escape") onClose() }
    addEventListener("keydown", esc)
    return () => removeEventListener("keydown", esc)
  }, [open, onClose])

  const apply = async () => {
    chat.setSettings(s)
    onClose()
    if (shared || wasShared) {
      try {
        const r = await fetch(url("settings"), { method: "POST", headers: apiHeaders(true), body: JSON.stringify({ defaults: shared ? sharedDefaults(s, projectionLoaded) : null }) })
        if (!r.ok) throw new Error(((await r.json().catch(() => ({}))) as { error?: { message?: string } }).error?.message || `HTTP ${r.status}`)
        toast("success", "Sampling saved", shared ? "Other apps (omp, API clients) use these settings from their next request." : "Other apps use their own settings again.")
      } catch (e) {
        toast("error", "Saved here, but not for other apps", (e as Error).message, 6000)
      }
      return
    }
    toast("success", "Sampling saved", s.temperature === 0 ? "Greedy: the same question gives the same answer." : "")
  }

  return (
    <div className={`fixed inset-0 z-30 overflow-hidden ${open ? "" : "pointer-events-none"}`}>
      <div
        aria-hidden
        onClick={onClose}
        className={`absolute inset-0 bg-black/30 transition-opacity duration-200 ${open ? "opacity-100" : "opacity-0"}`}
      />
      <aside
        aria-label="Sampling"
        aria-hidden={!open}
        inert={!open}
        className={`absolute inset-y-0 right-0 flex w-full max-w-sm flex-col bg-surface transition-transform duration-300 ease-[var(--ease)] ${open ? "translate-x-0 shadow-[-8px_0_40px_rgb(0_0_0/0.12)]" : "translate-x-full"}`}
      >
        <div className="flex items-center justify-between px-5 py-4">
          <h2 className="text-[15px] font-semibold">Sampling</h2>
          <Button kind="quiet" onClick={onClose} label="Close">Close</Button>
        </div>
        <div className="flex-1 overflow-y-auto px-5 pb-4">
          <Field label="Thinking" value={THINKING_TEXT[s.thinking]}>
            <Segmented label="Thinking" value={s.thinking} onChange={(v) => patch({ thinking: v })}
              options={[{ value: "none", label: "Off" }, { value: "low", label: "Low" }, { value: "medium", label: "Medium" }, { value: "high", label: "High" }]} />
          </Field>
          <Switch label="Show the thinking while it streams" checked={s.show} onChange={(v) => patch({ show: v })} />
          <Switch label="Show the prompt reading speed under each prompt" checked={s.prefill} onChange={(v) => patch({ prefill: v })} />
          <Field label="Temperature" value={s.temperature === 0 ? "0 · greedy" : s.temperature.toFixed(2)}>
            <input type="range" min={0} max={1.5} step={0.05} value={s.temperature} onChange={(e) => patch({ temperature: +e.target.value })} className="w-full accent-[var(--accent)]" />
          </Field>
          <Field label="Top-p" value={s.top_p.toFixed(2)}>
            <input type="range" min={0.05} max={1} step={0.05} value={s.top_p} disabled={s.temperature === 0} onChange={(e) => patch({ top_p: +e.target.value })} className="w-full accent-[var(--accent)] disabled:opacity-40" />
          </Field>
          <Field label="Top-k" value={s.top_k}>
            <input type="range" min={1} max={100} step={1} value={s.top_k} disabled={s.temperature === 0} onChange={(e) => patch({ top_k: +e.target.value })} className="w-full accent-[var(--accent)] disabled:opacity-40" />
          </Field>
          <Field label="Max new tokens" hint="Empty: as much as the context allows.">
            <input className={inputCls} inputMode="numeric" value={s.max} onChange={(e) => patch({ max: e.target.value })} placeholder="No limit" />
          </Field>
          <Field label="Seed" hint="Empty: a new one each time.">
            <input className={inputCls} inputMode="numeric" value={s.seed} onChange={(e) => patch({ seed: e.target.value })} placeholder="Random" />
          </Field>
          {projectionLoaded && (
            <Switch label="Speed projection" hint="The control vector this engine was started with." checked={s.esp !== false} onChange={(v) => patch({ esp: v })} />
          )}
          {mcp.servers.length > 0 && (
            <Switch label="Let the model use MCP tools" hint={mcp.tools ? `${mcp.tools} tools from ${mcp.servers.length} server${mcp.servers.length > 1 ? "s" : ""}.` : "No server is connected yet."} checked={s.mcp !== false} onChange={(v) => patch({ mcp: v })} />
          )}
          <Switch label="Use for other apps too" hint="Every client gets these as its defaults." checked={shared} onChange={setShared} />
        </div>
        <div className="flex items-center justify-between gap-2 border-t border-line px-5 py-3">
          <Button kind="quiet" onClick={() => setS({ ...DEFAULTS })}>Reset</Button>
          <Button kind="primary" onClick={() => void apply()}>Apply</Button>
        </div>
      </aside>
    </div>
  )
}
