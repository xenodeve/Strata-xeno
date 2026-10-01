import { useEffect, useState } from "react"
import { apiHeaders, getHealth, getMcp, NO_HEALTH, root, url, type Health, type McpInfo } from "../lib/api"
import { fmt } from "../lib/format"
import { store } from "../lib/store"
import { toast } from "../components/toast"
import { inputCls } from "../components/ui"

interface Engine { engine?: string; version?: string; context?: number; kv?: string; spec?: number; mtp_max?: number; cpu_isa?: string; gpu_arch?: string }

function Row({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-6 border-b border-line py-2 last:border-0">
      <dt className="text-ink-2">{k}</dt>
      <dd className="num m-0 text-right [overflow-wrap:anywhere]">{v}</dd>
    </div>
  )
}

const MCP_STATE: Record<string, string> = { ready: "Connected", starting: "Starting", failed: "Failed", stopped: "Stopped", idle: "Waiting" }

export function About() {
  const [health, setHealth] = useState<Health>(NO_HEALTH)
  const [engine, setEngine] = useState<Engine>({})
  const [mcp, setMcp] = useState<McpInfo>({ servers: [], tools: 0 })
  const [key, setKey] = useState(() => store.get("apikey", ""))

  useEffect(() => {
    void getHealth().then(setHealth).catch(() => {})
    void getMcp().then((m) => m && setMcp(m))
    void fetch(url("metrics"), { headers: apiHeaders() })
      .then((r) => (r.ok ? r.json() : null))
      .then((m: { engine?: Engine } | null) => m?.engine && setEngine(m.engine))
      .catch(() => {})
  }, [])

  return (
    <div className="max-w-[65ch] space-y-8">
      <section>
        <h1 className="text-xl font-semibold">About</h1>
        <p className="mt-2 text-ink-2">Strata runs a large mixture-of-experts model on this PC. Nothing leaves it.</p>
      </section>

      <section>
        <h2 className="text-[15px] font-semibold">This server</h2>
        <dl className="mt-2 m-0">
          <Row k="Model" v={health.model} />
          <Row k="Context" v={health.max_context ? `${fmt(health.max_context)} tokens` : "–"} />
          <Row k="Pictures" v={health.images ? "on" : "off"} />
          <Row k="Engine" v={engine.engine || engine.version || "–"} />
          {engine.cpu_isa && <Row k="CPU kernels" v={engine.cpu_isa} />}
          {engine.gpu_arch && <Row k="GPU architecture" v={engine.gpu_arch.replaceAll(",", ", ")} />}
        </dl>
      </section>

      <section>
        <h2 className="text-[15px] font-semibold">API key</h2>
        <p className="mt-1 text-[13px] text-ink-2">Only needed when the server was started with one. Kept in this browser.</p>
        <input
          className={`${inputCls} mt-2`}
          type="password"
          autoComplete="off"
          value={key}
          placeholder="No key"
          aria-label="API key"
          onChange={(e) => setKey(e.target.value)}
          onBlur={() => { store.set("apikey", key.trim()); toast("success", "API key saved", "Kept in this browser only.") }}
        />
      </section>

      {mcp.servers.length > 0 && (
        <section>
          <h2 className="text-[15px] font-semibold">MCP servers</h2>
          <p className="mt-1 text-[13px] text-ink-2">{fmt(mcp.tools)} tools; the model calls them when it decides to.</p>
          <ul className="mt-2 m-0 list-none p-0">
            {mcp.servers.map((s) => (
              <li key={s.name} className="border-b border-line py-2 last:border-0">
                <div className="flex items-baseline justify-between gap-4">
                  <span className="font-medium">{s.name}</span>
                  <span className={s.status === "failed" ? "text-bad" : "text-ink-2"}>{MCP_STATE[s.status] || s.status}</span>
                </div>
                <div className="text-[12px] text-ink-2">{s.transport} · {fmt(s.tools.length)} tools</div>
                {s.error && <div className="mt-1 text-[12px] text-bad [overflow-wrap:anywhere]">{s.error}</div>}
              </li>
            ))}
          </ul>
        </section>
      )}

      <section>
        <h2 className="text-[15px] font-semibold">The classic app</h2>
        <p className="mt-1 text-[13px] text-ink-2">
          Still here, with its Monitor, until the Live and Hardware pages replace it: <a href={`${root()}classic/`}>open the classic app</a>.
        </p>
      </section>
    </div>
  )
}
