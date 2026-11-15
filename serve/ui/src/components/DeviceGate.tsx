import { useEffect, useState, type ReactNode } from "react"
import { apiHeaders, url } from "../lib/api"
import { deviceGate, deviceName, setGate, useGate } from "../lib/device"
import { refreshTeam } from "../lib/keys"
import { t } from "../lib/i18n"
import { store } from "../lib/store"
import { Button, inputCls } from "./ui"

// The waiting screen (#310): a server in team mode lets a key's first use from a new device in only when the owner (or the key's holder,
// from a device already let in) approves it. Until then every request of this browser answers 403 device_pending with a code, and the
// page says so in words instead of failing piece by piece. It looks again every 5 s (GET /health: is the server there; GET /props: is
// the device let in yet) and opens the app the moment it is. Outside team mode it never shows: nothing answers device_pending there.
const EVERY_S = 5

type Check = "checking" | "waiting" | "down"

export function DeviceGate({ children }: { children: ReactNode }) {
  const gate = useGate()
  const [check, setCheck] = useState<Check>("waiting")
  const [left, setLeft] = useState(EVERY_S)
  const [key, setKey] = useState("")
  const [nonce, setNonce] = useState(0)                       // a new key: look again at once

  useEffect(() => {
    if (!gate) return
    let stop = false
    let timer = 0
    let tickTimer = 0
    const look = async () => {
      if (stop) return
      setCheck("checking")
      let up = false
      try { await fetch(url("health")); up = true } catch { up = false }   // any answer: the server is there (a 503 is an engine that stopped)
      if (stop) return
      if (!up) { setCheck("down"); wait(); return }
      try {
        const r = await fetch(url("props"), { headers: apiHeaders() })
        if (stop) return
        const g = await deviceGate(r)
        setGate(g)                                            // let in (or no longer a device matter: a 401 shows in the app): the app opens
        if (!g) { refreshTeam(); return }
        setCheck("waiting")
      } catch { if (!stop) setCheck("down") }
      wait()
    }
    const wait = () => {
      setLeft(EVERY_S)
      clearInterval(tickTimer)
      tickTimer = window.setInterval(() => setLeft((n) => Math.max(1, n - 1)), 1000)
      timer = window.setTimeout(() => { clearInterval(tickTimer); void look() }, EVERY_S * 1000)
    }
    void look()
    return () => { stop = true; clearTimeout(timer); clearInterval(tickTimer) }
  }, [gate === null, nonce])                                  // a new code or a refusal keeps the same loop going

  if (!gate) return <>{children}</>
  const broken = gate.kind === "pending" && gate.code === "----"
  const takeKey = () => {
    if (!key.trim()) return
    store.set("apikey", key.trim())
    setKey("")
    setNonce((n) => n + 1)
  }
  return (
    <main data-device-gate={gate.kind} className="grid min-h-dvh place-items-center bg-bg px-4 py-10">
      <div className="w-full max-w-[460px]">
        <h1 className="text-[22px] font-semibold tracking-[-0.02em]">
          {gate.kind === "refused" ? t("This device was refused") : t("This device waits for approval")}
        </h1>
        {gate.kind === "pending" && !broken && (
          <>
            <p className="mt-3 text-[14px] leading-relaxed text-ink-2">{t("This device waits for approval - code {code}. Ask the workstation's owner, or approve it from a device you already use.", { code: gate.code })}</p>
            <p className="mt-4 font-mono text-[32px] font-semibold tracking-[0.2em]" aria-label={t("Code {code}", { code: gate.code.split("").join(" ") })}>{gate.code}</p>
          </>
        )}
        {broken && <p className="mt-3 text-[14px] leading-relaxed text-ink-2">{t("This device waits for approval, but the server's keys file cannot be read or written, so the device is not recorded and has no code yet. The workstation's owner has to fix the keys file.")}</p>}
        {gate.kind === "refused" && <p className="mt-3 text-[14px] leading-relaxed text-ink-2">{t("The owner refused this device for this key, or the key has too many devices. Ask the workstation's owner to clear it; it then waits for approval again.")}</p>}
        <p className="mt-4 text-[13px] text-ink-2">{t("This device is called {name}.", { name: deviceName() })}</p>
        <p role="status" aria-live="polite" className="mt-1 text-[13px] text-ink-3">
          {check === "checking" ? t("Checking whether it is let in…") : check === "down" ? t("The server is not answering. Trying again in {n} s.", { n: left }) : t("Not yet. Checking again in {n} s.", { n: left })}
        </p>
        <form className="mt-6 border-t border-line pt-4" onSubmit={(e) => { e.preventDefault(); takeKey() }}>
          <label className="block text-[13px]" htmlFor="gate-key">{t("Have another key? Use it on this device:")}</label>
          <div className="mt-1.5 flex gap-2">
            <input id="gate-key" className={inputCls} type="password" autoComplete="off" value={key} placeholder={t("API key")} onChange={(e) => setKey(e.target.value)} />
            <span className="shrink-0 whitespace-nowrap"><Button type="submit" disabled={!key.trim()}>{t("Use this key")}</Button></span>
          </div>
        </form>
      </div>
    </main>
  )
}
