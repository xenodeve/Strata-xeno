import { useCallback, useEffect, useState } from "react"
import { getPending, postDevice, useTeam, type PendingDevice } from "../lib/keys"
import { t } from "../lib/i18n"
import { toast } from "./toast"
import { Button } from "./ui"

// A device asks to be let in (#310): a banner for each one this caller may decide - the owner, every device; a key's holder on a device
// already let in, her own key's. The server picks which (GET /devices/pending); this page only asks every 10 s while it is in view, and
// only on a server in team mode (outside it no device ever waits, and nothing is asked).
const EVERY_MS = 10_000

export function ApprovalNotice() {
  const team = useTeam()
  const [rows, setRows] = useState<PendingDevice[]>([])
  const [busy, setBusy] = useState<Record<string, "approve" | "refuse">>({})

  const look = useCallback(async () => { setRows((await getPending()) ?? []) }, [])
  useEffect(() => {
    if (!team) { setRows([]); return }
    let timer = 0
    let stop = false
    const tick = async () => {
      if (stop) return
      if (!document.hidden) await look()
      if (!stop) timer = window.setTimeout(tick, EVERY_MS)
    }
    void tick()
    const visible = () => { if (!document.hidden) { clearTimeout(timer); void tick() } }
    document.addEventListener("visibilitychange", visible)
    return () => { stop = true; clearTimeout(timer); document.removeEventListener("visibilitychange", visible) }
  }, [team, look])

  if (!rows.length) return null
  const decide = async (d: PendingDevice, how: "approve" | "refuse") => {
    const id = d.key + "/" + d.device
    setBusy((b) => ({ ...b, [id]: how }))
    const r = await postDevice(d.key, d.device, how)
    setBusy((b) => { const n = { ...b }; delete n[id]; return n })
    if ("error" in r) toast("error", how === "approve" ? t("The device was not approved") : t("The device was not refused"), r.error)
    else toast("success", how === "approve" ? t("Device approved") : t("Device refused"), d.label || d.device)
    await look()
  }
  return (
    <div className="mx-auto w-full max-w-5xl space-y-2 px-3 pt-3 sm:px-6" data-approval-notice>
      {rows.map((d) => {
        const id = d.key + "/" + d.device
        const doing = busy[id]
        return (
          <div key={id} role="alert" data-pending-device={d.device} className="toast-in flex flex-wrap items-center gap-x-4 gap-y-2 rounded-lg border border-line bg-surface px-4 py-3 shadow-[0_6px_24px_-12px_rgb(0_0_0/0.25)]">
            <div className="min-w-0 flex-1 basis-64 text-[13px]">
              <div className="font-medium">{t("A new device asks to use the key {key}", { key: d.key })}</div>
              <div className="mt-0.5 text-ink-2 [overflow-wrap:anywhere]">
                {t("{label} at {addr} · code {code}", { label: d.label || t("(no name)"), addr: d.addr || t("an unknown address"), code: d.code })}
              </div>
              <div className="mt-0.5 text-[12px] text-ink-3">{t("Approve it only if the device shows the same code.")}</div>
            </div>
            <div className="flex shrink-0 gap-2">
              <Button kind="primary" disabled={!!doing} onClick={() => void decide(d, "approve")}>{doing === "approve" ? t("Approving…") : t("Approve")}</Button>
              <Button disabled={!!doing} onClick={() => void decide(d, "refuse")}>{doing === "refuse" ? t("Refusing…") : t("Refuse")}</Button>
            </div>
          </div>
        )
      })}
    </div>
  )
}
