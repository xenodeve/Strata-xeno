import { useCallback, useEffect, useState } from "react"
import { deviceName, setDeviceName } from "../lib/device"
import { createKey, getKeys, logKind, postDevice, postKeys, refreshTeam, showFreshKey, type DeviceRow, type KeyRow, type KeysAction, type KeysAnswer, type KeysView, type LogRow, type Role } from "../lib/keys"
import { msg, t } from "../lib/i18n"
import { store } from "../lib/store"
import { serverHasKey } from "../lib/api"
import { when } from "./bits"
import { ConfirmDialog } from "./ConfirmDialog"
import { toast } from "./toast"
import { Button, Field, Segmented, inputCls } from "./ui"

// Keys and devices (#310, serve/identity.py), on the Settings page: the owner's list of named keys with the devices each one is used from,
// a new key (its text shown once, here, with a copy button), rename, owner or guest, revoke, a device approved, refused or cleared, and the
// log of what was done. A guest sees only this device's name. Outside team mode the owner sees one thing: making the first key, which turns
// team mode on. Never a key's text or hash, except the new key's in the answer to its create.
const ROLE: Record<Role, string> = { owner: msg("Owner"), guest: msg("Guest") }
const STATUS: Record<DeviceRow["status"], string> = { pending: msg("Waiting for approval"), approved: msg("Approved"), refused: msg("Refused") }
const LOG: Record<string, string> = {
  created: msg("{key}: made by {by}"), renamed: msg("{key}: renamed from {arg} by {by}"), role: msg("{key}: made {arg} by {by}"),
  revoked: msg("{key}: revoked by {by}"), pending: msg("{key}: a new device asked to be let in"), approved: msg("{key}: a device approved by {by}"),
  refused: msg("{key}: a device refused by {by}"), cleared: msg("{key}: a device cleared by {by}"),
}
const note = "mt-3 rounded-md bg-fill px-3 py-2 text-[12px] text-ink-2"

export function KeysSettings() {
  const [a, setA] = useState<KeysAnswer | undefined>(undefined)
  const load = useCallback(async () => { setA(await getKeys()) }, [])
  useEffect(() => { void load() }, [load])
  return (
    <section data-keys>
      <h2 className="text-[15px] font-semibold">{t("Keys and devices")}</h2>
      <p className="mt-1 text-[13px] text-ink-2">{t("Named keys let other people use this server, each with a key of their own. A key's first use from a new device waits until the owner approves it.")}</p>
      {a === undefined ? <p className="mt-3 text-[13px] text-ink-3">{t("Reading the keys…")}</p>
        : a.kind === "none" ? <p role="note" className={note}>{t("This server does not manage named keys.")}</p>
        : a.kind === "error" ? <p role="alert" className="mt-3 text-[12.5px] text-bad">{a.message}</p>
        : a.kind === "not-owner" ? <p role="note" className={note}>{t("Keys are managed only on this PC itself, or when Strata has an API key and it is entered in Settings, under API key.")}</p>
        : a.kind === "guest" ? <><ThisDevice /><p role="note" className={note}>{t("Keys are managed with an owner key. A new device of your own key waits for approval; you can approve it from a device already let in, at the top of the page.")}</p></>
        : <Owner view={a.view} reload={load} />}
    </section>
  )
}

/** This device's name, as the owner sees it beside the device (sent with every request, so a change shows on the next one). */
function ThisDevice() {
  const [name, setName] = useState(deviceName)
  return (
    <div className="mt-3">
      <Field label={t("Name of this device")} hint={t("The owner sees this name beside this device. Kept in this browser.")}>
        <input
          className={inputCls} value={name} maxLength={80} aria-label={t("Name of this device")}
          onChange={(e) => setName(e.target.value)}
          onBlur={() => { if (setDeviceName(name)) { setName(deviceName()); toast("success", t("Device name saved")) } else toast("error", t("The browser did not keep the name")) }}
        />
      </Field>
    </div>
  )
}

function Owner({ view, reload }: { view: KeysView; reload: () => Promise<void> }) {
  const [busy, setBusy] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)

  /** A change to the keys; null when it was done, else why not. */
  const act = async (body: KeysAction, done?: string): Promise<string | null> => {
    setBusy(true)
    setNotice(null)
    // A create: the first key turns team mode on, and from then on this page needs a key too, so createKey reads /health before it (after
    // it, "api_key" is true either way) and this browser takes the new key when it has none or one the server never asked for. The text
    // goes to the store App draws above the waiting screen, before anything can take this page away.
    const r = body.action === "create"
      ? await createKey(body, !view.team, { serverHasKey, post: postKeys, stored: () => store.get("apikey", ""), keep: (k) => store.set("apikey", k) })
      : await postKeys(body)
    setBusy(false)
    if ("error" in r) { setNotice(r.error); return r.error }
    if ("fresh" in r) showFreshKey(r.fresh)
    if (done) toast("success", done)
    await reload()
    refreshTeam()
    return null
  }
  const device = async (k: string, d: DeviceRow, how: "approve" | "refuse" | "clear") => {
    setBusy(true)
    setNotice(null)
    const r = await postDevice(k, d.device, how)
    setBusy(false)
    if ("error" in r) setNotice(r.error)
    else toast("success", how === "approve" ? t("Device approved") : how === "refuse" ? t("Device refused") : t("Device cleared"), d.label || d.device)
    await reload()
  }

  return (
    <>
      {!view.keys_file && <p role="note" className={note}>{t("This server was started without a run config file, so there is nowhere to keep keys. Start it with --config.")}</p>}
      {!view.team ? (
        <div className="mt-4" data-keys-first>
          <h3 className="text-[14px] font-semibold">{t("Turn on team mode by creating the first key")}</h3>
          <p className="mt-1 text-[13px] text-ink-2">{t("The first key is an owner key. From then on every request needs a key, this PC's too, and each key's first use from a new device waits for approval. This browser takes the new key as its own, unless it holds the server's API key.")}</p>
          {view.keys_file && <CreateKey first busy={busy} onCreate={(name, role) => act({ action: "create", name, role })} />}
        </div>
      ) : (
        <>
          <ThisDevice />
          <ul className="mt-3 space-y-2" data-key-list>
            {view.keys.map((k) => <KeyItem key={k.name} row={k} busy={busy} act={act} device={device} />)}
          </ul>
          <h3 className="mt-6 text-[14px] font-semibold">{t("New key")}</h3>
          <CreateKey busy={busy} onCreate={(name, role) => act({ action: "create", name, role })} />
          {view.retired.length > 0 && <p className="mt-2 text-[12px] text-ink-3">{t("Names used before, never given again: {names}", { names: view.retired.join(", ") })}</p>}
          <KeysLog rows={view.log} />
        </>
      )}
      {notice && <p role="alert" className="mt-2 text-[12px] text-bad">{notice}</p>}
    </>
  )
}

function CreateKey({ first, busy, onCreate }: { first?: boolean; busy: boolean; onCreate: (name: string, role: Role) => Promise<string | null> }) {
  const [name, setName] = useState("")
  const [role, setRole] = useState<Role>(first ? "owner" : "guest")
  const submit = async () => { if (name.trim() && !(await onCreate(name.trim(), first ? "owner" : role))) setName("") }
  return (
    <form className="mt-2 flex flex-wrap items-end gap-2" onSubmit={(e) => { e.preventDefault(); void submit() }}>
      <label className="min-w-0 flex-1 basis-48">
        <span className="text-[13px]">{t("Name")}</span>
        <input className={`${inputCls} mt-1.5`} value={name} maxLength={64} placeholder={first ? t("for example, your own name") : t("for example, ann")} onChange={(e) => setName(e.target.value)} />
      </label>
      {!first && <div className="w-48"><Segmented value={role} onChange={setRole} label={t("Role")} options={[{ value: "guest", label: t(ROLE.guest) }, { value: "owner", label: t(ROLE.owner) }]} /></div>}
      <Button type="submit" kind="primary" disabled={busy || !name.trim()}>{busy ? t("Making the key…") : first ? t("Make the owner key") : t("Make the key")}</Button>
    </form>
  )
}

function KeyItem({ row, busy, act, device }: {
  row: KeyRow; busy: boolean; act: (b: KeysAction, done?: string) => Promise<string | null>; device: (k: string, d: DeviceRow, how: "approve" | "refuse" | "clear") => Promise<void>
}) {
  const [renaming, setRenaming] = useState<string | null>(null)
  const [revoke, setRevoke] = useState(false)
  const seen = row.devices.reduce<number | null>((m, d) => (d.last_seen != null && (m == null || d.last_seen > m) ? d.last_seen : m), null)
  const other: Role = row.role === "owner" ? "guest" : "owner"
  return (
    <li data-key-name={row.name} className="rounded-md border border-line px-3 py-2">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        {renaming === null ? <span className="text-[13.5px] font-medium [overflow-wrap:anywhere]">{row.name}</span> : (
          <form className="flex min-w-0 flex-1 gap-2" onSubmit={(e) => { e.preventDefault(); void act({ action: "rename", name: row.name, to: renaming.trim() }, t("Key renamed")).then((err) => !err && setRenaming(null)) }}>
            <input className={inputCls} value={renaming} maxLength={64} aria-label={t("New name for {name}", { name: row.name })} autoFocus onChange={(e) => setRenaming(e.target.value)} />
            <Button type="submit" disabled={busy || !renaming.trim() || renaming.trim() === row.name}>{t("Rename")}</Button>
            <Button kind="quiet" onClick={() => setRenaming(null)}>{t("Cancel")}</Button>
          </form>
        )}
        <span className="text-[12px] text-ink-2">{t(ROLE[row.role])}{row.legacy && ` · ${t("the run config's API key")}`}</span>
        <span className="text-[12px] text-ink-3">
          {row.created != null && t("made {when}", { when: when(row.created) })}{seen != null && ` · ${t("last seen {when}", { when: when(seen) })}`}
        </span>
      </div>
      {!row.legacy && renaming === null && (
        <div className="mt-1.5 flex flex-wrap gap-1.5">
          <Button kind="quiet" disabled={busy} onClick={() => setRenaming(row.name)}>{t("Rename")}</Button>
          <Button kind="quiet" disabled={busy} onClick={() => void act({ action: "role", name: row.name, role: other }, other === "owner" ? t("Now an owner key") : t("Now a guest key"))}>{other === "owner" ? t("Make an owner") : t("Make a guest")}</Button>
          <Button kind="quiet" disabled={busy} onClick={() => setRevoke(true)}>{t("Revoke")}</Button>
        </div>
      )}
      {row.devices.length === 0 ? <p className="mt-1.5 text-[12px] text-ink-3">{t("Not used from any device yet.")}</p> : (
        <ul className="mt-1.5 space-y-1" data-key-devices>
          {row.devices.map((d) => (
            <li key={d.device} data-device={d.device} data-device-status={d.status} className="flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-line pt-1.5 text-[12.5px]">
              <span className="min-w-0 flex-1 basis-48 [overflow-wrap:anywhere]">
                {d.label || t("(no name)")}<span className="text-ink-3"> · {d.addr || t("an unknown address")}</span>
                <span className="block text-[11.5px] text-ink-3">{t(STATUS[d.status])}{d.decided_by && d.status !== "pending" && ` · ${t("by {name}", { name: d.decided_by })}`}{d.last_seen != null && ` · ${t("last seen {when}", { when: when(d.last_seen) })}`}</span>
              </span>
              <span className="flex shrink-0 gap-1">
                {d.status !== "approved" && <Button kind="quiet" disabled={busy} onClick={() => void device(row.name, d, "approve")}>{t("Approve")}</Button>}
                {d.status !== "refused" && <Button kind="quiet" disabled={busy} onClick={() => void device(row.name, d, "refuse")}>{t("Refuse")}</Button>}
                <Button kind="quiet" disabled={busy} label={t("Clear this device: it waits for approval again the next time it is used")} onClick={() => void device(row.name, d, "clear")}>{t("Clear")}</Button>
              </span>
            </li>
          ))}
        </ul>
      )}
      <ConfirmDialog
        open={revoke}
        title={t("Revoke the key {name}?", { name: row.name })}
        text={t("Every request with this key is refused from now on, and its runs still going are stopped. The name is never given to a key again.")}
        confirmLabel={t("Revoke")}
        onCancel={() => setRevoke(false)}
        onConfirm={() => { setRevoke(false); void act({ action: "revoke", name: row.name }, t("Key revoked")) }}
      />
    </li>
  )
}

function KeysLog({ rows }: { rows: LogRow[] }) {
  if (!rows.length) return null
  return (
    <>
      <h3 className="mt-6 text-[14px] font-semibold">{t("What was done")}</h3>
      <ul className="mt-2 space-y-0.5 text-[12.5px]" data-keys-log>
        {rows.slice().reverse().map((r, i) => {
          const { kind, arg } = logKind(r.what)
          const role = arg === "owner" || arg === "guest" ? t(ROLE[arg]) : arg
          return (
            <li key={i} className="flex gap-3">
              <span className="num shrink-0 text-ink-3">{when(r.t)}</span>
              <span className="min-w-0 [overflow-wrap:anywhere]">{kind in LOG ? t(LOG[kind], { key: r.key, by: r.by || t("the server"), arg: role }) : `${r.key}: ${r.what}`}</span>
            </li>
          )
        })}
      </ul>
    </>
  )
}
