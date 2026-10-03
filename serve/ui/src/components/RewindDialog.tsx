import { useEffect, useId, useRef, useState } from "react"
import { createPortal } from "react-dom"
import { applyRewind, previewRewind, rewindSummary, shortPath, type RewindFile, type RewindWhat } from "../lib/rewind"
import { cn } from "../lib/cn"
import { t } from "../lib/i18n"
import { Collapse, GlidePanel, useMounted } from "./motion"
import { toast } from "./toast"
import { Button } from "./ui"

// Rewind (issue #99): go back to before a prompt. The files the chat's tools changed from that prompt on are put back (a file the tools made is deleted), the conversation is cut at the
// prompt, or both. Before anything is done the dialog says which files, and which of them were changed by someone else since (they are left unless asked for). What the shell tool did
// to files is not tracked, and the dialog says so.

export interface RewindPrompt { index: number; text: string; checkpoint: string }

const WHAT: { value: RewindWhat; label: () => string; hint: () => string }[] = [
  { value: "both", label: () => t("The files and the conversation"), hint: () => t("Back to how it was before this prompt.") },
  { value: "files", label: () => t("Only the files"), hint: () => t("The conversation stays as it is.") },
  { value: "conversation", label: () => t("Only the conversation"), hint: () => t("The files stay as they are.") },
]

export function RewindDialog({ open, prompts, start, session, onCancel, onDone }: {
  open: boolean
  prompts: RewindPrompt[]                        // the prompts of the chat, oldest first
  start: number                                  // which of them it opens on (an index into `prompts`)
  session: string | null
  onCancel: () => void
  onDone: (conversation: { index: number } | null) => void       // the conversation is to be cut at this prompt, or not
}) {
  const mounted = useMounted(open, 220)
  const [shown, setShown] = useState(false)
  const [at, setAt] = useState(start)
  const [what, setWhat] = useState<RewindWhat>("both")
  const [files, setFiles] = useState<RewindFile[] | "loading" | "error" | null>(null)
  const [more, setMore] = useState(false)                              // also put back the files that were changed since
  const [busy, setBusy] = useState(false)
  const [problem, setProblem] = useState<string | null>(null)
  const panel = useRef<HTMLDivElement>(null)
  const was = useRef<Element | null>(null)
  const head = useId()
  const cancelRef = useRef(onCancel)
  cancelRef.current = onCancel

  useEffect(() => {
    if (!open) { setShown(false); return }
    setAt(Math.min(start, Math.max(0, prompts.length - 1))); setWhat("both"); setMore(false); setBusy(false); setProblem(null)
    was.current = document.activeElement
    const id = requestAnimationFrame(() => { setShown(true); panel.current?.querySelector<HTMLElement>("[data-first]")?.focus() })
    const key = (e: KeyboardEvent) => {
      if (e.key === "Escape") { e.preventDefault(); cancelRef.current() }
      else if (e.key === "Tab" && panel.current) {
        const f = [...panel.current.querySelectorAll<HTMLElement>("button:not(:disabled), input, select")]
        const first = f[0], last = f[f.length - 1]
        if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus() }
        else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus() }
      }
    }
    document.addEventListener("keydown", key)
    return () => { cancelAnimationFrame(id); document.removeEventListener("keydown", key); (was.current as HTMLElement | null)?.focus?.() }
  }, [open]) // eslint-disable-line react-hooks/exhaustive-deps

  const prompt = prompts[at]
  useEffect(() => {                                                    // what a rewind to this prompt would do to the files
    if (!open || !prompt || !session) { setFiles(null); return }
    let stale = false
    setFiles("loading")
    void previewRewind(session, prompt.checkpoint).then((r) => { if (!stale) setFiles("error" in r ? "error" : r.files) })
    return () => { stale = true }
  }, [open, at, session, prompt?.checkpoint]) // eslint-disable-line react-hooks/exhaustive-deps

  const list = Array.isArray(files) ? files : []
  const touches = list.filter((f) => f.action !== "skip")
  const changed = list.filter((f) => f.changed)
  const filesApply = what !== "conversation"
  const go = async () => {
    if (!prompt) return
    setBusy(true)
    setProblem(null)
    if (filesApply && session && touches.length) {
      const r = await applyRewind(session, prompt.checkpoint, more)
      if ("error" in r) { setBusy(false); setProblem(r.error); return }
      const s = rewindSummary(r)
      const parts = [s.restored ? t("{n} files put back", { n: s.restored }) : "", s.deleted ? t("{n} files deleted", { n: s.deleted }) : "", s.kept ? t("{n} left as you changed them", { n: s.kept }) : "", s.failed ? t("{n} could not be changed", { n: s.failed }) : ""].filter(Boolean)
      onDone(what === "both" ? { index: prompt.index } : null)
      toastDone(parts.join(" · ") || t("Nothing to put back."))
    } else {
      onDone(what !== "files" ? { index: prompt.index } : null)
    }
    setBusy(false)
  }

  if (!mounted) return null
  return createPortal(
    <div className="fixed inset-0 z-50 grid place-items-center p-4">
      <div aria-hidden className={cn("absolute inset-0 bg-black/45 transition-opacity duration-200", shown && open ? "opacity-100" : "opacity-0")} onClick={onCancel} />
      <div
        ref={panel}
        role="dialog"
        aria-modal="true"
        aria-labelledby={head}
        data-rewind
        className={cn("relative w-full max-w-[480px] overflow-hidden rounded-xl border border-line bg-surface shadow-[0_24px_64px_-16px_rgb(0_0_0/0.4)] transition-[opacity,transform] duration-200 ease-[var(--ease)]", shown && open ? "translate-y-0 scale-100 opacity-100" : "translate-y-2 scale-[0.98] opacity-0")}
      >
        <GlidePanel cap={() => innerHeight - 34} inner="p-5" className="[scrollbar-gutter:auto]">
          <h2 id={head} className="text-[16px] font-semibold">{t("Rewind to a prompt")}</h2>
          <p className="mt-1.5 text-[13px] leading-relaxed text-ink-2">{t("Go back to before this prompt: the files the coding tools changed from it on are put back, and the conversation is cut here.")}</p>

          {prompts.length > 1 ? (
            <label className="mt-3 block">
              <span className="text-[13px]">{t("Back to before")}</span>
              <select data-first className="mt-1.5 h-8 w-full rounded-sm border border-line bg-surface px-2 text-[13px]" value={at} onChange={(e) => setAt(+e.target.value)} aria-label={t("Back to before")}>
                {prompts.map((p, i) => <option key={p.checkpoint} value={i}>{`${i + 1}. ${p.text.replace(/\s+/g, " ").slice(0, 70)}`}</option>)}
              </select>
            </label>
          ) : (
            <p className="mt-3 rounded-sm bg-fill px-3 py-2 text-[13px] [overflow-wrap:anywhere]" data-first tabIndex={-1}>{prompt?.text.slice(0, 200)}</p>
          )}

          <div className="mt-3" data-rewind-files>
            <div className="mb-1.5 text-[13px]">{t("Files")}</div>
            {files === "loading" || files === null ? <p className="text-[12.5px] text-ink-3">{t("Looking at what changed…")}</p>
              : files === "error" ? <p className="text-[12.5px] text-bad">{t("The server could not say which files changed.")}</p>
              : list.length === 0 ? <p className="text-[12.5px] text-ink-2" data-rewind-none>{t("The coding tools changed no file from this prompt on.")}</p>
              : (
                <ul className="max-h-48 space-y-0.5 overflow-y-auto rounded-sm border border-line p-1">
                  {list.map((f) => (
                    <li key={f.path} data-rewind-file={f.path} data-action={f.action} data-changed={f.changed ? "" : undefined} className="flex items-center gap-2 px-1.5 py-1 text-[12.5px]">
                      <span className="min-w-0 flex-1 truncate font-mono" title={f.path}>{shortPath(f.path)}</span>
                      <span className={cn("shrink-0 text-[11.5px]", f.action === "delete" ? "text-bad" : f.action === "skip" ? "text-ink-3" : "text-ink-2")}>
                        {f.action === "delete" ? t("deleted: it was made") : f.action === "skip" ? t("cannot be put back: {why}", { why: f.why ?? "" }) : t("put back")}
                      </span>
                      {f.changed && <span className="shrink-0 rounded-sm bg-fill px-1.5 text-[11px] text-ink-2" title={t("Changed by you, or by a command, since the tools wrote it")}>{t("changed since")}</span>}
                    </li>
                  ))}
                </ul>
              )}
            <p className="mt-1.5 text-[12px] text-ink-3">{t("What commands did to files is not undone.")}</p>
            <Collapse open={changed.length > 0 && filesApply}>
              <label className="mt-2 flex items-start gap-2 text-[12.5px]"><input type="checkbox" checked={more} onChange={(e) => setMore(e.target.checked)} className="mt-0.5" /><span>{t("Also put back the files that were changed since (what was changed will be lost)")}</span></label>
            </Collapse>
          </div>

          <div role="radiogroup" aria-label={t("What to go back")} className="mt-3 space-y-1.5">
            {WHAT.map((w) => (
              <label key={w.value} className={cn("flex cursor-pointer items-start gap-2 rounded-sm border px-2.5 py-2 text-[13px] transition-colors", what === w.value ? "border-accent bg-fill" : "border-line hover:bg-hover")}>
                <input type="radio" name={head + "-what"} value={w.value} checked={what === w.value} onChange={() => setWhat(w.value)} className="mt-0.5" />
                <span><span className="block font-medium">{w.label()}</span><span className="block text-[12px] text-ink-2">{w.hint()}</span></span>
              </label>
            ))}
          </div>
          {problem && <p role="alert" className="mt-2 text-[12px] text-bad">{problem}</p>}

          <div className="mt-5 flex justify-end gap-2">
            <Button kind="quiet" onClick={onCancel}>{t("Cancel")}</Button>
            <Button kind="primary" disabled={busy || !prompt || files === "loading"} onClick={() => void go()}>{t("Rewind")}</Button>
          </div>
        </GlidePanel>
      </div>
    </div>,
    document.body,
  )
}

function toastDone(text: string) { toast("success", t("Rewound"), text) }
