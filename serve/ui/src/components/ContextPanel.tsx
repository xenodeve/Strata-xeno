import { msg, t } from "../lib/i18n"
import { fmt } from "../lib/format"
import type { ContextView, PartKey } from "../lib/context"
import { Button } from "./ui"

// What the context window holds (issue #99): how much is used, what it is made of, where it is compacted by itself. In the panel the context chip of the prompt bar opens, and in the
// right panel's Context tab. The styles are the prompt bar's (prompt-bar.css).

const PART_LABEL: Record<PartKey, string> = {
  conversation: msg("Conversation"), tools: msg("Tool calls and results"), summary: msg("Summary of earlier messages"), system: msg("Instructions, tools and memory"), free: msg("Free"),
}

export function ContextPanel({ view, canCompact, onCompact }: { view: ContextView; canCompact: boolean; onCompact: () => void }) {
  const shown = view.parts.filter((x) => x.key === "free" || x.tokens > 0)
  if (!view.known) return <p className="text-[13px] text-ink-2" data-context-panel>{t("The server did not say how big the context window is, so nothing can be shown.")}</p>
  return (
    <div data-context-panel>
      <div className="prompt-bar__ctx-head">
        <span className="prompt-bar__ctx-title">{t("Context window")}</span>
        <span className="num prompt-bar__ctx-pct" data-level={view.level}>{view.pct}%</span>
      </div>
      <p className="num mt-0.5 text-[12px] text-ink-2" data-context-figures>{view.exact ? "" : t("About") + " "}{t("{used} of {max} tokens", { used: fmt(view.used), max: fmt(view.max) })}</p>
      <div className="prompt-bar__ctx-bar" role="img" aria-label={t("What the context window holds")}>
        {shown.map((x) => <span key={x.key} data-part={x.key} style={{ flexGrow: x.tokens }} />)}
      </div>
      <ul className="prompt-bar__ctx-list">
        {shown.map((x) => (
          <li key={x.key} data-part={x.key}>
            <i aria-hidden data-part={x.key} />
            <span>{t(PART_LABEL[x.key])}</span>
            <span className="num">{fmt(x.tokens)}</span>
          </li>
        ))}
      </ul>
      {view.autoAt !== null && <p className="mt-2 text-[12px] text-ink-2" data-context-auto>{t("Compacted by itself at {n} tokens ({pct}%).", { n: fmt(view.autoAt), pct: Math.round((view.autoAt / view.max) * 100) })}</p>}
      {!view.exact && <p className="mt-1 text-[12px] text-ink-3">{t("The server has not reported the use yet: this is a guess from the text.")}</p>}
      <div className="mt-3"><Button onClick={onCompact} disabled={!canCompact}>{t("Compact now")}</Button></div>
    </div>
  )
}
