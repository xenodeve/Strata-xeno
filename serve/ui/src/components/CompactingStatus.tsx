import { Orb as CompactingOrb } from "@yogesharc/thinking-orbs"
import { t } from "../lib/i18n"

// While a conversation is being summarised (issue #99) its status is the orb that thinkingorbs.com made for exactly that, "compacting": the dots pack tight and spring back. It is
// a different library from the nine orbs of the other statuses (components/orb.tsx), drawn in the text colour, and a still picture under reduced motion.
export function CompactingStatus({ className = "" }: { className?: string }) {
  return (
    <span className={`inline-flex items-center gap-2 ${className}`} role="status" data-compacting-status>
      <CompactingOrb state="compacting" size={20} label={t("Compacting the conversation")} className="shrink-0 text-ink-2" />
      <span className="t-shimmer">{t("Compacting the conversation…")}</span>
    </span>
  )
}
