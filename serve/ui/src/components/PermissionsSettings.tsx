import { useState } from "react"
import { HugeiconsIcon } from "@hugeicons/react"
import { Cancel01Icon } from "@hugeicons/core-free-icons"
import { chat, useChatVersion } from "../lib/chat"
import { addPerm, loadPerms, removePerm, ruleSetOf, validRule, type Effect, type Scope } from "../lib/perms"
import { store } from "../lib/store"
import { t } from "../lib/i18n"
import { cn } from "../lib/cn"
import { Button, Segmented, inputCls } from "./ui"

// The rules of the chat's coding tools that last (issue #99): for everything and for each project, the ones that are allowed and the ones that are never allowed, in Claude Code's syntax.
// "Allow for this chat" is kept with the chat and is not here. The server decides: a "never" beats an allow, and no rule settles a secret, a change in .git or a dangerous command.

function Rules({ scope, effect, rules, onRemove }: { scope: Scope; effect: Effect; rules: string[]; onRemove: (rule: string) => void }) {
  if (!rules.length) return <p className="px-1 text-[12.5px] text-ink-3">{effect === "allow" ? t("Nothing is allowed here for good.") : t("Nothing is refused here for good.")}</p>
  return (
    <ul className="space-y-0.5">
      {rules.map((r) => (
        <li key={r} data-perm-rule={r} data-effect={effect} data-perm-scope={scope.kind === "everywhere" ? "everywhere" : scope.id} className="flex items-center gap-2 rounded-sm px-1.5 py-1 hover:bg-hover">
          <span className="min-w-0 flex-1 font-mono text-[12.5px] [overflow-wrap:anywhere]">{r}</span>
          <button type="button" aria-label={t("Remove the rule {rule}", { rule: r })} onClick={() => onRemove(r)} className="flex size-6 shrink-0 items-center justify-center rounded-sm text-ink-3 transition-colors hover:text-ink">
            <HugeiconsIcon icon={Cancel01Icon} size={13} aria-hidden />
          </button>
        </li>
      ))}
    </ul>
  )
}

function ScopeBlock({ scope, title, note, changed }: { scope: Scope; title: string; note?: string; changed: () => void }) {
  const [effect, setEffect] = useState<Effect>("allow")
  const [draft, setDraft] = useState("")
  const [problem, setProblem] = useState<string | null>(null)
  const set = ruleSetOf(loadPerms(store), scope)
  const key = scope.kind === "everywhere" ? "everywhere" : scope.id
  const add = () => {
    const text = draft.trim()
    if (!text) return
    if (!validRule(text)) { setProblem(t("That is not a rule. A rule is a tool's name, and between brackets what it is limited to: Bash(npm test:*)")); return }
    const known = set[effect]
    if (known.includes(text)) { setProblem(t("That rule is there already.")); return }
    if (!addPerm(store, scope, effect, text)) { setProblem(t("The rule could not be added: there may be too many.")); return }
    setDraft(""); setProblem(null); changed()
  }
  return (
    <section data-perm-block={key} className="border-t border-line py-3">
      <div className="flex items-baseline gap-2"><h3 className="text-[13px] font-semibold">{title}</h3>{note && <span className="text-[12px] text-ink-3">{note}</span>}</div>
      <div className="mt-2 grid gap-3 sm:grid-cols-2">
        <div><div className="mb-1 px-1 text-[11.5px] font-medium uppercase tracking-wide text-ink-3">{t("Allowed")}</div><Rules scope={scope} effect="allow" rules={set.allow} onRemove={(r) => { removePerm(store, scope, "allow", r); changed() }} /></div>
        <div><div className="mb-1 px-1 text-[11.5px] font-medium uppercase tracking-wide text-ink-3">{t("Never")}</div><Rules scope={scope} effect="deny" rules={set.deny} onRemove={(r) => { removePerm(store, scope, "deny", r); changed() }} /></div>
      </div>
      <form className="mt-3 flex flex-wrap items-center gap-2" onSubmit={(e) => { e.preventDefault(); add() }}>
        <Segmented label={t("Allow or never")} value={effect} onChange={(v) => setEffect(v as Effect)} options={[{ value: "allow", label: t("Allow") }, { value: "deny", label: t("Never") }]} />
        <input className={cn(inputCls, "min-w-[12rem] flex-1 font-mono")} value={draft} aria-label={t("A rule for {where}", { where: title })} placeholder="Bash(npm test:*)" autoComplete="off" spellCheck={false} onChange={(e) => { setDraft(e.target.value); setProblem(null) }} />
        <Button type="submit" disabled={!draft.trim()}>{t("Add")}</Button>
      </form>
      {problem && <p role="alert" className="mt-1.5 text-[12px] text-bad">{problem}</p>}
    </section>
  )
}

export function PermissionsSettings() {
  useChatVersion()
  const [, bump] = useState(0)
  const changed = () => bump((n) => n + 1)
  return (
    <section data-permissions>
      <h2 className="text-[15px] font-semibold">{t("Permissions")}</h2>
      <p className="mt-1 text-[13px] text-ink-2">{t("What the chat's coding tools may do without asking, for good. A card that asks can keep your answer here (More choices), or you can write a rule. \"Allow for this chat\" is kept with the chat only and is not listed.")}</p>
      <p className="mt-2 text-[13px] text-ink-2">{t("A rule is a tool and, between brackets, what it is limited to: Bash(npm test:*) for commands that start with npm test, Read(src/**) for files under src, Edit(docs/**), or just Read for all of that tool. \"Never\" wins over \"Allow\". No rule can allow a secret, a change in .git or a command that can do harm that is hard to undo: those ask every time.")}</p>
      <div className="mt-3">
        <ScopeBlock scope={{ kind: "everywhere" }} title={t("Everywhere")} note={t("every chat, every project")} changed={changed} />
        {chat.index.projects.map((p) => <ScopeBlock key={p.id} scope={{ kind: "project", id: p.id }} title={p.name} note={t("this project")} changed={changed} />)}
        {chat.index.projects.length === 0 && <p className="border-t border-line pt-3 text-[12.5px] text-ink-3">{t("Projects you make will be listed here, each with its own rules.")}</p>}
      </div>
    </section>
  )
}
