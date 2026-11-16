// Where a chat's data goes, in the line the page says it in (#310 final review I-2). "Nothing leaves it." is true only for a page opened
// on this PC itself, in a chat whose tools run here too. A browser on another PC sends its prompts - and what the tools read for them -
// to the PC that runs the model; a My PC chat also sends there what the tools read on the engineer's own PC, and its commands run there.
import { t } from "./i18n"
import type { Machine } from "./runner"

/** A page opened on this PC itself: localhost (or a name under .localhost), 127.0.0.0/8 or ::1 (`location.hostname` gives it as [::1]). */
export function isLoopbackHost(hostname: string): boolean {
  const h = hostname.toLowerCase().replace(/^\[(.*)\]$/, "$1")
  return h === "localhost" || h.endsWith(".localhost") || /^127(\.\d{1,3}){3}$/.test(h) || h === "::1"
}

/** The empty chat's line: which PC the model `name` runs on, and where the chat's data goes. `machine`: where the chat's tools run. */
export function chatWhere(name: string, hostname: string, machine: Machine): string {
  const here = isLoopbackHost(hostname)
  if (machine === "my") {
    return here
      ? t("{name} runs on this PC. The files the tools read on your PC come here. Commands run on your PC.", { name })
      : t("{name} runs on the server PC ({host}). Your prompts, and the files the tools read on your PC, go there. Commands run on your PC.", { name, host: hostname })
  }
  return here
    ? t("{name} runs on this PC. Nothing leaves it.", { name })
    : t("{name} runs on the server PC ({host}). Your prompts, and what the tools read for them, go there.", { name, host: hostname })
}

/** The About page's line, by the same rule (no chat: the Server PC's tools). */
export function aboutWhere(hostname: string): string {
  return isLoopbackHost(hostname)
    ? t("Strata runs a large mixture-of-experts model on this PC. Nothing leaves it.")
    : t("Strata runs a large mixture-of-experts model on the server PC ({host}). Your prompts, and what the tools read for them, go there.", { host: hostname })
}
