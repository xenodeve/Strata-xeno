import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import "./styles.css"
import "./prompt-bar.css"
import "./branched-menu.css"
import "./thought.css"
import { App } from "./App"
import { initLang } from "./lib/i18n"

initLang()
// #333: a file or a link dropped anywhere the page has no drop handler of its own must not navigate away from the app
// (a browser opens the file in place of the chat; Strata Desktop would open a dropped link in the system browser).
// The chat's own drop handler (pages/chat/Chat.tsx onDrop) still runs: this only stops the browser's default, and only
// for files and links outside a field (text, or a link dropped into the prompt box, is still put there as text).
const navigatesAway = (e: DragEvent) => {
  const types = Array.from(e.dataTransfer?.types ?? [])
  const field = e.target instanceof HTMLElement && (e.target.isContentEditable || /^(INPUT|TEXTAREA)$/.test(e.target.tagName))
  return types.includes("Files") || (types.includes("text/uri-list") && !field)
}
for (const kind of ["dragover", "drop"] as const) window.addEventListener(kind, (e) => { if (navigatesAway(e)) e.preventDefault() })
createRoot(document.getElementById("root")!).render(<StrictMode><App /></StrictMode>)
