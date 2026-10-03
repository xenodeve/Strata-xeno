import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import "./styles.css"
import "./prompt-bar.css"
import "./branched-menu.css"
import "./thought.css"
import { App } from "./App"
import { initLang } from "./lib/i18n"

initLang()
createRoot(document.getElementById("root")!).render(<StrictMode><App /></StrictMode>)
