import type { ReactNode } from "react"
import { About } from "./About"
import { Live } from "./Live"
import { Requests, RequestDetailPage } from "./Requests"
import { Chat } from "./chat/Chat"
import type { Route } from "../lib/router"

function Stub({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="max-w-[65ch]">
      <h1 className="text-xl font-semibold">{title}</h1>
      <p className="mt-2 text-ink-2">{children}</p>
    </section>
  )
}

// The next UI is built slice by slice (UI handoff 2026-10-01). Until each page lands it says so, and points at
// the classic app, which keeps working at classic/.
export function PageView({ route }: { route: Route }) {
  switch (route.page) {
    case "chat":
      return <Chat />
    case "live":
      return <Live />
    case "requests":
      return route.params[0] ? <RequestDetailPage id={route.params[0]} /> : <Requests />
    case "hardware":
      return <Stub title="Hardware">Each GPU, the CPU, RAM, PCIe and SSD, from what the engine itself did. Not built yet.</Stub>
    case "about":
      return <About />
  }
}
