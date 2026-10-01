import { About } from "./About"
import { Dashboard } from "./Dashboard"
import { GpuPage, Hardware, SsdPage } from "./Hardware"
import { Live } from "./Live"
import { Requests, RequestDetailPage } from "./Requests"
import { TracePage } from "./Trace"
import { Chat } from "./chat/Chat"
import type { Route } from "../lib/router"

export function PageView({ route }: { route: Route }) {
  switch (route.page) {
    case "chat":
      return <Chat />
    case "dashboard":
      return <Dashboard />
    case "live":
      return <Live />
    case "requests":
      return route.params[0] === "trace" ? <TracePage /> : route.params[0] ? <RequestDetailPage id={route.params[0]} /> : <Requests />
    case "hardware": {
      const [kind, n] = route.params
      return kind === "gpu" && n ? <GpuPage n={n} /> : kind === "ssd" && n ? <SsdPage n={n} /> : <Hardware />
    }
    case "about":
      return <About />
  }
}
