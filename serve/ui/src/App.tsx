import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react"
import {
  Activity01Icon, ComputerIcon, DashboardSquare01Icon, CpuIcon, InformationCircleIcon, Message01Icon, Moon02Icon, Sun03Icon, Task01Icon,
} from "@hugeicons/core-free-icons"
import { cn } from "./lib/cn"
import { href, PAGES, useRoute, type Page } from "./lib/router"
import { useTheme, type Theme } from "./lib/theme"
import { PageView } from "./pages"
import { ToastHost } from "./components/toast"

const NAV: Record<Page, { label: string; icon: IconSvgElement }> = {
  chat: { label: "Chat", icon: Message01Icon },
  dashboard: { label: "Dashboard", icon: DashboardSquare01Icon },
  live: { label: "Live", icon: Activity01Icon },
  requests: { label: "Requests", icon: Task01Icon },
  hardware: { label: "Hardware", icon: CpuIcon },
  about: { label: "About", icon: InformationCircleIcon },
}

const THEME_ICON: Record<Theme, IconSvgElement> = { system: ComputerIcon, light: Sun03Icon, dark: Moon02Icon }

export function App() {
  const route = useRoute()
  const { theme, cycle } = useTheme()
  return (
    <div className="mx-auto flex min-h-dvh max-w-5xl flex-col px-4 sm:px-6">
      <header className="flex items-center gap-3 py-4">
        <span className="text-[15px] font-semibold tracking-tight">Strata</span>
        <nav aria-label="Pages" className="-mx-1 flex min-w-0 flex-1 gap-0.5 overflow-x-auto px-1">
          {PAGES.map((p) => (
            <a
              key={p}
              href={href(p)}
              aria-current={route.page === p ? "page" : undefined}
              aria-label={NAV[p].label}
              className={cn(
                "flex h-8 shrink-0 items-center gap-1.5 rounded-sm px-2.5 text-[13px] no-underline transition-colors duration-150",
                route.page === p ? "bg-fill text-ink" : "text-ink-2 hover:bg-hover hover:text-ink",
              )}
            >
              <HugeiconsIcon icon={NAV[p].icon} size={16} strokeWidth={1.6} aria-hidden />
              <span className={route.page === p ? "" : "max-sm:sr-only"}>{NAV[p].label}</span>
            </a>
          ))}
        </nav>
        <button
          type="button"
          onClick={cycle}
          title={`Theme: ${theme}`}
          aria-label={`Theme: ${theme}. Change`}
          className="flex size-8 shrink-0 items-center justify-center rounded-sm text-ink-2 transition-colors duration-150 hover:bg-hover hover:text-ink"
        >
          <HugeiconsIcon icon={THEME_ICON[theme]} size={16} strokeWidth={1.6} aria-hidden />
        </button>
      </header>
      <ToastHost />
      <main key={route.page} className="page-in flex-1 pb-12 pt-4">
        <PageView route={route} />
      </main>
    </div>
  )
}
