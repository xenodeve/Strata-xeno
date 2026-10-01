import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react"
import {
  Activity01Icon, ComputerIcon, DashboardSquare01Icon, CpuIcon, InformationCircleIcon, Message01Icon, Moon02Icon, Sun03Icon, Task01Icon,
} from "@hugeicons/core-free-icons"
import { useEffect, useLayoutEffect, useRef, useState } from "react"
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
  const [scrolled, setScrolled] = useState(false)
  // A new page opens at its top. (Layout effect: it runs before the chat's own effect, which takes the page to the end of the chat.)
  const first = useRef(true)
  useLayoutEffect(() => {
    if (first.current) { first.current = false; return }
    scrollTo({ top: 0, behavior: "instant" as ScrollBehavior })
  }, [route.page])
  useEffect(() => {
    const on = () => setScrolled(scrollY > 4)
    on()
    addEventListener("scroll", on, { passive: true })
    return () => removeEventListener("scroll", on)
  }, [])
  return (
    <div className="flex min-h-dvh flex-col">
      {/* Stays at the top while the page scrolls under it: a hairline appears only once something is under it. */}
      <header
        className={cn(
          "sticky top-0 z-20 border-b backdrop-blur-md transition-[border-color,background-color] duration-300",
          scrolled ? "border-line bg-[color-mix(in_srgb,var(--bg)_82%,transparent)]" : "border-transparent bg-bg",
        )}
      >
      <div className="mx-auto flex max-w-5xl items-center gap-3 px-4 py-3 sm:px-6">
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
      </div>
      </header>
      <ToastHost />
      <main key={route.page} className="page-in mx-auto w-full max-w-5xl flex-1 px-4 pb-12 pt-4 sm:px-6">
        <PageView route={route} />
      </main>
    </div>
  )
}
