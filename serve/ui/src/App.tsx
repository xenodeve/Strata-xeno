import { useEffect, useLayoutEffect, useRef, useState } from "react"
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react"
import {
  Activity01Icon, ComputerIcon, CpuIcon, DashboardSquare01Icon, InformationCircleIcon, Message01Icon, Moon02Icon, Sun03Icon, Task01Icon,
} from "@hugeicons/core-free-icons"
import { cn } from "./lib/cn"
import { MetricsProvider, useMetrics } from "./lib/metrics"
import { href, PAGES, useRoute, type Page } from "./lib/router"
import { useTheme, type Theme } from "./lib/theme"
import { PageView } from "./pages"
import { StatusOrb } from "./components/live"
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

/** The page links, with one pill that glides to the current page (measured, then moved with transform and width). */
function Nav({ page }: { page: Page }) {
  const nav = useRef<HTMLElement>(null)
  const [pill, setPill] = useState<{ x: number; w: number } | null>(null)
  const [ready, setReady] = useState(false)               // no glide on the first placement
  useLayoutEffect(() => {
    const el = nav.current
    if (!el) return
    const place = () => {
      const a = el.querySelector<HTMLElement>('[aria-current="page"]')
      if (a) setPill({ x: a.offsetLeft, w: a.offsetWidth })
    }
    place()
    const ro = new ResizeObserver(place)
    ro.observe(el)
    document.fonts?.ready.then(place)
    const id = requestAnimationFrame(() => setReady(true))
    return () => { ro.disconnect(); cancelAnimationFrame(id) }
  }, [page])
  return (
    <nav ref={nav} aria-label="Pages" className="relative -mx-1 flex min-w-0 flex-1 gap-0.5 overflow-x-auto px-1">
      {pill && (
        <span
          aria-hidden
          className={cn("absolute inset-y-0 left-0 my-auto h-8 rounded-sm bg-fill", ready && "transition-[transform,width] duration-[420ms] ease-[var(--ease)]")}
          style={{ transform: `translateX(${pill.x}px)`, width: pill.w }}
        />
      )}
      {PAGES.map((p) => (
        <a
          key={p}
          href={href(p)}
          aria-current={page === p ? "page" : undefined}
          aria-label={NAV[p].label}
          className={cn(
            "relative flex h-8 shrink-0 items-center gap-1.5 rounded-sm px-2 text-[13px] sm:px-2.5 no-underline transition-colors duration-200",
            page === p ? "text-ink" : "text-ink-2 hover:text-ink",
          )}
        >
          <HugeiconsIcon icon={NAV[p].icon} size={16} strokeWidth={1.6} aria-hidden />
          <span className={page === p ? "" : "max-sm:sr-only"}>{NAV[p].label}</span>
        </a>
      ))}
    </nav>
  )
}

function Brand() {
  const { data, stale } = useMetrics()
  return (
    <a href={href("dashboard")} className="flex items-center gap-2 no-underline" aria-label="Strata, the dashboard">
      <StatusOrb live={data?.live ?? { state: "idle", queued: 0, tok_s: null }} stale={stale} size={20} />
      <span className="text-[15px] font-semibold tracking-[-0.02em] max-[430px]:sr-only">Strata</span>
    </a>
  )
}

function Shell() {
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
        <div className="mx-auto flex max-w-5xl items-center gap-2 px-3 py-3 sm:gap-4 sm:px-6">
          <Brand />
          <Nav page={route.page} />
          <button
            type="button"
            onClick={cycle}
            title={`Theme: ${theme}`}
            aria-label={`Theme: ${theme}. Change`}
            className="flex size-8 shrink-0 items-center justify-center rounded-sm text-ink-2 transition-[color,background-color,transform] duration-200 hover:bg-hover hover:text-ink active:scale-90"
          >
            <HugeiconsIcon icon={THEME_ICON[theme]} size={16} strokeWidth={1.6} aria-hidden />
          </button>
        </div>
      </header>
      <ToastHost />
      <main key={route.page} className="page-in mx-auto w-full max-w-5xl flex-1 px-4 pb-16 pt-6 sm:px-6">
        <PageView route={route} />
      </main>
    </div>
  )
}

export function App() {
  return <MetricsProvider><Shell /></MetricsProvider>
}
