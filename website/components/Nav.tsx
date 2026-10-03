"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { T, useLang, type L } from "@/lib/i18n";
import { GitHubButton } from "./GitHubButton";

export const SECTIONS: { id: string; label: L }[] = [
  { id: "story", label: { en: "Moods", th: "อารมณ์" } },
  { id: "demo", label: { en: "Demo", th: "Demo" } },
  { id: "app", label: { en: "Classic → new", th: "คลาสสิก → ใหม่" } },
  { id: "strata", label: { en: "The strata", th: "ชั้นของ Strata" } },
  { id: "numbers", label: { en: "Numbers", th: "ตัวเลข" } },
];

type Theme = "system" | "light" | "dark";
const THEME_KEY = "strata-xeno.theme";

function ThemeToggle() {
  const { lang } = useLang();
  const [theme, setTheme] = useState<Theme>("system");

  useEffect(() => {
    try {
      const saved = localStorage.getItem(THEME_KEY);
      if (saved === "light" || saved === "dark") setTheme(saved);
    } catch {
      /* ignore */
    }
  }, []);

  function cycle() {
    const next: Theme = theme === "system" ? "light" : theme === "light" ? "dark" : "system";
    setTheme(next);
    const root = document.documentElement;
    try {
      if (next === "system") {
        root.removeAttribute("data-theme");
        localStorage.removeItem(THEME_KEY);
      } else {
        root.setAttribute("data-theme", next);
        localStorage.setItem(THEME_KEY, next);
      }
    } catch {
      /* ignore */
    }
  }

  const name = { system: lang === "th" ? "ตามระบบ" : "System", light: lang === "th" ? "สว่าง" : "Light", dark: lang === "th" ? "มืด" : "Dark" }[theme];
  return (
    <button type="button" className="tool" onClick={cycle} aria-label={(lang === "th" ? "ธีม: " : "Theme: ") + name + (lang === "th" ? " — กดเพื่อเปลี่ยน" : " — change")}>
      <span className="tool__glyph" aria-hidden="true">
        {theme === "dark" ? "●" : theme === "light" ? "○" : "◐"}
      </span>
      <span className="tool__text">{name}</span>
    </button>
  );
}

function LangToggle() {
  const { lang, setLang } = useLang();
  return (
    <div className="lang" role="group" aria-label="Language / ภาษา">
      <button type="button" aria-pressed={lang === "en"} onClick={() => setLang("en")} lang="en">
        EN
      </button>
      <button type="button" aria-pressed={lang === "th"} onClick={() => setLang("th")} lang="th">
        ไทย
      </button>
    </div>
  );
}

export function DotMark() {
  // four bands of dots: the strata of the name
  const rows = [3, 5, 5, 3];
  return (
    <svg className="dotmark" viewBox="0 0 22 22" width="22" height="22" aria-hidden="true">
      {rows.map((n, r) =>
        Array.from({ length: n }, (_, c) => (
          <circle key={`${r}-${c}`} cx={11 + (c - (n - 1) / 2) * 3.6} cy={4 + r * 4.7} r={r === 0 ? 1.45 : r === 1 ? 1.35 : 1.2} />
        )),
      )}
    </svg>
  );
}

export function Nav() {
  const path = usePathname();
  const onDetails = path.startsWith("/details");
  return (
    <header className="nav">
      <div className="wrap nav__in">
        <Link className="nav__brand" href="/" aria-label="Strata-xeno">
          <DotMark />
          <span>Strata-xeno</span>
        </Link>
        <nav className="nav__links" aria-label="Sections">
          {SECTIONS.map((s) => (
            <Link key={s.id} href={`/#${s.id}`}>
              <T v={s.label} />
            </Link>
          ))}
          <Link href="/details" className="nav__details" aria-current={onDetails ? "page" : undefined}>
            <T v={{ en: "Details", th: "รายละเอียด" }} />
          </Link>
        </nav>
        <div className="nav__tools">
          <GitHubButton compact />
          <ThemeToggle />
          <LangToggle />
          <details className="nav__menu">
            <summary aria-label="Menu">
              <span aria-hidden="true">≡</span>
            </summary>
            <nav aria-label="Sections">
              {SECTIONS.map((s) => (
                <Link key={s.id} href={`/#${s.id}`}>
                  <T v={s.label} />
                </Link>
              ))}
              <Link href="/details" aria-current={onDetails ? "page" : undefined}>
                <T v={{ en: "Details", th: "รายละเอียด" }} />
              </Link>
            </nav>
          </details>
        </div>
      </div>
    </header>
  );
}
