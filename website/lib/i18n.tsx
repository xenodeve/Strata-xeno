"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

export type Lang = "en" | "th";
/** A string in both languages. English is the source; Thai is a full translation. */
export type L = { en: string; th: string };

type Ctx = { lang: Lang; setLang: (l: Lang) => void };
const LangContext = createContext<Ctx>({ lang: "en", setLang: () => {} });

const KEY = "strata-xeno.lang";

export function LangProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>("en");

  useEffect(() => {
    try {
      const saved = localStorage.getItem(KEY);
      if (saved === "en" || saved === "th") setLangState(saved);
      else if (navigator.language?.toLowerCase().startsWith("th")) setLangState("th");
    } catch {
      /* storage can be blocked: the page works without it */
    }
  }, []);

  useEffect(() => {
    document.documentElement.lang = lang;
  }, [lang]);

  const setLang = useCallback((l: Lang) => {
    setLangState(l);
    try {
      localStorage.setItem(KEY, l);
    } catch {
      /* ignore */
    }
  }, []);

  const value = useMemo(() => ({ lang, setLang }), [lang, setLang]);
  return <LangContext.Provider value={value}>{children}</LangContext.Provider>;
}

export function useLang() {
  return useContext(LangContext);
}

/**
 * A slash command or a skill (`/compact`, `/init`) is set in bold, as the app sets it in a message (its `.skill-tag`: weight 600). It is a
 * word that starts with a slash, comes after a space, a bracket, a quote or a backtick (so not a path such as `/v1/messages` or
 * `and/or`), and is not one of this site's own addresses. A bare `/` between backticks (the key that lists the commands) is bold too.
 */
const SLASH = /(^|[\s(,;:"“'`])(\/[a-z][a-z0-9_:-]*|(?<=`)\/(?=`))(?![\w/.-])/g;
const ADDRESSES = new Set(["/v1", "/classic", "/details", "/demo", "/metrics", "/health", "/next", "/film", "/harness"]);

export function slashed(text: string): ReactNode {
  if (!text.includes("/")) return text;
  const out: ReactNode[] = [];
  let at = 0;
  let k = 0;
  for (const m of text.matchAll(SLASH)) {
    const word = m[2];
    if (ADDRESSES.has(word)) continue;
    const start = (m.index ?? 0) + m[1].length;
    if (start > at) out.push(text.slice(at, start));
    out.push(
      <b key={k++} className="slash">
        {word}
      </b>,
    );
    at = start + word.length;
  }
  if (!out.length) return text;
  if (at < text.length) out.push(text.slice(at));
  return <>{out}</>;
}

/** Text in the current language. */
export function T({ v }: { v: L }) {
  const { lang } = useLang();
  return <>{slashed(v[lang])}</>;
}

export function useT() {
  const { lang } = useLang();
  return (v: L) => v[lang];
}
