"use client";

import { Children, useEffect, useMemo, useRef, useState, type CSSProperties, type ElementType, type ReactNode } from "react";
import { useT, type L } from "@/lib/i18n";
import { usePrefersReducedMotion } from "../Avatar";

/**
 * Text that moves. Each effect is written for this site (no library); they share one rule: the words are always in the
 * page for readers and search (the moving copy is `aria-hidden` where it is split), nothing moves with reduced motion,
 * and a page without scripts shows the finished text (the hidden start state exists only under `html.js`).
 */

/** True once the element has been mostly on screen, and stays true. */
export function useSeen<E extends HTMLElement>(threshold = 0.35): [React.RefObject<E | null>, boolean] {
  const ref = useRef<E>(null);
  const [seen, setSeen] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (typeof IntersectionObserver === "undefined") {
      setSeen(true);
      return;
    }
    const io = new IntersectionObserver(
      (e) => {
        if (e.some((x) => x.isIntersecting)) {
          setSeen(true);
          io.disconnect();
        }
      },
      { threshold },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [threshold]);
  return [ref, seen];
}

/** The text of an `L` in the current language. */
function useText(v: L | string): string {
  const t = useT();
  return typeof v === "string" ? v : t(v);
}

/** Words (Thai too: it has no spaces, so the browser's own segmenter finds them) with punctuation kept on its word. */
function wordsOf(text: string, lang: string): string[] {
  try {
    const seg = new Intl.Segmenter(lang === "th" ? "th" : "en", { granularity: "word" });
    const out: string[] = [];
    for (const s of seg.segment(text)) {
      if (s.isWordLike || out.length === 0 || /^\s+$/.test(s.segment)) out.push(s.segment);
      else out[out.length - 1] += s.segment; // punctuation joins the word before it
    }
    return out;
  } catch {
    return text.split(/(\s+)/).filter(Boolean);
  }
}
function graphemesOf(text: string): string[] {
  try {
    return Array.from(new Intl.Segmenter(undefined, { granularity: "grapheme" }).segment(text), (s) => s.segment);
  } catch {
    return Array.from(text);
  }
}

/* ── SlideUpText: words (or letters) rise out of a mask ─────────────────────────────────────────── */
type SlideProps = {
  /** An `L` (it follows the language) or a plain string. */
  v: L | string;
  split?: "words" | "chars";
  as?: ElementType;
  className?: string;
  /** Wait this long (ms) after it is seen. */
  delay?: number;
  /** Set it to run it on your own cue (a carousel slide that has just come on); otherwise it runs when seen. */
  play?: boolean;
  id?: string;
};
export function SlideUpText({ v, split = "words", as: Tag = "span", className = "", delay = 0, play, id }: SlideProps) {
  const t = useT();
  const text = useText(v);
  const lang = t({ en: "en", th: "th" });
  const [ref, seen] = useSeen<HTMLElement>();
  const on = play ?? seen;
  const parts = useMemo(() => (split === "chars" ? graphemesOf(text) : wordsOf(text, lang)), [text, split, lang]);
  let n = 0;
  return (
    <Tag ref={ref} id={id} className={"su " + (on ? "is-in " : "") + className} style={{ "--d": `${delay}ms` } as CSSProperties}>
      <span className="sr-only">{text}</span>
      <span aria-hidden="true">
        {parts.map((p, k) =>
          /^\s+$/.test(p) ? (
            p
          ) : (
            <span key={k} className="su__w">
              <span style={{ "--i": n++ } as CSSProperties}>{p}</span>
            </span>
          ),
        )}
      </span>
    </Tag>
  );
}

/** A SlideUpText that plays as soon as it is mounted: for text that is swapped by changing its key. */
export function SlideUpOnMount(props: Omit<SlideProps, "play">) {
  const [on, setOn] = useState(false);
  useEffect(() => {
    const id = requestAnimationFrame(() => requestAnimationFrame(() => setOn(true)));
    return () => cancelAnimationFrame(id);
  }, []);
  return <SlideUpText {...props} play={on} />;
}

/* ── HighlightedText: a highlighter sweeps behind the words ─────────────────────────────────────── */
export function HighlightedText({ children, delay = 0, from = "left", play, className = "" }: { children: ReactNode; delay?: number; from?: "left" | "right"; play?: boolean; className?: string }) {
  const [ref, seen] = useSeen<HTMLElement>(0.6);
  const on = play ?? seen;
  return (
    <mark ref={ref} className={"hl " + (on ? "is-in " : "") + className} data-from={from} style={{ "--d": `${delay}ms` } as CSSProperties}>
      {children}
    </mark>
  );
}

/* ── Marquee: a band that drifts sideways, pauses under the pointer ─────────────────────────────── */
export function Marquee({ children, duration = 40, reverse = false, className = "", label }: { children: ReactNode; duration?: number; reverse?: boolean; className?: string; label?: string }) {
  return (
    <div className={"mq " + className} data-reverse={reverse || undefined} style={{ "--dur": `${duration}s` } as CSSProperties} role="group" aria-label={label}>
      <div className="mq__track">
        <div className="mq__set">{children}</div>
        <div className="mq__set" aria-hidden="true">
          {children}
        </div>
      </div>
    </div>
  );
}

/* ── TextMarquee: a word rolls through a window, one at a time, next to a fixed phrase ──────────── */
export function TextMarquee({ prefix, items, suffix, interval = 1900, className = "" }: { prefix?: ReactNode; items: (L | string)[]; suffix?: ReactNode; interval?: number; className?: string }) {
  const t = useT();
  const reduced = usePrefersReducedMotion();
  const [ref, seen] = useSeen<HTMLDivElement>(0.3);
  const list = items.map((x) => (typeof x === "string" ? x : t(x)));
  const n = list.length;
  const [i, setI] = useState(0);
  const [snap, setSnap] = useState(false); // jump back to the first copy without a transition
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver((e) => setVisible(e.some((x) => x.isIntersecting)));
    io.observe(el);
    return () => io.disconnect();
  }, [ref]);

  useEffect(() => {
    if (reduced || !seen || !visible) return;
    const id = window.setInterval(() => setI((k) => k + 1), interval);
    return () => window.clearInterval(id);
  }, [reduced, seen, visible, interval]);

  // after the last real item the second copy of the first one is on show: go back to the start unseen
  useEffect(() => {
    if (i < n) return;
    const id = window.setTimeout(() => {
      setSnap(true);
      setI(0);
    }, 760);
    return () => window.clearTimeout(id);
  }, [i, n]);
  useEffect(() => {
    if (!snap) return;
    const id = requestAnimationFrame(() => requestAnimationFrame(() => setSnap(false)));
    return () => cancelAnimationFrame(id);
  }, [snap]);

  return (
    <div ref={ref} className={"tm " + className}>
      {prefix && <span className="tm__pre">{prefix}</span>}
      <span className="tm__win" aria-hidden="true">
        <span className="tm__col" data-snap={snap || undefined} style={{ transform: reduced ? undefined : `translateY(calc(${-i} * 1.25em))` }}>
          {[...list, list[0]].map((s, k) => (
            <span key={k} className="tm__item" data-on={k === i || (reduced && k === 0) || undefined}>
              {s}
            </span>
          ))}
        </span>
      </span>
      <span className="sr-only">{list.join(", ")}</span>
      {suffix && <span className="tm__post">{suffix}</span>}
    </div>
  );
}

/* ── GradientWaveText: a slow wave of colour runs along the letters ─────────────────────────────── */
export function GradientWaveText({ children, className = "", as: Tag = "span", style }: { children: ReactNode; className?: string; as?: ElementType; style?: CSSProperties }) {
  return (
    <Tag className={"gw " + className} style={style}>
      {children}
    </Tag>
  );
}

/* ── RandomizedText: letters scramble, then settle into the words from the left ─────────────────── */
const POOL_LATIN = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
const POOL_THAI = "กขคงจฉชซญดตถทนบปผฝพฟภมยรลวศษสหอฮ";
type RandProps = { v: L | string; split?: "chars" | "words"; as?: ElementType; className?: string; duration?: number; delay?: number; play?: boolean };
export function RandomizedText({ v, split = "chars", as: Tag = "span", className = "", duration = 900, delay = 0, play }: RandProps) {
  const t = useT();
  const text = useText(v);
  const lang = t({ en: "en", th: "th" });
  const reduced = usePrefersReducedMotion();
  const [ref, seen] = useSeen<HTMLElement>();
  const on = play ?? seen;
  const [phase, setPhase] = useState<"idle" | "run" | "done">("idle");
  const [shown, setShown] = useState(text);
  const parts = useMemo(() => (split === "words" ? wordsOf(text, lang) : graphemesOf(text)), [text, split, lang]);

  useEffect(() => {
    if (!on) return;
    if (reduced) {
      setPhase("done");
      return;
    }
    const pool = lang === "th" ? POOL_THAI : POOL_LATIN;
    const real = parts.filter((p) => !/^\s+$/.test(p)).length;
    const born = performance.now() + delay;
    let raf = 0;
    let last = 0;
    setPhase("run");
    const frame = (now: number) => {
      const e = now - born;
      if (e < 0) {
        raf = requestAnimationFrame(frame);
        return;
      }
      if (now - last > 42) {
        last = now;
        let k = 0;
        const out = parts.map((p) => {
          if (/^\s+$/.test(p)) return p;
          // the k-th letter settles at a time along the line, each scrambling for a third of the duration
          const settle = (k++ / Math.max(1, real)) * duration * 0.65 + duration * 0.35;
          if (e >= settle) return p;
          return split === "words" ? Array.from(p, (c) => (/\s/.test(c) ? c : pool[Math.floor(Math.random() * pool.length)])).join("") : pool[Math.floor(Math.random() * pool.length)];
        });
        setShown(out.join(""));
        if (e >= duration * 1.02) {
          setPhase("done");
          return;
        }
      }
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, [on, reduced, parts, lang, duration, delay, split]);

  return (
    <Tag ref={ref} className={"rt " + className} data-p={phase}>
      <span className="rt__real">{text}</span>
      {phase === "run" && (
        <span className="rt__ov" aria-hidden="true">
          {shown}
        </span>
      )}
    </Tag>
  );
}

/** Convenience: the children of a heading as one string. */
export function textOf(children: ReactNode): string {
  return Children.toArray(children)
    .map((c) => (typeof c === "string" || typeof c === "number" ? String(c) : ""))
    .join("");
}
