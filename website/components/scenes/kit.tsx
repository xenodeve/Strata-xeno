"use client";

import { memo, useEffect, useLayoutEffect, useRef, useState, type CSSProperties, type ReactNode } from "react";
import { ThinkingOrb, type OrbState } from "thinking-orbs";
import { usePrefersReducedMotion } from "../Avatar";
import { clamp01, ease, lerp, lin, seg } from "./Scene";

/**
 * The small parts every scene is made of. Each one is a plain function of the clock (`ms`), so a scene is just a list of
 * these placed in time. Layout is always reserved (a part that has not arrived is invisible, not absent), so nothing on a
 * scene ever jumps or reflows as the next part lands.
 */

/** Rises into place: opacity and a short slide, from `at` for `dur` ms. */
export function Rise({ ms, at, dur = 420, y = 10, className = "", style, children }: { ms: number; at: number; dur?: number; y?: number; className?: string; style?: CSSProperties; children: ReactNode }) {
  const p = seg(ms, at, dur);
  return (
    <div className={className} style={{ opacity: p, transform: p >= 1 ? undefined : `translate3d(0, ${(1 - p) * y}px, 0)`, ...style }}>
      {children}
    </div>
  );
}

/** Just fades, no slide. */
export function Fade({ ms, at, dur = 300, className = "", style, children }: { ms: number; at: number; dur?: number; className?: string; style?: CSSProperties; children: ReactNode }) {
  return (
    <div className={className} style={{ opacity: seg(ms, at, dur), ...style }}>
      {children}
    </div>
  );
}

/** Text that is typed out. The rest of it is there, invisible, so the line breaks never change as it grows. */
export function Typed({ text, ms, at, cps = 60, caret = false }: { text: string; ms: number; at: number; cps?: number; caret?: boolean }) {
  const n = Math.min(text.length, Math.max(0, Math.floor(((ms - at) / 1000) * cps)));
  const typing = caret && ms >= at && n < text.length;
  return (
    <>
      <span>{text.slice(0, n)}</span>
      {typing && <span className="a-stream" aria-hidden="true" />}
      <span className="a-ghost">{text.slice(n)}</span>
    </>
  );
}

export type Seg = string | { code: string };
/** Typed text with inline code in it: the count of letters runs across the pieces. */
export function RichTyped({ segs, ms, at, cps = 70 }: { segs: Seg[]; ms: number; at: number; cps?: number }) {
  let left = Math.max(0, Math.floor(((ms - at) / 1000) * cps));
  return (
    <>
      {segs.map((s, k) => {
        const text = typeof s === "string" ? s : s.code;
        const n = Math.min(text.length, left);
        left -= n;
        const body = (
          <>
            {text.slice(0, n)}
            <span className="a-ghost">{text.slice(n)}</span>
          </>
        );
        return typeof s === "string" ? <span key={k}>{body}</span> : <code key={k} className="a-code" style={{ opacity: n > 0 ? 1 : 0 }}>{body}</code>;
      })}
    </>
  );
}

/** A number that counts up (or down) over a stretch of the clock. */
export function Count({ ms, at, dur, from = 0, to, digits = 0, group = false }: { ms: number; at: number; dur: number; from?: number; to: number; digits?: number; group?: boolean }) {
  const v = lerp(from, to, ease(lin(ms, at, dur)));
  const s = v.toFixed(digits);
  return <>{group ? Number(s).toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits }) : s}</>;
}

/**
 * The app's orb: the thinking-orbs library (Libraries.dev, MIT) that the app itself draws for every status, one of its nine designs
 * for what is happening (lib/orbs.ts says which: searching while idle, listening while a prompt is read, solving while it thinks,
 * composing while it writes, connecting / weaving around a tool call). The library tunes two sizes by hand (64 and 20) and a third in between (32);
 * a slot of another size shows the nearest preset scaled to fit. Idle designs turn slowly (half speed), as in the app. The scenes are the app's
 * light skin, so the orb is always its light palette; with reduced motion it is a still frame.
 */
let dprUsers = 0;
let dprBoost = 1;
let dprReal = 1;
let dprSaved: PropertyDescriptor | null = null;

function OrbImpl({ design = "searching", size = 20, color, speed, rest }: { ms?: number; live?: boolean; size?: number; design?: string; color?: string; speed?: number; rest?: boolean }) {
  const reduced = usePrefersReducedMotion();
  const preset: 64 | 32 | 20 = size >= 48 ? 64 : size >= 26 ? 32 : 20;
  // The library draws its canvas at the preset's size times the screen's pixel ratio (at most 2), so a slot bigger than the preset is a
  // scaled picture. It is scaled by at most 2 and drawn at that many times the pixels (below), which keeps it sharp; a slot bigger than
  // that holds the orb at that size, centred, rather than a blurred one.
  const k = Math.min(2, size / preset);
  const idle = rest ?? design === "searching";
  // The library reads `devicePixelRatio` when it draws. For the length of the commit in which it does, it is asked for k times the
  // real ratio, so a canvas shown k times larger has k times the pixels. A layout effect runs before every passive effect of a commit
  // (the library's among them) and a passive effect of this component after the library's own (a child's come first), so the ratio
  // is raised before the library draws and put back after it. The browser's own property is saved and put back as it was.
  // The library draws once as it mounts, and again a moment later when it has found its theme and the visitor's motion setting (two
  // updates of its own state that follow the mount). Drawing again is what a bitmap of the wrong size would come from, so this orb is
  // drawn once more itself in the same pass as those updates: its layout effect below then runs before the library's second draw.
  const [, again] = useState(0);
  useEffect(() => {
    again(1);
  }, []);
  useLayoutEffect(() => {
    if (typeof window === "undefined") return;
    if (dprUsers === 0) {
      dprSaved = Object.getOwnPropertyDescriptor(window, "devicePixelRatio") ?? null;
      dprReal = window.devicePixelRatio || 1;
      Object.defineProperty(window, "devicePixelRatio", { configurable: true, get: () => dprReal * dprBoost });
    }
    dprUsers += 1;
    // twice the pixels the screen has (the library stops at 2): on an ordinary screen the little dots are then drawn at twice the
    // size and brought down by the browser, which is smoother than drawing them at the size they are shown, and a scaled orb has its factor too
    dprBoost = Math.max(dprBoost, k, 2 / dprReal);
  });
  useEffect(() => {
    if (typeof window === "undefined") return;
    dprUsers -= 1;
    if (dprUsers === 0) {
      if (dprSaved) Object.defineProperty(window, "devicePixelRatio", dprSaved);
      else delete (window as unknown as { devicePixelRatio?: number }).devicePixelRatio;
      dprSaved = null;
      dprBoost = 1;
    }
  });
  const shown = preset * k;
  return (
    <span className="a-orbslot" style={{ width: size, height: size }} aria-hidden="true">
      <span className="a-orbslot__in" style={{ left: (size - shown) / 2, top: (size - shown) / 2, ...(k === 1 ? null : { transform: `scale(${k})` }) }}>
        <ThinkingOrb state={design as OrbState} size={preset} theme="light" speed={speed ?? (idle ? 0.5 : 1)} paused={reduced} color={color} aria-hidden="true" />
      </span>
    </span>
  );
}

/** The orb is drawn again only when what it shows changes: the scene around it is drawn some thirty times a second, the orb is not. */
export const Orb = memo(OrbImpl, (a, b) => a.design === b.design && a.size === b.size && a.color === b.color && a.speed === b.speed && a.rest === b.rest);

/** A thin bar that fills to `v` (0..1) over a stretch. */
export function Meter({ ms, at, dur = 900, v, tone = "ink", height = 4 }: { ms: number; at: number; dur?: number; v: number; tone?: "ink" | "ok" | "warn" | "accent"; height?: number }) {
  return (
    <div className="a-meter" style={{ height }}>
      <div className={"a-meter__f a-meter__f--" + tone} style={{ width: `${clamp01(v) * seg(ms, at, dur) * 100}%` }} />
    </div>
  );
}

/** A line chart that draws itself left to right. `points` are 0..1 (height). */
export function Spark({ ms, at, dur = 1400, points, width = 300, height = 70, fill = true }: { ms: number; at: number; dur?: number; points: number[]; width?: number; height?: number; fill?: boolean }) {
  const pad = 2;
  const xs = points.map((_, i) => pad + (i / Math.max(1, points.length - 1)) * (width - pad * 2));
  const ys = points.map((p) => height - pad - clamp01(p) * (height - pad * 2));
  const d = xs.map((x, i) => `${i ? "L" : "M"}${x.toFixed(1)} ${ys[i].toFixed(1)}`).join(" ");
  const p = ease(lin(ms, at, dur));
  return (
    <svg className="a-spark" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-hidden="true">
      {[0.25, 0.5, 0.75].map((g) => (
        <line key={g} x1="0" x2={width} y1={height * g} y2={height * g} className="a-spark__grid" />
      ))}
      {fill && <path d={`${d} L${xs[xs.length - 1]} ${height} L${xs[0]} ${height} Z`} className="a-spark__area" style={{ opacity: p * 0.9 }} />}
      <path d={d} className="a-spark__line" pathLength={1} strokeDasharray="1" strokeDashoffset={1 - p} />
    </svg>
  );
}

/** The app's top bar: the brand mark, the pages, and the active one lit. */
export function AppNav({ active, items = ["Chat", "Dashboard", "Live", "Requests", "Hardware", "Settings", "About"] }: { active: string; items?: string[] }) {
  return (
    <div className="a-nav">
      <span className="a-nav__brand">
        <Orb ms={0} size={16} />
        Strata
      </span>
      {items.map((n) => (
        <span key={n} className={"a-nav__i" + (n === active ? " is-on" : "")}>
          {n}
        </span>
      ))}
      <span className="a-nav__end">Marks · EN / ไทย</span>
    </div>
  );
}

/** A framed place for a scene's pieces. */
export function Frame({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <div className={"a-frame " + className}>{children}</div>;
}

/**
 * A window onto content that is taller than it: the content slides up, from `from` for `dur` ms, by exactly what does not fit
 * (measured), so the latest part ends up in view, as in a chat that follows its newest message. If everything fits, nothing moves.
 */
export function Scroll({ ms, from, dur, className = "", children }: { ms: number; from: number; dur: number; className?: string; children: ReactNode }) {
  const outer = useRef<HTMLDivElement>(null);
  const inner = useRef<HTMLDivElement>(null);
  const [over, setOver] = useState(0);
  useEffect(() => {
    const o = outer.current;
    const i = inner.current;
    if (!o || !i) return;
    const measure = () => setOver(Math.max(0, i.offsetHeight - o.clientHeight));
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(o);
    ro.observe(i);
    return () => ro.disconnect();
  }, []);
  return (
    <div ref={outer} className={"a-scroll " + className}>
      <div ref={inner} style={over > 0 ? { transform: `translate3d(0, ${-over * seg(ms, from, dur)}px, 0)` } : undefined}>
        {children}
      </div>
    </div>
  );
}
