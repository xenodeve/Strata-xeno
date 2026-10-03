"use client";

import { useLayoutEffect, useRef, useState, type CSSProperties, type ReactNode } from "react";
import LatticeLoader from "../text/LatticeLoader";
import { clamp01, lin } from "./Scene";
import { DUR, EASE, EASE_DIGIT, EASE_LATTICE, EASE_REEL, EASE_SOFT, bezier } from "./motion";

/** CSS `ease`, the curve the soft collapse gives its opacity. */
const EASE_CSS = bezier(0.25, 0.1, 0.25, 1);

/**
 * The parts the app itself moves with, written again for a clock. Each one takes `ms` (the scene's clock) and the moment it
 * starts, and its timing, distance and curve are the app's (components/motion.ts says where each number comes from). They are
 * written from what the app does, in this site's own code: the app's components are not copied.
 */

const at = (ms: number, start: number, dur: number, ease: (t: number) => number = EASE) => ease(lin(ms, start, dur));

/* ── how a thing enters ─────────────────────────────────────────────── */

export type EnterKind = "msg" | "rise" | "row" | "page" | "swap" | "toast" | "panel" | "tip" | "orb";
const ENTER: Record<EnterKind, { dur: number; y: number; blur: number; scale: number }> = {
  msg: { dur: DUR.msg, y: 8, blur: 0, scale: 1 }, //          .msg-in: rise 300ms
  rise: { dur: DUR.rise, y: 8, blur: 0, scale: 1 }, //        .stagger > *, a page's content: rise 420ms
  row: { dur: DUR.row, y: -6, blur: 0, scale: 1 }, //         .row-in: from 6px above, 360ms
  page: { dur: DUR.page, y: 0, blur: 6, scale: 0.985 }, //    .page-in: sharpen 260ms
  swap: { dur: DUR.swap, y: 0, blur: 6, scale: 0.985 }, //    .swap-in: sharpen 280ms
  toast: { dur: DUR.toast, y: 8, blur: 0, scale: 0.98 }, //   .toast-in
  panel: { dur: DUR.panel, y: 4, blur: 0, scale: 1 }, //      .panel-in
  tip: { dur: DUR.tip, y: -3, blur: 0, scale: 1 }, //         .skill-tip
  orb: { dur: DUR.orb, y: 0, blur: 3, scale: 0.82 }, //       .orb-in
};

/** How a thing arrives, with the app's own keyframes: `msg` for a message, `row` for a row added to a list, `page` / `swap` for
 *  something that sharpens in from a blur, and so on. It is invisible until `start` and takes its place without moving anything. */
export function Enter({ ms, start, kind = "msg", delay = 0, className = "", style, children }: { ms: number; start: number; kind?: EnterKind; delay?: number; className?: string; style?: CSSProperties; children: ReactNode }) {
  const k = ENTER[kind];
  const p = at(ms, start + delay, k.dur);
  const done = p >= 1;
  const transform = done ? undefined : `translate3d(0, ${(1 - p) * k.y}px, 0)${k.scale !== 1 ? ` scale(${k.scale + (1 - k.scale) * p})` : ""}`;
  return (
    <div className={className} style={{ opacity: p, transform, filter: done || !k.blur ? undefined : `blur(${(1 - p) * k.blur}px)`, ...style }}>
      {children}
    </div>
  );
}

/** The children of a list arrive one after another, 50 ms apart (the app's `.stagger`; the seventh and later share 300 ms). */
export const stagger = (n: number) => Math.min(n, 6) * DUR.stagger;

/* ── a section that opens and closes ────────────────────────────────── */

/** The app's Collapse: the height goes through grid rows (0fr ↔ 1fr, 320 ms; 520 ms for `soft`) while the opacity rides along (240 / 360 ms),
 *  and what is inside it is clipped, so a section opens to exactly its own height. It closes at `closeAt` if given. */
export function Collapse({ ms, open, closeAt, soft = false, className = "", children }: { ms: number; open: number; closeAt?: number; soft?: boolean; className?: string; children: ReactNode }) {
  const d = soft ? DUR.collapseSoft : DUR.collapse;
  const o = soft ? DUR.collapseSoftOpacity : DUR.collapseOpacity;
  const curve = soft ? EASE_SOFT : EASE;
  const closing = closeAt !== undefined && ms >= closeAt;
  const rows = closing ? 1 - at(ms, closeAt!, d, curve) : at(ms, open, d, curve);
  const oc = soft ? EASE_CSS : EASE;
  const opacity = closing ? 1 - at(ms, closeAt!, o, oc) : at(ms, open, o, oc);
  return (
    <div className={className} style={{ display: "grid", gridTemplateRows: `${rows}fr`, opacity }}>
      <div style={{ minHeight: 0, overflow: "hidden" }}>{children}</div>
    </div>
  );
}

/* ── numbers ─────────────────────────────────────────────────────────── */

const isDigit = (c: string) => c >= "0" && c <= "9";
/** The stagger step of a character counted from the end of the figure: the last digit follows two steps behind, the one before it one. */
const staggerOf = (fromEnd: number) => (fromEnd === 0 ? 2 : fromEnd === 1 ? 1 : 0);

/**
 * Number pop-in: the digits that changed come in again, one after the other, from 6 px below and out of a 2 px blur, 320 ms with a
 * little overshoot; the last two follow a step (38 ms) behind. Digits that did not change stay still. With no `from` nothing moves
 * (the app does not animate what is there at the start).
 */
export function Pop({ ms, start, text, from }: { ms: number; start: number; text: string; from?: string }) {
  const now = [...text];
  const was = from === undefined ? null : [...from];
  return (
    <span className="a-popn">
      {now.map((c, j) => {
        const fromEnd = now.length - 1 - j;
        const changed = was !== null && isDigit(c) && was[was.length - 1 - fromEnd] !== c;
        if (!changed) return <span key={j}>{c}</span>;
        const raw = EASE_DIGIT(lin(ms, start + staggerOf(fromEnd) * DUR.digitStagger, DUR.digit));
        return (
          <span key={j} className="a-popn__d" style={{ opacity: clamp01(raw), transform: `translate3d(0, ${(1 - raw) * 6}px, 0)`, filter: raw >= 1 ? undefined : `blur(${(1 - clamp01(raw)) * 2}px)` }}>
            {c}
          </span>
        );
      })}
    </span>
  );
}

const STRIP = Array.from({ length: 50 }, (_, i) => i % 10).join("\n");
/** How many cells a reel turns from digit `a` to digit `b`: up when the figure grew, down when it fell, the short way round. */
const reelMove = (a: number, b: number, up: boolean) => (up ? (b - a + 10) % 10 : -((a - b + 10) % 10) || 0);

/**
 * The spinning counter: every digit is a clipped reel that turns to its new digit (up when the figure grew, down when it fell,
 * 450 ms, each reel starting 8 % of the length after the one before it) with a vertical streak while it moves. Used for a figure that goes up and down.
 * A figure that changes within 150 ms of its last change just shows the digits; one that changes often turns for 40 % of the time between.
 */
export function Reels({ ms, start, text, from, since }: { ms: number; start: number; text: string; from?: string; since?: number | null }) {
  const num = (s: string) => parseFloat(s.replace(/,/g, ""));
  const up = from === undefined ? true : num(text) >= num(from);
  const animate = from !== undefined && !(since != null && since < 150);
  const dur = since != null ? Math.round(Math.min(DUR.reel, Math.max(90, since * 0.4))) : DUR.reel;
  const now = [...text];
  const was = from === undefined ? null : [...from];
  let k = -1;
  return (
    <span className="a-reel" aria-hidden="true">
      <svg width="0" height="0" style={{ position: "absolute" }} aria-hidden="true">
        <filter id="a-reel-blur">
          <feGaussianBlur stdDeviation="0 2.5" />
        </filter>
      </svg>
      {now.map((c, j) => {
        if (!isDigit(c)) return <span key={j} className="a-reel__sep">{c}</span>;
        k += 1;
        const fromEnd = now.length - 1 - j;
        const old = was && isDigit(was[was.length - 1 - fromEnd] ?? "") ? +was[was.length - 1 - fromEnd] : +c;
        const move = animate ? reelMove(old, +c, up) : 0;
        const p = move === 0 ? 1 : EASE_REEL(lin(ms, start + Math.round(k * dur * 0.08), dur));
        const pos = 20 + old + move * p;
        return (
          <span key={j} className="a-reel__col">
            <span className="a-reel__strip" style={{ transform: `translateY(calc(var(--a-reel-cell) * ${-(move === 0 ? 20 + +c : pos)}))`, filter: p > 0 && p < 1 ? "url(#a-reel-blur)" : undefined }}>
              {STRIP}
            </span>
          </span>
        );
      })}
    </span>
  );
}

/**
 * A figure that is read off a live source and changes now and then: `steps` are the readings and the moment each arrives. A figure that
 * only grows (a count of tokens) pops its changed digits; one that goes up and down (a speed) turns its reels. The first reading just appears.
 */
export function Figure({ ms, steps, kind = "count", empty = "–" }: { ms: number; steps: { at: number; text: string }[]; kind?: "count" | "gauge"; empty?: string }) {
  let i = -1;
  for (let k = 0; k < steps.length; k++) if (ms >= steps[k].at) i = k;
  if (i < 0) return <span className="a-num">{empty}</span>;
  const cur = steps[i];
  const prev = i > 0 ? steps[i - 1] : null;
  return (
    <span className="a-num">
      <span className="a-sr">{cur.text}</span>
      {kind === "gauge" ? (
        <Reels ms={ms} start={cur.at} text={cur.text} from={prev?.text} since={prev ? cur.at - prev.at : null} />
      ) : (
        <span aria-hidden="true">
          <Pop ms={ms} start={cur.at} text={cur.text} from={prev?.text} />
        </span>
      )}
    </span>
  );
}

/* ── the loaders ─────────────────────────────────────────────────────── */

export type LatticePattern = "orbit" | "ripple" | "snake" | "spiral" | "arrow" | "dots";

/**
 * The 3 × 3 lattice of dots that lights in a pattern while the agent works and settles into a tick or a cross: the supplied React Bits
 * LatticeLoader (components/text/LatticeLoader.tsx), which is what the app itself uses beside a thought. Here it is only the lattice:
 * no label and no timer of its own (the thought line owns both), the app's own ok / error colours and its 4 px dots.
 */
export function Lattice({ pattern = "orbit", status = "working", cell = 4, gap = 1.5 }: { pattern?: LatticePattern; status?: "working" | "done" | "error"; cell?: number; gap?: number }) {
  return (
    <span className="a-lat" aria-hidden="true">
      <LatticeLoader label="" doneLabel="" errorLabel="" status={status} pattern={pattern} cellSize={cell} gap={gap} showTimer={false} doneColor="#1a8a4a" errorColor="#c6322b" style={{ gap: 0 }} />
    </span>
  );
}

export type MatrixVariant = "scan" | "twinkle" | "orbit" | "pulse";
const CORNERS = [0, 3, 12, 15];
const RING = [1, 2, 7, 11, 14, 13, 8, 4];
const INNER = [5, 6, 9, 10];
const TWINKLE = [7, 2, 11, 5, 14, 9, 0, 12, 3, 15, 6, 10, 13, 1, 8, 4];
const matrixDelay = (v: MatrixVariant, i: number): number | null => {
  if (v === "scan") return Math.round((i % 4) * (DUR.matrix / 10));
  if (v === "twinkle") return Math.round(TWINKLE[i] * (DUR.matrix / 16));
  if (v === "orbit") {
    const k = RING.indexOf(i);
    return k === -1 ? null : Math.round(k * (DUR.matrix / 8));
  }
  return Math.round((INNER.includes(i) ? 0 : 1) * (DUR.matrix * 0.16));
};

/** The 4 × 4 matrix of dots, one pulse of colour across it in a pattern of its own (1.2 s a cycle); the four corners are left out. */
export function Matrix({ variant, px = 18 }: { variant: MatrixVariant; px?: number }) {
  const dot = Math.max(2, Math.round(px / 9));
  const gap = Math.max(2, Math.round(px / 12));
  return (
    <span className="a-matrix" aria-hidden="true" style={{ "--m-dot": `${dot}px`, "--m-gap": `${gap}px` } as CSSProperties}>
      {Array.from({ length: 16 }, (_, i) => {
        if (CORNERS.includes(i)) return <i key={i} className="is-gap" />;
        const d = matrixDelay(variant, i);
        return <i key={i} style={d == null ? { animation: "none" } : { animationDelay: `${d}ms` }} />;
      })}
    </span>
  );
}

/** The app's loading style for an orb design: which loader stands for which state (lib/avatar.ts). */
export const LOADERS: Record<string, { family: "lattice"; pattern: LatticePattern } | { family: "matrix"; variant: MatrixVariant }> = {
  working: { family: "matrix", variant: "scan" },
  searching: { family: "lattice", pattern: "ripple" },
  solving: { family: "matrix", variant: "twinkle" },
  listening: { family: "lattice", pattern: "dots" },
  connecting: { family: "lattice", pattern: "snake" },
  weaving: { family: "lattice", pattern: "spiral" },
  composing: { family: "lattice", pattern: "arrow" },
  breathing: { family: "matrix", variant: "pulse" },
  shaping: { family: "lattice", pattern: "orbit" },
};

/** What the lattice beside a thought runs, by what the agent is doing (lib/orbs.ts latticePattern). */
export const latticeFor = (design: string | null): LatticePattern => (design === "searching" ? "ripple" : design === "connecting" ? "snake" : design === "weaving" ? "spiral" : "orbit");

/** Text whose colour is a band of full ink travelling left to right over half-strength ink: a live status (`.t-shimmer`, 2 s, linear). */
export function Shimmer({ children }: { children: ReactNode }) {
  return <span className="a-shimmer">{children}</span>;
}

/* ── the thinking line ───────────────────────────────────────────────── */

const clock = (ds: number) => (ds < 600 ? `${(ds / 10).toFixed(1)}s` : `${Math.floor(ds / 600)}m ${((ds % 600) / 10).toFixed(1)}s`);

/**
 * The thought line (components/thought.tsx): a lattice and a label that shimmers while it thinks, with a timer that counts in tenths of
 * a second; when it stops, the label becomes "Thought for" (the words that go up and blur out, the ones that come in rise from below,
 * 450 ms) and the timer slides to the end of the shorter text. The thought itself opens under it in a Collapse.
 */
export function ThoughtLine({ ms, start, stopAt, openAt, closeAt, seconds, pattern = "orbit", workText = "Thinking…", doneText = "Thought for", children }: {
  ms: number; start: number; stopAt?: number; openAt?: number; closeAt?: number; seconds?: number; pattern?: LatticePattern; workText?: string; doneText?: string; children?: ReactNode;
}) {
  const working = stopAt === undefined || ms < stopAt;
  const stack = useRef<HTMLSpanElement>(null);
  const work = useRef<HTMLSpanElement>(null);
  const done = useRef<HTMLSpanElement>(null);
  const [dx, setDx] = useState(0);
  useLayoutEffect(() => {
    const s = stack.current;
    const w = done.current;
    if (s && w) setDx(w.offsetWidth - s.offsetWidth);
  }, [workText, doneText]);
  const t = lin(ms, stopAt ?? Infinity, DUR.thoughtSettle);
  const settle = EASE(t);
  const ds = working ? Math.floor(Math.max(0, ms - start) / 100) : Math.round((seconds ?? ((stopAt ?? 0) - start) / 1000) * 10);
  // the chevron turns over as the thought opens and back as it closes (200 ms)
  const chev = closeAt !== undefined && ms >= closeAt ? 1 - at(ms, closeAt, 200) : openAt !== undefined ? at(ms, openAt, 200) : 0;
  return (
    <div className="a-thought">
      <div className="a-thought__head">
        <span className="a-thought__glyph">
          <Lattice pattern={pattern} status={working ? "working" : "done"} />
        </span>
        <span ref={stack} className="a-thought__label" aria-hidden="true">
          <span ref={work} className="a-thought__text" style={{ opacity: 1 - settle, filter: `blur(${settle * 2}px)`, transform: `translateY(${-3 * settle}px)` }}>
            <span className="a-thought__shimmer">{workText}</span>
          </span>
          <span ref={done} className="a-thought__text" style={{ opacity: 0.75 * settle, filter: `blur(${(1 - settle) * 2}px)`, transform: `translateY(${3 * (1 - settle)}px)` }}>
            {doneText}
          </span>
        </span>
        <span className="a-thought__timer" style={{ opacity: working ? 0.55 : 0.55 + 0.2 * settle, transform: `translateX(${dx * EASE_LATTICE(t)}px)` }}>
          {clock(ds)}
        </span>
        <span className="a-thought__chev" style={{ transform: `rotate(${180 * chev}deg)` }} aria-hidden="true">
          ⌄
        </span>
      </div>
      {children !== undefined && (
        <Collapse ms={ms} open={openAt ?? Infinity} closeAt={closeAt}>
          {children}
        </Collapse>
      )}
    </div>
  );
}

/**
 * The thought as it is written, in a window of a fixed height (components/reason.tsx, "Reasoning stream"): while it is written the window
 * stays at its end, stepping down every 840 ms (a smooth scroll of about 500 ms), and its edges fade only where there is more to see.
 */
export function ReasonStream({ ms, start, text, shown }: { ms: number; start: number; text: string; shown: number }) {
  const view = useRef<HTMLDivElement>(null);
  const run = useRef({ k: -1, from: 0, to: 0, t0: 0 });
  useLayoutEffect(() => {
    const el = view.current;
    if (!el) return;
    const end = Math.max(0, el.scrollHeight - el.clientHeight);
    const k = ms < start ? -1 : Math.floor((ms - start) / DUR.reasonHold);
    const r = run.current;
    if (k !== r.k) {
      if (k < r.k) Object.assign(r, { from: 0, to: 0, t0: start });
      else Object.assign(r, { from: el.scrollTop, to: end, t0: start + k * DUR.reasonHold });
      r.k = k;
    }
    el.scrollTop = r.from + (r.to - r.from) * at(ms, r.t0, DUR.reasonStep);
    el.toggleAttribute("data-above", el.scrollTop > 2);
    el.toggleAttribute("data-below", end - el.scrollTop > 2);
  });
  const n = Math.max(0, Math.min(text.length, Math.floor(shown)));
  return (
    <div ref={view} className="a-reason" role="presentation">
      <div className="a-reason__text">
        {text.slice(0, n)}
      </div>
    </div>
  );
}
