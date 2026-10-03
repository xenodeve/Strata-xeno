"use client";

import { Fragment, useLayoutEffect, useRef, type ReactNode } from "react";
import { Code } from "./code";
import { bezier } from "./motion";

/**
 * Parts for the classic app's scenes. The classic web UI (serve/web: app.js, app.css, components.css, tokens.css) is plain: no
 * entrance motion at all. Its few moves are CSS transitions on `--st-dur` / `--st-ease` (tokens.css: 160 ms,
 * cubic-bezier(.2, .7, .2, 1); not the new app's curve), caret blinks, and text that app.js writes into the page. Nothing here
 * eases in; what arrives, arrives in one frame.
 */

/** tokens.css `--st-ease`. */
export const ST_EASE = bezier(0.2, 0.7, 0.2, 1);
/** tokens.css `--st-dur`. */
export const ST_DUR = 160;
/** A classic transition that starts at `start`: `.st-chev` (transform), `.badges .st-badge` (opacity), `.st-progress__bar` (width). */
export const stEase = (ms: number, start: number, dur = ST_DUR) => ST_EASE(Math.min(1, Math.max(0, (ms - start) / dur)));

/* ── icons: the classic uses its own sprite; these are plain 24 px outlines of the same things ──────────────── */
const PATHS = {
  chat: <path d="M4 5h16v11H10l-4 3.5V16H4z" />,
  activity: <path d="M3 12h4l3-7 4 14 3-7h4" />,
  info: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 11v5M12 8h.01" />
    </>
  ),
  moon: <path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z" />,
  send: <path d="M21 3 3 10.5l7 2.5 2.5 7zM10 13 21 3" />,
  stop: <rect x="6" y="6" width="12" height="12" rx="2.5" />,
  attach: <path d="m20 11.5-8.5 8.5a5 5 0 0 1-7-7L13 4.5a3.3 3.3 0 0 1 4.7 4.7L9.2 17.7a1.7 1.7 0 0 1-2.4-2.4l7.7-7.7" />,
  newchat: <path d="M4 5h16v11H10l-4 3.5V16H4zM12 8v5M9.5 10.5h5" />,
  download: <path d="M12 4v11M7.5 10.5 12 15l4.5-4.5M5 19h14" />,
  settings: (
    <>
      <circle cx="12" cy="12" r="3" />
      <path d="M12 3v2.5M12 18.5V21M3 12h2.5M18.5 12H21M5.6 5.6l1.8 1.8M16.6 16.6l1.8 1.8M5.6 18.4l1.8-1.8M16.6 7.4l1.8-1.8" />
    </>
  ),
  thinking: <path d="M9 18h6M10 21h4M12 3a6 6 0 0 0-3.5 10.9V16h7v-2.1A6 6 0 0 0 12 3z" />,
  chevron: <path d="m6 9 6 6 6-6" />,
  copy: <path d="M9 9h10v10H9zM5 15V6a1 1 0 0 1 1-1h9" />,
  gauge: <path d="M4 17a8 8 0 1 1 16 0M12 17l4-5" />,
  gpu: (
    <>
      <rect x="3" y="6" width="18" height="11" rx="2" />
      <circle cx="9" cy="11.5" r="2.5" />
      <circle cx="15" cy="11.5" r="2.5" />
    </>
  ),
  layers: <path d="m12 4 9 4.5-9 4.5-9-4.5zM3 13l9 4.5 9-4.5" />,
  thermometer: <path d="M10 14.5V5a2 2 0 0 1 4 0v9.5a4 4 0 1 1-4 0z" />,
  bolt: <path d="M13 3 5 14h6l-1 7 8-11h-6z" />,
  link: <path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1" />,
  cpu: (
    <>
      <rect x="7" y="7" width="10" height="10" rx="1.5" />
      <path d="M10 3v4M14 3v4M10 17v4M14 17v4M3 10h4M3 14h4M17 10h4M17 14h4" />
    </>
  ),
  disk: (
    <>
      <rect x="3" y="7" width="18" height="10" rx="2" />
      <path d="M7 12h.01M11 12h6" />
    </>
  ),
} satisfies Record<string, ReactNode>;
export type IcoName = keyof typeof PATHS;

export function Ico({ n, size = 16, style }: { n: IcoName; size?: number; style?: React.CSSProperties }) {
  return (
    <svg className="cl-ico" viewBox="0 0 24 24" width={size} height={size} fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" style={style} aria-hidden="true">
      {PATHS[n]}
    </svg>
  );
}

/**
 * The chat's scroll box. app.js calls `scrollDown()` (scrollTop = scrollHeight, no easing) at every paint while the reader is
 * near the bottom, so the newest line is always in view and the view moves only as the answer grows (the answer is not
 * reserved: it is as tall as what has been written). Done here after every frame.
 */
export function StickBottom({ className, children }: { className?: string; children: ReactNode }) {
  const box = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const e = box.current;
    if (e) e.scrollTop = e.scrollHeight;
  });
  return (
    <div ref={box} className={className}>
      {children}
    </div>
  );
}

/* ── an answer as the classic draws it: its markdown is rebuilt from the text so far at every paint ──────────── */
type Block = { k: "p"; lines: string[] } | { k: "code"; lang: string; body: string; closed: boolean };

/** The text so far as blocks, the way the classic reads it: a fence opens only once its line is complete (until then it is text), a block without its closing fence is still being written. */
export function blocks(text: string): Block[] {
  const out: Block[] = [];
  const paras = (s: string) => {
    let cur: string[] = [];
    for (const l of s.split("\n")) {
      if (l.trim() === "") {
        if (cur.length) out.push({ k: "p", lines: cur });
        cur = [];
      } else cur.push(l);
    }
    if (cur.length) out.push({ k: "p", lines: cur });
  };
  let rest = text;
  for (;;) {
    const open = /(^|\n)```([^\n`]*)\n/.exec(rest);
    if (!open) {
      paras(rest);
      break;
    }
    paras(rest.slice(0, open.index));
    rest = rest.slice(open.index + open[0].length);
    const lang = open[2].trim();
    const end = /(^|\n)```[ \t]*(\n|$)/.exec(rest);
    if (!end) {
      out.push({ k: "code", lang, body: rest, closed: false });
      break;
    }
    out.push({ k: "code", lang, body: rest.slice(0, end.index), closed: true });
    rest = rest.slice(end.index + end[0].length);
  }
  return out;
}

/** Inline code: only a closed pair of backticks turns into code; an open one stays as the character. */
function inline(s: string): ReactNode[] {
  const out: ReactNode[] = [];
  let last = 0;
  let k = 0;
  for (const m of s.matchAll(/`([^`\n]+)`/g)) {
    const at = m.index ?? 0;
    if (at > last) out.push(s.slice(last, at));
    out.push(
      <code key={k++} className="cl-ic">
        {m[1]}
      </code>,
    );
    last = at + m[0].length;
  }
  if (last < s.length) out.push(s.slice(last));
  return out;
}

const isWord = (c: string) => /\w/.test(c);
/** A block that is still being written is coloured again word by word: the word in the making waits for the character after it. */
const byWords = (body: string, closed: boolean) => {
  if (closed) return body;
  let n = body.length;
  while (n > 0 && isWord(body[n - 1])) n--;
  return body.slice(0, n);
};

export function Reply({ text }: { text: string }) {
  return (
    <>
      {blocks(text).map((b, i) =>
        b.k === "p" ? (
          <p key={i}>
            {b.lines.map((l, j) => (
              <Fragment key={j}>
                {j > 0 && <br />}
                {inline(l)}
              </Fragment>
            ))}
          </p>
        ) : (
          <div key={i} className="l-code cl-code">
            <div className="l-code__h">
              <span>{b.lang || "code"}</span>
              <span className="cl-ibtn" aria-hidden="true">
                <Ico n="copy" size={14} />
              </span>
            </div>
            <pre className="mk">
              <Code code={byWords(b.body, b.closed)} lang={b.lang} />
            </pre>
          </div>
        ),
      )}
    </>
  );
}

/** A sparkline as the classic's `spark()` draws it: a path set in one go (no drawing-in), y = 30 - value * 26 in a 100 x 32 box. */
export function Line({ values, also, tone }: { values?: number[]; /** the speed card's second series (prefill), a darker line over the first */ also?: number[]; tone?: "warn" | "info" }) {
  const path = (v: number[]) => v.map((x, i) => `${i ? "L" : "M"}${((i / (v.length - 1)) * 100).toFixed(2)},${(30 - x * 26).toFixed(2)}`).join("");
  const one = (v: number[] | undefined) => {
    const d = v && v.length > 1 ? path(v) : "";
    return (
      <>
        <path d={d ? `${d}L100,32L0,32Z` : ""} fill="currentColor" opacity=".12" />
        <path d={d} fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" strokeLinecap="round" vectorEffect="non-scaling-stroke" />
      </>
    );
  };
  return (
    <svg className={"cl-spark" + (tone ? " cl-spark--" + tone : "")} viewBox="0 0 100 32" preserveAspectRatio="none" aria-hidden="true">
      {one(values)}
      {also && <g className="cl-prefill">{one(also)}</g>}
    </svg>
  );
}
