"use client";

import { lazy, memo, Suspense, useEffect, useMemo, useState, type CSSProperties, type ReactNode } from "react";
import type { BotAvatarFace, BotAvatarState, BotAvatarType } from "@/vendor/bot-avatars/index.es.js";
import { usePrefersReducedMotion } from "../Avatar";
import { lin } from "./Scene";
import { Orb } from "./kit";
import { bezier, DUR, EASE, EASE_SOFT } from "./motion";
import { Enter, Shimmer } from "./parts";

/**
 * The parts of the side-panel and settings scenes that the shared parts do not cover: how the app's panels stretch where their content
 * arrives, its soft collapse, its small switch, its Loading line, how a person types, and the picker's avatars. Every number is the app's,
 * and the comment on each says which file it is from (serve/ui/src/...). Written again for a clock; no component of the app is copied.
 */

/** The English of the app and its Thai (`i18n/th*.ts`, merged in the order the app merges them). */
const TH: Record<string, string> = {
  "Git": "Git",
  "Plan": "วางแผน",
  "Skills": "Skill",
  "Memory": "หน่วยความจำ",
  "Context": "บริบท",
  "Reading the repository…": "กำลังอ่านรีโพซิทอรี…",
  "Not staged": "ยังไม่ staged",
  "Branches": "Branch",
  "Worktrees": "Worktree",
  "Recent commits": "commit ล่าสุด",
  "To-do list": "รายการงาน",
  "{done} of {total} done": "เสร็จ {done} จาก {total}",
  "{done} of {n} steps done": "ทำแล้ว {done} จาก {n} ขั้นตอน",
  "Looking for notes…": "กำลังหาโน้ต…",
  "The chat is handed these notes after its rules. They never change what it may do.": "แชทได้รับโน้ตเหล่านี้ต่อจากกฎของมัน โน้ตไม่เปลี่ยนสิ่งที่มันทำได้",
  "The project's own: {name}": "ของโปรเจกต์เอง: {name}",
  "its instructions for you": "คำสั่งที่มันมีให้คุณ",
  "always on": "เปิดเสมอ",
  "Context window": "หน้าต่างบริบท",
  "{used} of {max} tokens": "{used} จาก {max} โทเคน",
  "Conversation": "บทสนทนา",
  "Tool calls and results": "การเรียก tool และผลลัพธ์",
  "Instructions, tools and memory": "คำสั่ง, tool และ memory",
  "Free": "ว่าง",
  "Compacted by itself at {n} tokens ({pct}%).": "สรุปให้เองเมื่อถึง {n} โทเคน ({pct}%)",
  "Compact now": "สรุปเดี๋ยวนี้",
  "Settings": "ตั้งค่า",
  "How the app looks and what it connects to. The tools the model may use are set here too.": "หน้าตาของแอปและสิ่งที่เชื่อมต่อ รวมถึงเครื่องมือที่โมเดลใช้ได้ ตั้งได้ที่นี่",
  "General": "ทั่วไป",
  "MCP tools": "เครื่องมือ MCP",
  "Import": "นำเข้า",
  "Status marks": "ตัวแสดงสถานะ",
  "API key": "คีย์ API",
  "Coding tools": "เครื่องมือเขียนโค้ด",
  "Permissions": "สิทธิ์",
  "Hooks": "ฮุก",
  "Web access": "การเข้าถึงเว็บ",
  "Sub-agents": "ตัวช่วย (sub-agent)",
  "Servers": "เซิร์ฟเวอร์",
  "Limits": "ขีดจำกัด",
  "MCP servers": "เซิร์ฟเวอร์ MCP",
  "Allowed": "อนุญาต",
  "Never": "ไม่เคย",
  "Allow": "อนุญาต",
  "Add": "เพิ่ม",
  "Everywhere": "ทุกที่",
  "this project": "โปรเจกต์นี้",
  "every chat, every project": "ทุกแชท ทุกโปรเจกต์",
  "Nothing is allowed here for good.": "ยังไม่มีอะไรอนุญาตถาวรที่นี่",
  "Nothing is refused here for good.": "ยังไม่มีอะไรห้ามถาวรที่นี่",
  "What the chat's coding tools may do without asking, for good. A card that asks can keep your answer here (More choices), or you can write a rule. \"Allow for this chat\" is kept with the chat only and is not listed.": "สิ่งที่เครื่องมือเขียนโค้ดของแชททำได้โดยไม่ต้องถาม แบบถาวร การ์ดที่ถามเก็บคำตอบของคุณไว้ที่นี่ได้ (ตัวเลือกเพิ่ม) หรือจะเขียนกฎเองก็ได้ \"อนุญาตสำหรับแชทนี้\" เก็บไว้กับแชทเท่านั้นและไม่แสดงที่นี่",
  "A rule is a tool and, between brackets, what it is limited to: Bash(npm test:*) for commands that start with npm test, Read(src/**) for files under src, Edit(docs/**), or just Read for all of that tool. \"Never\" wins over \"Allow\". No rule can allow a secret, a change in .git or a command that can do harm that is hard to undo: those ask every time.": "กฎคือ tool และในวงเล็บคือสิ่งที่จำกัดไว้ เช่น Bash(npm test:*) สำหรับคำสั่งที่ขึ้นต้นด้วย npm test, Read(src/**) สำหรับไฟล์ใต้ src, Edit(docs/**) หรือแค่ Read สำหรับ tool นั้นทั้งหมด \"ไม่เคย\" ชนะ \"อนุญาต\" ไม่มีกฎใดอนุญาต secret การแก้ใน .git หรือคำสั่งที่ทำลายแล้วกู้ยากได้ เรื่องเหล่านี้ถามทุกครั้ง",
  "New project": "โปรเจกต์ใหม่",
  "Project name": "ชื่อโปรเจกต์",
  "Folders": "โฟลเดอร์",
  "Cancel": "ยกเลิก",
  "Create project": "สร้างโปรเจกต์",
  "Main": "หลัก",
  "Make main": "ตั้งเป็นหลัก",
  "Browse": "เลือกโฟลเดอร์",
  "A project is one or more folders on this PC, such as the worktrees of one repository. The coding tools of its chats work in them: files inside are free to read and change, everything else asks you first. The first folder is the main one: commands run there.": "โปรเจกต์คือโฟลเดอร์หนึ่งโฟลเดอร์ขึ้นไปบนเครื่องนี้ เช่น worktree ของ repository เดียวกัน เครื่องมือเขียนโค้ดของแชทในโปรเจกต์ทำงานในโฟลเดอร์เหล่านั้น ไฟล์ข้างในอ่านและแก้ได้เลย อย่างอื่นจะถามคุณก่อน โฟลเดอร์แรกคือโฟลเดอร์หลัก คำสั่งจะรันที่นั่น",
  "Orbs": "Orb",
  "Orbs + Loading": "Orb + ตัวโหลด",
  "Loading only": "ตัวโหลดอย่างเดียว",
  "Avatar": "อวาตาร์",
  "Default": "ค่าเริ่มต้น",
  "By status": "ตามสถานะ",
  "Random": "สุ่ม",
  "What shows that something is happening: thinking, answering, reading.": "สิ่งที่บอกว่ากำลังมีอะไรเกิดขึ้น: กำลังคิด กำลังตอบกลับ กำลังอ่าน",
  "What shows that something is happening: thinking, answering, reading. Pick one; the same list opens from the button at the top.":"สิ่งที่บอกว่ากำลังมีอะไรเกิดขึ้น: กำลังคิด กำลังตอบกลับ กำลังอ่าน เลือกได้หนึ่งแบบ รายการเดียวกันนี้เปิดได้จากปุ่มด้านบน",
  "A dotted ball that takes a form for each status.": "ลูกบอลจุดที่เปลี่ยนรูปทรงตามสถานะ",
  "Orbs, with a lattice of dots beside the thinking. How it starts.": "Orb และมีตารางจุดข้างตอนคิด เป็นค่าเริ่มต้น",
  "Loaders of dots, a pattern for each status. No orbs.": "ตัวโหลดจุดอย่างเดียว ลวดลายต่างกันตามสถานะ ไม่มี orb",
  "A small character with a shape and a mood of its own.": "ตัวละครเล็ก ๆ ที่มีรูปร่างและอารมณ์ของตัวเอง",
  "Which avatar: one for every status, a shape for each status, or a random one for each place.": "เลือกอวาตาร์ตัวเดียวสำหรับทุกสถานะ ให้แต่ละสถานะมีรูปร่างของตัวเอง หรือสุ่มให้แต่ละจุด",
};

/** The words of the app in Thai, or the English itself where the app has no Thai for it. `vars` fills {name} in either language. */
export function tr(lang: "en" | "th", key: string, vars?: Record<string, string | number>): string {
  const s = lang === "th" ? TH[key] ?? key : key;
  return vars ? s.replace(/\{(\w+)\}/g, (m, k: string) => (k in vars ? String(vars[k]) : m)) : s;
}

/* ── curves and clocks ───────────────────────────────────────────────── */

/** CSS's own `ease` (the curve of a transition that names none, and of `transition: background-color 200ms`). */
export const EASE_CSS = bezier(0.25, 0.1, 0.25, 1);
/** Tailwind's default transition curve, cubic-bezier(0.4, 0, 0.2, 1): `transition-colors`, `transition-transform duration-200`, `transition-[width] duration-300`. */
export const EASE_TW = EASE_SOFT;
/** Tailwind's `transition-colors` with no duration: 150 ms. */
export const TW_MS = 150;

/** Progress of a stretch of the clock with a curve. */
export const at = (ms: number, start: number, dur: number, ease: (t: number) => number = EASE) => ease(lin(ms, start, dur));

/** `active:scale-[0.98]` of the app's Button (150 ms transition): the scale it has while it is pressed at `from` and let go 120 ms later. */
export function press(ms: number, from: number): number {
  const d = ms - from;
  if (d < 0 || d > 300) return 1;
  const down = EASE_TW(lin(d, 0, TW_MS));
  const up = EASE_TW(lin(d, 120, TW_MS));
  return 1 - 0.02 * down * (1 - up);
}

/** The panel body when the side panel is opened: the whole of it fades in from nothing, 260 ms (`.side-in`, orb-fade). */
export const SIDE_IN = 260;

/* ── how a person types ──────────────────────────────────────────────── */

// The gap before the next key: not even, as a hand is not (the same text always takes the same time).
const gap = (text: string, j: number) => 58 + ((j * 7 + 3) % 5) * 14 + (text[j] === " " ? 36 : 0);

/** How many characters of `text` are typed at `ms` by someone who starts at `start`. */
export function typed(text: string, ms: number, start: number): number {
  let t = start;
  for (let j = 0; j < text.length; j++) {
    if (ms < t) return j;
    t += gap(text, j);
  }
  return text.length;
}
/** When the last key of `text` is struck. */
export function typedEnd(text: string, start: number): number {
  let t = start;
  for (let j = 0; j < text.length - 1; j++) t += gap(text, j);
  return t;
}

/** The caret of a field being typed in: a thin bar in the accent colour (`caret-color`), not the block that follows a streamed answer. */
export function Caret() {
  return <span className="a-caret a-caret--blink" aria-hidden="true" />;
}

/* ── layout that is reserved ─────────────────────────────────────────── */

/**
 * A scene is a function of the clock, and so is its height (a section that stretches open is shorter before it does). The page around it
 * must not move for that, so the last frame is drawn once, out of sight, and gives the stage its size; the live frame sits on top of it.
 * `render(ms, ghost)`: `ghost` is true for the copy that is only there to take up room.
 */
export function Stage({ ms, end, lang, className = "", center = false, render }: { ms: number; end: number; lang: string; className?: string; center?: boolean; render: (ms: number, ghost: boolean) => ReactNode }) {
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const ghost = useMemo(() => render(end, true), [end, lang]);
  return (
    <div className={"sp-stage " + className}>
      <div className="sp-ghost" aria-hidden="true">
        {ghost}
      </div>
      <div className={"sp-live" + (center ? " sp-live--c" : "")}>{render(ms, false)}</div>
    </div>
  );
}

/* ── how a thing opens ───────────────────────────────────────────────── */

/**
 * A panel that stretches where its content arrived (lib/glide.ts `stretch()`): the new part goes from no height to its height,
 * 420 ms, cubic-bezier(0.4, 0, 0.2, 1), clipped while it grows, with no fade and no slide: only the height moves.
 */
export function Glide({ ms, start, className = "", children }: { ms: number; start: number; className?: string; children: ReactNode }) {
  const p = at(ms, start, DUR.glide, EASE_SOFT);
  if (p >= 1) return <div className={className}>{children}</div>;
  return (
    <div className={className} style={{ display: "grid", gridTemplateRows: `${p}fr` }}>
      <div style={{ minHeight: 0, overflow: "hidden" }}>{children}</div>
    </div>
  );
}

/**
 * `Collapse soft` of the app (components/motion.tsx, styles.css `.collapse-soft`): a section that is a good part of a page stretches open
 * over 520 ms, cubic-bezier(0.4, 0, 0.2, 1), while its opacity follows over 360 ms with CSS's `ease`.
 */
export function Fold({ ms, open, children }: { ms: number; open: number; children: ReactNode }) {
  const rows = at(ms, open, DUR.collapseSoft, EASE_SOFT);
  const opacity = at(ms, open, DUR.collapseSoftOpacity, EASE_CSS);
  if (ms < open) return null;
  return (
    <div style={{ display: "grid", gridTemplateRows: `${rows}fr`, opacity }}>
      <div style={{ minHeight: 0, overflow: "hidden" }}>{children}</div>
    </div>
  );
}

/** The arrow of a heading that opens a section: it turns over in 200 ms (`transition-transform duration-200`). `from` and `to` are degrees. */
export const turn = (ms: number, start: number, from = 0, to = 180) => from + (to - from) * at(ms, start, 200, EASE_TW);

/* ── the switch of a row (`prompt-bar__sw`) ──────────────────────────── */

/**
 * The small switch: 32 × 18, a 14 px knob that moves 14 px in 200 ms (the app's ease), a track that goes from the grey to green in 200 ms
 * (CSS `ease`), a knob that goes from half-strength ink to full white. `flips` are the moments it is switched; it starts off.
 */
export function MiniSwitch({ ms, flips }: { ms: number; flips: number[] }) {
  const done = flips.filter((f) => ms >= f);
  const on = done.length % 2 === 1;
  const last = done.length ? done[done.length - 1] : null;
  const t = last === null ? 1 : lin(ms, last, 200);
  const x = on ? EASE(t) : 1 - EASE(t);
  const c = on ? EASE_CSS(t) : 1 - EASE_CSS(t);
  const settled = last === null || t >= 1;
  return (
    <span className="sp-sw" role="presentation" style={{ background: settled ? (on ? "var(--a-ok)" : undefined) : `color-mix(in srgb, var(--a-ok) ${c * 100}%, color-mix(in srgb, var(--a-ink) 18%, transparent))` }} aria-hidden="true">
      <i style={{ transform: `translateX(${14 * x}px)`, background: on ? "#fff" : "var(--a-ink)", opacity: 0.55 + 0.45 * c }} />
    </span>
  );
}

/* ── a line that waits ───────────────────────────────────────────────── */

/** `Loading` of the app: an orb (the `connecting` form, 20 px, which dissolves in over 450 ms) and the words, shimmering, in a `py-6` line. */
export function LoadingLine({ ms, start = 0, text }: { ms: number; start?: number; text: string }) {
  return (
    <div className="sp-loading" role="presentation">
      <div className="sp-status">
        <Enter ms={ms} start={start} kind="orb" className="sp-orbslot">
          <Orb ms={ms} live size={20} design="connecting" />
        </Enter>
        <Shimmer>{text}</Shimmer>
      </div>
    </div>
  );
}

/* ── icons (drawn here: the app uses Hugeicons, which this site does not ship) ── */

const ICONS: Record<string, ReactNode> = {
  chevron: <path d="M6 9l6 6 6-6" />,
  close: <path d="M6 6l12 12M18 6L6 18" />,
  refresh: <path d="M20 12a8 8 0 1 1-2.6-5.9M20 4v4.5h-4.5" />,
  folder: <path d="M3 7.5A2.5 2.5 0 0 1 5.5 5H9l2 2.5h7.5A2.5 2.5 0 0 1 21 10v7.5a2.5 2.5 0 0 1-2.5 2.5h-13A2.5 2.5 0 0 1 3 17.5z" />,
  branch: (
    <>
      <circle cx="6" cy="5.5" r="2.3" />
      <circle cx="6" cy="18.5" r="2.3" />
      <circle cx="18" cy="8.5" r="2.3" />
      <path d="M6 8v8M18 10.8c0 3.7-3.5 3.4-8.2 5.4" />
    </>
  ),
  commit: (
    <>
      <circle cx="12" cy="12" r="3.6" />
      <path d="M3 12h5.4M15.6 12H21" />
    </>
  ),
  circle: <circle cx="12" cy="12" r="9.5" />,
  check: (
    <>
      <circle cx="12" cy="12" r="9.5" />
      <path d="M8 12.4l2.8 2.8 5.2-5.6" />
    </>
  ),
  book: <path d="M12 6.8C10 5.2 7 4.7 3.5 5.2v13c3.5-.5 6.5 0 8.5 1.6 2-1.6 5-2.1 8.5-1.6v-13C17 4.7 14 5.2 12 6.8zM12 6.8v13" />,
  layers: <path d="M12 3.5l9 5-9 5-9-5zM3 13l9 5 9-5" />,
  // settings menu
  paint: <path d="M4 5h16v14H4zM8 9.5h3M8 14h8" />,
  key: (
    <>
      <circle cx="8" cy="15.5" r="3.7" />
      <path d="M10.6 12.9L20 3.5M16.2 7.3l2.4 2.4" />
    </>
  ),
  terminal: <path d="M3.5 5h17v14h-17zM7 10l3 2.2L7 14.4M12.5 15h4.5" />,
  shield: <path d="M12 3l8 3v5.5c0 4.5-3.4 8-8 9.5-4.6-1.5-8-5-8-9.5V6z" />,
  hook: (
    <>
      <circle cx="12" cy="6.5" r="2.5" />
      <circle cx="6" cy="17.5" r="2.5" />
      <circle cx="18" cy="17.5" r="2.5" />
      <path d="M12 9v4l-4.7 3M12 13l4.7 3" />
    </>
  ),
  globe: <path d="M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM3 12h18M12 3c3.2 3.2 3.2 14.8 0 18M12 3c-3.2 3.2-3.2 14.8 0 18" />,
  group: (
    <>
      <circle cx="9" cy="8" r="3" />
      <circle cx="17.2" cy="9" r="2.4" />
      <path d="M3 19c0-3 2.8-5 6-5s6 2 6 5M16 14.2c3 0 5 1.4 5 4" />
    </>
  ),
  server: <path d="M4 4.5h16v6H4zM4 13.5h16v6H4zM7.5 7.5h.01M7.5 16.5h.01" />,
  plug: <path d="M9 3v5M15 3v5M6 8h12v3a6 6 0 0 1-12 0zM12 17v4" />,
  download: <path d="M12 4v11M7 11l5 5 5-5M5 20h14" />,
  brain: <path d="M12 5v14M9.5 5.5a3 3 0 0 0-3.2 3.3A3 3 0 0 0 5 13.5a3 3 0 0 0 2.6 4A3 3 0 0 0 12 19a3 3 0 0 0 4.4-1.5 3 3 0 0 0 2.6-4 3 3 0 0 0-1.3-4.7 3 3 0 0 0-3.2-3.3A3 3 0 0 0 12 5a3 3 0 0 0-2.5.5z" />,
};

export type IconName = keyof typeof ICONS;
export function Ico({ n, size = 14, style, className }: { n: IconName; size?: number; style?: CSSProperties; className?: string }) {
  return (
    <svg viewBox="0 0 24 24" width={size} height={size} fill="none" stroke="currentColor" strokeWidth={1.7} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" className={"sp-ico " + (className ?? "")} style={style}>
      {ICONS[n]}
    </svg>
  );
}

/* ── an avatar of the picker ─────────────────────────────────────────── */

// bot-avatars (the app's own, vendored): loaded when the first one is wanted, as the app does.
const Bot = lazy(() => import("@/vendor/bot-avatars/index.es.js").then((m) => ({ default: m.BotAvatar })));

/** One avatar as the app draws it in the picker (`BotTile`: 44 px, smooth shading, not following the pointer) or in a row (20 px, eyes). */
export const BotMark = memo(function BotMark({ type, size = 44, state = "default", face }: { type: BotAvatarType; size?: number; state?: BotAvatarState; face?: BotAvatarFace }) {
  const reduced = usePrefersReducedMotion();
  // the library touches the document when it is loaded: it is asked for only once the page is in the browser (as components/Avatar.tsx does)
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  if (!mounted) return <span style={{ display: "block", width: size, height: size }} />;
  return (
    <Suspense fallback={<span style={{ display: "block", width: size, height: size }} />}>
      <Bot type={type} state={state} size={size} shading="smooth" interactive={false} paused={reduced} {...(face ? { face } : null)} aria-hidden="true" role="presentation" />
    </Suspense>
  );
});
