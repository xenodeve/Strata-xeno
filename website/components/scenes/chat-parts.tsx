"use client";

import { useLayoutEffect, useRef, type CSSProperties, type ReactNode } from "react";
import { lin } from "./Scene";
import { DUR, EASE, EASE_SOFT } from "./motion";
import { Orb } from "./kit";
import { Code } from "./code";
import { Collapse, Enter, Pop, Shimmer, ThoughtLine, type LatticePattern } from "./parts";

/**
 * The parts of the app's chat, written for a clock (components/scenes/chat.tsx plays them). Everything here is a function of `ms`.
 * Each part says which part of the app it follows: pages/chat/Messages.tsx and AgentCall.tsx (the message, the tool card, the status
 * under the answer), components/PromptBar.tsx and prompt-bar.css (the composer), components/Sidebar.tsx + BranchedMenu.tsx (the list),
 * lib/chat.ts (how a reply streams in), lib/sendfx.ts (the prompt that rises out of the composer), lib/orbs.ts + lib/status.ts
 * (which orb, which words), components/orb.tsx (an orb that changes form dissolves for 450 ms).
 */

export type Lang = "en" | "th";
export type OrbDesign = "working" | "searching" | "solving" | "listening" | "connecting" | "weaving" | "composing" | "breathing" | "shaping";
export type ModeKey = "ask" | "plan" | "auto";

/* ── the app's words (serve/ui/src, English; Thai from i18n/th-chat.ts and th-core.ts) ───────────────────────────── */

const EN = {
  waiting: "Waiting for the model…",
  reading: "Reading the prompt…",
  thinking: "Thinking…",
  thoughts: "Thoughts",
  thoughtFor: "Thought for",
  stWriting: "Writing",
  stAsking: "Waiting for you",
  stRunning: "Running",
  stDone: "Done",
  askQ: "Run this command?",
  allow: "Allow",
  deny: "Deny",
  why: "It writes to a file or hides what runs (a substitution, a redirection).",
  allowed: "You allowed it.",
  you: "You · {time}",
  prefill: "Prefill {rate} tok/s mean",
  tokens: "{n} tokens",
  toolCalls: "{n} tool calls",
  tokS: "{n} tok/s",
  copy: "Copy",
  message: "Message",
  busy: "Message: it is sent when the answer ends",
  newChat: "New chat",
  projects: "Projects",
  recents: "Recents",
  empty: "Empty",
  noChats: "No conversations yet",
  hello: "What can I help with?",
  lede: "{name} runs on this PC. Nothing leaves it.",
  writingCall: "Writing the call to {tool}…",
  runningTool: "Running {tool}…",
  readingResult: "Reading the tool's result…",
  answering: "Answering…",
  waitingYou: "Waiting for you…",
  ask: "Ask",
  plan: "Plan",
  auto: "Auto",
  xhigh: "XHigh",
  modeAsk: "Files inside the folder are free to read and change. Everything else asks you first.",
  modePlan: "Nothing is changed. The model reads, then sends a plan for you to approve.",
  modeAuto: "A second check decides what would ask you. What is risky is blocked or asked; dangerous commands and secrets always ask you.",
  noProject: "No project",
};
export type Words = typeof EN;
const TH: Words = {
  waiting: "กำลังรอโมเดล…",
  reading: "กำลังอ่าน prompt…",
  thinking: "กำลังคิด…",
  thoughts: "ความคิด",
  thoughtFor: "คิดไป",
  stWriting: "กำลังเขียน",
  stAsking: "รอคุณ",
  stRunning: "กำลังรัน",
  stDone: "เสร็จ",
  askQ: "รันคำสั่งนี้ไหม",
  allow: "อนุญาต",
  deny: "ไม่อนุญาต",
  why: "เขียนลงไฟล์ หรือซ่อนสิ่งที่จะรัน (การแทนที่คำสั่ง การ redirect)",
  allowed: "คุณอนุญาต",
  you: "คุณ · {time}",
  prefill: "prefill ค่าเฉลี่ย {rate} tok/s",
  tokens: "{n} token",
  toolCalls: "เรียกเครื่องมือ {n} ครั้ง",
  tokS: "{n} tok/s",
  copy: "คัดลอก",
  message: "ข้อความ",
  busy: "ข้อความ: จะส่งเมื่อคำตอบจบ",
  newChat: "แชทใหม่",
  projects: "โปรเจกต์",
  recents: "ล่าสุด",
  empty: "ว่าง",
  noChats: "ยังไม่มีการสนทนา",
  hello: "มีอะไรให้ช่วย",
  lede: "{name} ทำงานบนเครื่องนี้ ไม่มีข้อมูลออกจากเครื่อง",
  writingCall: "กำลังเขียนการเรียก {tool}…",
  runningTool: "กำลังรัน {tool}…",
  readingResult: "กำลังอ่านผลของ tool…",
  answering: "กำลังตอบกลับ…",
  waitingYou: "รอคุณอยู่…",
  ask: "ถาม",
  plan: "แผน",
  auto: "อัตโนมัติ",
  xhigh: "สูงมาก",
  modeAsk: "ไฟล์ในโฟลเดอร์อ่านและแก้ได้เลย อย่างอื่นจะถามคุณก่อน",
  modePlan: "ไม่มีอะไรถูกแก้ โมเดลจะอ่านแล้วส่งแผนมาให้คุณอนุมัติ",
  modeAuto: "การตรวจรอบที่สองตัดสินสิ่งที่ปกติจะถามคุณ สิ่งที่เสี่ยงจะถูกบล็อกหรือถามคุณ ส่วนคำสั่งอันตรายและความลับจะถามคุณเสมอ",
  noProject: "ไม่อยู่ในโปรเจกต์",
};
export const wordsOf = (lang: Lang): Words => (lang === "th" ? TH : EN);
const fill = (s: string, v: Record<string, string | number>) => s.replace(/\{(\w+)\}/g, (_, k: string) => String(v[k] ?? ""));

/* ── the one real run these scenes show (its text and figures) ──────────────────────────────────────────────────── */

export const PROMPT = 'Read notes.py, then add a --version flag that prints "notes-app 0.1.0" and exits. Keep the change small.';
export const PROJECT = "notes-app";
export const MODEL = "Swift Flash Next · IQ2_XS"; // the model of the run (the Dashboard scene shows the same name)
export const FILE = "C:\\demo\\notes-app\\notes.py";
export const BASH_NOTE = "Run --version and list to check behaviour";
export const BASH_CMD = 'python notes.py --version; echo "exit=$?"; python notes.py list; echo "exit=$?"';
export const THOUGHT_SECS = 76.3; // "Thought for 1m 16.3s": what the first thought says once the answer begins (the classic scene shows the same figure)
export const TIME = "03:16 PM";

export type Seg = string | { code: string };
export const REPLY_A: Seg[] = ["Added a ", { code: "--version" }, " branch at the top of ", { code: "main" }, ", before the notes are loaded, so it prints and exits without touching the database:"];
export const REPLY_B: Seg[] = ["Ran it: ", { code: "python notes.py --version" }, " → ", { code: "notes-app 0.1.0" }, ", exit 0. ", { code: "python notes.py list" }, " still works (exit 0, no notes stored)."];
export const CODE_TEXT = ["def main(argv: list[str]) -> int:", '    if argv[:1] == ["--version"]:', '        print("notes-app 0.1.0")', "        return 0", "    notes = load()", "    ..."].join("\n");

/* ── how a reply streams in (lib/chat.ts): the text arrives a token at a time and the page is drawn once a frame ─── */

// the text as the model writes it: a name between backticks
const plain = (s: Seg[]) => s.map((x) => (typeof x === "string" ? x : "`" + x.code + "`")).join("");
// a token is a word (with the space before it) or one mark: close to what a tokenizer cuts, and what the code is re-coloured at
const TOKEN = /\s*[A-Za-z0-9_]+|\s*[^\sA-Za-z0-9_]|\s+/g;
const cutsOf = (s: string): number[] => {
  const out = [0];
  for (const m of s.matchAll(TOKEN)) out.push(m.index! + m[0].length);
  return out;
};
const CUTS_A = cutsOf(plain(REPLY_A));
const CUTS_C = cutsOf(CODE_TEXT);
const CUTS_B = cutsOf(plain(REPLY_B));
const NA = CUTS_A.length - 1;
const NC = CUTS_C.length - 1;
const NB = CUTS_B.length - 1;
const FENCE_OPEN = 3; // ```  python  newline
const FENCE_SHUT = 2; // newline  ```
/** The tokens of the answer's text, as the model wrote it: the paragraph, a blank line, the fenced code, a blank line, the paragraph. */
export const ANSWER_TOKENS = NA + 1 + FENCE_OPEN + NC + FENCE_SHUT + 1 + NB;

export type ReplyShown = { a: number; code: number | null; shut: boolean; b: number; caret: "a" | "code" | "b" | null };
/** What has arrived after `k` tokens of the answer: characters of each block. */
export function shownAt(k: number): ReplyShown {
  const n = Math.max(0, Math.floor(k));
  const codeAt = NA + 1 + FENCE_OPEN;
  const shutAt = codeAt + NC + FENCE_SHUT;
  const bAt = shutAt + 1;
  const a = CUTS_A[Math.min(NA, n)];
  const code = n >= codeAt ? CUTS_C[Math.min(NC, n - codeAt)] : null;
  const shut = n >= shutAt;
  const b = n > bAt ? CUTS_B[Math.min(NB, n - bAt)] : 0;
  const done = n >= ANSWER_TOKENS;
  return { a, code, shut, b, caret: done ? null : n > bAt ? "b" : code !== null ? "code" : "a" };
}

/* ── the run, as a clock (everything the chat does, in the order the app does it) ────────────────────────────────── */

export const RATE = 63.1; // tok/s of the run (the line under the answer says so)
const tk = (n: number) => Math.round((n / RATE) * 1000); // ms the model takes to write n tokens at that speed

// tokens written, by the part of the answer; they add up to the figures the run showed: 363 when the Edit result is read, 434 when the
// command asks, 642 at the end (the split between thinking and calls inside those is only a way to move the count)
const T_R1 = 100;
const T_C1 = 24;
const T_R2 = 104;
const T_C2 = 135;
const T_R3 = 36;
const T_C3 = 35;
export const TOKENS_ASK = T_R1 + T_C1 + T_R2 + T_C2 + T_R3 + T_C3; // 434
export const TOKENS_EDIT = T_R1 + T_C1 + T_R2 + T_C2; // 363
export const TOKENS_ALL = 642;
const T_R4 = Math.max(8, TOKENS_ALL - TOKENS_ASK - ANSWER_TOKENS);

export type Run = ReturnType<typeof buildRun>;
/** `wait`: how long the command's question stays open before it is answered; `ran`: how long the command runs after that. */
export function buildRun(S: number, o: { wait?: number; ran?: number } = {}) {
  const wait = o.wait ?? 1400;
  const ran = o.ran ?? 450;
  const r0 = S + 550; // the server begins to read the prompt (until then: "Waiting for the model…")
  const g1 = r0 + 650; // the prompt is read; the model begins to think
  const a1 = g1 + tk(T_R1); // the call to Read starts to be written
  const c1 = a1 + tk(T_C1); // ... is made
  const d1 = c1 + 300; // ... has its result
  const g2 = d1 + 500; // the server reads the result; the model thinks again
  const a2 = g2 + tk(T_R2);
  const c2 = a2 + tk(T_C2);
  const d2 = c2 + 300;
  const g3 = d2 + 700;
  const a3 = g3 + tk(T_R3);
  const ask = a3 + tk(T_C3); // the command asks
  const ans = ask + wait; // the server took the answer
  const press = Math.max(ask, ans - 150); // Allow is pressed (the buttons dim, then the answer is taken)
  const d3 = ans + ran;
  const g4 = d3 + 500;
  const aS = g4 + tk(T_R4); // the answer's text begins
  const end = aS + tk(ANSWER_TOKENS); // the stream ends
  const landed = end + DUR.handoverDelay + DUR.handoverIn;
  // the tokens, from the moments the model writes: a piece that is written has its tokens in a straight line, the rest holds still
  const pieces: [number, number, number, number][] = [
    [g1, a1, 0, T_R1],
    [a1, c1, T_R1, T_R1 + T_C1],
    [g2, a2, TOKENS_EDIT - T_C2 - T_R2, TOKENS_EDIT - T_C2],
    [a2, c2, TOKENS_EDIT - T_C2, TOKENS_EDIT],
    [g3, a3, TOKENS_EDIT, TOKENS_EDIT + T_R3],
    [a3, ask, TOKENS_EDIT + T_R3, TOKENS_ASK],
    [g4, aS, TOKENS_ASK, TOKENS_ASK + T_R4],
    [aS, end, TOKENS_ASK + T_R4, TOKENS_ASK + T_R4 + ANSWER_TOKENS],
  ];
  const tokens = (t: number) => {
    let v = 0;
    for (const [from, to, a, b] of pieces) {
      if (t >= to) v = b;
      else if (t >= from) return a + ((b - a) * (t - from)) / (to - from);
    }
    return v;
  };
  const generating = (t: number) => pieces.some(([from, to]) => t >= from && t < to);
  return { S, r0, g1, a1, c1, d1, g2, a2, c2, d2, g3, a3, ask, press, ans, d3, g4, aS, end, landed, tokens, generating };
}

/* ── how a clocked page follows its newest part, and reserves its room ──────────────────────────────────────────── */

/** The list is shown at its end: what does not fit slides up, at every frame, exactly as the app's page follows the answer (Chat.tsx). */
export function Follow({ className = "", children }: { className?: string; children: ReactNode }) {
  const outer = useRef<HTMLDivElement>(null);
  const inner = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const o = outer.current;
    const i = inner.current;
    if (!o || !i) return;
    const over = Math.max(0, i.offsetHeight - o.clientHeight);
    i.style.transform = over > 0 ? `translate3d(0, ${-over}px, 0)` : "";
  });
  return (
    <div ref={outer} className={"c-follow " + className}>
      <div ref={inner}>{children}</div>
    </div>
  );
}

/** A scene that grows as it plays would move what is below it on the page; this keeps the room of its last frame from the first. */
export function Reserve({ children, last }: { children: ReactNode; last: ReactNode }) {
  return (
    <div className="c-reserve">
      <div className="c-reserve__live">{children}</div>
      <div className="c-reserve__last" aria-hidden="true">
        {last}
      </div>
    </div>
  );
}

/* ── orbs (components/orb.tsx): one slot; a changed form dissolves out while the new one resolves in, 450 ms ───────── */

export type OrbStep = { at: number; design: OrbDesign };
export function OrbSwap({ ms, steps, size = 20 }: { ms: number; steps: OrbStep[]; size?: number }) {
  let i = -1;
  for (let k = 0; k < steps.length; k++) if (ms >= steps[k].at) i = k;
  const box: CSSProperties = { width: size, height: size };
  if (i < 0) return <span className="c-orb" style={box} />;
  const cur = steps[i];
  const prev = i > 0 ? steps[i - 1] : null;
  const p = EASE(lin(ms, cur.at, DUR.orb));
  return (
    <span className="c-orb" style={box}>
      {prev && p < 1 && (
        <span className="c-orb__l" aria-hidden="true" style={{ opacity: 1 - p, transform: `scale(${1 + 0.16 * p})`, filter: `blur(${3 * p}px)` }}>
          <Orb ms={ms} live design={prev.design} size={size} />
        </span>
      )}
      <span className="c-orb__l" aria-hidden="true" style={p >= 1 ? undefined : { opacity: p, transform: `scale(${0.82 + 0.18 * p})`, filter: `blur(${3 * (1 - p)}px)` }}>
        <Orb ms={ms} live design={cur.design} size={size} />
      </span>
    </span>
  );
}

/** An orb beside a status line whose words shimmer (StatusLabel). */
export function StatusLabel({ ms, steps, label, size = 20 }: { ms: number; steps: OrbStep[]; label: string; size?: number }) {
  return (
    <span className="c-sl">
      <OrbSwap ms={ms} steps={steps} size={size} />
      <Shimmer>{label}</Shimmer>
    </span>
  );
}

/* ── the thought (Messages.tsx Thinking / RoundThought) ─────────────────────────────────────────────────────────── */

const clamp = (x: number, a: number, b: number) => Math.min(b, Math.max(a, x));

/**
 * The thought line. While it works it shimmers and counts; when it stops the label settles for 450 ms. The app counts time only while
 * a thought works: once it has stopped it says "Thoughts", with no figure, until the answer begins and `seconds` is known ("Thought for 1m 16.3s").
 */
export function Thought({ ms, start, stopAt, timedFrom, seconds, pattern, w }: { ms: number; start: number; stopAt: number; timedFrom?: number; seconds?: number; pattern: LatticePattern; w: Words }) {
  const timed = ms < stopAt || (timedFrom !== undefined && ms >= timedFrom);
  const known = timedFrom !== undefined && ms >= timedFrom;
  return (
    <div className={"c-th" + (timed ? "" : " is-untimed")}>
      <ThoughtLine ms={ms} start={start} stopAt={stopAt} seconds={known ? seconds : undefined} pattern={pattern} workText={w.thinking} doneText={known ? w.thoughtFor : w.thoughts} />
    </div>
  );
}
/** A thought of an earlier round: drawn once the next tool has started, already finished (the working one is another element, so nothing settles). */
export function DoneThought({ ms, pattern, w }: { ms: number; pattern: LatticePattern; w: Words }) {
  return (
    <div className="c-th is-untimed">
      <ThoughtLine ms={ms} start={0} stopAt={-DUR.thoughtSettle * 4} seconds={0} pattern={pattern} workText={w.thinking} doneText={w.thoughts} />
    </div>
  );
}

/* ── a tool call (AgentCall.tsx) ────────────────────────────────────────────────────────────────────────────────── */

export type ToolSpec = {
  name: string;
  arg: string;
  start: number; // the call begins to be written ("Writing")
  call: number; // the call is made: its arguments are known and it runs
  done: number; // its result arrived
  time: string;
  ask?: { at: number; press: number; answer: number; command: string };
};
const Chev = ({ open }: { open?: number }) => (
  <svg className="c-chev" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={open ? { transform: `rotate(${180 * open}deg)` } : undefined}>
    <path d="m6 9 6 6 6-6" />
  </svg>
);

export function ToolCard({ ms, tool, w }: { ms: number; tool: ToolSpec; w: Words }) {
  if (ms < tool.start) return null;
  const asking = !!tool.ask && ms >= tool.ask.at && ms < tool.ask.answer;
  const answered = !!tool.ask && ms >= tool.ask.answer;
  const done = ms >= tool.done;
  const writing = !asking && !done && ms < tool.call && !(tool.ask && ms >= tool.ask.at);
  const busy = !asking && !done;
  const orbAt = tool.ask && ms >= tool.ask.answer ? tool.ask.answer : tool.start; // an orb that comes back after the question is a new one
  const state = done ? w.stDone : asking ? w.stAsking : writing ? w.stWriting : w.stRunning;
  const known = ms >= tool.call;
  const ask = tool.ask;
  const dim = ask ? EASE_SOFT(lin(ms, ask.press, 150)) : 0;
  return (
    <div className="a-card c-tool" data-state={done ? "done" : asking ? "asking" : writing ? "writing" : "running"}>
      <div className="c-tool__h">
        {busy && <OrbSwap ms={ms} steps={[{ at: orbAt, design: "connecting" }]} size={20} />}
        <span className="c-tool__n">{busy ? <Shimmer>{tool.name}</Shimmer> : tool.name}</span>
        <span className="c-tool__a">{known && <Code code={tool.arg} lang="" />}</span>
        <span className={"c-tool__s" + (asking ? " is-asking" : "")}>{state}</span>
        {done && <span className="c-tool__t">{tool.time}</span>}
        <Chev open={ask ? (asking ? EASE_SOFT(lin(ms, ask.at, 200)) : answered ? 1 - EASE_SOFT(lin(ms, ask.answer, 200)) : 0) : 0} />
      </div>
      {ask && asking && (
        <div className="c-ask">
          <div className="c-ask__q a-ask__q">{w.askQ}</div>
          <pre className="a-ask__cmd">
            <Code code={ask.command} lang="bash" />
          </pre>
          <div className="c-ask__why">{w.why}</div>
          <div className="c-ask__b" style={{ opacity: 1 - 0.6 * dim }}>
            <span className="c-btn c-btn--ink">{w.allow}</span>
            <span className="c-btn">{w.deny}</span>
          </div>
        </div>
      )}
      {ask && answered && <div className="c-ask__done a-ask__q">{w.allowed}</div>}
    </div>
  );
}

/* ── the answer (Prose + the code block of lib/markdown.ts; the caret of `.streaming`) ──────────────────────────────── */

function Caret() {
  return <span className="a-stream" aria-hidden="true" />;
}

/** Inline text with names in it. A name that is still being written is plain text with its backtick, as it is in the app, until it closes. */
function Inline({ segs, chars, caret }: { segs: Seg[]; chars: number; caret: boolean }) {
  let left = chars;
  return (
    <p>
      {segs.map((s, k) => {
        const text = typeof s === "string" ? s : s.code;
        if (left <= 0) return null;
        if (typeof s === "string") {
          const n = Math.min(text.length, left);
          left -= n;
          return <span key={k}>{text.slice(0, n)}</span>;
        }
        const n = Math.min(text.length + 2, left);
        left -= n;
        if (n < text.length + 2) return <span key={k}>{("`" + text + "`").slice(0, n)}</span>;
        return (
          <code key={k} className="a-code mk">
            <Code code={text} lang="bash" />
          </code>
        );
      })}
      {caret && <Caret />}
    </p>
  );
}

function CodeBlock({ code, caret, w }: { code: string; caret: boolean; w: Words }) {
  return (
    <div className="a-codeb c-code">
      <div className="a-codeb__h c-code__h">
        <span>python</span>
        <span className="c-code__c">{w.copy}</span>
      </div>
      <pre className="a-codeb__p">
        <code className="hljs">
          <Code code={code} lang="python" />
        </code>
      </pre>
      {caret && (
        <div className="c-code__after">
          <Caret />
        </div>
      )}
    </div>
  );
}

/** The answer as far as `k` tokens of it have arrived. */
export function Reply({ k, w }: { k: number; w: Words }) {
  const s = shownAt(k);
  return (
    <div className="c-prose">
      {s.a > 0 && <Inline segs={REPLY_A} chars={s.a} caret={s.caret === "a"} />}
      {s.code !== null && <CodeBlock code={CODE_TEXT.slice(0, s.code)} caret={s.caret === "code"} w={w} />}
      {s.b > 0 && <Inline segs={REPLY_B} chars={s.b} caret={s.caret === "b"} />}
    </div>
  );
}

/* ── what is under the reply (Messages.tsx: the status, the live count, the line of figures) ───────────────────────── */

export type StatusItem = { at: number; design: OrbDesign; label: string };
export type StatusRun = { from: number; to: number; items: StatusItem[] };

/** What the agent is doing now, in words (lib/status.ts), as the runs of the status that the run goes through. Thinking has no status line: the thought has its own. */
export function statusRuns(r: Run, w: Words, asks = true): StatusRun[] {
  const call = (tool: string) => fill(w.writingCall, { tool });
  const run = (tool: string) => fill(w.runningTool, { tool });
  return [
    { from: r.a1, to: r.g2, items: [{ at: r.a1, design: "connecting", label: call("Read") }, { at: r.c1, design: "connecting", label: run("Read") }, { at: r.d1, design: "listening", label: w.readingResult }] },
    { from: r.a2, to: r.g3, items: [{ at: r.a2, design: "connecting", label: call("Edit") }, { at: r.c2, design: "connecting", label: run("Edit") }, { at: r.d2, design: "listening", label: w.readingResult }] },
    {
      from: r.a3,
      to: r.g4,
      items: [
        { at: r.a3, design: "connecting", label: call("Bash") },
        ...(asks ? [{ at: r.ask, design: "listening" as const, label: w.waitingYou }] : []),
        { at: asks ? r.ans : r.ask, design: "connecting", label: run("Bash") },
        { at: r.d3, design: "listening", label: w.readingResult },
      ],
    },
    { from: r.aS, to: r.end, items: [{ at: r.aS, design: "composing", label: w.answering }] },
  ];
}

/** The count of tokens written so far, beside the status: drawn every 250 ms (LiveCount), as plain text; the speed only while the model writes. */
function LiveCount({ ms, r, w }: { ms: number; r: Run; w: Words }) {
  if (ms < r.g1 || ms >= r.end) return null;
  const tick = r.g1 + Math.floor((ms - r.g1) / 250) * 250;
  const n = Math.floor(r.tokens(tick));
  if (n < 1) return null;
  return (
    <span className="c-live num" role="status">
      {fill(w.tokens, { n })}
      {r.generating(ms) ? " · " + fill(w.tokS, { n: RATE.toFixed(1) }) : ""}
    </span>
  );
}

/** The live status gives way to the figures (Handover): the status keeps its place, fading, blurring and rising away, while the figures arrive from a little below. */
function Handover({ ms, runs, end, meta }: { ms: number; runs: StatusRun[]; end: number; meta: ReactNode }) {
  const cur = runs.find((x) => ms >= x.from && ms < x.to + DUR.handoverOut) ?? null;
  let out: ReactNode = null;
  if (cur) {
    let item = cur.items[0];
    for (const it of cur.items) if (ms >= it.at) item = it;
    const leaving = ms >= cur.to;
    const p = EASE(lin(ms, cur.from, DUR.swap));
    const q = EASE(lin(ms, cur.to, DUR.handoverOut));
    const style: CSSProperties = leaving
      ? { opacity: 1 - q, filter: `blur(${3 * q}px)`, transform: `translate3d(0, ${-4 * q}px, 0)` }
      : p >= 1
        ? {}
        : { opacity: p, filter: `blur(${(1 - p) * 6}px)`, transform: `scale(${0.985 + 0.015 * p})` };
    out = (
      <span className="c-handover__out" style={style}>
        <StatusLabel ms={ms} steps={cur.items.map((it) => ({ at: it.at, design: it.design }))} label={item.label} />
      </span>
    );
  }
  const m = EASE(lin(ms, end + DUR.handoverDelay, DUR.handoverIn));
  const shown = ms >= end;
  return (
    <span className="c-handover">
      <span className="c-handover__in" style={{ opacity: shown ? m : 0, filter: shown && m < 1 ? `blur(${3 * (1 - m)}px)` : undefined, transform: shown && m < 1 ? `translate3d(0, ${4 * (1 - m)}px, 0)` : undefined }}>
        {shown && meta}
      </span>
      {out}
    </span>
  );
}

const CopyIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <rect x="9" y="9" width="11" height="11" rx="2.5" />
    <path d="M5 15V6.5A2.5 2.5 0 0 1 7.5 4H15" />
  </svg>
);

/** The row under a reply: the status (or, once the reply is written, its figures), the live count, the copy button. */
export function Footer({ ms, r, w, asks = true }: { ms: number; r: Run; w: Words; asks?: boolean }) {
  const meta = (
    <span className="a-stats c-stats num">
      {fill(w.tokens, { n: TOKENS_ALL })} · {fill(w.tokS, { n: RATE.toFixed(1) })} · {fill(w.toolCalls, { n: 3 })}
    </span>
  );
  return (
    <div className="c-foot">
      <Handover ms={ms} runs={statusRuns(r, w, asks)} end={r.end} meta={meta} />
      <LiveCount ms={ms} r={r} w={w} />
      {ms >= r.end && (
        <span className="c-copy">
          <CopyIcon />
        </span>
      )}
    </div>
  );
}

/* ── the prompt that was sent (Messages.tsx, sendfx.ts) ─────────────────────────────────────────────────────────── */

/** The prompt rises out of the composer to its place: 460 ms, from as far below as the composer is (24 to 260 px), faded in on the way. */
export function SendRise({ ms, at, children }: { ms: number; at: number; children: ReactNode }) {
  const el = useRef<HTMLDivElement>(null);
  const rise = useRef(120);
  useLayoutEffect(() => {
    const e = el.current;
    const from = e?.closest(".c-app")?.querySelector(".c-pb__input");
    if (!e || !from) return;
    const keep = e.style.transform;
    e.style.transform = "none";
    rise.current = clamp(from.getBoundingClientRect().top - e.getBoundingClientRect().top, 24, 260);
    e.style.transform = keep;
  }, []);
  const p = EASE(lin(ms, at, 460));
  return (
    <div
      ref={el}
      className="c-um"
      style={{ opacity: p < 0.35 ? p / 0.35 : 1, transform: p >= 1 ? undefined : `translate3d(0, ${rise.current * (1 - p)}px, 0) scale(${0.97 + 0.03 * p})` }}
    >
      {children}
    </div>
  );
}

/** The user's message: the bubble, who and when, and the speed the prompt was read at (a line that fades in when its words change). */
export function UserMessage({ ms, from, readDone, w, rise }: { ms: number; from: number; readDone: number; w: Words; rise: boolean }) {
  const [pre, post] = w.prefill.split("{rate}");
  const body = (
    <>
      <div className="c-um__b">{PROMPT}</div>
      <div className="c-um__m num">{fill(w.you, { time: TIME })}</div>
      <div className="c-um__m num" style={{ opacity: EASE(lin(ms, ms >= readDone ? readDone : from, DUR.fadeSwap)) }}>
        {ms < readDone ? (
          <span>{w.reading}</span>
        ) : (
          <>
            <span>{pre}</span>
            <Pop ms={ms} start={readDone} text="696" />
            <span>{post}</span>
          </>
        )}
      </div>
    </>
  );
  return rise ? <SendRise ms={ms} at={from}>{body}</SendRise> : <Enter ms={ms} start={from} kind="msg" className="c-um">{body}</Enter>;
}

/* ── the composer (PromptBar.tsx, prompt-bar.css) ───────────────────────────────────────────────────────────────── */

const ARROW = [12, 4.5, 18.5, 11, 14.25, 11, 14.25, 19.5, 9.75, 19.5, 9.75, 11, 5.5, 11];
const SQUARE = [12, 6, 18, 6, 18, 12, 18, 18, 6, 18, 6, 12, 6, 6];
const mix = (a: number, b: number, t: number) => a + (b - a) * t;
const pathAt = (t: number) => {
  let d = "";
  for (let i = 0; i < ARROW.length; i += 2) d += `${i ? "L" : "M"}${mix(ARROW[i], SQUARE[i], t).toFixed(2)} ${mix(ARROW[i + 1], SQUARE[i + 1], t).toFixed(2)}`;
  return d + "Z";
};
const inOut = (t: number) => (t < 0.5 ? 8 * t ** 4 : 1 - (-2 * t + 2) ** 4 / 2);

/** How far the send arrow has turned into the stop square at `ms` (0 arrow, 1 square), and which way it is turning: 240 ms, with a squash and a tilt. */
export function morph(ms: number, busyAt: number, idleAt: number): { t: number; dir: 1 | -1 } {
  if (ms < busyAt) return { t: 0, dir: 1 };
  if (ms < busyAt + 240) return { t: inOut(lin(ms, busyAt, 240)), dir: 1 };
  if (ms < idleAt) return { t: 1, dir: 1 };
  return { t: 1 - inOut(lin(ms, idleAt, 240)), dir: -1 };
}

function SendGlyph({ t, dir }: { t: number; dir: 1 | -1 }) {
  const goo = Math.sin(t * Math.PI);
  const sx = 1 - 0.12 * goo;
  return (
    <svg className="c-pb__glyph" viewBox="0 0 24 24" aria-hidden="true" fill="currentColor" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" style={goo > 0.001 ? { transform: `rotate(${dir * 8 * goo}deg) scale(${sx}, ${1 / sx})` } : undefined}>
      <path d={pathAt(t)} />
    </svg>
  );
}

const Ring = ({ pct }: { pct: number }) => {
  const r = 5.5;
  const c = 2 * Math.PI * r;
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true" className="c-pb__ring">
      <circle cx="7" cy="7" r={r} fill="none" strokeWidth="2" className="c-pb__ring-t" />
      <circle cx="7" cy="7" r={r} fill="none" strokeWidth="2" strokeLinecap="round" strokeDasharray={`${(c * pct) / 100} ${c}`} transform="rotate(-90 7 7)" className="c-pb__ring-f" />
    </svg>
  );
};

// the sparks that rise behind the field at the top thinking level while it is focused and nothing is being written (PromptBar.tsx draws them on a canvas)
const SPARKS = Array.from({ length: 16 }, (_, i) => ({ x: (i * 37 + 11) % 100, d: 2.6 + ((i * 7) % 5) * 0.55, w: -(((i * 5) % 9) * 0.45), s: ((i * 13) % 11) - 5, r: 0.9 + ((i * 3) % 4) * 0.35 }));

const Plus = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
    <path d="M12 5v14M5 12h14" />
  </svg>
);
const Shield = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" aria-hidden="true">
    <path d="M12 3 5 6v5.5c0 4.4 3 7.6 7 9.5 4-1.9 7-5.1 7-9.5V6z" />
  </svg>
);
const Sparkle = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" aria-hidden="true">
    <path d="m10 4 1.8 5.2L17 11l-5.2 1.8L10 18l-1.8-5.2L3 11l5.2-1.8zM19 14l.8 2.2L22 17l-2.2.8L19 20l-.8-2.2L16 17l2.2-.8z" />
  </svg>
);

export type BarProps = {
  w: Words;
  text: string; // what is in the field
  caret?: boolean; // the caret of the field blinks
  busy?: boolean; // an answer is being written: the placeholder says so, the arrow is a square
  send: { t: number; dir: 1 | -1 };
  armed: boolean;
  pct: number;
  mode?: { label: string; color?: string; on?: number };
  sparks?: boolean;
  focused?: boolean;
  children?: ReactNode; // a menu above the bar
};

export function PromptBar({ w, text, caret = true, busy = false, send, armed, pct, mode, sparks = false, focused = true, children }: BarProps) {
  return (
    <div className="c-pb">
      {children}
      <div className="c-pb__field" data-focus={focused ? "" : undefined} data-max="">
        {sparks && (
          <span className="c-pb__sparks" aria-hidden="true">
            {SPARKS.map((s, i) => (
              <i key={i} style={{ left: `${s.x}%`, width: s.r * 2, height: s.r * 2, animationDuration: `${s.d + 1.6}s`, animationDelay: `${s.w}s`, ["--sway" as string]: `${s.s}px` }} />
            ))}
          </span>
        )}
        <div className="c-pb__input">
          {text ? (
            <>
              {text}
              {caret && focused && <span className="c-caret" aria-hidden="true" />}
            </>
          ) : (
            <>
              {caret && focused && <span className="c-caret" aria-hidden="true" />}
              <span className="c-pb__ph">{busy ? w.busy : w.message}</span>
            </>
          )}
        </div>
        <div className="c-pb__bar">
          <span className="c-pb__tool">
            <Plus />
          </span>
          <span className="c-pb__pick" style={{ color: mode?.color, background: mode?.on ? `color-mix(in srgb, var(--a-ink) ${8 * mode.on}%, transparent)` : undefined }}>
            <Shield />
            <span>{mode?.label ?? w.ask}</span>
          </span>
          <span className="c-pb__pick" data-max="">
            <Sparkle />
            <span>{w.xhigh}</span>
          </span>
          <span className="c-pb__spacer" />
          <span className="c-pb__pick">
            <Ring pct={pct} />
            <span className="num">{Math.round(pct)}%</span>
          </span>
          <span className="c-pb__send" data-armed={armed ? "" : undefined}>
            <SendGlyph t={send.t} dir={send.dir} />
          </span>
        </div>
      </div>
    </div>
  );
}

/* ── the list at the side (Sidebar.tsx + BranchedMenu.tsx) ──────────────────────────────────────────────────────── */

const clip = (s: string, n: number) => (s.length > n ? s.slice(0, n).trimEnd() + "…" : s); // sessions.ts: a title is the first prompt, cut at 40

/** Projects and Recents. The chat is made when the prompt is sent: it takes its place in the project at once, the line is drawn to it, the marker comes to the
 *  project, and a dot beside it pulses while it answers (amber while it waits for you). */
export function Side({ ms, w, chatAt, askFrom, askTo, endAt }: { ms: number; w: Words; chatAt: number; askFrom: number; askTo: number; endAt: number }) {
  const made = ms >= chatAt;
  const dot = made && ms < endAt ? (ms >= askFrom && ms < askTo ? "asking" : "answering") : null;
  return (
    <aside className="c-side">
      <div className="c-side__new">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M12 4H8a4 4 0 0 0-4 4v8a4 4 0 0 0 4 4h8a4 4 0 0 0 4-4v-4M17 3v6M14 6h6" />
        </svg>
        <span>{w.newChat}</span>
      </div>
      <section>
        <div className="c-side__h">
          <span>{w.projects}</span>
          <span className="c-side__plus">+</span>
        </div>
        <nav className="c-bm">
          <span className="c-bm__mark" style={{ opacity: made ? EASE(lin(ms, chatAt, 150)) : 0 }} aria-hidden="true" />
          <div className="c-bm__headrow">
            <span className="c-bm__head a-side__p">{PROJECT}</span>
            <span className="c-bm__n num">{made ? 1 : 0}</span>
          </div>
          {made ? (
            <div className="c-bm__tree">
              <svg width="40" height="48" className="c-bm__lines" aria-hidden="true">
                <path className="c-bm__base" d="M 14 0 V 14 A 10 10 0 0 0 24 24 H 32" />
                <path className="c-bm__reach" d="M 14 0 V 14 A 10 10 0 0 0 24 24 H 32" />
              </svg>
              <div className="c-bm__row">
                <span className="c-bm__item">{clip(PROMPT.replace(/\s+/g, " "), 40)}</span>
                {dot && <span className="c-run-dot" data-running={dot} />}
              </div>
            </div>
          ) : (
            <p className="c-bm__empty">{w.empty}</p>
          )}
        </nav>
      </section>
      <section>
        <nav className="c-bm">
          <div className="c-bm__headrow">
            <span className="c-bm__head">{w.recents}</span>
          </div>
          <p className="c-bm__empty">{made ? w.empty : w.noChats}</p>
        </nav>
      </section>
    </aside>
  );
}

/* ── the empty chat (Chat.tsx): the orb that says what is happening, the question, the project, the line under it ──── */

export function Hero({ ms, w, steps, size = 160 }: { ms: number; w: Words; steps: OrbStep[]; size?: number }) {
  return (
    <div className="c-hero">
      <OrbSwap ms={ms} steps={steps} size={size} />
      <h1 className="c-hero__h">{w.hello}</h1>
      <span className="c-hero__pick">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M3 7.5A2.5 2.5 0 0 1 5.5 5H9l2 2.5h7.5A2.5 2.5 0 0 1 21 10v7.5a2.5 2.5 0 0 1-2.5 2.5h-13A2.5 2.5 0 0 1 3 17.5z" />
        </svg>
        <span>{PROJECT}</span>
        <Chev />
      </span>
      <p className="c-hero__lede">{fill(w.lede, { name: MODEL })}</p>
    </div>
  );
}

/** The hero that closes when the first message is added (Collapse, `instant`: it is open at the start). */
export function HeroCollapse({ ms, closeAt, children }: { ms: number; closeAt: number; children: ReactNode }) {
  return (
    <Collapse ms={ms} open={-DUR.collapse * 4} closeAt={closeAt} className="c-heroc">
      {children}
    </Collapse>
  );
}
