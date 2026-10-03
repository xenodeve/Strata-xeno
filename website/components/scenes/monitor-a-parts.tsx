"use client";

import { useId, type CSSProperties, type ReactNode } from "react";
import { clamp01, lin } from "./Scene";
import { AppNav, Orb, Scroll } from "./kit";
import { Collapse, Enter } from "./parts";
import { DUR, EASE, EASE_DIGIT } from "./motion";

/**
 * The Dashboard and Live pages of the app (serve/ui/src/pages/Dashboard.tsx, Live.tsx), as scenes. They are written from what those pages
 * do when they are opened, not from a screenshot:
 *
 *  - the page sharpens in from a blur (`.page-in`, 260 ms) and its blocks rise one after another, 50 ms apart (`.stagger`, `main > * > *`,
 *    rise 420 ms); a block that is added to the page (`Reveal`) opens through grid rows (`.collapse-grid`, 320 ms);
 *  - the headline is a `Swap`: it sharpens in (`.swap-in`, 280 ms) when it first shows or when the state changes;
 *  - the orb that stands for the server is the one `serverDesign` (lib/orbs.ts) picks: "searching" while idle, "solving" while the model thinks;
 *    it comes in over 450 ms (`.orb-in`) and, when the design changes, the old one goes out (`.orb-out`);
 *  - what is on the page when it opens is not animated: a figure only moves when it changes (Pop for a count, 320 ms at most, shorter when
 *    the readings come fast: `pace()` in lib/pop.ts); a bar is only set; a chart is drawn at once, and again whenever its data changes
 *    (components/Chart.tsx calls uPlot's setData: nothing draws itself over time);
 *  - the page polls the server every 500 ms while a request runs and every 1000 ms when idle (lib/metrics.ts).
 */

/* ── the app's own words ─────────────────────────────────────────────── */

type Lang = "en" | "th";
type Words = {
  nav: string[];
  navDash: string;
  navLive: string;
  ready: string;
  model: string;
  last: string;
  speed: string;
  open: string;
  decNow: string;
  idle: string;
  lastReq: string;
  prefillLast: string;
  decLast: string;
  mean: string;
  req: string;
  since: string;
  read: string;
  cached: string;
  written: string;
  avg: string;
  hw: string;
  pcie: string;
  pcie2: string;
  modelOn: string;
  experts: string;
  e: [string, string, string, string];
  over: string;
  modelH: string;
  server: string;
  tools: string;
  thinking: (n: string) => string;
  windowed: string;
  decode: string;
  prefill: string;
  now: string;
  meanReq: string;
  elapsed: string;
  lastMin: string;
  cards: string;
  gpu: (n: number) => string;
  recent: string;
  all: string;
  row: (read: string, cached: string | null, out: string) => string;
};

/** Every string is the app's own: the English of serve/ui/src, the Thai of serve/ui/src/i18n. */
const W: Record<Lang, Words> = {
  en: {
    nav: ["Chat", "Dashboard", "Live", "Requests", "Hardware", "Settings", "About"],
    navDash: "Dashboard",
    navLive: "Live",
    ready: "Ready",
    model: "Swift Flash Next · IQ2_XS · 3.7 bits per weight · 256K context",
    last: "Last request · 64.0 tok/s decode · 31 tokens read, 4,742 cached · 3.3 s",
    speed: "Speed",
    open: "Open",
    decNow: "Decode now",
    idle: "idle",
    lastReq: "Last request",
    prefillLast: "Prefill, last",
    decLast: "Decode of the last 4 requests",
    mean: "mean",
    req: "Requests",
    since: "Since start",
    read: "Read",
    cached: "Cached",
    written: "Written",
    avg: "Avg decode",
    hw: "Hardware",
    pcie: "Gen 4 ×16",
    pcie2: "Gen 1 ×4",
    modelOn: "Model on WD_BLACK SN850X 1000GB · NVMe SSD",
    experts: "Where the last request's experts ran",
    e: ["Primary GPU", "Secondary GPU", "RAM over PCIe", "RAM on the CPU"],
    over: "Over 2.66 s of decode",
    modelH: "Model",
    server: "This server",
    tools: "15 tools",
    thinking: (n) => `Thinking · ${n} tokens`,
    windowed: "windowed over 2 s",
    decode: "Decode",
    prefill: "Prefill",
    now: "now",
    meanReq: "mean of this request",
    elapsed: "Elapsed",
    lastMin: "Decode, last minute",
    cards: "Cards",
    gpu: (n) => `GPU ${n}`,
    recent: "Recent requests",
    all: "All",
    row: (r, c, o) => `${r} read${c ? ` · ${c} cached` : ""} → ${o}`,
  },
  th: {
    nav: ["แชท", "ภาพรวม", "สด", "คำขอ", "ฮาร์ดแวร์", "ตั้งค่า", "ประมาณ"],
    navDash: "ภาพรวม",
    navLive: "สด",
    ready: "พร้อม",
    model: "Swift Flash Next · IQ2_XS · 3.7 บิตต่อน้ำหนัก · 256K context",
    last: "คำขอล่าสุด · decode 64.0 tok/s · อ่าน 31 token, cache 4,742 · 3.3 s",
    speed: "ความเร็ว",
    open: "เปิด",
    decNow: "decode ตอนนี้",
    idle: "ว่าง",
    lastReq: "คำขอล่าสุด",
    prefillLast: "prefill ล่าสุด",
    decLast: "decode ของ 4 คำขอล่าสุด",
    mean: "ค่าเฉลี่ย",
    req: "คำขอ",
    since: "ตั้งแต่เริ่ม",
    read: "อ่าน",
    cached: "อยู่ใน cache",
    written: "เขียน",
    avg: "decode เฉลี่ย",
    hw: "ฮาร์ดแวร์",
    pcie: "Gen 4 ×16",
    pcie2: "Gen 1 ×4",
    modelOn: "โมเดลอยู่บน WD_BLACK SN850X 1000GB · NVMe SSD",
    experts: "ส่วน expert ของคำขอล่าสุดทำงานที่ไหน",
    e: ["GPU หลัก", "GPU รอง", "RAM ผ่าน PCIe", "RAM บน CPU"],
    over: "decode ใช้เวลา 2.66 s",
    modelH: "โมเดล",
    server: "เซิร์ฟเวอร์นี้",
    tools: "15 เครื่องมือ",
    thinking: (n) => `กำลังคิด · ${n} token`,
    windowed: "เฉลี่ยในช่วง 2 s",
    decode: "Decode",
    prefill: "Prefill",
    now: "ตอนนี้",
    meanReq: "ค่าเฉลี่ยของคำขอนี้",
    elapsed: "เวลาที่ผ่านไป",
    lastMin: "decode, 1 นาทีล่าสุด",
    cards: "การ์ด",
    gpu: (n) => `GPU ${n}`,
    recent: "คำขอที่ผ่านมา",
    all: "ทั้งหมด",
    row: (r, c, o) => `อ่าน ${r}${c ? ` · cache ${c}` : ""} → ${o}`,
  },
};
// "of Gen 4 ×16" in the app (th: "จาก Gen 4 ×16"), after the GPU's own "Gen 1 ×4"
const OF: Record<Lang, string> = { en: "of Gen 4 ×16", th: "จาก Gen 4 ×16" };

/* ── what one real run showed ────────────────────────────────────────── */

const PROMPT = 'Read notes.py, then add a --version flag that prints "notes-app 0.1.0" and exits. Keep the change small.';
const REQUESTS = [
  { when: "Oct 3, 03:17:57 PM", read: "31", cached: "4,742", out: "171", prefill: "48", dec: "64.0" },
  { when: "Oct 3, 03:16:51 PM", read: "229", cached: "4,439", out: "74", prefill: "68", dec: "74.2" },
  { when: "Oct 3, 03:16:43 PM", read: "376", cached: "3,727", out: "336", prefill: "172", dec: "61.1" },
  { when: "Oct 3, 03:16:37 PM", read: "3,664", cached: null, out: "61", prefill: "696", dec: "65.1" },
];
/** Decode speed of the last four requests, oldest first (`recent` of the Dashboard: the newest twelve, reversed). */
const DECODE = [65.1, 61.1, 74.2, 64.0];

/* ── the orb in its slot ─────────────────────────────────────────────── */

/**
 * An orb in a fixed slot (components/orb.tsx `Orb`): it comes in over 450 ms from a little smaller and out of a blur (`.orb-in`); when the
 * design changes the old one swells a little and blurs away (`.orb-out`) while the new one comes in, 450 ms, the old one removed at 480.
 * `phases`: the design from each moment on (lib/orbs.ts says which design stands for which state). The orbs of the app always move.
 */
export function OrbSlot({ ms, size, phases }: { ms: number; size: number; phases: { at: number; design: string }[] }) {
  let i = 0;
  for (let k = 0; k < phases.length; k++) if (ms >= phases[k].at) i = k;
  const cur = phases[i];
  const p = EASE(lin(ms, cur.at, DUR.orb));
  const prev = i > 0 ? phases[i - 1] : null;
  const out = prev && ms < cur.at + 480 ? EASE(lin(ms, cur.at, DUR.orb)) : null;
  return (
    <span className="ma-orb" style={{ width: size, height: size }}>
      {prev && out !== null && (
        <span className="ma-orb__l" style={{ opacity: 1 - out, transform: `scale(${1 + 0.16 * out})`, filter: out > 0 ? `blur(${out * 3}px)` : undefined }}>
          <Orb ms={ms} live size={size} design={prev.design} />
        </span>
      )}
      <span className="ma-orb__l" style={{ opacity: p, transform: p >= 1 ? undefined : `scale(${0.82 + 0.18 * p})`, filter: p >= 1 ? undefined : `blur(${(1 - p) * 3}px)` }}>
        <Orb ms={ms} live size={size} design={cur.design} />
      </span>
    </span>
  );
}

/* ── figures ─────────────────────────────────────────────────────────── */

type Step = { at: number; text: string };
const NUM = /\d(?:[\d.,]*\d)?/g;
/** The text cut into plain runs and number runs ("1,240.5" is one number), as lib/pop.ts popRuns does. */
function cut(text: string): { num: boolean; text: string }[] {
  const out: { num: boolean; text: string }[] = [];
  let at = 0;
  for (const m of text.matchAll(NUM)) {
    if (m.index! > at) out.push({ num: false, text: text.slice(at, m.index) });
    out.push({ num: true, text: m[0] });
    at = m.index! + m[0].length;
  }
  if (at < text.length) out.push({ num: false, text: text.slice(at) });
  return out;
}
const tier = (fromEnd: number) => (fromEnd === 0 ? 2 : fromEnd === 1 ? 1 : 0);

/**
 * Text whose numbers pop in when a reading changes (components/pop.tsx). `steps` are the readings and when each arrives. Only the characters
 * that changed come in again, one after the other, from 6 px below and out of a 2 px blur, with a little overshoot; the last two follow a step
 * behind. The first reading is just there. How long it takes depends on how fast the readings come (`pace` in lib/pop.ts): never more than
 * 320 ms, 40 % of the time since the last change, never under 90 ms, and not at all when it changed less than 150 ms ago; the step between
 * digits is 12 % of that.
 */
export function PopText({ ms, steps }: { ms: number; steps: Step[] }) {
  let i = 0;
  for (let k = 0; k < steps.length; k++) if (ms >= steps[k].at) i = k;
  const cur = steps[i];
  const prev = i > 0 ? steps[i - 1] : null;
  const since = prev ? cur.at - prev.at : 0;
  const animate = prev !== null && since >= 150;
  const dur = animate ? Math.round(Math.min(DUR.digit, Math.max(90, since * 0.4))) : 0;
  const gap = Math.round(dur * 0.12);
  const now = cut(cur.text);
  const was = prev ? cut(prev.text) : null;
  return (
    <>
      {now.map((r, ri) => {
        if (!r.num) return <span key={ri}>{r.text}</span>;
        const chars = [...r.text];
        const w = was?.[ri];
        const before = w?.num ? [...w.text] : [];
        return (
          <span key={ri} className="ma-pop">
            {chars.map((c, j) => {
              const fromEnd = chars.length - 1 - j;
              const same = before.length > fromEnd && before[before.length - 1 - fromEnd] === c;
              if (!animate || same) return <span key={j}>{c}</span>;
              const raw = EASE_DIGIT(lin(ms, cur.at + tier(fromEnd) * gap, dur));
              const style: CSSProperties = { opacity: clamp01(raw), transform: `translate3d(0, ${(1 - raw) * 6}px, 0)`, filter: raw >= 1 ? undefined : `blur(${(1 - clamp01(raw)) * 2}px)` };
              return (
                <span key={j} className="ma-pop__d" style={style}>
                  {c}
                </span>
              );
            })}
          </span>
        );
      })}
    </>
  );
}

/* ── the chart (components/Chart.tsx, uPlot) ─────────────────────────── */

/** Where the gridlines are: the first of 1 · 2 · 2.5 · 5 × 10ⁿ that leaves 30 px between two of them. */
function gridAt(top: number, plot: number): number[] {
  const need = (30 * top) / plot;
  const mag = Math.pow(10, Math.floor(Math.log10(need)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((v) => v >= need) ?? need;
  const out: number[] = [];
  for (let v = 0; v <= top + 1e-9; v += step) out.push(v);
  return out;
}

/**
 * The line chart of the app: ink on the page, a wash under the line that fades to nothing at the baseline (16 % → 0), gridlines in the
 * page's line colour, an axis of values on the left; the scale starts at 0 and ends 12 % above the highest value. It is drawn whole, as it
 * is: the app hands the chart its data and uPlot draws that, so a series that grows is a series that is drawn again, longer. The x axis is
 * the index of the reading, and the first and last readings sit at the two edges. `labels` false: no values on the axis (the reading
 * scale of the run is not known).
 */
export function PlotChart({ data, height, labels = true, className = "" }: { data: number[]; height: number; labels?: boolean; className?: string }) {
  const id = "ma" + useId().replace(/[^a-zA-Z0-9]/g, "");
  const hi = Math.max(...data);
  const top = Math.max(1, hi * 1.12);
  const plot = height - 8 - 6;
  const grid = gridAt(top, plot);
  const n = data.length;
  const X = (k: number) => (n < 2 ? 0 : (k / (n - 1)) * 1000);
  const Y = (v: number) => 1000 - (v / top) * 1000;
  const line = data.map((v, k) => `${k ? "L" : "M"}${X(k).toFixed(1)} ${Y(v).toFixed(1)}`).join(" ");
  return (
    <div className={"a-spark ma-chart " + className} style={{ height }}>
      {labels &&
        grid.map((v) => (
          <span key={v} className="ma-chart__y" style={{ top: 8 + plot * (1 - v / top) }}>
            {+v.toFixed(1)}
          </span>
        ))}
      <div className={"ma-chart__plot" + (labels ? "" : " is-bare")}>
        <svg viewBox="0 0 1000 1000" preserveAspectRatio="none" aria-hidden="true">
          <defs>
            <linearGradient id={id} gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="0" y2="1000">
              <stop offset="0" style={{ stopColor: "var(--a-ink)", stopOpacity: 0.16 }} />
              <stop offset="1" style={{ stopColor: "var(--a-ink)", stopOpacity: 0 }} />
            </linearGradient>
          </defs>
          {grid.map((v) => (
            <line key={v} x1="0" x2="1000" y1={Y(v)} y2={Y(v)} className="ma-chart__g" />
          ))}
          <path d={`${line} L${X(n - 1)} 1000 L${X(0)} 1000 Z`} fill={`url(#${id})`} />
          <path d={line} className="ma-chart__l" />
        </svg>
      </div>
    </div>
  );
}

/* ── the blocks of a page ────────────────────────────────────────────── */

/** `Block` of the Dashboard (a heading that opens its page) and `Section` of Live (a heading and a quiet word at its right). */
function Block({ title, aside, live = false, children }: { title: string; aside?: ReactNode; live?: boolean; children: ReactNode }) {
  return (
    <section className={"ma-block" + (live ? " is-live" : "")}>
      <div className="ma-block__h">
        <b>{title}</b>
        {aside !== undefined && <span>{aside}</span>}
      </div>
      <div className="ma-block__b">{children}</div>
    </section>
  );
}
const OpenLink = ({ w }: { w: Words }) => (
  <>
    {w.open} <span aria-hidden="true">→</span>
  </>
);

const Unit = ({ children }: { children: ReactNode }) => <span className="ma-unit"> {children}</span>;
const Fact = ({ k, children }: { k: string; children: ReactNode }) => (
  <div>
    <dt>{k}</dt>
    <dd>{children}</dd>
  </div>
);

function Meter({ f }: { f: number }) {
  return (
    <div className="ma-bar">
      <i style={{ width: `${clamp01(f) * 100}%` }} />
    </div>
  );
}

/** `Chart` of the Dashboard: the speed of the last requests. */
function SpeedBlock({ w, chart }: { w: Words; chart: number }) {
  const mean = DECODE.reduce((a, b) => a + b, 0) / DECODE.length;
  return (
    <Block title={w.speed} aside={<OpenLink w={w} />}>
      <dl className="ma-facts">
        <Fact k={w.decNow}>
          <span className="ma-dim">{w.idle}</span>
        </Fact>
        <Fact k={w.lastReq}>
          64.0<Unit>tok/s</Unit>
        </Fact>
        <Fact k={w.prefillLast}>
          48<Unit>tok/s</Unit>
        </Fact>
      </dl>
      <div className="ma-chart-h">
        <span>{w.decLast}</span>
        <span>
          {w.mean} {mean.toFixed(1)}
        </span>
      </div>
      <PlotChart data={DECODE} height={chart} />
    </Block>
  );
}

/** `RequestList` of the Requests page: every row comes in at once from 6 px above (`.row-in`, 360 ms), as a list does when it is added to. */
function RequestList({ ms, w, rows }: { ms: number; w: Words; rows: number }) {
  return (
    <ul className="ma-reqs">
      {REQUESTS.slice(0, rows).map((r) => (
        <li key={r.when}>
          <Enter ms={ms} start={0} kind="row" className="ma-req">
            <div className="ma-req__t">
              <div className="ma-req__p">{PROMPT}</div>
              <div className="ma-req__m">
                {r.when} · OpenAI · Chrome · {w.tools}
              </div>
            </div>
            <div className="ma-req__n">
              <span>{w.row(r.read, r.cached, r.out)}</span>
              <span>
                {r.prefill}
                <Unit>tok/s</Unit>
              </span>
              <span>
                {r.dec}
                <Unit>tok/s</Unit>
              </span>
            </div>
          </Enter>
        </li>
      ))}
    </ul>
  );
}

const Chevron = () => (
  <svg className="ma-fold__c" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M6 9l6 6 6-6" />
  </svg>
);
/** `Disclosure` (components/motion.tsx), closed: its title, a hint after it and a chevron. */
const Fold = ({ title, hint }: { title: string; hint: string }) => (
  <section className="ma-fold">
    <div className="ma-fold__b">
      <span>
        <b>{title}</b>
        <em>{hint}</em>
      </span>
      <Chevron />
    </div>
  </section>
);

/** One card of the Hardware block: its orb (`gpuDesign`: "working" from 5 % busy, "searching" at rest), name, load, bar and link. */
function Gpu({ ms, design, name, arch, load, f, mem, link }: { ms: number; design: string; name: string; arch: string; load: string; f: number; mem: string; link: ReactNode }) {
  return (
    <div className="m-gpu">
      <div className="ma-gpu__l">
        <span className="ma-gpu__n">
          <OrbSlot ms={ms} size={20} phases={[{ at: 0, design }]} />
          {name} <i>{arch}</i>
        </span>
        <span className="ma-gpu__v">{load}</span>
      </div>
      <Meter f={f} />
      <div className="ma-gpu__s">
        {mem} · {link}
      </div>
    </div>
  );
}

function HardwareBlock({ ms, w, lang }: { ms: number; w: Words; lang: Lang }) {
  return (
    <Block title={w.hw} aside={<OpenLink w={w} />}>
      <Gpu ms={ms} design="working" name="RTX 4070 SUPER" arch="sm89" load="12% · 56°C" f={10.2 / 12} mem="10.2 / 12.0 GB" link={w.pcie} />
      <Gpu
        ms={ms}
        design="searching"
        name="RTX 5060 Ti"
        arch="sm120"
        load="0% · 39°C"
        f={14.7 / 15.9}
        mem="14.7 / 15.9 GB"
        link={
          <>
            {w.pcie2}
            <span className="ma-unit"> {OF[lang]}</span>
          </>
        }
      />
      <div className="ma-duo">
        <div>
          <div className="ma-duo__h">
            <span>CPU · AVX-VNNI</span>
            <b>37%</b>
          </div>
          <Meter f={0.37} />
        </div>
        <div>
          <div className="ma-duo__h">
            <span>RAM</span>
            <b>41.2 / 47.7 GB</b>
          </div>
          <Meter f={41.2 / 47.7} />
        </div>
      </div>
      <p className="ma-dim ma-model-on">{w.modelOn}</p>
    </Block>
  );
}

function ExpertsBlock({ w }: { w: Words }) {
  const share = [52, 26, 0, 23];
  const sum = 52 + 26 + 0 + 23;
  return (
    <Block title={w.experts} aside={<OpenLink w={w} />}>
      <div className="m-stack">
        {share.map((v, k) => (
          <i key={k} style={{ width: `${(v / sum) * 100}%`, opacity: 1 - k * 0.24 }} />
        ))}
      </div>
      <dl className="ma-legend">
        {w.e.map((n, k) => (
          <div key={n}>
            <span className="ma-legend__d" style={{ opacity: 1 - k * 0.24 }} />
            <dt>{n}</dt>
            <dd>{share[k]}%</dd>
          </div>
        ))}
      </dl>
      <p className="ma-dim ma-over">{w.over}</p>
    </Block>
  );
}

function RequestsBlock({ ms, w }: { ms: number; w: Words }) {
  return (
    <Block title={w.req} aside={<OpenLink w={w} />}>
      <dl className="ma-facts">
        <Fact k={w.since}>4</Fact>
        <Fact k={w.read}>4.3k</Fact>
        <Fact k={w.cached}>13k</Fact>
        <Fact k={w.written}>642</Fact>
        <Fact k={w.avg}>
          63.1<Unit>tok/s</Unit>
        </Fact>
      </dl>
      <div className="ma-reqs-gap">
        <RequestList ms={ms} w={w} rows={3} />
      </div>
    </Block>
  );
}

/* ── the Dashboard ───────────────────────────────────────────────────── */

/** The head of the Dashboard: the orb (64 px, "searching" while idle), the state as a sentence at display size (a `Swap`), the model, the last request. */
function DashHead({ ms, w }: { ms: number; w: Words }) {
  return (
    <Enter ms={ms} start={0} kind="rise" className="ma-head">
      <OrbSlot ms={ms} size={64} phases={[{ at: 0, design: "searching" }]} />
      <div className="ma-head__t">
        <div className="m-head__h ma-h1 is-dash">
          <Enter ms={ms} start={0} kind="swap" style={{ display: "inline-block" }}>
            {w.ready}
          </Enter>
        </div>
        <p className="ma-lede">{w.model}</p>
        <p className="ma-last">{w.last}</p>
      </div>
    </Enter>
  );
}

/** The whole Dashboard: the page sharpens in; the head and the grid rise (0 and 50 ms); in each column the blocks rise 50 ms apart. The
 *  block of where the experts ran is a `Reveal`: it opens through grid rows when it is added. What is on the page is not animated. */
export function DashboardBody({ ms, lang }: { ms: number; lang: Lang }) {
  const w = W[lang];
  return (
    <div className="a-app">
      <AppNav active={w.navDash} items={w.nav} />
      <Scroll ms={ms} from={900} dur={1400}>
        <Enter ms={ms} start={0} kind="page" className="m-page">
          <DashHead ms={ms} w={w} />
          <Enter ms={ms} start={0} kind="rise" delay={50} className="ma-cols">
            <div className="ma-blocks">
              <Enter ms={ms} start={0} kind="rise">
                <SpeedBlock w={w} chart={120} />
              </Enter>
              <Enter ms={ms} start={0} kind="rise" delay={50}>
                <RequestsBlock ms={ms} w={w} />
              </Enter>
            </div>
            <div className="ma-blocks">
              <Enter ms={ms} start={0} kind="rise">
                <HardwareBlock ms={ms} w={w} lang={lang} />
              </Enter>
              <Enter ms={ms} start={0} kind="rise" delay={50}>
                <Collapse ms={ms} open={16}>
                  <ExpertsBlock w={w} />
                </Collapse>
              </Enter>
              <Enter ms={ms} start={0} kind="rise" delay={100}>
                <Fold title={w.modelH} hint="IQ2_XS · 3.69 bpw · 79.6 GB" />
                <Fold title={w.server} hint="0.1.37 · 256K context · 1/1 MCP" />
              </Enter>
            </div>
          </Enter>
        </Enter>
      </Scroll>
    </div>
  );
}

/** The Dashboard at rest, as far as its Speed block: the head and the last four requests. */
export function ReadyBody({ ms, lang }: { ms: number; lang: Lang }) {
  const w = W[lang];
  return (
    <Enter ms={ms} start={0} kind="page" className="m-page">
      <DashHead ms={ms} w={w} />
      <Enter ms={ms} start={0} kind="rise" delay={50}>
        <SpeedBlock w={w} chart={132} />
      </Enter>
    </Enter>
  );
}

/* ── Live ────────────────────────────────────────────────────────────── */

/** The run the Live scenes show: the model has been thinking for 3.3 s when the page opens and the page reads the server every 500 ms
 *  (lib/metrics.ts, while a request runs) until 9.3 s, where the count of tokens is 452. The count is the one figure of the run that
 *  moves between its two ends, and it moves with the time (452 over 9.3 s). */
const RUN = { from: 3.3, to: 9.3, tokens: 452, poll: 500 };
const readings = (): { at: number; e: number; n: number }[] => {
  const out: { at: number; e: number; n: number }[] = [];
  const count = Math.round(((RUN.to - RUN.from) * 1000) / RUN.poll);
  for (let k = 0; k <= count; k++) {
    const e = RUN.from + (k * RUN.poll) / 1000;
    out.push({ at: k * RUN.poll, e, n: Math.round((RUN.tokens * e) / RUN.to) });
  }
  return out;
};
const POLLS = readings();
export const LIVE_END = POLLS[POLLS.length - 1].at;

/** The decode of the last minute: flat, a first run, flat, and the run now under way, whose points reach the chart as the page reads the server. */
const MINUTE = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9, 0.62, 0.55, 0.66, 0.72, 0.67, 0.64, 0.58, 0.7, 0.66, 0, 0, 0, 0, 0, 0, 0, 0, 0.75, 0.66, 0.7, 0.79, 0.81, 0.77];
// the page opens 3.3 s into the request: three points of the run now under way are on the chart; the other three come in one every 2 s
const MINUTE_AT_OPEN = MINUTE.length - 3;
const MINUTE_EVERY = 2000;

/** The Live page while the model thinks (its orb "solving"): `full` adds the cards and the recent requests. */
export function LiveBody({ ms, lang, full }: { ms: number; lang: Lang; full: boolean }) {
  const w = W[lang];
  const num = (n: number) => n.toLocaleString("en-US");
  const tokens: Step[] = POLLS.map((p) => ({ at: p.at, text: w.thinking(num(p.n)) }));
  const elapsed: Step[] = POLLS.map((p) => ({ at: p.at, text: p.e.toFixed(1) }));
  const series = MINUTE.slice(0, Math.min(MINUTE.length, MINUTE_AT_OPEN + Math.floor(ms / MINUTE_EVERY)));
  const page = (
    <Enter ms={ms} start={0} kind="page" className="m-page">
      <Enter ms={ms} start={0} kind="rise" className="ma-head is-live">
        <OrbSlot ms={ms} size={32} phases={[{ at: 0, design: "solving" }]} />
        <div className="ma-head__t">
          <div className="m-head__h ma-h1">
            <Enter ms={ms} start={0} kind="swap" style={{ display: "inline-block" }}>
              <PopText ms={ms} steps={tokens} />
            </Enter>
          </div>
          <p className="ma-lede is-sub">{w.model}</p>
        </div>
      </Enter>
      <Enter ms={ms} start={0} kind="rise" delay={50}>
        <Block title={w.speed} aside={w.windowed} live>
          <dl className="ma-rows">
            <div>
              <dt>
                {w.decode}
                <small>{w.now}</small>
              </dt>
              <dd>
                49.5<Unit>tok/s</Unit>
              </dd>
            </div>
            <div>
              <dt>
                {w.decode}
                <small>{w.meanReq}</small>
              </dt>
              <dd>
                57.7<Unit>tok/s</Unit>
              </dd>
            </div>
            <div>
              <dt>
                {w.prefill}
                <small>{w.meanReq}</small>
              </dt>
              <dd>
                52<Unit>tok/s</Unit>
              </dd>
            </div>
            <div>
              <dt>{w.elapsed}</dt>
              <dd>
                <PopText ms={ms} steps={elapsed} />
                <Unit>s</Unit>
              </dd>
            </div>
          </dl>
          <div className="ma-chart-wrap">
            <div className="ma-chart-t">{w.lastMin}</div>
            <PlotChart data={series} height={110} labels={false} />
          </div>
        </Block>
      </Enter>
      {full && (
        <>
          <Enter ms={ms} start={0} kind="rise" delay={100}>
            <Block title={w.cards} aside={w.hw} live>
              <dl className="ma-rows">
                <div>
                  <dt>{w.gpu(0)}</dt>
                  <dd>11% · 10.2 / 12.0 GB · 58°C</dd>
                </div>
                <div>
                  <dt>{w.gpu(1)}</dt>
                  <dd>99% · 14.7 / 15.9 GB · 54°C</dd>
                </div>
              </dl>
            </Block>
          </Enter>
          <Enter ms={ms} start={0} kind="rise" delay={150}>
            <Block title={w.recent} aside={w.all} live>
              <RequestList ms={ms} w={w} rows={4} />
            </Block>
          </Enter>
        </>
      )}
    </Enter>
  );
  return full ? (
    <div className="a-app">
      <AppNav active={w.navLive} items={w.nav} />
      {page}
    </div>
  ) : (
    page
  );
}
