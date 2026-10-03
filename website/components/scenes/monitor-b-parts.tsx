"use client";

import type { ReactNode } from "react";
import { Code } from "./code";
import { AppNav, Orb } from "./kit";
import { Collapse, Enter, stagger } from "./parts";

/**
 * The Requests and Hardware pages of the app, and the blocks of the Dashboard that come from them (the hardware block, where the
 * experts ran, the list of requests), written for a clock. The figures are the ones one real run showed (the same run as the
 * chat scenes). Where the app moves, the scene moves with the app's numbers; where the app does not move, neither does the scene:
 *
 *  - a page opens with `.page-in` (260 ms, sharpening from a 6 px blur) and every child of the page rises with `.stagger`
 *    (420 ms, 50 ms between children; the seventh and later share 300 ms), in styles.css and App.tsx (`main.page-in`);
 *  - a row added to a list comes in with `.row-in` (360 ms, from 6 px above), every row at once: RequestList (pages/Requests.tsx);
 *  - an orb comes in with `.orb-in` (450 ms) when it is mounted: components/orb.tsx; a GPU's orb is chosen by gpuDesign (lib/orbs.ts);
 *  - the figures on the Hardware page and in the Dashboard's hardware block are plain text (pages/Hardware.tsx, pages/Dashboard.tsx),
 *    and a bar's width only transitions when it changes after it has been drawn (Bar, `duration-500`): on a page that opens, both
 *    are simply there. Nothing counts up and no bar fills.
 */

type Lang = "en" | "th";
/** The strings of the Dashboard's blocks that the caller already holds (the scene's own word table). */
type Hw = { hw: string; open: string; experts: string; e1: string; e2: string; e3: string; e4: string; over: string; modelOn: string; pcie: string; pcie2: string; tools: string; rd: string };

/** The words of the two pages: English from the app's source, Thai from its own tables (i18n/th-requests.ts, th-core.ts). */
const B = {
  en: {
    nav: ["Chat", "Dashboard", "Live", "Requests", "Hardware", "Settings", "About"],
    reqTitle: "Requests",
    reqSub: "4 kept on this PC. Hover a speed for its spread.",
    keep: "Prompts are kept as 200 characters. Keep the whole prompt of the next:",
    tlA: "Have an engine timeline (",
    tlB: ")? ",
    tlLink: "Open it in the viewer",
    tlEnd: ".",
    tools: "15 tools",
    hwTitle: "Hardware",
    hwSub: "What each part is doing for the model. A figure the engine does not measure says so.",
    gpus: "GPUs",
    memory: "Memory",
    busy: (n: number) => `${n}% busy`,
    threads: "20 threads",
    kernels: "Expert kernels in use",
    q2: "Q2_0 rows",
    pool: "Pool workers",
    used: "Used",
    bw: "Bandwidth",
    bwHint: "the CPU pool's own, per request",
    nm: "not measured",
    pcie: "Gen 4 ×16",
    pcie2: "Gen 1 ×4 of Gen 4 ×16",
    read: (read: string, cached: string, out: string) => `${read} read${cached ? ` · ${cached} cached` : ""} → ${out}`,
  },
  th: {
    nav: ["แชท", "ภาพรวม", "สด", "คำขอ", "ฮาร์ดแวร์", "ตั้งค่า", "ประมาณ"],
    reqTitle: "คำขอ",
    reqSub: "เก็บคำขอไว้ในเครื่องนี้ 4 รายการ วางเมาส์เหนือความเร็วเพื่อดูช่วงของค่า",
    keep: "prompt จะถูกเก็บไว้ 200 ตัวอักษร เก็บ prompt ทั้งหมดของคำขอถัดไป:",
    tlA: "มี timeline ของ engine (",
    tlB: ") ไหม? ",
    tlLink: "เปิดในตัวดู",
    tlEnd: "",
    tools: "15 เครื่องมือ",
    hwTitle: "ฮาร์ดแวร์",
    hwSub: "แต่ละส่วนกำลังทำอะไรให้โมเดล ตัวเลขที่ engine ไม่ได้วัดจะบอกไว้ว่ายังไม่ได้วัด",
    gpus: "GPU",
    memory: "หน่วยความจำ",
    busy: (n: number) => `ใช้งาน ${n}%`,
    threads: "20 เธรด",
    kernels: "kernel ของ expert ที่ใช้",
    q2: "แถว Q2_0",
    pool: "worker ของพูล",
    used: "ใช้ไป",
    bw: "แบนด์วิดท์",
    bwHint: "ของพูล CPU เอง ต่อคำขอ",
    nm: "ยังไม่ได้วัด",
    pcie: "Gen 4 ×16",
    pcie2: "Gen 1 ×4 จาก Gen 4 ×16",
    read: (read: string, cached: string, out: string) => `อ่าน ${read}${cached ? ` · cache ${cached}` : ""} → ${out}`,
  },
} as const;

/* ── the pieces ──────────────────────────────────────────────────────── */

/** A card's orb (pages/Hardware.tsx GpuOrb, lib/orbs.ts gpuDesign): held back by a limit is struggling (solving), 60 % and over is
 *  weaving (parallel strands), 5 % and over is working, and below that the card is idle: the searching globe, slowly (the app draws it at 30 fps).
 *  It comes in with `.orb-in` (450 ms) when it is mounted at `start`. */
export function gpuDesign(util: number | null, held = false): "solving" | "weaving" | "working" | "searching" {
  if (held) return "solving";
  const u = util ?? 0;
  return u >= 60 ? "weaving" : u >= 5 ? "working" : "searching";
}
function GpuMark({ ms, start, util }: { ms: number; start: number; util: number }) {
  return (
    <Enter ms={ms} start={start} kind="orb" className="mb-orb">
      <Orb ms={ms} live size={20} design={gpuDesign(util)} />
    </Enter>
  );
}

/** The app's Bar: a 4 px track and its fill. It has no fill-in of its own: it is drawn at its width. */
function Bar({ f }: { f: number }) {
  return (
    <div className="a-meter mb-bar">
      <div className="a-meter__f" style={{ width: `${Math.min(1, Math.max(0, f)) * 100}%` }} />
    </div>
  );
}

/** A part of a page: a heading and its content (components/bits.tsx Section). */
function Sec({ title, aside, children }: { title: string; aside?: string; children: ReactNode }) {
  return (
    <section className="m-sec">
      <div className="m-sec__h">
        <b>{title}</b>
        {aside && <span>{aside}</span>}
      </div>
      {children}
    </section>
  );
}

/** One device on the Hardware page: an optional mark, its name and one quiet line, then what it has to say (pages/Hardware.tsx Device). */
function Device({ title, sub, lead, className = "", children }: { title: string; sub?: string; lead?: ReactNode; className?: string; children: ReactNode }) {
  return (
    <div className={"mb-dev " + className}>
      <div className="mb-dev__h">
        {lead}
        <div>
          <div className="mb-dev__n">{title}</div>
          {sub && <div className="mb-dev__s">{sub}</div>}
        </div>
      </div>
      <div className="mb-dev__b">{children}</div>
    </div>
  );
}

/** A label and its value on one line, with a quiet hint (components/bits.tsx Row). `hot`: the row the caption speaks of, drawn as the app draws a row under the pointer. */
function Row({ k, hint, hot = false, children }: { k: string; hint?: string; hot?: boolean; children: ReactNode }) {
  return (
    <div className={"mb-row" + (hot ? " mb-row--hot" : "")}>
      <dt>
        {k}
        {hint && <small>{hint}</small>}
      </dt>
      <dd>{children}</dd>
    </div>
  );
}

/** The page: it sharpens in (`.page-in`), and what is on it rises one child after the next (`.stagger`, `i` is the child's place on the page). */
function Page({ ms, children }: { ms: number; children: ReactNode }) {
  return (
    <Enter ms={ms} start={0} kind="page" className="m-page">
      {children}
    </Enter>
  );
}
function Child({ ms, i, className, children }: { ms: number; i: number; className?: string; children: ReactNode }) {
  return (
    <Enter ms={ms} start={0} kind="rise" delay={stagger(i)} className={className}>
      {children}
    </Enter>
  );
}

/** When the last child of a page of `n` children has landed. */
const settled = (n: number) => stagger(n - 1) + 420;

/* ── the requests ────────────────────────────────────────────────────── */

const PROMPT = 'Read notes.py, then add a --version flag that prints "notes-app 0.1.0" and exits. Ke...';
const ROWS = [
  { when: "Oct 3, 03:17:57 PM", prompt: PROMPT, read: "31", cached: "4,742", out: "171", prefill: "48", dec: "64.0" },
  { when: "Oct 3, 03:16:51 PM", prompt: PROMPT, read: "229", cached: "4,439", out: "74", prefill: "68", dec: "74.2" },
  { when: "Oct 3, 03:16:43 PM", prompt: PROMPT, read: "376", cached: "3,727", out: "336", prefill: "172", dec: "61.1" },
  { when: "Oct 3, 03:16:37 PM", prompt: 'Read notes.py, then add a --version flag that prints "notes-app 0.1.0" and exits. Keep the change small.', read: "3,664", cached: "", out: "61", prefill: "696", dec: "65.1" },
];

/** The rows of the app's request list (pages/Requests.tsx RequestList). A row that is added comes in with `.row-in` (360 ms, from 6 px
 *  above); the rows that are there together come in together, with no step between them. `at` is when they are added. */
export function RequestRows({ ms, at, w, n = 3, lang }: { ms: number; at: number; w: Pick<Hw, "tools">; n?: number; lang: Lang }) {
  const b = B[lang];
  return (
    <div className="mb-reqs">
      {ROWS.slice(0, n).map((r) => (
        <Enter key={r.when} ms={ms} start={at} kind="row" className="mb-req">
          <div className="mb-req__a">
            <div className="mb-req__t">{r.prompt}</div>
            <div className="mb-req__m">
              {r.when} · OpenAI · Chrome · {w.tools}
            </div>
          </div>
          <div className="mb-req__n">
            <span className="mb-u">{b.read(r.read, r.cached, r.out)}</span>
            <span>
              {r.prefill}
              <span className="mb-u"> tok/s</span>
            </span>
            <span>
              {r.dec}
              <span className="mb-u"> tok/s</span>
            </span>
          </div>
        </Enter>
      ))}
    </div>
  );
}

export function RequestsPage({ ms, lang }: { ms: number; lang: Lang }) {
  const b = B[lang];
  return (
    <div className="a-app">
      <AppNav active={b.nav[3]} items={[...b.nav]} />
      <Page ms={ms}>
        <Child ms={ms} i={0} className="m-title">
          {b.reqTitle}
        </Child>
        <Child ms={ms} i={1} className="m-sub">
          {b.reqSub}
        </Child>
        <Child ms={ms} i={2} className="mb-keep">
          <span>{b.keep}</span>
          <span className="m-pill">1</span>
          <span className="m-pill">5</span>
          <span className="m-pill">10</span>
        </Child>
        <Child ms={ms} i={3} className="mb-tl">
          {b.tlA}
          <code className="mk">
            <Code lang="bash" code="STRATA_TIMELINE" />
          </code>
          {b.tlB}
          <span className="mb-a">{b.tlLink}</span>
          {b.tlEnd}
        </Child>
        <Child ms={ms} i={4} className="mb-list">
          <RequestRows ms={ms} at={0} w={{ tools: b.tools }} n={4} lang={lang} />
        </Child>
      </Page>
    </div>
  );
}
export const REQUESTS_DURATION = settled(5);

/* ── the hardware ────────────────────────────────────────────────────── */

const GPUS = [
  { name: "NVIDIA GeForce RTX 4070 SUPER", sm: "sm89", util: 2, gb: "10.2 / 12.0 GB", f: 10.2 / 12, temp: 56, power: 41, link: "pcie" as const },
  { name: "NVIDIA GeForce RTX 5060 Ti", sm: "sm120", util: 0, gb: "14.7 / 15.9 GB", f: 14.7 / 15.9, temp: 39, power: 6, link: "pcie2" as const },
];

function GpusSection({ ms, lang, start }: { ms: number; lang: Lang; start: number }) {
  const b = B[lang];
  return (
    <Sec title={b.gpus}>
      {GPUS.map((g) => (
        <Device key={g.name} className="m-gpu" title={g.name} sub={g.sm} lead={<GpuMark ms={ms} start={start} util={g.util} />}>
          <Bar f={g.f} />
          <div className="mb-figs">
            <span>{b.busy(g.util)}</span>
            <span>{g.gb}</span>
            <span>{g.temp}°C</span>
            <span>{g.power} W</span>
            <span>{b[g.link]}</span>
          </div>
        </Device>
      ))}
    </Sec>
  );
}

function CpuSection({ lang }: { lang: Lang }) {
  const b = B[lang];
  return (
    <Sec title="CPU">
      <Device title="13th Gen Intel(R) Core(TM) i5-13500" sub={b.threads}>
        <dl className="mb-rows">
          <Row k={b.kernels} hint={b.q2} hot>
            AVX-VNNI
          </Row>
          <Row k={b.pool}>
            <span className="mb-n">13</span>
          </Row>
        </dl>
      </Device>
    </Sec>
  );
}

function MemorySection({ lang }: { lang: Lang }) {
  const b = B[lang];
  return (
    <Sec title={b.memory}>
      <Device title="RAM">
        <Bar f={41.0 / 47.7} />
        <dl className="mb-rows">
          <Row k={b.used}>
            <span className="mb-n">41.0 / 47.7 GB</span>
          </Row>
          <Row k={b.bw} hint={b.bwHint}>
            <span className="mb-nm">{b.nm}</span>
          </Row>
        </dl>
      </Device>
    </Sec>
  );
}

/** The Hardware page, opened. The orbs of the two idle cards come in with the page (they are mounted with it). */
export function HardwarePage({ ms, lang }: { ms: number; lang: Lang }) {
  const b = B[lang];
  return (
    <div className="a-app">
      <AppNav active={b.nav[4]} items={[...b.nav]} />
      <Page ms={ms}>
        <Child ms={ms} i={0} className="m-title">
          {b.hwTitle}
        </Child>
        <Child ms={ms} i={1} className="m-sub">
          {b.hwSub}
        </Child>
        <Child ms={ms} i={2}>
          <GpusSection ms={ms} lang={lang} start={0} />
        </Child>
        <Child ms={ms} i={3}>
          <CpuSection lang={lang} />
        </Child>
        <Child ms={ms} i={4}>
          <MemorySection lang={lang} />
        </Child>
      </Page>
    </div>
  );
}
export const HARDWARE_DURATION = settled(5);

/** The middle of the Hardware page: its GPUs and its CPU, the third and fourth children of the page. */
export function GpusPage({ ms, lang }: { ms: number; lang: Lang }) {
  return (
    <Page ms={ms}>
      <Child ms={ms} i={2}>
        <GpusSection ms={ms} lang={lang} start={0} />
      </Child>
      <Child ms={ms} i={3}>
        <CpuSection lang={lang} />
      </Child>
    </Page>
  );
}
export const GPUS_DURATION = settled(4);

/* ── the Dashboard's blocks that come from these pages ───────────────── */

/** The Dashboard's Hardware block (pages/Dashboard.tsx): a row for each card with its orb, its load and temperature as plain text, a bar, and the
 *  memory and the link; then the CPU and the RAM as two bars. It rises (`.stagger`, 420 ms) at `at`. */
export function Hardware({ ms, at, w }: { ms: number; at: number; w: Hw }) {
  return (
    <Enter ms={ms} start={at} kind="rise" className="m-sec">
      <div className="m-sec__h">
        <b>{w.hw}</b>
        <span>{w.open}</span>
      </div>
      {[
        { n: "RTX 4070 SUPER", sm: "sm89", util: 12, temp: 56, gb: "10.2 / 12.0 GB", f: 10.2 / 12, link: w.pcie },
        { n: "RTX 5060 Ti", sm: "sm120", util: 0, temp: 39, gb: "14.7 / 15.9 GB", f: 14.7 / 15.9, link: w.pcie2 },
      ].map((g) => (
        <div key={g.n} className="m-gpu mb-gpu">
          <div className="mb-gpu__n">
            <span className="mb-gpu__who">
              <GpuMark ms={ms} start={at} util={g.util} />
              {g.n} <span>{g.sm}</span>
            </span>
            <span className="mb-gpu__v">
              {g.util}% · {g.temp}°C
            </span>
          </div>
          <Bar f={g.f} />
          <div className="mb-gpu__s">
            {g.gb} · {g.link}
          </div>
        </div>
      ))}
      <div className="m-duo">
        <div>
          <div className="m-duo__h">
            CPU · AVX-VNNI <b>37%</b>
          </div>
          <Bar f={0.37} />
        </div>
        <div>
          <div className="m-duo__h">
            RAM <b>41.2 / 47.7 GB</b>
          </div>
          <Bar f={41.2 / 47.7} />
        </div>
      </div>
      <div className="m-dim">{w.modelOn}</div>
    </Enter>
  );
}

/** The Dashboard's "Where the last request's experts ran" (pages/Dashboard.tsx, inside a Reveal): the block rises and opens like a Collapse (320 ms) once
 *  its data is there. The bar is drawn at its widths (it only transitions when they change), the dots take the ink at 100, 76, 52 and 28 %. */
export function Experts({ ms, at, w }: { ms: number; at: number; w: Hw }) {
  const parts: [string, number][] = [
    [w.e1, 52],
    [w.e2, 26],
    [w.e3, 0],
    [w.e4, 23],
  ];
  const tone = (i: number) => 1 - i * 0.24;
  return (
    <Enter ms={ms} start={at} kind="rise">
      <Collapse ms={ms} open={at + 16}>
        <section className="m-sec">
          <div className="m-sec__h">
            <b>{w.experts}</b>
            <span>{w.open}</span>
          </div>
          <div className="m-stack">
            {parts.map(([n, v], i) => (
              <i key={n} style={{ flexGrow: v, background: "var(--a-ink)", opacity: tone(i) }} />
            ))}
          </div>
          <div className="mb-legend">
            {parts.map(([n, v], i) => (
              <span key={n}>
                <i className="mb-dot" style={{ opacity: tone(i) }} />
                <em>{n}</em>
                <b>{v}%</b>
              </span>
            ))}
          </div>
          <div className="m-dim">{w.over}</div>
        </section>
      </Collapse>
    </Enter>
  );
}
