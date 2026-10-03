"use client";

import { useLayoutEffect, useMemo, useRef, useState } from "react";
import { lerp, lin, type SceneDef } from "./Scene";
import { DUR, EASE, EASE_SOFT } from "./motion";
import { AppNav } from "./kit";
import { Enter } from "./parts";
import {
  buildRun,
  DoneThought,
  Follow,
  Footer,
  Hero,
  HeroCollapse,
  morph,
  PromptBar,
  PROMPT,
  RATE,
  Reply,
  Reserve,
  Side,
  StatusLabel,
  Thought,
  ToolCard,
  UserMessage,
  wordsOf,
  BASH_CMD,
  BASH_NOTE,
  FILE,
  THOUGHT_SECS,
  type Lang,
  type ModeKey,
  type Run,
  type ToolSpec,
  type Words,
} from "./chat-parts";

/**
 * The chat of the app, from one real run (a prompt that reads a file, edits it, runs a command that asks first, and answers), played from a
 * clock. The run is written once (buildRun in chat-parts.tsx, in the order and at the speed the app does it) and every scene here shows a part of
 * it: the whole page (chatRun), or the part of the conversation that says one thing (the cards, the status, the question, the answer).
 * The real source: pages/chat/Chat.tsx, Messages.tsx, AgentCall.tsx, components/PromptBar.tsx, Sidebar.tsx, thought.tsx, lib/chat.ts.
 */

const RUN = buildRun(700);

/** The three calls of the run, as the cards show them. Without `asks` the command runs without a question (a rule allows it). */
function toolsOf(r: Run, asks: boolean): { read: ToolSpec; edit: ToolSpec; bash: ToolSpec } {
  return {
    read: { name: "Read", arg: FILE, start: r.a1, call: r.c1, done: r.d1, time: "0.0 s" },
    edit: { name: "Edit", arg: FILE, start: r.a2, call: r.c2, done: r.d2, time: "0.0 s" },
    bash: {
      name: "Bash",
      arg: BASH_NOTE,
      start: r.a3,
      call: r.ask,
      done: r.d3,
      time: "61.8 s",
      ask: asks ? { at: r.ask, press: r.press, answer: r.ans, command: BASH_CMD } : undefined,
    },
  };
}

type ThreadOpts = {
  /** The user's message: its prompt rises out of the composer (the whole page), or it simply arrives. */
  user?: "rise" | "plain" | null;
  /** The parts of the conversation that began before this moment are left out. */
  skipBefore?: number;
  asks?: boolean;
};

/** The conversation of the run at moment `ms` of it: the message of the user, and the answer with its thoughts, cards, text and the row under it. */
function Thread({ ms, r, w, o }: { ms: number; r: Run; w: Words; o: ThreadOpts }) {
  const from = o.skipBefore ?? 0;
  const asks = o.asks ?? true;
  const T = toolsOf(r, asks);
  const at = (t: number) => ms >= t && t >= from;
  const waiting = ms >= r.S && ms < r.g1 && from <= r.S;
  return (
    <>
      {o.user && ms >= r.S && <UserMessage ms={ms} from={r.S} readDone={r.g1} w={w} rise={o.user === "rise"} />}
      {ms >= r.S && (
        <Enter ms={ms} start={r.S} kind="msg" className="c-am">
          {waiting && (
            <div className="c-wait">
              <StatusLabel ms={ms} steps={[{ at: r.S, design: "breathing" }, { at: r.r0, design: "listening" }]} label={ms < r.r0 ? w.waiting : w.reading} />
            </div>
          )}
          {at(r.g1) && <Thought ms={ms} start={r.g1} stopAt={r.a1} timedFrom={r.aS} seconds={THOUGHT_SECS} pattern="orbit" w={w} />}
          {at(r.a1) && <ToolCard ms={ms} tool={T.read} w={w} />}
          {at(r.g2) && (ms < r.a2 ? <Thought key="tail2" ms={ms} start={r.g2} stopAt={r.a2} pattern="spiral" w={w} /> : <DoneThought key="r1" ms={ms} pattern="snake" w={w} />)}
          {at(r.a2) && <ToolCard ms={ms} tool={T.edit} w={w} />}
          {at(r.g3) && (ms < r.a3 ? <Thought key="tail3" ms={ms} start={r.g3} stopAt={r.a3} pattern="spiral" w={w} /> : <DoneThought key="r2" ms={ms} pattern="snake" w={w} />)}
          {at(r.a3) && <ToolCard ms={ms} tool={T.bash} w={w} />}
          {at(r.g4) && <Thought ms={ms} start={r.g4} stopAt={r.aS} pattern="spiral" w={w} />}
          {at(r.aS) && <Reply k={((ms - r.aS) / 1000) * RATE} w={w} />}
          <Footer ms={ms} r={r} w={w} asks={asks} />
        </Enter>
      )}
    </>
  );
}

/* ── the whole page, as the run goes ─────────────────────────────────────────────────────────────────────────────── */

function ChatRunScene({ ms, lang }: { ms: number; lang: Lang }) {
  const w = wordsOf(lang);
  const r = RUN;
  const busy = ms >= r.S && ms < r.end;
  const send = morph(ms, r.S, r.end);
  const pct = ms >= r.end ? 2 * EASE(lin(ms, r.end, 400)) : 0;
  return (
    <div className="a-app c-app c-run c" data-lang={lang}>
      <AppNav active="Chat" />
      <div className="c-split">
        <Side ms={ms} w={w} chatAt={r.S} askFrom={r.ask} askTo={r.ans} endAt={r.end} />
        <section className="c-chat">
          <Follow>
            <div className="c-list">
              {ms < r.S + 600 && (
                <HeroCollapse ms={ms} closeAt={r.S}>
                  <Hero ms={ms} w={w} steps={[{ at: -DUR.orb * 2, design: "listening" }]} />
                </HeroCollapse>
              )}
              <Thread ms={ms} r={r} w={w} o={{ user: "rise" }} />
            </div>
          </Follow>
          <div className="c-dock">
            <PromptBar w={w} text={ms < r.S ? PROMPT : ""} busy={busy} send={send} armed={ms < r.S || busy} pct={pct} sparks={ms < r.S || ms >= r.end} />
          </div>
        </section>
      </div>
    </div>
  );
}

/** The whole run as the new chat shows it. */
export const chatRun: SceneDef = {
  duration: RUN.landed,
  hold: 4200,
  Body: ({ ms, lang }) => <ChatRunScene ms={ms} lang={lang} />,
};

/* ── the parts of it that say one thing ──────────────────────────────────────────────────────────────────────────── */

/** A piece of the run, from moment `off` of it: what had happened by then is already there and does not move. */
function Slice({ ms, lang, run, off, duration, o }: { ms: number; lang: Lang; run: Run; off: number; duration: number; o: ThreadOpts }) {
  const w = wordsOf(lang);
  const render = (t: number) => (
    <div className="c-pad c" data-lang={lang}>
      <div className="c-list">
        <Thread ms={t + off} r={run} w={w} o={o} />
      </div>
    </div>
  );
  // the last frame is made once: it only keeps the room
  const last = useMemo(() => render(duration), [lang, run, off, duration, o]); // eslint-disable-line react-hooks/exhaustive-deps
  return <Reserve last={last}>{render(ms)}</Reserve>;
}


const O_TOOLS: ThreadOpts = { asks: false };
const O_STATUS: ThreadOpts = { user: "plain" };
const O_ASK: ThreadOpts = {};
const O_ANSWER: ThreadOpts = { skipBefore: RUN.g4 };
const R_STATUS = RUN;
// the command that runs without asking (a rule of the chat allows it): its card is written, runs, and is done
const R_TOOLS = buildRun(700, { wait: 0, ran: 700 });
// the card that asks, for as long as it takes to read it (in the run it is a person deciding)
const R_ASK = buildRun(700, { wait: 2100 });

/** Mood "connecting": the model reaches for a tool, and every call is a card with its time. */
const TOOLS_AT = RUN.a2;
const TOOLS_END = R_TOOLS.d3 + 450 - TOOLS_AT; // the status that follows the result has just settled
export const toolsScene: SceneDef = {
  duration: TOOLS_END,
  Body: ({ ms, lang }) => <Slice ms={ms} lang={lang} run={R_TOOLS} off={TOOLS_AT} duration={TOOLS_END} o={O_TOOLS} />,
};

/** What the chat shows while a call's result is being read: the question above, the cards, one status that never goes quiet. */
const STATUS_AT = R_STATUS.g2;
const STATUS_END = R_STATUS.d2 + 650 - STATUS_AT;
export const statusScene: SceneDef = {
  duration: STATUS_END,
  Body: ({ ms, lang }) => <Slice ms={ms} lang={lang} run={R_STATUS} off={STATUS_AT} duration={STATUS_END} o={O_STATUS} />,
};

/** Mood "breathing": the command waits for you. */
const ASK_AT = R_ASK.g3;
const ASK_END = R_ASK.d3 + 450 - ASK_AT;
export const permissionScene: SceneDef = {
  duration: ASK_END,
  Body: ({ ms, lang }) => <Slice ms={ms} lang={lang} run={R_ASK} off={ASK_AT} duration={ASK_END} o={O_ASK} />,
};

/** Mood "composing": the answer streams in, with its speed underneath. */
const ANSWER_AT = RUN.g4;
const ANSWER_END = RUN.landed - ANSWER_AT;
export const answerScene: SceneDef = {
  duration: ANSWER_END,
  Body: ({ ms, lang }) => <Slice ms={ms} lang={lang} run={RUN} off={ANSWER_AT} duration={ANSWER_END} o={O_ANSWER} />,
};

/* ── mood "listening": the empty chat, and the question going in ──────────────────────────────────────────────────── */

const TYPE_AT = 900;
const TYPE_CPS = 26; // a person typing, not the model writing
const LISTEN_END = TYPE_AT + Math.ceil((PROMPT.length / TYPE_CPS) * 1000) + 500;

function ListeningBody({ ms, lang }: { ms: number; lang: Lang }) {
  const w = wordsOf(lang);
  const n = Math.min(PROMPT.length, Math.max(0, Math.floor(((ms - TYPE_AT) / 1000) * TYPE_CPS)));
  return (
    <div className="c-pad c c-lis" data-lang={lang}>
      <Hero ms={ms} w={w} steps={[{ at: -DUR.orb * 2, design: "searching" }, { at: TYPE_AT + 1000 / TYPE_CPS, design: "listening" }]} />
      <div className="c-lis__dock">
        <PromptBar w={w} text={PROMPT.slice(0, n)} send={{ t: 0, dir: 1 }} armed={n > 0} pct={0} sparks />
      </div>
    </div>
  );
}
function ListeningScene({ ms, lang }: { ms: number; lang: Lang }) {
  const last = useMemo(() => <ListeningBody ms={LISTEN_END} lang={lang} />, [lang]);
  return (
    <Reserve last={last}>
      <ListeningBody ms={ms} lang={lang} />
    </Reserve>
  );
}
export const listeningScene: SceneDef = {
  duration: LISTEN_END,
  Body: ({ ms, lang }) => <ListeningScene ms={ms} lang={lang} />,
};

/* ── the three coding modes, from the chip on the bar (AgentControls.tsx ModePicker) ─────────────────────────────── */

const MODES: ModeKey[] = ["ask", "plan", "auto"];
const OPEN_AT = 600;
const PICK_AT = [0, 2300, 3900]; // when each mode was chosen
const MODES_END = PICK_AT[2] + 420 + 700;
const INK70 = "color-mix(in srgb, var(--a-ink) 70%, transparent)";
const TINT: Record<ModeKey, string> = { ask: INK70, plan: "var(--a-accent)", auto: "color-mix(in srgb, #d98a00 80%, var(--a-ink))" };

/** The text of the mode in use. When it is replaced by one of another height the box stretches to it (420 ms, an even stretch), and what is under it follows. */
function ModeText({ ms, texts, idx }: { ms: number; texts: string[]; idx: number }) {
  const ghosts = useRef<(HTMLParagraphElement | null)[]>([]);
  const [h, setH] = useState<number[] | null>(null);
  useLayoutEffect(() => {
    const measure = () => {
      const next = ghosts.current.map((g) => g?.offsetHeight ?? 0);
      setH((prev) => (prev && prev.every((v, i) => v === next[i]) ? prev : next));
    };
    measure();
    const ro = new ResizeObserver(measure);
    for (const g of ghosts.current) if (g) ro.observe(g);
    return () => ro.disconnect();
  }, [texts.join("|")]); // eslint-disable-line react-hooks/exhaustive-deps
  const before = Math.max(0, idx - 1);
  const e = idx === 0 ? 1 : EASE_SOFT(lin(ms, PICK_AT[idx], DUR.glide));
  return (
    <div className="c-seg__txt" style={h ? { height: lerp(h[before], h[idx], e) } : undefined}>
      {texts.map((t, i) => (
        <p key={i} ref={(el) => { ghosts.current[i] = el; }} className="c-seg__ghost" aria-hidden="true">
          {t}
        </p>
      ))}
      <span>{texts[idx]}</span>
    </div>
  );
}

function ModesBody({ ms, lang }: { ms: number; lang: Lang }) {
  const w = wordsOf(lang);
  const idx = ms < PICK_AT[1] ? 0 : ms < PICK_AT[2] ? 1 : 2;
  const k = ms < PICK_AT[1] ? 0 : ms < PICK_AT[2] ? lerp(0, 1, EASE(lin(ms, PICK_AT[1], DUR.glide))) : lerp(1, 2, EASE(lin(ms, PICK_AT[2], DUR.glide)));
  const pop = EASE(lin(ms, OPEN_AT, 180));
  const mode = MODES[idx];
  const swap = idx === 0 ? 1 : lin(ms, PICK_AT[idx], 150); // the chip's colour moves to the new mode's in 150 ms
  const color = idx === 0 ? TINT.ask : `color-mix(in srgb, ${TINT[mode]} ${swap * 100}%, ${TINT[MODES[idx - 1]]})`;
  const label = { ask: w.ask, plan: w.plan, auto: w.auto };
  const texts = [w.modeAsk, w.modePlan, w.modeAuto];
  return (
    <div className="c-pad c c-modes" data-lang={lang}>
      <PromptBar w={w} text="" send={{ t: 0, dir: 1 }} armed={false} pct={0} mode={{ label: label[mode], color, on: ms >= OPEN_AT ? lin(ms, OPEN_AT, 150) : 0 }}>
        {ms >= OPEN_AT && (
          <div className="c-pb__menu" style={{ opacity: pop, transform: pop >= 1 ? undefined : `translate3d(0, ${4 * (1 - pop)}px, 0) scale(${0.98 + 0.02 * pop})` }}>
            <div className="c-pb__menu-in">
              <div className="c-seg">
                <span className="c-seg__pill" style={{ transform: `translateX(calc(${k} * (100% + 2px)))` }} />
                {MODES.map((m, i) => {
                  const near = Math.max(0, 1 - Math.abs(k - i));
                  return (
                    <span key={m} className="c-seg__b" style={{ fontWeight: near > 0.5 ? 500 : 400, color: `color-mix(in srgb, var(--a-ink) ${near * 100}%, color-mix(in srgb, var(--a-ink) 62%, var(--a-surface)))` }}>
                      {label[m]}
                    </span>
                  );
                })}
              </div>
              <ModeText ms={ms} texts={texts} idx={idx} />
            </div>
          </div>
        )}
      </PromptBar>
    </div>
  );
}
export const modesScene: SceneDef = {
  duration: MODES_END,
  Body: ({ ms, lang }) => <ModesBody ms={ms} lang={lang} />,
};
