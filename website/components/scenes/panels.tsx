"use client";

import type { ReactNode } from "react";
import { lerp, type SceneDef } from "./Scene";
import { Collapse } from "./parts";
import { AppNav } from "./kit";
import { HarnessIcon, type HarnessId } from "../HarnessIcon";
import { EASE_SOFT } from "./motion";
import { at, EASE_TW, Glide, Ico, LoadingLine, MiniSwitch, SIDE_IN, Stage, tr, type IconName } from "./panels-parts";

/**
 * The side panel of the chat: Git, Plan, Skills, Memory, Context (components/SidePanel.tsx, ContextPanel.tsx). The text is the app's own,
 * from one real run. What moves is what moves in the app:
 *  - the panel opens by fading in, 260 ms, nothing else (`.side-in`); the tabs have no entrance of their own;
 *  - a tab that has to ask the server (Git, Memory) shows the `Loading` line first, and what arrives takes the place of it by stretching
 *    from no height to its height, 420 ms, with no fade and no slide (GlidePanel / lib/glide.ts);
 *  - a section opens as a Collapse (320 ms), a switch as the prompt bar's small switch (200 ms);
 *  - the figures are plain text: they change, they do not count.
 */

type Lang = "en" | "th";
const TABS: [string, IconName][] = [
  ["Git", "branch"],
  ["Plan", "check"],
  ["Skills", "book"],
  ["Memory", "book"],
  ["Context", "layers"],
];

/**
 * The panel as the app draws it where there is not room beside the chat (components/SidePanel.tsx, `PanelDock`, below 1280 px, which is
 * the width of these screens): a sheet from the right edge, `w-[min(92vw,380px)] bg-surface p-3 shadow-xl`, over the chat that stays
 * where it was under a 40 % black veil. The veil fades in and the sheet slides in, both over 300 ms (`duration-300`, the app's ease);
 * the panel's own body does not fade in a sheet (that is only for the docked panel). The tab in use has the `fill` behind it; the others are quiet;
 * and the cross at the end of the tabs closes it.
 */
const OPEN_AT = 120;
const SHEET_MS = 300;
function Dock({ ms, lang, on, children }: { ms: number; lang: Lang; on: string; children: ReactNode }) {
  const p = at(ms, OPEN_AT, SHEET_MS);
  const veil = at(ms, OPEN_AT, SHEET_MS, EASE_SOFT);
  return (
    <div className="sp-sheetpage">
      <div className="sp-sheetpage__nav">
        <AppNav active="Chat" />
      </div>
      <div className="sp-behind" aria-hidden="true">
        <BehindChat lang={lang} />
      </div>
      <div className="sp-veil" style={{ opacity: veil }} aria-hidden="true" />
      <aside className="sp-sheet" style={{ transform: p >= 1 ? undefined : `translate3d(${(1 - p) * 100}%, 0, 0)` }}>
        <div className="sp-dock">
          <div className="sp-tabrow">
            <div className="sp-tabs" aria-hidden="true">
              {TABS.map(([k, icon]) => (
                <span key={k} className={"sp-tab" + (k === on ? " is-on" : "")}>
                  <Ico n={icon} size={14} />
                  <span>{tr(lang, k)}</span>
                </span>
              ))}
            </div>
            <span className="sp-x" aria-hidden="true">
              <Ico n="close" size={15} />
            </span>
          </div>
          <div className="sp-pane">{children}</div>
        </div>
      </aside>
    </div>
  );
}

/** The chat that stays under the sheet: the real run the Plan tab belongs to (a to-do list asked for, written, not acted on). */
function BehindChat({ lang }: { lang: Lang }) {
  return (
    <div className="sp-bc">
      <div className="sp-bc__ub">Without changing any file, write a three-step todo list (TodoWrite) for adding a &quot;delete&quot; command to notes.py.</div>
      <div className="sp-bc__meta">You · 03:30 PM</div>
      <div className="sp-bc__th">
        <Ico n="check" size={13} /> {tr(lang, "Thoughts")}
      </div>
      <p>Todo list created (no files touched):</p>
      <ol>
        <li>
          Add an <code>elif argv[:1] == [&quot;delete&quot;]:</code> branch in <code>main()</code> that removes <code>notes[int(argv[1]) - 1]</code> and saves.
        </li>
        <li>
          Update the usage line to mention <code>delete &lt;n&gt;</code>, and guard against a missing or out-of-range index (print one line, return 2).
        </li>
        <li>
          Verify: <code>python notes.py list</code>, <code>python notes.py delete 1</code>, <code>python notes.py list</code> — note gone, exit codes 0.
        </li>
      </ol>
      <p>Say the word and I&apos;ll start on step 1.</p>
    </div>
  );
}

/* ── Plan: the model's to-do list, as the last TodoWrite left it ───────────────────────────────── */

// The text of the steps as the model wrote them: the app shows it as it is (plain text, its backticks and all).
const TODOS = [
  "Add an `elif argv[:1] == [\"delete\"]:` branch in main() that removes notes[int(argv[1]) - 1] from the loaded list and saves it",
  "Update the usage line to mention `delete <n>` and guard against a missing/out-of-range index (print one line, return 2)",
  "Run `python notes.py list`, then `python notes.py delete 1` and `python notes.py list` to confirm the note is gone and exit codes are 0",
];
// When each TodoWrite lands. A step that is finished is ticked at once (no spinner, no drawing of the tick); the bar under the header
// is the only thing that moves, 300 ms (`transition-[width] duration-300`).
const TICKS = [2200, 3900, 5500];
const BAR_MS = 300;

function PlanBody({ ms, lang }: { ms: number; lang: Lang }) {
  const done = TICKS.filter((t) => ms >= t).length;
  let frac = 0;
  TICKS.forEach((t, k) => {
    if (ms >= t) frac = lerp(k / 3, (k + 1) / 3, at(ms, t, BAR_MS, EASE_TW));
  });
  return (
    <div>
      <div className="sp-plan__h">
        <span>{tr(lang, "To-do list")}</span>
        <span className="num">{tr(lang, "{done} of {total} done", { done, total: 3 })}</span>
      </div>
      <div className="sp-plan__bar" aria-hidden="true">
        <div style={{ width: `${frac * 100}%` }} />
      </div>
      <div className="sp-todos">
        <div className="sp-todos__h">{tr(lang, "{done} of {n} steps done", { done, n: 3 })}</div>
        <ul>
          {TODOS.map((x, i) => (
            <li key={i} className={i < done ? "is-done" : undefined}>
              <Ico n={i < done ? "check" : "circle"} size={15} />
              <span>{x}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

export const planScene: SceneDef = {
  duration: TICKS[2] + BAR_MS,
  Body: ({ ms, lang }) => (
    <Stage
      ms={ms}
      end={TICKS[2] + BAR_MS}
      lang={lang}
      render={(m) => (
        <Dock ms={m} lang={lang} on="Plan">
          <PlanBody ms={m} lang={lang} />
        </Dock>
      )}
    />
  ),
};

/* ── Git: the folder, the branch, what is not staged, the sections ─────────────────────────────── */

// The server answers (the Loading line goes, the view stretches in). The time is the answer of a local git call, not a figure of the app.
const GIT_LOAD = 1300;

function Section({ title, count, deg, children }: { title: string; count: number; deg: number; children?: ReactNode }) {
  return (
    <section className="sp-sec">
      <div className="sp-sec__h">
        <span className="sp-sec__t">{title}</span>
        <span className="num sp-sec__n">{count}</span>
        <Ico n="chevron" size={12} style={{ transform: `rotate(${deg}deg)` }} />
      </div>
      {children}
    </section>
  );
}

function GitView({ ms, lang }: { ms: number; lang: Lang }) {
  return (
    <div className="sp-gitview">
      <div className="sp-branchrow">
        <span className="sp-branch">
          <Ico n="branch" size={13} />
          feature/delete-note
        </span>
      </div>
      <div>
        <div className="sp-grp">
          <span>{tr(lang, "Not staged")}</span>
          <span className="num">1</span>
        </div>
        <div className="sp-file">
          <span className="sp-st">M</span>
          <span className="sp-file__p">README.md</span>
          <Ico n="chevron" size={12} />
        </div>
      </div>
      <Section title={tr(lang, "Branches")} count={3} deg={0} />
      <Section title={tr(lang, "Worktrees")} count={2} deg={0} />
      {/* Recent commits is open from the start: its Collapse plays its opening when the view is put on the page (one frame later) */}
      <Section title={tr(lang, "Recent commits")} count={1} deg={180}>
        <Collapse ms={ms} open={GIT_LOAD + 16}>
          <div className="sp-sec__c">
            <div className="sp-commit">
              <Ico n="commit" size={13} />
              <span>
                <b>notes-app: add, list and --version</b>
                <small className="num">dcb23cc · Demo · {lang === "th" ? "3 ต.ค." : "Oct 3"}</small>
              </span>
            </div>
          </div>
        </Collapse>
      </Section>
    </div>
  );
}

export const gitScene: SceneDef = {
  duration: GIT_LOAD + 16 + 420,
  Body: ({ ms, lang }) => (
    <Stage
      ms={ms}
      end={GIT_LOAD + 16 + 420}
      lang={lang}
      render={(m) => (
        <Dock ms={m} lang={lang} on="Git">
          <div className="sp-git">
            <div className="sp-gitpath">
              <span className="sp-path">C:\demo\notes-app</span>
              <span className="sp-refresh">
                <Ico n="refresh" size={14} />
              </span>
            </div>
            {m < GIT_LOAD ? (
              <LoadingLine ms={m} text={tr(lang, "Reading the repository…")} />
            ) : (
              <Glide ms={m} start={GIT_LOAD}>
                <GitView ms={m} lang={lang} />
              </Glide>
            )}
          </div>
        </Dock>
      )}
    />
  ),
};

/* ── Memory: notes the chat is handed, each with its switch ────────────────────────────────────── */

const MEM_LOAD = 1200;
// The Claude Code switch is turned on and, a moment later, off again: the notes of other apps are off unless they are switched on.
const MEM_FLIPS = [2800, 4600];

function Source({ title, icon, size, file, switchFlips, on }: { title: string; icon?: HarnessId; size: string; file: string; switchFlips?: { ms: number; flips: number[] }; on: boolean }) {
  return (
    <section className={"sp-msrc" + (on ? "" : " is-off")}>
      <div className="sp-msrc__top">
        <div className="sp-msrc__txt">
          <div className="sp-msrc__t">
            {icon && <HarnessIcon id={icon} size={18} className="sp-hicon" />}
            {title}
          </div>
          <div className="num sp-msrc__s">{size}</div>
        </div>
        {switchFlips && <MiniSwitch ms={switchFlips.ms} flips={switchFlips.flips} />}
      </div>
      <ul>
        <li>
          <div className="sp-mfile">
            <span>{file}</span>
            <Ico n="chevron" size={12} />
          </div>
        </li>
      </ul>
    </section>
  );
}

export const memoryScene: SceneDef = {
  duration: MEM_FLIPS[1] + 200,
  Body: ({ ms, lang }) => (
    <Stage
      ms={ms}
      end={MEM_FLIPS[1] + 200}
      lang={lang}
      render={(m) => {
        const claudeOn = MEM_FLIPS.filter((f) => m >= f).length % 2 === 1;
        return (
          <Dock ms={m} lang={lang} on="Memory">
            {m < MEM_LOAD ? (
              <LoadingLine ms={m} text={tr(lang, "Looking for notes…")} />
            ) : (
              <Glide ms={m} start={MEM_LOAD}>
                <div className="sp-mem">
                  <p className="sp-mem__lede">{tr(lang, "The chat is handed these notes after its rules. They never change what it may do.")}</p>
                  <Source title={tr(lang, "The project's own: {name}", { name: "notes-app" })} size={`202 B · ${tr(lang, "always on")}`} file="CLAUDE.md" on />
                  <Source icon="claude-code" title={`Claude Code · ${tr(lang, "its instructions for you")}`} size="2.6 KB" file="~/.claude/CLAUDE.md" switchFlips={{ ms: m, flips: MEM_FLIPS }} on={claudeOn} />
                  <Source icon="codex" title={`Codex · ${tr(lang, "its instructions for you")}`} size="2.6 KB" file="~/.codex/AGENTS.md" switchFlips={{ ms: m, flips: [] }} on={false} />
                </div>
              </Glide>
            )}
          </Dock>
        );
      }}
    />
  ),
};

/* ── Context: how full the window is, by part ──────────────────────────────────────────────────── */

// The figures of the run. The server has reported the use, so the figure is exact; the window is 262,144 tokens; the parts are what the app
// weighs from the text (conversation, tool calls and results) and what the server adds (its instructions, the tools, the memory).
const USED = 5820;
const MAX = 262144;
const AUTO_AT = 249036;
const PARTS: [string, string, number][] = [
  ["conversation", "Conversation", 907],
  ["tools", "Tool calls and results", 1139],
  ["system", "Instructions, tools and memory", 3774],
  ["free", "Free", 256324],
];
const fmt = (n: number) => n.toLocaleString("en-US");

export const contextScene: SceneDef = {
  // nothing on this tab moves once the panel is open: the scene is the panel opening, and then it rests
  duration: SIDE_IN + 40,
  hold: 14000,
  Body: ({ ms, lang }) => (
    <Stage
      ms={ms}
      end={SIDE_IN + 40}
      lang={lang}
      render={(m) => (
        <Dock ms={m} lang={lang} on="Context">
          <div>
            <div className="sp-ctx__head">
              <span className="sp-ctx__title">{tr(lang, "Context window")}</span>
              <span className="num sp-ctx__pct">{Math.round((USED / MAX) * 100)}%</span>
            </div>
            <p className="num sp-ctx__fig">{tr(lang, "{used} of {max} tokens", { used: fmt(USED), max: fmt(MAX) })}</p>
            <div className="sp-ctx__bar" aria-hidden="true">
              {PARTS.map(([k, , n]) => (
                <span key={k} data-part={k} style={{ flexGrow: n }} />
              ))}
            </div>
            <ul className="sp-ctx__list">
              {PARTS.map(([k, name, n]) => (
                <li key={k}>
                  <i data-part={k} />
                  <span>{tr(lang, name)}</span>
                  <span className="num">{fmt(n)}</span>
                </li>
              ))}
            </ul>
            <p className="sp-ctx__auto">{tr(lang, "Compacted by itself at {n} tokens ({pct}%).", { n: fmt(AUTO_AT), pct: Math.round((AUTO_AT / MAX) * 100) })}</p>
            <div className="sp-ctx__btn">
              <span className="sp-btn">{tr(lang, "Compact now")}</span>
            </div>
          </div>
        </Dock>
      )}
    />
  ),
};

/** The panel body on its own, for a caller that wants the frame of the side panel round something else. */
export function Panel({ children }: { children: ReactNode }) {
  return <div className="sp-dock">{children}</div>;
}
