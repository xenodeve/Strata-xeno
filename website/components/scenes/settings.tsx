"use client";

import type { CSSProperties, ReactNode } from "react";
import type { BotAvatarType } from "@/vendor/bot-avatars/index.es.js";
import { lerp, type SceneDef } from "./Scene";
import { Enter, Lattice, Matrix } from "./parts";
import { Orb } from "./kit";
import { at, BotMark, Caret, EASE_TW, Fold, Glide, Ico, press, Stage, tr, typed, type IconName } from "./panels-parts";

/**
 * Settings and dialogs of the app, as scenes: Settings › Permissions (pages/Settings.tsx, PermissionsSettings.tsx, BranchedMenu.tsx), the
 * "New project" dialog (NewProjectDialog.tsx, FoldersEditor.tsx), and the Marks picker (App.tsx AvatarMenu, StatusMarks.tsx). The words are
 * the app's own, from one real run; what moves is what moves in the app (the comments say which file each number is from).
 */

type Lang = "en" | "th";

/** A mix of a colour with the app's `line`, for a border that changes: `t` 0..1. */
const mixTo = (color: string, t: number) => `color-mix(in srgb, ${color} ${(t * 100).toFixed(1)}%, var(--sp-line))`;

/** 0..1: how much a field has the focus at `ms`, for the stretches of the clock it holds it (the border takes 150 ms to change: `transition-colors`). */
function focus(ms: number, ranges: [number, number][]): number {
  let f = 0;
  for (const [s, e] of ranges) f = Math.max(f, at(ms, s, 150, EASE_TW) * (1 - at(ms, e, 150, EASE_TW)));
  return f;
}
const inRange = (ms: number, ranges: [number, number][]) => ranges.some(([s, e]) => ms >= s && ms < e);

/* ── Settings › Permissions ─────────────────────────────────────────────────────────────────────── */

// The menu of the Settings page (BranchedMenu): sections with their topics, every section open.
const MENU: { head: string; items: [string, IconName][] }[] = [
  { head: "General", items: [["Status marks", "paint"], ["API key", "key"], ["Coding tools", "terminal"], ["Permissions", "shield"], ["Hooks", "hook"], ["Web access", "globe"], ["Sub-agents", "group"]] },
  { head: "MCP tools", items: [["Servers", "server"], ["Limits", "plug"]] },
  { head: "Import", items: [["Skills", "book"], ["MCP servers", "download"], ["Memory", "brain"]] },
];

// BranchedMenu's geometry: the rows are 36 px, the branches leave a trunk 14 px in, turn with a radius of 10 and end 8 px short of the labels.
const PAD = 6;
const ROW = 36;
const INDENT = 40;
const TRUNK = 14;
const R = 10;
const END = INDENT - 8;
const rowY = (k: number) => PAD + k * ROW + ROW / 2;
const branch = (k: number) => `M ${TRUNK} ${rowY(k) - R} A ${R} ${R} 0 0 0 ${TRUNK + R} ${rowY(k)} H ${END}`;
const reach = (k: number) => `M ${TRUNK} 0 V ${rowY(k) - R} A ${R} ${R} 0 0 0 ${TRUNK + R} ${rowY(k)} H ${END}`;

/** The menu as it stands once the page is open: the line is drawn to the topic in use, the marker sits beside its section (neither moves on the first showing). */
function Menu({ lang, active }: { lang: Lang; active: string }) {
  return (
    <nav className="sp-bm" aria-hidden="true">
      <span className="sp-bm__mark" />
      {MENU.map((s) => {
        const n = s.items.length;
        const h = PAD * 2 + n * ROW;
        const length = (k: number) => rowY(k) - R + (Math.PI * R) / 2 + (END - TRUNK - R);
        return (
          <div key={s.head} className="sp-bm__sec">
            <div className="sp-bm__head">{tr(lang, s.head)}</div>
            <div className="sp-bm__tree" style={{ height: h }}>
              <svg className="sp-bm__lines" width={INDENT} height={h} style={{ width: INDENT, height: h, maxWidth: "none" }}>
                <path className="sp-bm__base" d={`M ${TRUNK} 0 V ${rowY(n - 1) - R}`} />
                {s.items.map((_, k) => (
                  <path key={k} className="sp-bm__base" d={branch(k)} />
                ))}
                {s.items.map(([name], k) => (
                  <path key={name} className="sp-bm__reach" d={reach(k)} style={{ strokeDasharray: length(k), strokeDashoffset: name === active ? 0 : length(k) }} />
                ))}
              </svg>
              {s.items.map(([name, icon]) => (
                <div key={name} className={"sp-bm__item" + (name === active ? " is-on" : "")}>
                  <Ico n={icon} size={16} />
                  <span>{tr(lang, name)}</span>
                </div>
              ))}
            </div>
          </div>
        );
      })}
    </nav>
  );
}

/** The Settings page: its title and line rise (420 ms, the second 50 ms after the first), the menu stands beside the topic, and the topic comes in with `panel-in`. */
function SettingsPage({ ms, lang, active, children }: { ms: number; lang: Lang; active: string; children: ReactNode }) {
  return (
    <div className="sp-page">
      <Enter ms={ms} start={0} kind="rise" className="sp-page__head">
        <h1>{tr(lang, "Settings")}</h1>
        <p>{tr(lang, "How the app looks and what it connects to. The tools the model may use are set here too.")}</p>
      </Enter>
      <Enter ms={ms} start={0} delay={50} kind="rise" className="sp-page__row">
        <aside className="sp-page__menu">
          <Menu lang={lang} active={active} />
        </aside>
        <Enter ms={ms} start={0} kind="panel" className="sp-page__topic">
          {children}
        </Enter>
      </Enter>
    </div>
  );
}

const RULES = [
  { text: "Bash(npm test:*)", type: 1500, add: 3400 },
  { text: "Read(src/**)", type: 4500, add: 6100 },
];

/** The two ways to say what a rule does; "Allow" is the one in use (the pill that glides to the other is not drawn: nobody switches it here). */
function Segmented({ lang }: { lang: Lang }) {
  return (
    <div className="sp-seg" aria-hidden="true">
      <span className="is-on">{tr(lang, "Allow")}</span>
      <span>{tr(lang, "Never")}</span>
    </div>
  );
}

function ScopeBlock({ ms, lang, title, note, live }: { ms: number; lang: Lang; title: string; note: string; live: boolean }) {
  const added = live ? RULES.filter((r) => ms >= r.add) : [];
  // what is typed in the field now: the rule being written, from its first key to the moment it is added
  const cur = live ? RULES.find((r) => ms >= r.type && ms < r.add) : undefined;
  const draft = cur ? cur.text.slice(0, typed(cur.text, ms, cur.type)) : "";
  const focused = live && ms >= RULES[0].type - 100;
  const f = live ? focus(ms, [[RULES[0].type - 100, Infinity]]) : 0;
  const lastAdd = live ? [...RULES].reverse().find((r) => ms >= r.add) : undefined;
  return (
    <section className="sp-scope">
      <div className="sp-scope__h">
        <h3>{title}</h3>
        <span>{note}</span>
      </div>
      <div className="sp-scope__cols">
        <div>
          <div className="sp-scope__l">{tr(lang, "Allowed")}</div>
          {added.length === 0 ? (
            <p className="sp-scope__none">{tr(lang, "Nothing is allowed here for good.")}</p>
          ) : (
            <ul className="sp-rules">
              {added.map((r) => (
                <li key={r.text}>
                  <span>{r.text}</span>
                  <Ico n="close" size={13} />
                </li>
              ))}
            </ul>
          )}
        </div>
        <div>
          <div className="sp-scope__l">{tr(lang, "Never")}</div>
          <p className="sp-scope__none">{tr(lang, "Nothing is refused here for good.")}</p>
        </div>
      </div>
      <div className="sp-form">
        <Segmented lang={lang} />
        <div className="sp-in sp-in--mono" style={{ borderColor: mixTo("var(--a-accent)", f) }}>
          {draft ? (
            <>
              {draft}
              <Caret />
            </>
          ) : (
            <>
              {focused && <Caret />}
              <span className="sp-ph">Bash(npm test:*)</span>
            </>
          )}
        </div>
        <span className={"sp-btn" + (draft ? "" : " is-off")} style={{ transform: lastAdd ? `scale(${press(ms, lastAdd.add)})` : undefined }}>
          {tr(lang, "Add")}
        </span>
      </div>
    </section>
  );
}

const PERM_END = RULES[1].add + 300;

export const permissionsScene: SceneDef = {
  duration: PERM_END,
  Body: ({ ms, lang }) => (
    <Stage
      ms={ms}
      end={PERM_END}
      lang={lang}
      render={(m) => (
        <SettingsPage ms={m} lang={lang} active="Permissions">
          <section>
            <h2 className="sp-h2">{tr(lang, "Permissions")}</h2>
            <p className="sp-p">{tr(lang, "What the chat's coding tools may do without asking, for good. A card that asks can keep your answer here (More choices), or you can write a rule. \"Allow for this chat\" is kept with the chat only and is not listed.")}</p>
            <p className="sp-p sp-p--2">{tr(lang, "A rule is a tool and, between brackets, what it is limited to: Bash(npm test:*) for commands that start with npm test, Read(src/**) for files under src, Edit(docs/**), or just Read for all of that tool. \"Never\" wins over \"Allow\". No rule can allow a secret, a change in .git or a command that can do harm that is hard to undo: those ask every time.")}</p>
            <div className="sp-scopes">
              <ScopeBlock ms={m} lang={lang} title={tr(lang, "Everywhere")} note={tr(lang, "every chat, every project")} live />
              <ScopeBlock ms={m} lang={lang} title="notes-app" note={tr(lang, "this project")} live={false} />
            </div>
          </section>
        </SettingsPage>
      )}
    />
  ),
};

/* ── the "New project" dialog: more than one folder ─────────────────────────────────────────────── */

// The dialog is opened, then written: the name, a folder, a word more on the name, the second folder. Times are in ms.
const P = {
  nameType: 500,
  nameEnter: 1500,
  path1Type: 1800,
  addClick: 3500,
  added1: 3620,
  nameAgain: 4300,
  nameMore: 4450,
  path2Click: 6000,
  path2Type: 6150,
  enter2: 8200,
  added2: 8320,
  create: 9300,
};
const NAME = "notes-app";
const MORE = " (2 worktrees)";
const PATH1 = "C:\\demo\\notes-app";
const PATH2 = "C:\\demo\\notes-app-wt";
const PROJECT_END = P.create + 400;
const NAME_FOCUS: [number, number][] = [[16, P.nameEnter], [P.nameAgain, P.path2Click]];
const PATH_FOCUS: [number, number][] = [[P.nameEnter, P.addClick], [P.path2Click, Infinity]];

/** A folder of the project: its path with the end (the name) in full ink and the folders above it quiet. */
function FolderRow({ path, main, other, lang }: { path: string; main: boolean; other: boolean; lang: Lang }) {
  const m = /^(.*[\\/])([^\\/]+)$/.exec(path);
  return (
    <div className="sp-folder">
      <Ico n="folder" size={14} />
      <span className="sp-folder__p">
        {m && <span className="sp-folder__d">{m[1]}</span>}
        <b>{m ? m[2] : path}</b>
      </span>
      {main && <span className="sp-tag">{tr(lang, "Main")}</span>}
      {other && <span className="sp-mk">{tr(lang, "Make main")}</span>}
      <span className="sp-x">
        <Ico n="close" size={13} />
      </span>
    </div>
  );
}

function ProjectDialog({ ms, lang }: { ms: number; lang: Lang }) {
  // the name: "notes-app", then a word more
  const name = NAME.slice(0, typed(NAME, ms, P.nameType)) + (ms >= P.nameMore ? MORE.slice(0, typed(MORE, ms, P.nameMore)) : "");
  // the folder field: the first path, which is added; later the second one
  let text = "";
  if (ms >= P.path1Type && ms < P.added1) text = ms >= P.addClick ? PATH1 : PATH1.slice(0, typed(PATH1, ms, P.path1Type));
  else if (ms >= P.path2Type && ms < P.added2) text = ms >= P.enter2 ? PATH2 : PATH2.slice(0, typed(PATH2, ms, P.path2Type));
  // while the server checks a folder the Add button waits
  const checking = (ms >= P.addClick && ms < P.added1) || (ms >= P.enter2 && ms < P.added2);
  const folders = ms >= P.added2 ? 2 : ms >= P.added1 ? 1 : 0;
  const ready = name.length > 0 && (folders > 0 || text.length > 0);
  const nameF = focus(ms, NAME_FOCUS);
  const pathF = focus(ms, PATH_FOCUS);
  // the dialog comes in over 200 ms with the app's ease (translateY 8 px, scale .98), the backdrop (black at 45 %) fades beside it
  const enter = at(ms, 16, 200);
  const veil = at(ms, 16, 200, EASE_TW) * 0.45;
  return (
    <div className="sp-dlgbg" style={{ "--veil": veil } as CSSProperties}>
      <div className="sp-dlg" style={{ opacity: enter, transform: `translateY(${lerp(8, 0, enter)}px) scale(${lerp(0.98, 1, enter)})` }}>
        <h2>{tr(lang, "New project")}</h2>
        <p className="sp-dlg__p">{tr(lang, "A project is one or more folders on this PC, such as the worktrees of one repository. The coding tools of its chats work in them: files inside are free to read and change, everything else asks you first. The first folder is the main one: commands run there.")}</p>
        <div className="sp-dlg__l">{tr(lang, "Project name")}</div>
        <div className="sp-in" style={{ borderColor: mixTo("var(--a-accent)", nameF) }}>
          {name}
          {inRange(ms, NAME_FOCUS) && <Caret />}
        </div>
        <div className="sp-dlg__f">
          <div className="sp-dlg__l sp-dlg__l--f">{tr(lang, "Folders")}</div>
          {folders > 0 && (
            <div className="sp-folders-wrap">
              <Glide ms={ms} start={P.added1}>
                <ul className="sp-folders">
                  <li>
                    <FolderRow path={PATH1} main={folders > 1} other={false} lang={lang} />
                  </li>
                  {folders > 1 && (
                    <li style={{ marginTop: 4 }}>
                      <Glide ms={ms} start={P.added2}>
                        <FolderRow path={PATH2} main={false} other lang={lang} />
                      </Glide>
                    </li>
                  )}
                </ul>
              </Glide>
            </div>
          )}
          <div className="sp-addrow">
            <div className="sp-in sp-in--mono" style={{ borderColor: mixTo("var(--a-accent)", pathF) }}>
              {text}
              {inRange(ms, PATH_FOCUS) && <Caret />}
              {!text && <span className="sp-ph">C:/work/my-project</span>}
            </div>
            <span className="sp-btn">{tr(lang, "Browse")}</span>
            <span className={"sp-btn" + (text && !checking ? "" : " is-off")} style={{ transform: ms >= P.addClick && ms < P.addClick + 300 ? `scale(${press(ms, P.addClick)})` : undefined }}>
              {tr(lang, "Add")}
            </span>
          </div>
        </div>
        <div className="sp-dlg__b">
          <span className="sp-btn sp-btn--quiet">{tr(lang, "Cancel")}</span>
          <span className={"sp-btn sp-btn--ink" + (ready ? "" : " is-off")} style={{ transform: ms >= P.create ? `scale(${press(ms, P.create)})` : undefined }}>
            {tr(lang, "Create project")}
          </span>
        </div>
      </div>
    </div>
  );
}

export const projectScene: SceneDef = {
  duration: PROJECT_END,
  Body: ({ ms, lang }) => <Stage ms={ms} end={PROJECT_END} lang={lang} center render={(m) => <ProjectDialog ms={m} lang={lang} />} />,
};

/* ── Marks: how the app shows that something is happening, and the eighteen avatars ──────────────── */

type Kind = "orbs" | "mixed" | "loading" | "bots";
const OPTIONS: { kind: Kind; name: string; text: string }[] = [
  { kind: "orbs", name: "Orbs", text: "A dotted ball that takes a form for each status." },
  { kind: "mixed", name: "Orbs + Loading", text: "Orbs, with a lattice of dots beside the thinking. How it starts." },
  { kind: "loading", name: "Loading only", text: "Loaders of dots, a pattern for each status. No orbs." },
  { kind: "bots", name: "Avatar", text: "A small character with a shape and a mood of its own." },
];
// the eighteen shapes of the app, in its own order
const BOT_TYPES: BotAvatarType[] = ["clover", "flower", "triangle", "square", "blob", "ghost", "circle", "drop", "star", "droid", "mech", "alien", "hexagon", "cat", "cloud", "pill", "pebble", "puddle"];
// what the three marks of a row stand for: thinking, answering, an idle server (solving, composing, searching)
const SAMPLE = ["solving", "composing", "searching"] as const;
// the avatar that stands for each of them when the choice is "by status" (lib/avatar.ts)
const AUTO: Record<(typeof SAMPLE)[number], BotAvatarType> = { solving: "cat", composing: "pill", searching: "droid" };

const HOVER = 1300; // the pointer comes to the Avatar row: its marks are loaded and show
const PICK = 1800; // and the row is chosen
// the choice walks along the shapes
const WALK: [number, BotAvatarType][] = [
  [3600, "ghost"],
  [5000, "star"],
  [6400, "droid"],
  [7800, "mech"],
];
const MARKS_END = 7800 + 400;

/** Three marks of a way: what it shows for thinking, answering and an idle server. The Avatar row shows dotted rings until it is chosen or looked at. */
function Preview({ ms, kind, shown, pick, ghost }: { ms: number; kind: Kind; shown: boolean; pick: BotAvatarType | null; ghost: boolean }) {
  if (kind === "bots") {
    if (!shown) return <>{SAMPLE.map((d) => <span key={d} className="sp-ring" />)}</>;
    return (
      <>
        {SAMPLE.map((d) => (
          <Enter key={d} ms={ms} start={HOVER} kind="orb" className="sp-mark">
            {ghost ? <span style={{ display: "block", width: 20, height: 20 }} /> : <BotMark type={pick ?? AUTO[d]} size={20} face="eyes" state={d === "searching" ? "default" : "working"} />}
          </Enter>
        ))}
      </>
    );
  }
  return (
    <>
      {SAMPLE.map((d, i) => {
        if (kind === "mixed" && i === 0) return <Lattice key={d} status="working" pattern="orbit" />;
        if (kind === "loading") {
          if (d === "solving") return <span key={d} className="sp-mark"><Matrix variant="twinkle" px={18} /></span>;
          return <span key={d} className="sp-mark"><Lattice pattern={d === "composing" ? "arrow" : "ripple"} status="working" cell={4} gap={1.4} /></span>;
        }
        return (
          <span key={d} className="sp-mark">
            <Orb ms={ms} live size={20} design={d} />
          </span>
        );
      })}
    </>
  );
}

/** How much `key` is the choice of the picker at `ms`: it goes to 1 (or back to 0) over the 150 ms of `transition-colors` after the choice changes. */
function chosen(ms: number, key: string): number {
  let idx = -1;
  WALK.forEach(([w], i) => {
    if (ms >= w) idx = i;
  });
  const now = idx < 0 ? "auto" : WALK[idx][1];
  const was = idx <= 0 ? "auto" : WALK[idx - 1][1];
  const e = idx < 0 ? 1 : at(ms, WALK[idx][0], 150, EASE_TW);
  return (key === now ? e : 0) + (key === was && key !== now ? 1 - e : 0);
}

function MarksPicker({ ms, lang, ghost }: { ms: number; lang: Lang; ghost: boolean }) {
  const cur = [...WALK].reverse().find(([t]) => ms >= t) ?? null;
  const picked = ms >= PICK;
  const hovered = ms >= HOVER;
  const t1 = at(ms, HOVER, 150, EASE_TW); // the row is looked at: `hover:bg-hover`
  const t2 = at(ms, PICK, 150, EASE_TW); // and chosen: `border-ink bg-fill`
  return (
    <div className="sp-pop-in">
      <p className="sp-pop__lede">{tr(lang, "What shows that something is happening: thinking, answering, reading.")}</p>
      <div className="sp-opts" role="presentation">
        {OPTIONS.map((o) => {
          // the selected look of a row: the way in use (the one it starts with, then Avatar)
          const sel = o.kind === "mixed" ? 1 - t2 : o.kind === "bots" ? t2 : 0;
          const wash = o.kind === "bots" ? t1 * (1 - t2) * 0.08 + t2 * 0.12 : o.kind === "mixed" ? sel * 0.12 : 0;
          const style: CSSProperties = { borderColor: `color-mix(in srgb, var(--a-ink) ${(11 + 89 * sel).toFixed(1)}%, transparent)`, background: `color-mix(in srgb, var(--a-ink) ${(wash * 100).toFixed(1)}%, transparent)` };
          const on = o.kind === "mixed" ? !picked : o.kind === "bots" ? picked : false;
          return (
            <div key={o.kind} className="sp-opt" style={style}>
              <span className="sp-opt__marks">
                <Preview ms={ms} kind={o.kind} shown={o.kind === "bots" ? hovered : true} pick={cur ? cur[1] : null} ghost={ghost} />
              </span>
              <span className="sp-opt__t">
                <b>
                  {tr(lang, o.name)}
                  {o.kind === "mixed" && <em>{tr(lang, "Default")}</em>}
                </b>
                <small>{tr(lang, o.text)}</small>
              </span>
              <span className={"sp-radio" + (on ? " is-on" : "")} />
            </div>
          );
        })}
      </div>
      <Fold ms={ms} open={PICK + 16}>
        <div className="sp-which">
          <p>{tr(lang, "Which avatar: one for every status, a shape for each status, or a random one for each place.")}</p>
          <div className="sp-tiles">
            <TileText label={tr(lang, "By status")} s={chosen(ms, "auto")} />
            <TileText label={tr(lang, "Random")} s={0} />
            {BOT_TYPES.map((b) => (
              <span key={b} className="sp-tile" style={tileStyle(chosen(ms, b))}>
                {ghost ? <span style={{ display: "block", width: 44, height: 44 }} /> : <BotMark type={b} size={44} />}
              </span>
            ))}
          </div>
        </div>
      </Fold>
    </div>
  );
}

const tileStyle = (s: number): CSSProperties => ({ borderColor: `color-mix(in srgb, var(--a-ink) ${(11 + 89 * s).toFixed(1)}%, transparent)`, background: `color-mix(in srgb, var(--a-ink) ${(s * 12).toFixed(1)}%, transparent)` });

function TileText({ label, s }: { label: string; s: number }) {
  return (
    <span className="sp-tile sp-tile--t" style={tileStyle(s)}>
      {label}
    </span>
  );
}

export const marksScene: SceneDef = {
  duration: MARKS_END,
  Body: ({ ms, lang }) => (
    <Stage
      ms={ms}
      end={MARKS_END}
      lang={lang}
      render={(m, ghost) => (
        <div className="sp-popwrap">
          <Enter ms={m} start={0} kind="toast" className="sp-pop">
            <MarksPicker ms={m} lang={lang} ghost={ghost} />
          </Enter>
        </div>
      )}
    />
  ),
};
