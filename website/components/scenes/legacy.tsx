"use client";

import { type SceneDef } from "./Scene";
import { Ico, Line, Reply, StickBottom, stEase, type IcoName } from "./legacy-parts";

/**
 * The classic app (serve/web: index.html, app.js, app.css, components.css, tokens.css), as scenes. It is plain: nothing
 * eases in or slides, no part is reserved and faded up. What its script writes into the page appears in that frame:
 * app.js paints once per animation frame while a reply streams, and asks the server for /metrics once a second while the
 * page is open. The only moves of its own are the ones its stylesheet has: the thought's chevron turning (160 ms), the
 * Monitor's badges and bars (160 ms, `--st-ease`), and the blinking caret. English only (the classic has no translations).
 */

/* ── what the run said, as the classic shows it ──────────────────────────────────────────────────────────────── */
const PROMPT = 'Read notes.py, then add a --version flag that prints "notes-app 0.1.0" and exits. Keep the change small.';
const CODE = ["def main(argv: list[str]) -> int:", '    if argv[:1] == ["--version"]:', '        print("notes-app 0.1.0")', "        return 0", "    notes = load()", "    ..."].join("\n");
/** The answer as the model wrote it (markdown); the classic turns it into paragraphs and a block as it arrives. */
const SRC =
  "Added a `--version` branch at the top of `main`, before the notes are loaded, so it prints and exits without touching the database:\n\n```python\n" +
  CODE +
  "\n```\n\nRan it: `python notes.py --version` → `notes-app 0.1.0`, exit 0. `python notes.py list` still works (exit 0, no notes stored).";

/* ── the chat's clock ──
 * The real wait (a thought of 76.3 s) is cut down; what is kept is the order and the answer's own rate: the last request of the run
 * wrote 171 tokens at 64.2 tok/s (the Monitor's table), and the classic paints each token as it arrives. */
const TOKENS = 171;
const RATE = 64.2; // tok/s
const T_THINK = 700; // the first thought token: the thought block appears, open
const T_TEXT = 2600; // the first answer token: the thought closes and is named
const T_END = T_TEXT + Math.round((TOKENS / RATE) * 1000); // [DONE]: the caret goes, the copy button and the hint come back
const POLL0 = 300; // /metrics is asked every 1000 ms; the header pill follows it

/** The header: the brand, the three tabs, the status pill (written by each poll) and the theme button. */
function Bar({ tab, pill, state }: { tab: "chat" | "monitor"; pill: string; state: "idle" | "generating" }) {
  const tabs: [string, IcoName, "chat" | "monitor" | null][] = [
    ["Chat", "chat", "chat"],
    ["Monitor", "activity", "monitor"],
    ["About", "info", null],
  ];
  return (
    <div className="cl-header">
      <span className="cl-brand">Strata</span>
      <span className="cl-tabs">
        {tabs.map(([name, icon, id]) => (
          <span key={name} className={"cl-tab" + (id === tab ? " is-on" : "")}>
            <Ico n={icon} />
            {name}
          </span>
        ))}
      </span>
      <span className="cl-spacer" />
      <span className="cl-pill" data-state={state}>
        <i />
        <span className="cl-pill__t">{pill}</span>
      </span>
      <span className="cl-ibtn" aria-hidden="true">
        <Ico n="moon" size={18} />
      </span>
    </div>
  );
}

export const legacyChat: SceneDef = {
  duration: T_END + 150,
  Body: ({ ms }) => {
    const busy = ms < T_END;
    // the characters of the answer that have arrived: tokens at the run's rate, spread over what is written
    const tokens = ms < T_TEXT ? 0 : Math.min(TOKENS, Math.floor(((ms - T_TEXT) * RATE) / 1000) + 1);
    const n = Math.round((tokens * SRC.length) / TOKENS);
    // the status pill is rewritten by a poll, once a second
    const tick = ms < POLL0 ? -1 : POLL0 + Math.floor((ms - POLL0) / 1000) * 1000;
    const generating = tick >= T_THINK && tick < T_END;
    const open = ms < T_TEXT; // the thought is open while it streams, closed once the answer starts
    // the chevron: the one transition in the message (turns back when the thought closes)
    const turn = open ? 180 : 180 * (1 - stEase(ms, T_TEXT));

    return (
      <div className="l-app cl-app">
        <Bar tab="chat" state={generating ? "generating" : "idle"} pill={generating ? `Generating · ${RATE.toFixed(1)} tok/s` : "Idle"} />
        <div className="cl-view">
          <StickBottom className="cl-scroll">
            <div className="cl-chat">
              <div className="cl-msg cl-msg--user">
                <div className="cl-bubble">{PROMPT}</div>
                <div className="cl-meta">You · 03:16 PM</div>
              </div>
              <div className="cl-msg cl-msg--asst">
                {ms >= T_THINK && (
                  <div className="cl-think">
                    <div className="cl-think__sum">
                      <Ico n="thinking" size={16} />
                      <span>{open ? "Thinking…" : "Thought for 76.3 s"}</span>
                      <Ico n="chevron" size={16} style={{ marginLeft: "auto", transform: `rotate(${turn}deg)` }} />
                    </div>
                    {open && <div className="cl-think__body" />}
                  </div>
                )}
                <div className={"cl-bubble" + (tokens > 0 && busy ? " is-cursor" : "")}>
                  {ms >= T_THINK && tokens === 0 && <span className="cl-writing">Writing</span>}
                  {tokens > 0 && <Reply text={SRC.slice(0, n)} />}
                </div>
                <div className="cl-meta">
                  {!busy && (
                    <span className="cl-ibtn" aria-hidden="true">
                      <Ico n="copy" size={16} />
                    </span>
                  )}
                </div>
              </div>
            </div>
          </StickBottom>
          <div className="cl-dock">
            <div className="cl-composer">
              <div className="cl-ta">Ask anything…</div>
              <div className="cl-composer__bar">
                {(["attach", "newchat", "download", "settings"] as IcoName[]).map((i) => (
                  <span key={i} className="cl-ibtn" aria-hidden="true">
                    <Ico n={i} size={18} />
                  </span>
                ))}
                <span className="cl-spacer" />
                {!busy && <span className="cl-hint">Shift+Enter: new line</span>}
                {busy && (
                  <span className="cl-btn cl-btn--2">
                    <Ico n="stop" size={14} />
                    Stop
                  </span>
                )}
                <span className={"cl-btn cl-btn--send" + (busy ? " is-off" : "")} aria-hidden="true">
                  <Ico n="send" size={18} />
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  },
};

/* ── the Monitor tab ─────────────────────────────────────────────────────────────────────────────────────────────
 * The page's markup is there from the start with its dashes; the first answer of /metrics (the page polls on load) fills every
 * card, the table and the bars in one go. A second poll a second later writes the same values: nothing on screen moves. */
const T_POLL = 250;

type Spec = { icon: IcoName; label: string; value: string; unit?: string; sub?: string; series: number[]; tone?: "warn" | "info" };
// the figures are the ones this scene already had (the run's last request: 1,625 tokens read, 147 written at 49.9 tok/s, prefill 488);
// the curves are only the shape of a history line, not recorded values
const SERIES = {
  speed: [0.05, 0.05, 0.62, 0.7, 0.66, 0.05, 0.05, 0.6, 0.64, 0.05, 0.05, 0.05, 0.7, 0.66, 0.68, 0.05, 0.05, 0.62, 0.7, 0.66],
  prefill: [0.05, 0.05, 0.4, 0.5, 0.05, 0.05, 0.45, 0.05, 0.05, 0.05, 0.5, 0.5, 0.05, 0.05, 0.5, 0.05, 0.05, 0.4, 0.5, 0.05],
  gpu: [0.02, 0.02, 0.5, 0.6, 0.02, 0.02, 0.4, 0.02, 0.02, 0.02, 0.5, 0.55, 0.02, 0.02, 0.5, 0.02, 0.02, 0.4, 0.5, 0.02],
  vram: [0.8, 0.82, 0.86, 0.88, 0.88, 0.88, 0.89, 0.89, 0.89, 0.89, 0.89, 0.89, 0.89, 0.89, 0.89, 0.89, 0.89, 0.89, 0.89, 0.89],
  temp: [0.4, 0.42, 0.55, 0.6, 0.58, 0.5, 0.48, 0.55, 0.6, 0.55, 0.5, 0.48, 0.5, 0.58, 0.6, 0.55, 0.5, 0.5, 0.52, 0.5],
  power: [0.1, 0.1, 0.4, 0.5, 0.1, 0.1, 0.35, 0.1, 0.1, 0.1, 0.45, 0.5, 0.1, 0.1, 0.4, 0.1, 0.1, 0.35, 0.4, 0.1],
  pcie: [0.05, 0.05, 0.3, 0.4, 0.05, 0.05, 0.3, 0.05, 0.05, 0.05, 0.35, 0.4, 0.05, 0.05, 0.3, 0.05, 0.05, 0.3, 0.3, 0.05],
  cpu: [0.1, 0.15, 0.3, 0.25, 0.12, 0.1, 0.2, 0.22, 0.14, 0.12, 0.25, 0.3, 0.16, 0.14, 0.2, 0.18, 0.15, 0.2, 0.18, 0.18],
};
const CARDS: Spec[] = [
  { icon: "gpu", label: "GPU load", value: "1", unit: "%", sub: "GPU 0 1% · GPU 1 0%", series: SERIES.gpu },
  { icon: "layers", label: "VRAM", value: "24.9", unit: "/ 28 GB", sub: "GPU 0 10.2 GB · GPU 1 14.7 GB", series: SERIES.vram },
  { icon: "thermometer", label: "GPU temp", value: "57", unit: "°C", sub: "GPU 0 57° · GPU 1 38°", series: SERIES.temp, tone: "warn" },
  { icon: "bolt", label: "Power", value: "47", unit: "W", sub: "of 400 W limit", series: SERIES.power },
  { icon: "link", label: "PCIe", value: "Gen4", unit: "x16", sub: "to GPU 9.9 MB/s", series: SERIES.pcie, tone: "info" },
  { icon: "cpu", label: "CPU", value: "18", unit: "%", sub: "20 threads", series: SERIES.cpu },
  { icon: "disk", label: "Disk read", value: "–", sub: "needs psutil (setup installs it)", series: [], tone: "info" },
];

/** One of the eight cards; before the first poll it holds the page's own dashes. */
function Gauge({ loaded, spec }: { loaded: boolean; spec: Spec }) {
  return (
    <div className="cl-card cl-m">
      <div className="cl-m__l">
        <Ico n={spec.icon} />
        {spec.label}
      </div>
      <div className="cl-m__v">
        {loaded ? spec.value : "–"}
        {loaded && spec.unit && <small>{spec.unit}</small>}
      </div>
      <div className="cl-m__s">{loaded ? spec.sub : ""}</div>
      <Line values={loaded ? spec.series : undefined} tone={spec.tone} />
    </div>
  );
}

const REQS: [string, string, string, string, string, string, string, string][] = [
  ["03:21:28 PM", "Done", "1,625", "0", "147", "49.9", "70.7%", "6.3 s"],
  ["03:21:00 PM", "Max tokens", "73", "0", "900", "59.3", "72.4%", "16.6 s"],
  ["03:20:31 PM", "Max tokens", "84", "0", "700", "53.8", "70.1%", "14.9 s"],
  ["03:17:57 PM", "Done", "4,773", "4,742", "171", "64.2", "69.5%", "3.3 s"],
  ["03:16:51 PM", "Done", "4,668", "4,439", "74", "72.9", "73.9%", "4.4 s"],
];

/** The page's `.st-progress`: the bar's width follows its figure over `--st-dur`. */
function Progress({ pct, tone }: { pct: number; tone?: "warn" }) {
  return (
    <div className="cl-progress">
      <div className={"cl-progress__bar" + (tone ? " cl-progress__bar--" + tone : "")} style={{ width: `${pct}%` }} />
    </div>
  );
}

const BADGES: [string, string][] = [
  ["Idle", ""],
  ["Reading", " cl-badge--info"],
  ["Generating", " cl-badge--ok"],
  ["Queued", " cl-badge--warn"],
  ["Error", " cl-badge--err"],
];

export const legacyMonitor: SceneDef = {
  duration: T_POLL + 200,
  hold: 7000,
  Body: ({ ms }) => {
    const loaded = ms >= T_POLL;
    const move = stEase(ms, T_POLL); // 160 ms: the bars and the badge that light up with the first poll
    return (
      <div className="l-app cl-app">
        <Bar tab="monitor" state="idle" pill={loaded ? "Idle" : "Connecting…"} />
        <div className="cl-view">
          <div className="cl-mon">
            <div className="cl-card cl-state">
              <div className="cl-state__h">
                <span className="cl-title">Model state</span>
                <span className="cl-badges">
                  {BADGES.map(([name, cls], k) => (
                    <span key={name} className={"cl-badge" + cls} style={{ opacity: k === 0 ? 0.38 + 0.62 * move : 0.38 }}>
                      {name}
                    </span>
                  ))}
                </span>
              </div>
              <div className="cl-state__row">
                <span>Waiting for a request</span>
                <span className="cl-muted">{loaded ? "last: 147 tokens at 49.9 tok/s" : ""}</span>
              </div>
              <Progress pct={0} />
            </div>

            <div className="cl-metrics">
              <div className="cl-card cl-m">
                <div className="cl-m__l">
                  <Ico n="gauge" />
                  Speed
                </div>
                <div className="cl-speed">
                  <div>
                    <div className="cl-m__v">
                      {loaded ? "49.9" : "-"}
                      {loaded && <small>t/s</small>}
                    </div>
                    <div className="cl-m__s">{loaded ? "Decode last request" : "Decode"}</div>
                  </div>
                  <div>
                    <div className="cl-m__v">
                      {loaded ? "488" : "-"}
                      {loaded && <small>t/s</small>}
                    </div>
                    <div className="cl-m__s">{loaded ? "Prefill last request" : "Prefill"}</div>
                  </div>
                </div>
                <Line values={loaded ? SERIES.speed : undefined} also={loaded ? SERIES.prefill : undefined} />
              </div>
              {CARDS.map((c) => (
                <Gauge key={c.label} loaded={loaded} spec={c} />
              ))}
            </div>

            <div className="cl-row">
              <div className="cl-card cl-ctx">
                <span className="cl-title">Context fill</span>
                <div className="cl-gauge">
                  <svg viewBox="0 0 120 120" aria-hidden="true">
                    <circle className="cl-gauge__track" cx="60" cy="60" r="50" fill="none" strokeWidth="10" strokeLinecap="round" strokeDasharray="235.6 314.2" transform="rotate(135 60 60)" />
                    {/* 1,772 of 256K is under 1 %: the arc would be only its round cap, so the page hides it (opacity 0) */}
                    <circle className="cl-gauge__fill" cx="60" cy="60" r="50" fill="none" strokeWidth="10" strokeLinecap="round" strokeDasharray="0 314.2" transform="rotate(135 60 60)" style={{ opacity: 0 }} />
                  </svg>
                  <div className="cl-gauge__label">
                    <span className="cl-gauge__pct">{loaded ? "1%" : "0%"}</span>
                    <span className="cl-muted">{loaded ? "1.8k / 256K" : "–"}</span>
                  </div>
                </div>
                <div className="cl-bar-row">
                  <span>Experts in VRAM</span>
                  <span className="cl-muted">–</span>
                </div>
                <Progress pct={0} />
                <div className="cl-bar-row">
                  <span>System RAM</span>
                  <span className="cl-muted">{loaded ? "41.2 / 48 GB" : "–"}</span>
                </div>
                <Progress pct={loaded ? (41.2 / 47.7) * 100 * move : 0} />
                <div className="cl-bar-row">
                  <span>GPU temperature</span>
                  <span className="cl-muted">{loaded ? "57 °C" : "–"}</span>
                </div>
                <Progress pct={loaded ? 57 * move : 0} tone="warn" />
              </div>

              <div className="cl-card cl-req">
                <div className="cl-req__h">
                  <span className="cl-title">Recent requests</span>
                </div>
                <table className="cl-table">
                  <thead>
                    <tr>
                      {["Time", "Status", "Prompt", "Reused", "Output", "Tok/s", "Hit rate", "Duration"].map((h, k) => (
                        <th key={h} className={k > 1 ? "num" : undefined}>
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {loaded ? (
                      REQS.map((r) => (
                        <tr key={r[0]}>
                          {r.map((c, k) => (
                            <td key={k} className={k > 1 ? "num" : undefined}>
                              {k === 1 ? <span className="cl-badge">{c}</span> : c}
                            </td>
                          ))}
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan={8} className="cl-muted">
                          No requests yet
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  },
};
