"use client";

import { useEffect, useRef, useState, type CSSProperties } from "react";
import { T, useT, type L } from "@/lib/i18n";
import { usePrefersReducedMotion } from "../Avatar";
import PixelSwap from "../PixelSwap";
import { AppScene, type SceneId } from "../scenes";
import { HighlightedText, SlideUpText } from "../text/Fx";

type SceneRef = { id: SceneId; alt: L };
/** A callout on the new screen: the part of the scene it points at (a selector) and what it says. */
type Pin = { sel: string; nth?: number; label: L };
type Pair = { id: string; title: string; tab: L; note: L; old: SceneRef; next: SceneRef; pins: Pin[] };

const PAIRS: Pair[] = [
  {
    id: "chat",
    title: "Strata · Chat",
    tab: { en: "Chat", th: "แชต" },
    note: {
      en: "The same conversation. Before: plain text. After: a card for every tool call.",
      th: "บทสนทนาเดียวกัน ก่อน: ข้อความธรรมดา หลัง: การ์ดสำหรับทุกการเรียกเครื่องมือ",
    },
    old: {
      id: "legacy-chat",
      alt: {
        en: "Before: the classic app's chat shows the conversation as plain text, with the code in a dark block.",
        th: "ก่อน: แชตของแอปคลาสสิกแสดงบทสนทนาเป็นข้อความธรรมดา โค้ดอยู่ในบล็อกสีเข้ม",
      },
    },
    next: {
      id: "chat-run",
      alt: {
        en: "After: the new app's chat runs the same conversation with a card for each tool call, a question before the command, the answer and a projects sidebar.",
        th: "หลัง: แชตของแอปใหม่รันบทสนทนาเดียวกันพร้อมการ์ดสำหรับทุกการเรียกเครื่องมือ คำถามก่อนรันคำสั่ง คำตอบ และแถบโปรเจกต์",
      },
    },
    pins: [
      { sel: ".a-side__p", label: { en: "Projects in the sidebar", th: "โปรเจกต์ในแถบข้าง" } },
      { sel: ".a-card", nth: 1, label: { en: "A card for every tool call", th: "การ์ดสำหรับทุกการเรียกเครื่องมือ" } },
      { sel: ".a-ask__q", label: { en: "It asks first", th: "มันถามก่อน" } },
      { sel: ".a-stats", label: { en: "Speed under every reply", th: "ความเร็วใต้ทุกคำตอบ" } },
    ],
  },
  {
    id: "monitor",
    title: "Strata · Dashboard",
    tab: { en: "Monitor", th: "Monitor" },
    note: {
      en: "Before: one page of gauges. After: Dashboard, Live, Requests and Hardware.",
      th: "ก่อน: หน้าเดียวเต็มไปด้วยเกจ หลัง: Dashboard, Live, Requests และ Hardware",
    },
    old: {
      id: "legacy-monitor",
      alt: { en: "Before: the classic app's Monitor tab with gauges and a request table.", th: "ก่อน: แท็บ Monitor ของแอปคลาสสิกพร้อมเกจและตารางคำขอ" },
    },
    next: {
      id: "dashboard",
      alt: {
        en: "After: the new app's Dashboard with speed, hardware, requests and where the experts ran.",
        th: "หลัง: Dashboard ของแอปใหม่พร้อมความเร็ว ฮาร์ดแวร์ คำขอ และที่ที่ expert ทำงาน",
      },
    },
    pins: [
      { sel: ".m-head__h", label: { en: "Ready, at a glance", th: "สถานะ ดูได้ทันที" } },
      { sel: ".a-spark", label: { en: "Speed, charted", th: "ความเร็ว เป็นกราฟ" } },
      { sel: ".m-gpu", label: { en: "Each part of the hardware", th: "ฮาร์ดแวร์ทีละส่วน" } },
      { sel: ".m-stack", label: { en: "Where the experts ran", th: "ที่ที่ expert ทำงาน" } },
    ],
  },
];

const clamp = (x: number, a = 0, b = 1) => Math.min(b, Math.max(a, x));

/** The opacity a part really has: its own and everything around it. */
function seen(el: Element, root: Element): number {
  let o = 1;
  for (let n: Element | null = el; n && n !== root; n = n.parentElement) o *= Number(getComputedStyle(n).opacity);
  return o;
}

type Spot = { x: number; y: number; show: boolean; flip: boolean };
type Lean = (e: React.PointerEvent<HTMLElement>) => void;

/**
 * One pair: an app window (title bar and a Classic / New rail), the classic scene and the new scene on a pixel swap,
 * a line of light that crosses with the pixels, and callouts. A callout is measured from the part of the new scene it
 * points at and shows once that part has landed, so it never points at an empty place.
 */
function PairView({ p, on, swapped, run, reduced, onSwap, lean, level, t }: { p: Pair; on: boolean; swapped: boolean; run: number; reduced: boolean; onSwap: (v: boolean) => void; lean: Lean; level: Lean; t: ReturnType<typeof useT> }) {
  const stage = useRef<HTMLDivElement>(null);
  const [spots, setSpots] = useState<Spot[]>([]);

  useEffect(() => {
    const el = stage.current;
    if (!el || reduced) return;
    if (!on || !swapped) {
      setSpots((s) => (s.some((x) => x.show) ? s.map((x) => ({ ...x, show: false })) : s));
      return;
    }
    const measure = () => {
      const box = el.getBoundingClientRect();
      const layer = el.querySelector(".pixel-swap__layer:nth-child(2)");
      if (!layer || box.width < 10) return;
      const next: Spot[] = p.pins.map((pin) => {
        const target = layer.querySelectorAll(pin.sel)[pin.nth ?? 0];
        if (!target) return { x: 0, y: 0, show: false, flip: false };
        const r = target.getBoundingClientRect();
        const x = r.right - box.left + 12;
        const y = r.top + r.height / 2 - box.top;
        const visible = seen(target, layer) > 0.96 && r.width > 0 && y > 64 && y < box.height - 24;
        return { x: Math.round(x), y: Math.round(y), show: visible, flip: x > box.width * 0.6 };
      });
      setSpots((prev) => (prev.length === next.length && prev.every((s, i) => s.x === next[i].x && s.y === next[i].y && s.show === next[i].show) ? prev : next));
    };
    measure();
    const id = window.setInterval(measure, 220);
    return () => window.clearInterval(id);
  }, [on, swapped, reduced, p.pins]);

  const classic = t({ en: "Classic", th: "คลาสสิก" });
  const fresh = t({ en: "New", th: "ใหม่" });
  const face = (s: SceneRef, isNew: boolean) => (
    <figure className="sc__face">
      <AppScene id={s.id} alt={s.alt} active={on && (isNew ? swapped : !swapped)} replay={false} className="sc__scene" />
    </figure>
  );

  return (
    <div className="sc__tilt" onPointerMove={lean} onPointerLeave={level}>
      <div className="sc__window">
        <div className="sc__chrome" aria-hidden="true">
          <i />
          <i />
          <i />
          <span className="sc__title mono">{p.title}</span>
          <span className="sc__rail mono" data-new={swapped || undefined}>
            <b>{classic}</b>
            <u />
            <b>{fresh}</b>
          </span>
        </div>
        <div className="sc__stage" ref={stage}>
          {reduced ? (
            <div className="sc__stack">
              {face(p.old, false)}
              {face(p.next, true)}
            </div>
          ) : (
            <>
              <PixelSwap
                className="sc__swap"
                aspectRatio="auto"
                trigger="click"
                pattern="left-to-right"
                randomness={0.18}
                pixelSize={74}
                pixelScale={0.3}
                duration={1500}
                pixelDuration={480}
                active={swapped}
                onActiveChange={onSwap}
                ariaLabel={t({ en: "Classic screen, then the new one. Click or press Enter to swap them.", th: "หน้าจอคลาสสิก แล้วตามด้วยหน้าจอใหม่ คลิกหรือกด Enter เพื่อสลับ" })}
                firstContent={face(p.old, false)}
                secondContent={face(p.next, true)}
              />
              {run > 0 && <span key={run} className="sc__scan" aria-hidden="true" />}
              <div className="sc__pins" aria-hidden="true">
                {p.pins.map((pin, j) => {
                  const s = spots[j];
                  return (
                    <span key={j} className={"sc__pin" + (s?.flip ? " is-left" : "")} data-show={s?.show || undefined} style={{ left: s?.x ?? 0, top: s?.y ?? 0 } as CSSProperties}>
                      <i />
                      <em className="mono">{t(pin.label)}</em>
                    </span>
                  );
                })}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

/**
 * Before and after, as a pixel swap that the page's own scroll sets off. The classic screen comes first and plays; as the
 * visitor scrolls, it breaks into pixels from the left edge and becomes the new screen, which starts to play as it lands:
 * the chat first and then the monitor. The scroll is the browser's own (a tall section with a pinned stage). A click or
 * Enter swaps it by hand, and scrolling hands it back. With reduced motion nothing is pinned or animated: the two screens
 * are shown one above the other, each on its last frame.
 */
export function Compare() {
  const t = useT();
  const reduced = usePrefersReducedMotion();
  const root = useRef<HTMLElement>(null);
  const [phase, setPhase] = useState(0);
  // the swap of each pair: set by the scroll, or by a click until the next scroll
  const [swapped, setSwapped] = useState<boolean[]>(() => PAIRS.map(() => false));
  const hand = useRef<(boolean | null)[]>(PAIRS.map(() => null));
  // each change of a pair's swap sends one bright line across it, in step with the pixels
  const [runs, setRuns] = useState<number[]>(() => PAIRS.map(() => 0));
  const was = useRef<boolean[]>(PAIRS.map(() => false));
  useEffect(() => {
    const changed = swapped.map((v, k) => v !== was.current[k]);
    if (changed.some(Boolean)) {
      setRuns((r) => r.map((n, k) => (changed[k] ? n + 1 : n)));
      was.current = swapped;
    }
  }, [swapped]);

  useEffect(() => {
    if (reduced) return;
    const el = root.current;
    if (!el) return;
    let raf = 0;
    const frame = () => {
      raf = 0;
      const r = el.getBoundingClientRect();
      const total = Math.max(1, r.height - window.innerHeight);
      const p = clamp(-r.top / total);
      // each pair has half of the scroll: the classic screen rests, it swaps, the new screen rests
      setSwapped((prev) =>
        {
          const next = PAIRS.map((_, k) => {
            const hold = hand.current[k];
            if (hold !== null) return hold;
            const local = clamp((p - k * 0.5) / 0.5);
            // a little hysteresis so it does not flutter at the threshold
            return prev[k] ? local > 0.3 : local > 0.38;
          });
          return next.every((v, k) => v === prev[k]) ? prev : next;
        },
      );
      setPhase((prev) => {
        const next = p < 0.5 ? 0 : 1;
        return prev === next ? prev : next;
      });
    };
    const on = () => {
      if (!raf) raf = requestAnimationFrame(frame);
    };
    // scrolling hands the swaps back to the page
    const onScroll = () => {
      hand.current = PAIRS.map(() => null);
      on();
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", on);
    on();
    return () => {
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", on);
      if (raf) cancelAnimationFrame(raf);
    };
  }, [reduced]);

  function jump(k: number) {
    const el = root.current;
    if (!el) return;
    if (reduced) {
      el.querySelectorAll(".sc__pair")[k]?.scrollIntoView({ block: "center", behavior: "auto" });
      return;
    }
    const top = el.getBoundingClientRect().top + window.scrollY;
    const total = el.offsetHeight - window.innerHeight;
    // land on the pause before the swap of that pair, so the story starts from the classic screen
    window.scrollTo({ top: top + total * (k * 0.5 + 0.03), behavior: "smooth" });
  }
  function setHand(k: number, v: boolean) {
    hand.current[k] = v;
    setSwapped((prev) => prev.map((x, i) => (i === k ? v : x)));
  }

  // the window leans a little toward the pointer
  const lean: Lean = (e) => {
    if (reduced || e.pointerType !== "mouse") return;
    const r = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width - 0.5) * 2;
    const y = ((e.clientY - r.top) / r.height - 0.5) * 2;
    e.currentTarget.style.setProperty("--rx", `${(-y * 2.6).toFixed(2)}deg`);
    e.currentTarget.style.setProperty("--ry", `${(x * 3.4).toFixed(2)}deg`);
  };
  const level: Lean = (e) => {
    e.currentTarget.style.setProperty("--rx", "0deg");
    e.currentTarget.style.setProperty("--ry", "0deg");
  };

  const pair = PAIRS[phase];

  return (
    <section ref={root} className={"sc" + (reduced ? " sc--static" : "")} id="app" aria-labelledby="app-title">
      <div className="sc__sticky">
        <div className="wrap">
          <header className="sc__head">
            <h2 id="app-title" className="h2">
              <SlideUpText v={{ en: "From classic to", th: "จากคลาสสิก สู่" }} />{" "}
              <HighlightedText delay={400}>
                <SlideUpText v={{ en: "new.", th: "ใหม่" }} delay={200} />
              </HighlightedText>
            </h2>
            <p className="shead__lede">
              <T
                v={{
                  en: "Keep scrolling: the classic screen breaks into pixels and becomes the new one, already at work. Or click it.",
                  th: "เลื่อนต่อไป: หน้าจอคลาสสิกจะแตกเป็นพิกเซลแล้วกลายเป็นหน้าจอใหม่ที่ทำงานอยู่แล้ว หรือจะคลิกก็ได้",
                }}
              />
            </p>
          </header>

          <div className="tabs tabs--center sc__tabs" role="group" aria-label={t({ en: "Which screen", th: "หน้าจอไหน" })}>
            {PAIRS.map((p, k) => (
              <button key={p.id} type="button" className="tabs__tab" aria-pressed={phase === k} onClick={() => jump(k)}>
                <T v={p.tab} />
              </button>
            ))}
          </div>

          <div className="sc__frames">
            {PAIRS.map((p, k) => (
              <div key={p.id} className={"sc__pair" + (phase === k ? " is-on" : "")}>
                <PairView p={p} on={phase === k} swapped={swapped[k]} run={runs[k]} reduced={reduced} onSwap={(v) => setHand(k, v)} lean={lean} level={level} t={t} />
              </div>
            ))}
          </div>

          <div className="sc__bar">
            <p className="cap mono" aria-live="polite">
              <T v={pair.note} />
            </p>
            <div className="sc__acts">
              <button type="button" className="btn btn--ghost" onClick={() => setHand(phase, !swapped[phase])}>
                <T v={swapped[phase] ? { en: "Back to classic", th: "กลับไปคลาสสิก" } : { en: "Swap to the new one", th: "สลับเป็นหน้าใหม่" }} /> <span aria-hidden="true">⇄</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
