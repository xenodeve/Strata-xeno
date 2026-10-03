"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { SLIDES } from "@/data/story";
import { botAvatarPalette } from "@/vendor/bot-avatars/index.es.js";
import { T, useT } from "@/lib/i18n";
import { Avatar, usePrefersReducedMotion, useSize } from "../Avatar";
import { AppScene } from "../scenes";
import { HighlightedText, SlideUpText } from "../text/Fx";

const clamp = (x: number, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const smooth = (x: number) => x * x * (3 - 2 * x);

/**
 * The nine moods, played by the page's own scroll. The stage is pinned and the scroll moves through the moods: each one
 * holds still for a while, then glides to the next. The scroll is the browser's own, nothing takes it over. The buttons,
 * the arrow keys and a swipe move the page to the matching place; with reduced motion the stage is not pinned and the
 * moods are changed with the same controls.
 */
export function Story() {
  const t = useT();
  const reduced = usePrefersReducedMotion();
  const size = useSize(240, 170);
  const n = SLIDES.length;

  const root = useRef<HTMLElement>(null);
  const drag = useRef<{ x: number; y: number } | null>(null);
  const [i, setI] = useState(0);
  const [hop, setHop] = useState(false);
  const slide = SLIDES[i];

  // scroll -> where the track is (a fraction, with a rest on each mood) and which mood is on
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
      const f = p * (n - 1);
      const k = Math.floor(f);
      const frac = f - k;
      // rest for the first quarter and the last quarter of each step, glide in the middle half
      const g = k >= n - 1 ? n - 1 : k + smooth(clamp((frac - 0.25) / 0.5));
      el.style.setProperty("--f", g.toFixed(4));
      const idx = Math.round(g);
      setI((prev) => (prev === idx ? prev : idx));
    };
    const on = () => {
      if (!raf) raf = requestAnimationFrame(frame);
    };
    // When the scrolling stops between two moods, in the middle of the glide, it is taken on to the mood it was heading for (the next
    // one if the page was moving down, the one before if up), so the story never rests half way between two faces.
    let idle = 0;
    let lastY = window.scrollY;
    let dir = 1;
    const settle = () => {
      const r = el.getBoundingClientRect();
      const total = r.height - window.innerHeight;
      if (total <= 0) return;
      const p = -r.top / total;
      if (p <= 0 || p >= 1) return; // above or below the story: nothing to settle
      const f = p * (n - 1);
      const k = Math.floor(f);
      const frac = f - k;
      if (frac < 0.25 || frac > 0.75) return; // already at rest on a mood
      const to = Math.max(0, Math.min(n - 1, dir >= 0 ? k + 1 : k));
      window.scrollTo({ top: r.top + window.scrollY + (total * to) / (n - 1) + (to === 0 ? 2 : 0), behavior: "smooth" });
    };
    const onScrollSettle = () => {
      const y = window.scrollY;
      if (Math.abs(y - lastY) > 0.5) dir = y > lastY ? 1 : -1;
      lastY = y;
      window.clearTimeout(idle);
      idle = window.setTimeout(settle, 160);
    };
    window.addEventListener("scroll", on, { passive: true });
    window.addEventListener("scroll", onScrollSettle, { passive: true });
    window.addEventListener("resize", on);
    on();
    return () => {
      window.removeEventListener("scroll", on);
      window.removeEventListener("scroll", onScrollSettle);
      window.removeEventListener("resize", on);
      window.clearTimeout(idle);
      if (raf) cancelAnimationFrame(raf);
    };
  }, [reduced, n]);

  // with reduced motion there is no scroll story: the track follows the chosen mood at once
  useEffect(() => {
    if (reduced) root.current?.style.setProperty("--f", String(i));
  }, [reduced, i]);

  // a swap is a little hop: the new character jumps in, then settles into its own mood
  useEffect(() => {
    setHop(true);
    const id = window.setTimeout(() => setHop(false), 900);
    return () => window.clearTimeout(id);
  }, [i]);

  const goTo = useCallback(
    (to: number) => {
      const k = Math.max(0, Math.min(n - 1, to));
      const el = root.current;
      if (!el) return;
      if (reduced) {
        setI(k);
        return;
      }
      const top = el.getBoundingClientRect().top + window.scrollY;
      const total = el.offsetHeight - window.innerHeight;
      window.scrollTo({ top: top + (total * k) / (n - 1) + (k === 0 ? 2 : 0), behavior: "smooth" });
    },
    [n, reduced],
  );

  function onKey(e: React.KeyboardEvent) {
    // the up and down arrows belong to the page's scroll; only the sideways ones and Home / End are taken
    const map: Record<string, number> = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: n - 1 };
    if (!(e.key in map)) return;
    e.preventDefault();
    goTo(map[e.key]);
  }
  function onDown(e: React.PointerEvent) {
    if (e.pointerType === "mouse" && e.button !== 0) return;
    drag.current = { x: e.clientX, y: e.clientY };
  }
  function onUp(e: React.PointerEvent) {
    const d = drag.current;
    drag.current = null;
    if (!d) return;
    const dx = e.clientX - d.x;
    const dy = e.clientY - d.y;
    if (Math.abs(dx) > 48 && Math.abs(dx) > Math.abs(dy) * 1.4) goTo(i + (dx < 0 ? 1 : -1));
  }

  return (
    <section ref={root} className={"story" + (reduced ? " story--static" : "")} id="story" aria-labelledby="story-title">
      <div className="story__sticky">
        <div className="wrap">
          <header className="story__head">
            <h2 id="story-title" className="h2">
              <SlideUpText v={{ en: "Nine moods.", th: "เก้าอารมณ์" }} />{" "}
              <HighlightedText delay={500}>
                <SlideUpText v={{ en: "One rule.", th: "หนึ่งกฎ" }} delay={160} />
              </HighlightedText>
            </h2>
            <p className="shead__lede">
              <T v={{ en: "It never goes quiet. Scroll: every moment has a face and a word.", th: "ไม่เคยเงียบ เลื่อนลง: ทุกช่วงเวลามีหน้าตาและมีคำพูด" }} />
            </p>
          </header>

          <div className="story__frame" role="region" aria-roledescription="carousel" aria-label={t({ en: "The nine moods of the app", th: "เก้าอารมณ์ของแอป" })}>
            <div className="story__prog" aria-hidden="true" style={{ "--last": n - 1 } as React.CSSProperties}>
              <i />
              <span className="mono">
                {String(i + 1).padStart(2, "0")} / {String(n).padStart(2, "0")}
              </span>
            </div>

            <div
              className="story__stage"
              role="group"
              tabIndex={0}
              onKeyDown={onKey}
              onPointerDown={onDown}
              onPointerUp={onUp}
              onPointerCancel={() => (drag.current = null)}
              style={{ "--glow": botAvatarPalette[slide.avatar] } as React.CSSProperties}
              aria-label={t({ en: "Scroll, or use the left and right arrow keys, to change the mood", th: "เลื่อนหน้า หรือใช้ปุ่มลูกศรซ้ายและขวาเพื่อเปลี่ยนอารมณ์" })}
            >
              <div className="story__avatar">
                <div key={slide.avatar} className="story__bot">
                  <Avatar type={slide.avatar} state={hop ? "working" : slide.state} size={size} label={`${slide.avatar} avatar`} />
                </div>
                <p className="story__status mono" aria-hidden="true">
                  <span className="story__dot" /> {slide.status}
                </p>
              </div>

              <div className="story__viewport">
                <div className="story__track" aria-live="polite">
                  {SLIDES.map((s, k) => (
                    <div
                      key={s.id}
                      className={"slide" + (k === i ? " is-active" : "")}
                      role="group"
                      aria-roledescription="slide"
                      aria-label={`${k + 1} / ${n}`}
                      aria-hidden={k !== i}
                      inert={k !== i}
                    >
                      <h3 className="slide__title">
                        <SlideUpText v={s.title} play={k === i} />
                      </h3>
                      <p className="slide__line">
                        <T v={s.line} />
                      </p>
                      {s.scene ? (
                        <AppScene id={s.scene.id} alt={s.scene.alt} active={k === i} className="slide__scene" glide />
                      ) : (
                        <p className="slide__quiet mono">
                          <T v={{ en: "No screen: nothing is happening.", th: "ไม่มีหน้าจอ: ไม่มีอะไรเกิดขึ้น" }} />
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="story__ctl">
              <button type="button" className="round" onClick={() => goTo(i - 1)} disabled={i === 0} aria-label={t({ en: "Previous mood", th: "อารมณ์ก่อนหน้า" })}>
                <span aria-hidden="true">←</span>
              </button>
              <div className="dots" role="group" aria-label={t({ en: "Choose a mood", th: "เลือกอารมณ์" })}>
                {SLIDES.map((s, k) => (
                  <button key={s.id} type="button" className="dots__d" aria-current={k === i} aria-label={`${k + 1}: ${s.status}`} onClick={() => goTo(k)} />
                ))}
              </div>
              <button type="button" className="round" onClick={() => goTo(i + 1)} disabled={i === n - 1} aria-label={t({ en: "Next mood", th: "อารมณ์ถัดไป" })}>
                <span aria-hidden="true">→</span>
              </button>
              <a className="story__skip mono" href="#demo">
                <T v={{ en: "Skip", th: "ข้าม" }} />
              </a>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
