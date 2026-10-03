"use client";

import Link from "next/link";
import { useEffect, useRef, useState, type CSSProperties } from "react";
import { SLIDES } from "@/data/story";
import { T, useT } from "@/lib/i18n";
import { botAvatarPalette, type BotAvatarType } from "@/vendor/bot-avatars/index.es.js";
import { Avatar, usePrefersReducedMotion, useSize } from "../Avatar";
import { GitHubButton } from "../GitHubButton";
import { GradientWaveText, RandomizedText, SlideUpOnMount, SlideUpText } from "../text/Fx";
import { ParticleField } from "../text/Words";

/** The eighteen shapes of the app's Marks picker, in the picker's own order. */
const SHAPES: BotAvatarType[] = ["clover", "flower", "triangle", "square", "blob", "ghost", "circle", "drop", "star", "droid", "mech", "alien", "hexagon", "cat", "cloud", "pill", "pebble", "puddle"];

const HEAD = {
  en: { lead: "A local AI that", em: "shows its work." },
  th: { lead: "AI บนเครื่องคุณ", em: "ที่โชว์ว่ากำลังทำอะไร" },
};

const EVERY = 3300;
/** The nine moods' own colours: the points that drift behind the avatar are these. */
const MOOD_COLORS = SLIDES.map((s) => botAvatarPalette[s.avatar]);

/**
 * First screen: one idea, one living mark. The mark walks through the app's nine moods by itself (the ring around it
 * is the nine, the glow takes the colour of the one that is on) until you pick a shape of your own from the eighteen.
 */
export function Hero() {
  const t = useT();
  const lang = t({ en: "en", th: "th" }) as "en" | "th";
  const size = useSize(340, 200);
  const reduced = usePrefersReducedMotion();
  const root = useRef<HTMLElement>(null);
  const stage = useRef<HTMLDivElement>(null);
  const raf = useRef(0);
  const [mood, setMood] = useState(1); // the mood that is on: it starts on "thinks", the cat
  const [pick, setPick] = useState<BotAvatarType | null>(null); // a shape the visitor chose
  const [hop, setHop] = useState(0);
  const [seen, setSeen] = useState(true);
  const [held, setHeld] = useState(false); // pointer or focus on the stage: hold the mood still
  const slide = SLIDES[mood];
  const type = pick ?? slide.avatar;
  const glow = botAvatarPalette[type];

  // the hero is only walked through while it is on screen, and never for a visitor who asks for less motion
  useEffect(() => {
    const el = root.current;
    if (!el || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver((e) => setSeen(e.some((x) => x.isIntersecting)), { threshold: 0.25 });
    io.observe(el);
    return () => io.disconnect();
  }, []);
  useEffect(() => {
    if (reduced || pick || held || !seen) return;
    const id = window.setInterval(() => setMood((m) => (m + 1) % SLIDES.length), EVERY);
    return () => window.clearInterval(id);
  }, [reduced, pick, held, seen]);

  // the stage leans a little toward the pointer and the dotted strata drift against it: depth, not decoration
  function onMove(e: React.PointerEvent) {
    if (reduced || e.pointerType !== "mouse") return;
    const el = stage.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width - 0.5) * 2;
    const y = ((e.clientY - r.top) / r.height - 0.5) * 2;
    cancelAnimationFrame(raf.current);
    raf.current = requestAnimationFrame(() => {
      el.style.setProperty("--px", x.toFixed(3));
      el.style.setProperty("--py", y.toFixed(3));
    });
  }
  function onLeave() {
    stage.current?.style.setProperty("--px", "0");
    stage.current?.style.setProperty("--py", "0");
    setHeld(false);
  }
  function chooseMood(k: number) {
    setPick(null);
    setMood(k);
    setHop((h) => h + 1);
  }

  return (
    <section ref={root} className="hero" id="top" aria-labelledby="hero-title">
      <div className="wrap hero__grid">
        <div className="hero__text">
          <p className="eyebrow mono">
            <span className="eyebrow__dot" aria-hidden="true" />
            Strata-xeno
          </p>
          <h1 id="hero-title" className="display">
            <SlideUpText v={HEAD[lang].lead} />{" "}
            <GradientWaveText className="hero__em">
              <SlideUpText v={HEAD[lang].em} delay={260} />
            </GradientWaveText>
          </h1>
          <p className="lede">
            <RandomizedText
              split="words"
              duration={1300}
              delay={500}
              v={{
                en: "A fork of Strata with a new web app, real coding tools, and memory that gives RAM back as VRAM grows.",
                th: "fork ของ Strata ที่มีเว็บแอปใหม่ เครื่องมือเขียนโค้ดจริง และหน่วยความจำที่คืน RAM ให้เมื่อ VRAM มากขึ้น",
              }}
            />
          </p>
          <div className="cta">
            <a className="btn btn--primary" href="#story">
              <T v={{ en: "Meet the nine moods", th: "ดูเก้าอารมณ์" }} />
            </a>
            <Link className="btn" href="/details">
              <T v={{ en: "All the details", th: "รายละเอียดทั้งหมด" }} /> <span aria-hidden="true">→</span>
            </Link>
            <GitHubButton />
          </div>
        </div>

        <div
          className="hero__stage"
          ref={stage}
          onPointerMove={onMove}
          onPointerEnter={() => setHeld(true)}
          onPointerLeave={onLeave}
          onFocus={() => setHeld(true)}
          onBlur={() => setHeld(false)}
        >
          <div className="hero__glow" style={{ "--glow": glow } as CSSProperties} aria-hidden="true" />
          <ParticleField className="hero__particles" colors={MOOD_COLORS} count={90} size={260} />
          <div className="hero__strata" aria-hidden="true">
            <i />
            <i />
            <i />
            <i />
          </div>

          <div className="hero__orbit">
            <svg className="hero__ring" viewBox="0 0 100 100" aria-hidden="true">
              <circle cx="50" cy="50" r="49" />
            </svg>
            <div className="hero__bot" key={hop}>
              <Avatar type={type} state={pick ? "default" : slide.state} size={size} label={`${type} avatar`} />
            </div>
            <div className="hero__moods" role="group" aria-label={t({ en: "The app's nine moods. Choose one.", th: "เก้าอารมณ์ของแอป เลือกได้" })}>
              {SLIDES.map((s, k) => {
                const a = (k / SLIDES.length) * 360 - 90;
                const on = !pick && k === mood;
                return (
                  <button
                    key={s.id}
                    type="button"
                    className={"hero__mood" + (on ? " is-on" : "")}
                    aria-pressed={on}
                    aria-label={s.status}
                    title={s.status}
                    style={{ "--a": `${a}deg`, "--c": botAvatarPalette[s.avatar] } as CSSProperties}
                    onClick={() => chooseMood(k)}
                  />
                );
              })}
            </div>
          </div>

          <p className="hero__status" aria-live="off">
            <span className="hero__sdot" style={{ background: glow }} aria-hidden="true" />
            <span className="mono hero__sname">{pick ? pick : slide.status}</span>
            <span className="hero__sline">
              {pick ? (
                <T v={{ en: "your pick", th: "ที่คุณเลือก" }} />
              ) : (
                <SlideUpOnMount key={slide.id + lang} v={slide.title} />
              )}
            </span>
          </p>

          <div className="picker" role="group" aria-label={t({ en: "Pick a shape. These are the app's own avatars.", th: "เลือกรูปทรง นี่คือ avatar ตัวจริงของแอป" })}>
            {SHAPES.map((s) => (
              <button
                key={s}
                type="button"
                className="picker__b"
                aria-pressed={s === pick}
                aria-label={s}
                title={s}
                style={{ "--c": botAvatarPalette[s] } as CSSProperties}
                onClick={() => {
                  setPick(s);
                  setHop((h) => h + 1);
                }}
              />
            ))}
          </div>
          <p className="cap mono hero__cap">
            <T
              v={{
                en: "The app's own avatars. It walks through the nine moods by itself; pick a shape to keep it.",
                th: "avatar ตัวจริงของแอป มันเปลี่ยนเก้าอารมณ์เอง เลือกรูปทรงเพื่อให้อยู่ตัวนั้น",
              }}
            />
          </p>
        </div>
      </div>
    </section>
  );
}
