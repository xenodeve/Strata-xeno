"use client";

import { useCallback, useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { T, useT, type L } from "@/lib/i18n";
import { usePrefersReducedMotion } from "../Avatar";
import { EASE } from "./motion";

/**
 * A scene is a piece of the app's interface, rebuilt for this page and played from a clock: everything on it is a plain
 * function of `ms`, the time since the scene started. That makes it exact (the same frame every time), cheap (one small
 * tree, no animation library) and scrubbable. It starts when it is on screen, pauses when it is not, rests on its last
 * frame for a moment and plays again. With reduced motion it is only its last frame.
 *
 * The text on a scene is copied from real runs of the real app; none of the app's own component code is used.
 */
export type SceneBodyProps = { ms: number; lang: "en" | "th"; done: boolean };
export type SceneDef = {
  /** When the last thing has landed, in ms. */
  duration: number;
  /** How long to rest on the last frame before it plays again. */
  hold?: number;
  Body: (p: SceneBodyProps) => ReactNode;
};

export const clamp01 = (x: number) => Math.min(1, Math.max(0, x));
/** The app's one easing (`--ease`, cubic-bezier(0.23, 1, 0.32, 1)), as a curve of 0..1: fast out, soft settle. */
export const ease = (x: number) => EASE(clamp01(x));
/** Progress 0..1 of the stretch [start, start + dur] of the clock, eased. */
export const seg = (ms: number, start: number, dur: number) => ease((ms - start) / dur);
/** Progress 0..1 of the stretch, linear. */
export const lin = (ms: number, start: number, dur: number) => clamp01((ms - start) / dur);
export const at = (ms: number, start: number) => ms >= start;
export const lerp = (a: number, b: number, p: number) => a + (b - a) * p;

type Props = {
  def: SceneDef;
  alt: L;
  /** Only the active scene plays (a carousel slide, the visible half of a pair). */
  active?: boolean;
  className?: string;
  /** Show the Replay button. */
  replay?: boolean;
  /** Show the app in this language whatever language the page is in. */
  lang?: "en" | "th";
  /** The box takes the height of what is on the scene and glides to it when that changes (the app's own stretch: 420 ms,
   *  cubic-bezier(0.4, 0, 0.2, 1)), so a scene that grows as it plays opens downwards instead of jumping. For a scene that
   *  sits in a box of its own height (the compare window) leave it off. */
  glide?: boolean;
};

export function Scene({ def, alt, active = true, className = "", replay = true, lang: forced, glide = false }: Props) {
  const t = useT();
  const lang = forced ?? (t({ en: "en", th: "th" }) as "en" | "th");
  const reduced = usePrefersReducedMotion();
  const box = useRef<HTMLDivElement>(null);
  const elapsed = useRef(0);
  // Before it is on its way it shows its last frame (what a reader without scripts gets); once it is near the screen it is
  // put back to the start, out of sight, and plays when it is properly in view.
  const [ms, setMs] = useState(def.duration);
  const [armed, setArmed] = useState(false);
  const [visible, setVisible] = useState(false);
  const [run, setRun] = useState(0);
  const hold = def.hold ?? 3600;

  // glide: the height of the box follows the height of its content. The first measure is set at once; what comes after glides.
  const inner = useRef<HTMLDivElement>(null);
  const bar = useRef<HTMLElement>(null);
  const [h, setH] = useState<number | null>(null);
  const [glides, setGlides] = useState(false);
  useLayoutEffect(() => {
    if (!glide) return;
    const el = inner.current;
    const out = box.current;
    if (!el || !out) return;
    const frame = out.offsetHeight - out.clientHeight; // the border
    const measure = () => setH(el.offsetHeight + frame);
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    const id = requestAnimationFrame(() => setGlides(true));
    return () => {
      ro.disconnect();
      cancelAnimationFrame(id);
    };
  }, [glide]);

  useEffect(() => {
    const el = box.current;
    if (!el || typeof IntersectionObserver === "undefined") return;
    const near = new IntersectionObserver((e) => e.some((x) => x.isIntersecting) && setArmed(true), { rootMargin: "300px 0px" });
    const seen = new IntersectionObserver((e) => setVisible(e.some((x) => x.isIntersecting)), { threshold: 0.35 });
    near.observe(el);
    seen.observe(el);
    return () => {
      near.disconnect();
      seen.disconnect();
    };
  }, []);

  // armed: back to the start, out of sight
  useEffect(() => {
    if (armed && !reduced) {
      elapsed.current = 0;
      setMs(0);
    }
  }, [armed, reduced]);

  // a slide that has just become the active one starts from the top
  const wasActive = useRef(active);
  useEffect(() => {
    if (active && !wasActive.current && !reduced) {
      elapsed.current = 0;
      setMs(0);
    }
    wasActive.current = active;
  }, [active, reduced]);

  useEffect(() => {
    if (!armed || reduced || !visible || !active) return;
    let raf = 0;
    const cycle = def.duration + hold;
    let t0 = performance.now() - elapsed.current;
    let last = 0;
    // The Body is drawn only while something on it is moving (up to `duration`) and once more at its last frame. For the rest of the
    // round, which is most of it for a short scene, nothing on it changes, so it is not drawn again; the foot bar and the fade-out at
    // the end of the rest are written straight to their elements (no render for them).
    let settled = false;
    const loop = (now: number) => {
      let e = now - t0;
      if (e > cycle) {
        t0 = now;
        e = 0;
      }
      elapsed.current = e;
      if (e < def.duration) {
        settled = false;
        if (now - last > 28) {
          last = now;
          setMs(e);
        }
      } else if (!settled) {
        settled = true;
        setMs(def.duration);
      }
      if (bar.current) bar.current.style.transform = `scaleX(${clamp01(e / cycle)})`;
      if (inner.current) inner.current.style.opacity = String(clamp01((cycle - e) / 500));
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [armed, reduced, visible, active, def.duration, hold, run]);

  const again = useCallback(() => {
    elapsed.current = 0;
    setMs(0);
    setRun((r) => r + 1);
  }, []);

  const shown = reduced ? def.duration : ms;
  // (the last moment of the rest fades the scene out, so a new round starts from nothing without a cut: written by the loop above)
  return (
    <div ref={box} className={"scene " + className + (glide ? " scene--glide" : "") + (glide && glides && !reduced ? " is-gliding" : "")} style={glide && h != null ? { height: h } : undefined}>
      <div role="img" aria-label={t(alt)}>
        <div ref={inner} className="scene__in" aria-hidden="true">
          {def.Body({ ms: shown, lang, done: shown >= def.duration })}
        </div>
      </div>
      {!reduced && armed && (
        <span className="scene__bar" aria-hidden="true">
          <i ref={bar} style={{ transform: "scaleX(0)" }} />
        </span>
      )}
      {replay && !reduced && (
        <button type="button" className="scene__again" onClick={again} aria-label={t({ en: "Play this scene again", th: "เล่นฉากนี้อีกครั้ง" })}>
          <span aria-hidden="true">↻</span>
          <T v={{ en: "Replay", th: "เล่นซ้ำ" }} />
        </button>
      )}
    </div>
  );
}
