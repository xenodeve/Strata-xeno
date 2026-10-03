"use client";

import { useEffect, useRef, type CSSProperties } from "react";

/**
 * ParticleText: a word made of points. The points start scattered, gather into the letters (each one starting a little later than the
 * one before, by `stagger`), drift a little while they rest, and are pushed away from the pointer, taking `highlightColor` while they
 * are displaced. `trigger` says what gathers them: "hover" (the pointer enters, or focus arrives; they scatter again when it
 * leaves), "visible" (once, when it is on screen) or "mount". Where there is no hover (a touch screen) "hover" gathers when it is on
 * screen, so the word is never left as dust. With reduced motion the word is simply there, still.
 *
 * Written for this site from the usage of the component that was supplied (its props), as this site's own code. It draws with the 2D
 * canvas: the text is drawn once into a hidden canvas, sampled every `density` pixels, and each sample is a point. It only runs while
 * it is on screen.
 */
export type ParticleTextProps = {
  text: string;
  /** Radius-ish size of a point, in px. */
  particleSize?: number;
  /** Distance in px between samples of the text: smaller is denser. */
  density?: number;
  color?: string;
  highlightColor?: string;
  /** How far, in px, the points start from their place. */
  scatter?: number;
  /** How long a point takes to arrive, in ms. */
  gatherDuration?: number;
  /** The spread, in ms, of the moments the points start. */
  stagger?: number;
  /** How far, in px, the pointer pushes a point at its strongest. */
  pointerRepel?: number;
  /** How near the pointer has to be to push. */
  repelRadius?: number;
  /** How much a resting point wanders, in px. */
  idleDrift?: number;
  trigger?: "hover" | "visible" | "mount";
  /** Any CSS length, `clamp()` included. */
  fontSize?: string;
  fontWeight?: number | string;
  fontFamily?: string;
  glow?: boolean;
  className?: string;
  style?: CSSProperties;
};

const clamp01 = (x: number) => Math.min(1, Math.max(0, x));
/** The site's ease: fast out, soft settle (the app's `--ease`, cubic-bezier(0.23, 1, 0.32, 1), close enough for a point's flight). */
const ease = (x: number) => 1 - Math.pow(1 - clamp01(x), 4);
const MAX_POINTS = 9000;

export default function ParticleText({
  text,
  particleSize = 2,
  density = 4,
  color = "#ffffff",
  highlightColor = "#8b5cf6",
  scatter = 180,
  gatherDuration = 1600,
  stagger = 420,
  pointerRepel = 40,
  repelRadius = 120,
  idleDrift = 0.7,
  trigger = "hover",
  fontSize = "clamp(3rem, 12vw, 8rem)",
  fontWeight = 800,
  fontFamily = "inherit",
  glow = false,
  className = "",
  style,
}: ParticleTextProps) {
  const box = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const el = box.current;
    const cv = canvas.current;
    if (!el || !cv) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    const reduced = typeof matchMedia !== "undefined" && matchMedia("(prefers-reduced-motion: reduce)").matches;
    const canHover = typeof matchMedia !== "undefined" && matchMedia("(hover: hover)").matches;

    let W = 0;
    let H = 0;
    let dpr = 1;
    // the points, in flat arrays: where each belongs, where it starts, when it starts, how far it has come, and how the pointer has moved it
    let n = 0;
    let tx = new Float32Array(0);
    let ty = new Float32Array(0);
    let sx = new Float32Array(0);
    let sy = new Float32Array(0);
    let delay = new Float32Array(0);
    let phase = new Float32Array(0);
    let from = new Float32Array(0);
    let to = new Float32Array(0);
    let ox = new Float32Array(0);
    let oy = new Float32Array(0);
    let goal = 0; // 1: gathering, 0: scattered
    let goalAt = 0;
    let pointer: { x: number; y: number } | null = null;
    let onScreen = false;
    let raf = 0;
    let last = 0;
    let started = false;
    let idle = false; // only the slow drift is left

    const layout = () => {
      W = el.clientWidth;
      H = el.clientHeight;
      if (W < 10 || H < 10) return;
      dpr = Math.min(2, window.devicePixelRatio || 1);
      cv.width = Math.round(W * dpr);
      cv.height = Math.round(H * dpr);

      // the size of the text: the CSS length asked for, as pixels
      const probe = document.createElement("span");
      probe.style.cssText = `position:absolute;visibility:hidden;white-space:nowrap;font-size:${fontSize};`;
      el.appendChild(probe);
      let px = parseFloat(getComputedStyle(probe).fontSize) || 96;
      el.removeChild(probe);
      const family = fontFamily === "inherit" ? getComputedStyle(el).fontFamily : fontFamily;

      const off = document.createElement("canvas");
      off.width = W;
      off.height = H;
      const o = off.getContext("2d", { willReadFrequently: true });
      if (!o) return;
      o.font = `${fontWeight} ${px}px ${family}`;
      const wide = o.measureText(text).width;
      if (wide > W * 0.92) {
        px = Math.floor((px * W * 0.92) / wide);
        o.font = `${fontWeight} ${px}px ${family}`;
      }
      o.textAlign = "center";
      o.textBaseline = "middle";
      o.fillStyle = "#000";
      o.fillText(text, W / 2, H / 2 + px * 0.04);
      const data = o.getImageData(0, 0, W, H).data;

      // sample the letters; if there would be too many points, sample more sparsely
      let step = Math.max(1, density);
      let xs: number[] = [];
      let ys: number[] = [];
      for (let tries = 0; tries < 4; tries++) {
        xs = [];
        ys = [];
        for (let y = 0; y < H; y += step) {
          for (let x = 0; x < W; x += step) {
            if (data[(Math.floor(y) * W + Math.floor(x)) * 4 + 3] > 128) {
              xs.push(x);
              ys.push(y);
            }
          }
        }
        if (xs.length <= MAX_POINTS) break;
        step *= 1.3;
      }
      n = xs.length;
      tx = Float32Array.from(xs);
      ty = Float32Array.from(ys);
      sx = new Float32Array(n);
      sy = new Float32Array(n);
      delay = new Float32Array(n);
      phase = new Float32Array(n);
      from = new Float32Array(n);
      to = new Float32Array(n);
      ox = new Float32Array(n);
      oy = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const a = Math.random() * Math.PI * 2;
        const d = scatter * (0.35 + Math.random() * 0.65);
        sx[i] = tx[i] + Math.cos(a) * d;
        sy[i] = ty[i] + Math.sin(a) * d;
        delay[i] = Math.random() * stagger;
        phase[i] = Math.random() * Math.PI * 2;
        from[i] = reduced ? 1 : goal ? 1 : 0;
        to[i] = from[i];
      }
    };

    // how far point `i` has come (0: scattered, 1: in its place) at `now`
    const current = (i: number, now: number) => from[i] + (to[i] - from[i]) * ease((now - goalAt - delay[i]) / gatherDuration);
    const set = (g: 0 | 1, now: number) => {
      if (reduced || g === goal) return;
      // where each point is now becomes where it sets out from, so a change of mind in mid-flight never jumps
      for (let i = 0; i < n; i++) {
        from[i] = current(i, now);
        to[i] = g;
      }
      goal = g;
      goalAt = now;
    };

    // Draws a frame and says how much the picture can still change by itself: 2 a point on its way or pushed (every frame), 1 only the
    // slow drift of resting points (half the frames are enough), 0 nothing (the loop can stop until something wakes it).
    const draw = (now: number, dt: number): 0 | 1 | 2 => {
      let push = 0;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      const r = particleSize * 0.5 + 0.35;
      const amp = reduced ? 0 : idleDrift * 1.6;
      const k = 1 - Math.pow(0.001, dt / 1000); // how far an offset moves toward its goal this frame, whatever the frame rate
      ctx.fillStyle = color;
      ctx.beginPath();
      const hi: number[] = [];
      for (let i = 0; i < n; i++) {
        const g = reduced ? 1 : current(i, now);
        let x = sx[i] + (tx[i] - sx[i]) * g;
        let y = sy[i] + (ty[i] - sy[i]) * g;
        if (g > 0.98 && amp) {
          x += Math.sin(now * 0.0011 + phase[i]) * amp;
          y += Math.cos(now * 0.0009 + phase[i] * 1.3) * amp;
        }
        let gx = 0;
        let gy = 0;
        if (pointer && !reduced) {
          const dx = x - pointer.x;
          const dy = y - pointer.y;
          const d = Math.hypot(dx, dy);
          if (d < repelRadius && d > 0.001) {
            const f = Math.pow(1 - d / repelRadius, 2) * pointerRepel;
            gx = (dx / d) * f;
            gy = (dy / d) * f;
          }
        }
        ox[i] += (gx - ox[i]) * k;
        oy[i] += (gy - oy[i]) * k;
        x += ox[i];
        y += oy[i];
        push += Math.abs(ox[i]) + Math.abs(oy[i]);
        if (Math.abs(ox[i]) + Math.abs(oy[i]) > pointerRepel * 0.18) {
          hi.push(x, y);
          continue;
        }
        ctx.moveTo(x + r, y);
        ctx.arc(x, y, r, 0, 6.2832);
      }
      ctx.fill();
      if (hi.length) {
        if (glow) {
          ctx.globalAlpha = 0.16;
          ctx.fillStyle = highlightColor;
          ctx.beginPath();
          for (let j = 0; j < hi.length; j += 2) {
            ctx.moveTo(hi[j] + r * 3.2, hi[j + 1]);
            ctx.arc(hi[j], hi[j + 1], r * 3.2, 0, 6.2832);
          }
          ctx.fill();
          ctx.globalAlpha = 1;
        }
        ctx.fillStyle = highlightColor;
        ctx.beginPath();
        for (let j = 0; j < hi.length; j += 2) {
          ctx.moveTo(hi[j] + r * 1.15, hi[j + 1]);
          ctx.arc(hi[j], hi[j + 1], r * 1.15, 0, 6.2832);
        }
        ctx.fill();
      }
      if (pointer || push > 0.05 || now - goalAt < stagger + gatherDuration + 80) return 2;
      return goal === 1 && amp > 0 ? 1 : 0;
    };

    const frame = (now: number) => {
      raf = 0;
      if (!onScreen) return;
      // the drift of resting points is slow: it is drawn every other frame; and when nothing can change, not at all
      if (idle && last && now - last < 30) {
        raf = requestAnimationFrame(frame);
        return;
      }
      const dt = last ? Math.min(64, now - last) : 16;
      last = now;
      const busy = draw(now, dt);
      idle = busy === 1;
      if (!reduced && busy > 0) raf = requestAnimationFrame(frame);
    };
    const run = () => {
      if (!raf && onScreen) {
        last = 0;
        raf = requestAnimationFrame(frame);
      }
    };

    const gatherNow = () => {
      set(1, performance.now());
      run();
    };
    const scatterNow = () => {
      set(0, performance.now());
      run();
    };

    const setup = () => {
      layout();
      if (reduced) {
        draw(performance.now(), 16);
        return;
      }
      run();
    };

    const io = new IntersectionObserver(
      (e) => {
        onScreen = e.some((x) => x.isIntersecting);
        if (onScreen) {
          if (!started) {
            started = true;
            setup();
          }
          if (trigger === "mount" || (trigger === "visible" && !goal) || (trigger === "hover" && !canHover && !goal)) {
            // the word is gathered when it comes into view, a moment after (so the gathering is seen)
            window.setTimeout(() => onScreen && gatherNow(), 250);
          }
          run();
        }
      },
      { threshold: 0.35 },
    );
    io.observe(el);

    const move = (e: PointerEvent) => {
      const r = el.getBoundingClientRect();
      pointer = { x: e.clientX - r.left, y: e.clientY - r.top };
      idle = false;
      run();
    };
    const enter = (e: PointerEvent) => {
      if (e.pointerType === "touch") return;
      move(e);
      if (trigger === "hover") gatherNow();
    };
    const leave = () => {
      pointer = null;
      if (trigger === "hover" && canHover) scatterNow();
    };
    el.addEventListener("pointerenter", enter);
    el.addEventListener("pointermove", move);
    el.addEventListener("pointerleave", leave);
    const focus = () => trigger === "hover" && gatherNow();
    el.addEventListener("focusin", focus);

    const ro = new ResizeObserver(() => {
      if (!started) return;
      layout();
      if (reduced) draw(performance.now(), 16);
    });
    ro.observe(el);
    // the face of the text is known once the fonts are: lay out again then
    void document.fonts?.ready.then(() => {
      if (started) {
        layout();
        if (reduced) draw(performance.now(), 16);
      }
    });

    return () => {
      io.disconnect();
      ro.disconnect();
      if (raf) cancelAnimationFrame(raf);
      el.removeEventListener("pointerenter", enter);
      el.removeEventListener("pointermove", move);
      el.removeEventListener("pointerleave", leave);
      el.removeEventListener("focusin", focus);
    };
  }, [text, particleSize, density, color, highlightColor, scatter, gatherDuration, stagger, pointerRepel, repelRadius, idleDrift, trigger, fontSize, fontWeight, fontFamily, glow]);

  return (
    <div ref={box} className={"particle-text " + className} style={{ position: "relative", width: "100%", height: "100%", ...style }}>
      {/* the word as text, for a page without scripts and for assistive technology; the canvas is only its picture */}
      <span className="particle-text__still" style={{ fontSize, fontWeight, fontFamily, color }}>
        {text}
      </span>
      <canvas ref={canvas} aria-hidden="true" style={{ position: "absolute", inset: 0, width: "100%", height: "100%", display: "block" }} />
    </div>
  );
}
