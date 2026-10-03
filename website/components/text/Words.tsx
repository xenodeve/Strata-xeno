"use client";

import dynamic from "next/dynamic";
import { useEffect, useRef, useState } from "react";
import { usePrefersReducedMotion } from "../Avatar";
import { useT, type L } from "@/lib/i18n";

// The two canvas effects load only when their section is built on the client: they are not part of the first paint.
const TechText = dynamic(() => import("./TechText"), { ssr: false });
const WarpText = dynamic(() => import("./WarpText"), { ssr: false });
const Particles = dynamic(() => import("./Particles"), { ssr: false });
const Aurora = dynamic(() => import("./Aurora"), { ssr: false });

/** The page's ink and accent colours as #rrggbb, kept in step with the theme (the canvases need real colours, not var()). */
export function useInk(): { ink: string; accent: string; light: boolean } {
  const [c, setC] = useState({ ink: "#131316", accent: "#0a62d6", light: true });
  useEffect(() => {
    const read = () => {
      const cs = getComputedStyle(document.documentElement);
      const hex = (name: string, fallback: string) => {
        const v = cs.getPropertyValue(name).trim();
        return /^#[0-9a-f]{3,8}$/i.test(v) ? v : fallback;
      };
      const bg = hex("--bg", "#f5f5f7").replace("#", "");
      const full = bg.length === 3 ? bg.replace(/./g, (x) => x + x) : bg.slice(0, 6);
      const lum = (parseInt(full.slice(0, 2), 16) * 0.299 + parseInt(full.slice(2, 4), 16) * 0.587 + parseInt(full.slice(4, 6), 16) * 0.114) / 255;
      setC({ ink: hex("--ink", "#131316"), accent: hex("--accent", "#0a62d6"), light: lum > 0.5 });
    };
    read();
    const mo = new MutationObserver(read);
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    mq.addEventListener("change", read);
    return () => {
      mo.disconnect();
      mq.removeEventListener("change", read);
    };
  }, []);
  return c;
}

/** WebGL 2 is there and the visitor has not turned the canvases off with ?webgl=0. */
export function useCanvasOk(): boolean | null {
  const [ok, setOk] = useState<boolean | null>(null);
  useEffect(() => {
    let yes = false;
    try {
      yes = !!document.createElement("canvas").getContext("webgl2") && new URLSearchParams(location.search).get("webgl") !== "0";
    } catch {
      yes = false;
    }
    setOk(yes);
  }, []);
  return ok;
}

/** A big word that bends around the pointer (WebGL). Without WebGL it is the same word, plain. */
export function WarpWord({ v, className = "" }: { v: L; className?: string }) {
  const t = useT();
  const text = t(v);
  const th = t({ en: "en", th: "th" }) === "th";
  const { ink } = useInk();
  const ok = useCanvasOk();
  return (
    <div className={"tw tw--warp " + className}>
      {ok ? (
        <WarpText
          text={text}
          color={ink}
          fontFamily={th ? '"Anuphan Variable", sans-serif' : '"Instrument Serif", Georgia, serif'}
          fontWeight={th ? 600 : 400}
          fontSize="clamp(3.4rem, 11vw, 9.5rem)"
          letterSpacing={th ? "0em" : "-0.025em"}
          lineHeight={1}
          warpStrength={0.1}
          speed={0.5}
          pointerInfluence={0.38}
          pointerStrength={0.45}
          refraction={0.016}
        />
      ) : (
        <p className="tw__plain" aria-label={text}>
          {text}
        </p>
      )}
    </div>
  );
}

/** A word to play with: move over it and a letter is picked out in a frame; drag a letter and it springs back. */
export function TechWord({ text, className = "" }: { text: string; className?: string }) {
  const { ink, accent } = useInk();
  const ok = useCanvasOk();
  return (
    <div className={"tw tw--tech " + className}>
      {ok ? (
        <TechText text={text} fontFamily={'"Instrument Serif", Georgia, serif'} fontWeight={400} fontSize={260} letterSpacing={-0.02} color={ink} accentColor={accent} reach={190} softness={0.7} specks={14} />
      ) : (
        <p className="tw__plain" aria-label={text}>
          {text}
        </p>
      )}
    </div>
  );
}

/**
 * A field of soft points that drift (WebGL). It exists only while it is on screen, never for a visitor who asks for less
 * motion, and not without WebGL: then the place behind it is simply empty, nothing is hidden by it.
 */
export function ParticleField({ colors, count = 120, className = "", size = 90, spread = 10, speed = 0.08 }: { colors: string[]; count?: number; className?: string; size?: number; spread?: number; speed?: number }) {
  const ok = useCanvasOk();
  const reduced = usePrefersReducedMotion();
  const box = useRef<HTMLDivElement>(null);
  const [on, setOn] = useState(false);
  const [dpr, setDpr] = useState(1);
  useEffect(() => {
    setDpr(Math.min(window.devicePixelRatio || 1, 1.5));
    const el = box.current;
    if (!el || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver((e) => setOn(e.some((x) => x.isIntersecting)), { rootMargin: "80px" });
    io.observe(el);
    return () => io.disconnect();
  }, []);
  return (
    <div ref={box} className={"pfield " + className} aria-hidden="true">
      {ok && !reduced && on && <Particles particleColors={colors} particleCount={count} particleSpread={spread} speed={speed} alphaParticles particleBaseSize={size} sizeRandomness={1} cameraDistance={20} pixelRatio={dpr} />}
    </div>
  );
}

/**
 * A slow aurora (WebGL) behind a place. It follows the theme (a light page gets the light version), exists only while it is
 * on screen, never for a visitor who asks for less motion, and not without WebGL: the place behind it is then just the page.
 */
export function AuroraField({ colors, className = "", amplitude = 1, blend = 0.55, speed = 0.7 }: { colors: [string, string, string]; className?: string; amplitude?: number; blend?: number; speed?: number }) {
  const ok = useCanvasOk();
  const reduced = usePrefersReducedMotion();
  const { light } = useInk();
  const box = useRef<HTMLDivElement>(null);
  const [on, setOn] = useState(false);
  useEffect(() => {
    const el = box.current;
    if (!el || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver((e) => setOn(e.some((x) => x.isIntersecting)), { rootMargin: "80px" });
    io.observe(el);
    return () => io.disconnect();
  }, []);
  return (
    <div ref={box} className={"afield " + className} aria-hidden="true">
      {ok && !reduced && on && <Aurora colorStops={colors} amplitude={amplitude} blend={blend} lightMode={light} speed={speed} />}
    </div>
  );
}
