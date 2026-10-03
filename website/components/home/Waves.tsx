"use client";

import dynamic from "next/dynamic";
import { useEffect, useRef, useState } from "react";
import { botAvatarPalette } from "@/vendor/bot-avatars/index.es.js";
import { T } from "@/lib/i18n";
import { usePrefersReducedMotion } from "../Avatar";

// WebGPU, and only on the client: nothing of it is part of the first paint.
const ShapeWaves = dynamic(() => import("../text/ShapeWaves"), { ssr: false });

/**
 * A dark band of dots that take the shapes of the app's marks; the name is cut out of the field and a pointer sends ripples
 * through it. It draws with WebGPU (the supplied ShapeWaves). It is decorative: where the browser has no WebGPU, or the
 * visitor asks for less motion, or it fails, what stays is a still band of dots with the name on it.
 */
export function Waves() {
  const reduced = usePrefersReducedMotion();
  const box = useRef<HTMLElement>(null);
  const [gpu, setGpu] = useState(false);
  const [near, setNear] = useState(false);
  const [broken, setBroken] = useState(false);
  const [key, setKey] = useState(0);
  const played = useRef(false);

  useEffect(() => {
    setGpu(typeof navigator !== "undefined" && "gpu" in navigator && new URLSearchParams(location.search).get("webgl") !== "0");
    const el = box.current;
    if (!el || typeof IntersectionObserver === "undefined") return;
    // it is built when the band is a screen away, and its entrance plays when the band is really in view
    const near = new IntersectionObserver((e) => e.some((x) => x.isIntersecting) && setNear(true), { rootMargin: "600px 0px" });
    const seen = new IntersectionObserver(
      (e) => {
        if (!played.current && e.some((x) => x.isIntersecting && x.intersectionRatio >= 0.4)) {
          played.current = true;
          setKey(1);
        }
      },
      { threshold: [0.4] },
    );
    near.observe(el);
    seen.observe(el);
    return () => {
      near.disconnect();
      seen.disconnect();
    };
  }, []);

  const on = gpu && near && !broken;

  return (
    <section ref={box} className="waves" aria-hidden="true">
      <div className="waves__still">
        <span className="waves__word">Strata-xeno</span>
      </div>
      {on && (
        <ShapeWaves
          className="waves__gl"
          text="Strata-xeno"
          fontFamily='"Inter Variable", system-ui, sans-serif'
          fontWeight={700}
          textSize={0.5}
          shapes="mixed"
          cellSize={13}
          dotSize={0.72}
          color="#64646e"
          hoverColor={botAvatarPalette.clover}
          backgroundColor="#0b0b0d"
          speed={0.8}
          brightness={0.45}
          fade={0.3}
          interactive={!reduced}
          splashRadius={70}
          splashStrength={0.5}
          glow={0.4}
          intro
          introDuration={1.8}
          introKey={key}
          onError={(e) => {
            console.warn("The shapes band could not start (a still band is shown instead):", e);
            setBroken(true);
          }}
        />
      )}
      <p className="waves__hint mono">
        <T v={{ en: "Move over it.", th: "ลองเลื่อนเมาส์ผ่าน" }} />
      </p>
    </section>
  );
}
