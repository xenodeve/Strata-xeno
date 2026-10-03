"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { allocate, share, STRATA, type Alloc, type StratumId } from "@/lib/strata";
import type { SceneApi } from "@/lib/strataScene";
import { T, useT, type L } from "@/lib/i18n";
import Link from "next/link";
import { usePrefersReducedMotion } from "../Avatar";
import { Reveal, useInView } from "../Reveal";
import { HighlightedText, RandomizedText, SlideUpText } from "../text/Fx";

const LABEL: Record<StratumId, { name: L; hint: L }> = {
  gpu1: { name: { en: "GPU 1", th: "GPU 1" }, hint: { en: "VRAM of the primary card", th: "VRAM ของการ์ดหลัก" } },
  gpu2: { name: { en: "GPU 2", th: "GPU 2" }, hint: { en: "VRAM of the second card", th: "VRAM ของการ์ดใบที่สอง" } },
  ram: { name: { en: "RAM", th: "RAM" }, hint: { en: "system memory", th: "หน่วยความจำของระบบ" } },
  nvme: { name: { en: "NVMe", th: "NVMe" }, hint: { en: "capacity mode only", th: "เฉพาะ capacity mode" } },
};

/** The same four strata as plain bars and text: the reader view, and the fallback. */
function StrataReader({ alloc, large }: { alloc: Alloc; large?: boolean }) {
  return (
    <figure className={"reader" + (large ? " reader--large" : "")}>
      <ol className="reader__rows">
        {STRATA.map((id) => (
          <li key={id} className={`reader__row reader__row--${id}`}>
            <span className="reader__name">
              <b><T v={LABEL[id].name} /></b> <small><T v={LABEL[id].hint} /></small>
            </span>
            <span className="reader__bar" aria-hidden="true">
              <span style={{ width: `${share(alloc[id])}%` }} />
            </span>
            <span className="reader__val num">{share(alloc[id])} %</span>
          </li>
        ))}
      </ol>
      <figcaption className="cap mono">
        <T
          v={{
            en: "Share of a schematic set of experts in each stratum.",
            th: "สัดส่วนของ expert ชุดจำลองในแต่ละชั้น",
          }}
        />
      </figcaption>
    </figure>
  );
}

/** A number that moves to its new value (ease-out, `ms`) instead of jumping: here the 0 / 1 of the Capacity switch. */
function useEased(target: number, reduced: boolean, ms = 450): number {
  const [v, setV] = useState(target);
  const cur = useRef(target);
  useEffect(() => {
    if (reduced) {
      cur.current = target;
      setV(target);
      return;
    }
    const from = cur.current;
    const t0 = performance.now();
    let raf = 0;
    const tick = (now: number) => {
      const p = Math.min(1, (now - t0) / ms);
      cur.current = from + (target - from) * (1 - Math.pow(1 - p, 4));
      setV(cur.current);
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, reduced, ms]);
  return v;
}

export function Strata() {
  const t = useT();
  const [vram, setVram] = useState(0.55); // what the picture shows: it follows the slider, a little behind
  const [goal, setGoal] = useState<number | null>(null); // where the slider has been put (null until it has been touched)
  const vramRef = useRef(0.55);
  vramRef.current = vram;
  const [capacity, setCapacity] = useState(false);
  const [mode, setMode] = useState<"pending" | "webgl" | "fallback">("pending");

  const reduced = usePrefersReducedMotion();
  const [introRef, introSeen] = useInView<HTMLDivElement>("0px 0px -25% 0px");
  const touched = useRef(false);
  const stageRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sceneRef = useRef<SceneApi | null>(null);
  const stateRef = useRef({ vram, capacity });
  stateRef.current = { vram, capacity };

  const alloc = allocate(vram, capacity);
  const gpuShare = share(alloc.gpu1 + alloc.gpu2);
  // the readout moves between the two modes: how much of the way to Capacity mode the figures are (0 .. 1), so RAM gives its share
  // to NVMe over a moment and the NVMe line opens, rather than both jumping
  const c = useEased(capacity ? 1 : 0, reduced);
  const off = allocate(vram, false);
  const on = allocate(vram, true);
  const shownRam = Math.round(share(off.ram) + (share(on.ram) - share(off.ram)) * c);
  const shownNvme = Math.round(share(off.nvme) + (share(on.nvme) - share(off.nvme)) * c);

  // start the scene once, when the section is close to the screen
  useEffect(() => {
    const stage = stageRef.current;
    if (!stage) return;
    const params = new URLSearchParams(window.location.search);
    const weak = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
    if (params.get("webgl") === "0" || (typeof weak === "number" && weak <= 2)) {
      setMode("fallback");
      return;
    }
    let cancelled = false;
    let io: IntersectionObserver | undefined;
    let ro: ResizeObserver | undefined;
    let visible = false;
    const apply = () => sceneRef.current?.setActive(visible && !document.hidden);
    const onVis = () => apply();

    const start = async () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      try {
        const { createScene } = await import("@/lib/strataScene");
        if (cancelled) return;
        const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
        const scene = createScene(canvas, { reducedMotion: reduced, onLost: () => setMode("fallback") });
        if (!scene) {
          setMode("fallback");
          return;
        }
        sceneRef.current = scene;
        scene.setState(stateRef.current.vram, stateRef.current.capacity);
        scene.resize();
        ro = new ResizeObserver(() => scene.resize());
        ro.observe(canvas);
        setMode("webgl");
        apply();
      } catch {
        if (!cancelled) setMode("fallback");
      }
    };

    io = new IntersectionObserver(
      (entries) => {
        visible = entries.some((e) => e.isIntersecting);
        if (visible && !sceneRef.current) void start();
        apply();
      },
      { rootMargin: "300px 0px" },
    );
    io.observe(stage);
    document.addEventListener("visibilitychange", onVis);
    return () => {
      cancelled = true;
      io?.disconnect();
      ro?.disconnect();
      document.removeEventListener("visibilitychange", onVis);
      sceneRef.current?.dispose();
      sceneRef.current = null;
    };
  }, []);

  useEffect(() => {
    sceneRef.current?.setState(vram, capacity);
  }, [vram, capacity]);

  // the picture eases toward the slider (the app's own ease-out, about a quarter of a second to settle) instead of jumping with every step of it
  useEffect(() => {
    if (goal === null) return;
    if (reduced) {
      setVram(goal);
      return;
    }
    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      const dt = Math.min(64, now - last);
      last = now;
      const cur = vramRef.current;
      const next = cur + (goal - cur) * (1 - Math.pow(0.0004, dt / 1000));
      if (Math.abs(goal - next) < 0.0008) {
        setVram(goal);
        return;
      }
      setVram(next);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [goal, reduced]);

  // when it arrives it plays once: from no free VRAM up to the starting point, so the idea is seen before it is touched
  useEffect(() => {
    if (!introSeen || reduced || touched.current) return;
    let raf = 0;
    const t0 = performance.now();
    const tick = (now: number) => {
      if (touched.current) return;
      const p = Math.min(1, (now - t0) / 2200);
      setVram(0.55 * (1 - Math.pow(1 - p, 3)));
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    setVram(0);
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [introSeen, reduced]);

  // drag to turn; a vertical swipe still scrolls the page (touch-action: pan-y)
  const drag = useRef<{ x: number; t: number; v: number } | null>(null);
  const onDown = useCallback((e: React.PointerEvent) => {
    drag.current = { x: e.clientX, t: performance.now(), v: 0 };
    sceneRef.current?.setDragging(true);
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
  }, []);
  const onMove = useCallback((e: React.PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    const dx = e.clientX - d.x;
    const now = performance.now();
    d.v = (dx * 0.008) / Math.max(0.008, (now - d.t) / 1000);
    d.x = e.clientX;
    d.t = now;
    sceneRef.current?.rotateBy(dx * 0.008);
  }, []);
  const onUp = useCallback(() => {
    const d = drag.current;
    drag.current = null;
    sceneRef.current?.setDragging(false);
    if (d && Math.abs(d.v) > 0.4) sceneRef.current?.fling(d.v);
  }, []);
  const onKey = useCallback((e: React.KeyboardEvent) => {
    if (e.key === "ArrowLeft") {
      sceneRef.current?.rotateBy(-0.25);
      e.preventDefault();
    } else if (e.key === "ArrowRight") {
      sceneRef.current?.rotateBy(0.25);
      e.preventDefault();
    }
  }, []);

  return (
    <section className="section section--deep" id="strata" aria-labelledby="strata-title">
      <div className="wrap">
        <Reveal as="header" className="shead shead--center">
          <h2 id="strata-title" className="h2">
            <SlideUpText v={{ en: "Every GiB a GPU owns is", th: "ทุก GiB ที่ GPU ถือ คือ" }} />{" "}
            <HighlightedText delay={500}>
              <SlideUpText v={{ en: "a GiB less RAM.", th: "RAM ที่ลดลง 1 GiB" }} delay={200} />
            </HighlightedText>
          </h2>
          <p className="shead__lede">
            <RandomizedText split="words" duration={1100} v={{ en: "Drag the slider. The more VRAM, the less RAM.", th: "ลากแถบ ยิ่งมี VRAM มาก ก็ยิ่งใช้ RAM น้อย" }} />
          </p>
        </Reveal>

        <div className="strata" ref={introRef}>
          <div className="strata__stage" ref={stageRef}>
            {mode !== "fallback" ? (
              <div
                className="strata__canvaswrap"
                tabIndex={0}
                role="group"
                aria-label={t({
                  en: "Core sample of the memory strata. Drag or use the left and right arrow keys to turn it.",
                  th: "ตัวอย่างแท่งชั้นหน่วยความจำ ลากหรือกดลูกศรซ้าย/ขวาเพื่อหมุน",
                })}
                onKeyDown={onKey}
              >
                <canvas
                  ref={canvasRef}
                  className="strata__canvas"
                  onPointerDown={onDown}
                  onPointerMove={onMove}
                  onPointerUp={onUp}
                  onPointerCancel={onUp}
                />
                <ol className="strata__labels" aria-hidden="true">
                  {STRATA.map((id) => (
                    <li key={id}>
                      <b className="mono"><T v={LABEL[id].name} /></b>
                      <small><T v={LABEL[id].hint} /></small>
                    </li>
                  ))}
                </ol>
                {mode === "pending" && (
                  <p className="strata__wait mono" role="status">
                    <T v={{ en: "Loading the 3D view…", th: "กำลังโหลดภาพ 3 มิติ…" }} />
                  </p>
                )}
              </div>
            ) : (
              <StrataReader alloc={alloc} large />
            )}
            <p className="strata__note mono">
              <T
                v={{
                  en: "Illustration of the principle. Expert counts and sizes are schematic: this is not a measurement and not live telemetry.",
                  th: "ภาพอธิบายหลักการ จำนวนและขนาด expert เป็นแบบจำลอง ไม่ใช่ค่าที่วัด และไม่ใช่ telemetry สด",
                }}
              />
            </p>
          </div>

          <div className="strata__side">
            <div className="controls">
              <label className="control" htmlFor="vram">
                <span className="control__label">
                  <T v={{ en: "Free VRAM", th: "VRAM ว่าง" }} />
                </span>
                <input
                  id="vram"
                  className="slider"
                  type="range"
                  min={0}
                  max={1000}
                  step={1}
                  value={Math.round((goal ?? vram) * 1000)}
                  style={{ "--p": `${((goal ?? vram) * 100).toFixed(2)}%` } as React.CSSProperties}
                  onChange={(e) => {
                    touched.current = true;
                    setGoal(Number(e.target.value) / 1000);
                  }}
                  aria-valuetext={t({ en: `${Math.round((goal ?? vram) * 100)} % of the schematic maximum`, th: `${Math.round((goal ?? vram) * 100)}% ของค่าสูงสุดในแบบจำลอง` })}
                />
                <span className="control__scale mono" aria-hidden="true">
                  <span><T v={{ en: "less", th: "น้อย" }} /></span>
                  <span><T v={{ en: "more", th: "มาก" }} /></span>
                </span>
              </label>
              <div className="switch-row">
                <div className="switch-row__text">
                  <span id="cap-label" className="switch-row__label">
                    <T v={{ en: "Capacity mode", th: "Capacity mode" }} />
                  </span>
                  <small id="cap-hint">
                    <T
                      v={{
                        en: "RAM becomes a bounded cache; the rest of the experts sit on NVMe.",
                        th: "RAM กลายเป็น cache ที่จำกัดขนาด ส่วน expert ที่เหลืออยู่บน NVMe",
                      }}
                    />
                  </small>
                </div>
                <button
                  type="button"
                  role="switch"
                  aria-checked={capacity}
                  aria-labelledby="cap-label"
                  aria-describedby="cap-hint"
                  className="switch"
                  onClick={() => {
                    touched.current = true;
                    setCapacity(!capacity);
                  }}
                >
                  <span className="switch__knob" aria-hidden="true" />
                </button>
              </div>
            </div>

            <div className="readout" aria-live="polite">
              <p>
                <b className="num">{gpuShare} %</b> <T v={{ en: "of the experts are owned by a GPU", th: "ของ expert ถูกถือโดย GPU" }} />
              </p>
              <p>
                <b className="num">{shownRam} %</b> <T v={{ en: "stay in RAM", th: "อยู่ใน RAM" }} />
              </p>
              <div className="readout__nvme" aria-hidden={!capacity} style={{ gridTemplateRows: `${c}fr`, opacity: c }}>
                <div>
                  <p>
                    <b className="num">{shownNvme} %</b> <T v={{ en: "wait on NVMe", th: "รออยู่บน NVMe" }} />
                  </p>
                </div>
              </div>
            </div>

          </div>
        </div>

        <p className="strata__more">
          <Link className="btn" href="/details#dynamic-experts">
            <T v={{ en: "How it is built", th: "สร้างอย่างไร" }} /> <span aria-hidden="true">→</span>
          </Link>
          <Link className="btn btn--link" href="/details#ram">
            <T v={{ en: "The measured numbers", th: "ตัวเลขที่วัดได้" }} />
          </Link>
        </p>
      </div>
    </section>
  );
}
