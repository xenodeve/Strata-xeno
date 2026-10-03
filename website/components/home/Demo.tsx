"use client";

import { useEffect, useRef, useState } from "react";
import { T, useT, type L } from "@/lib/i18n";
import { Reveal, useInView } from "../Reveal";
import { HighlightedText, SlideUpText } from "../text/Fx";

const SCENES: { id: string; hash: string; label: L; hint?: L }[] = [
  { id: "chat", hash: "#/chat", label: { en: "Chat", th: "แชต" }, hint: { en: "Type “agent demo” to see the coding tools ask before they act.", th: "พิมพ์ “agent demo” เพื่อดูเครื่องมือเขียนโค้ดถามก่อนลงมือ" } },
  { id: "dashboard", hash: "#/dashboard", label: { en: "Dashboard", th: "Dashboard" } },
  { id: "live", hash: "#/live", label: { en: "Live", th: "Live" } },
  { id: "requests", hash: "#/requests", label: { en: "Requests", th: "Requests" } },
  { id: "hardware", hash: "#/hardware", label: { en: "Hardware", th: "Hardware" } },
  { id: "settings", hash: "#/settings", label: { en: "Settings", th: "Settings" }, hint: { en: "Try Status marks: pick Avatar.", th: "ลอง Status marks: เลือก Avatar" } },
];

const SRC = "/demo/next/";
// The app lays itself out by the width of its own window: from 1280 px it docks its side panel beside the chat, below that it draws the panel
// as a sheet over it. The frame is a little narrower than that, so it is given a window of 1290 px and shown scaled to fit (the app is then
// drawn as it is on a desktop, the panel docked, at about 96 %); on a screen too narrow for that (under 960 px) it is shown as it is.
const WIDE = 1290;
const NARROWEST = 960;

/**
 * The real app, running. It is served by the real server code of Strata-xeno (serve/*.py) in its no-GPU mock mode
 * (`npm run demo`): a script answers instead of the model, the engine's statistics are fixtures. This site only frames it.
 */
export function Demo() {
  const t = useT();
  const [rootRef, seen] = useInView<HTMLDivElement>("200px 0px 200px 0px");
  const frame = useRef<HTMLIFrameElement>(null);
  const [scene, setScene] = useState(SCENES[0]);
  const [up, setUp] = useState<"checking" | "up" | "down">("checking");
  const [loaded, setLoaded] = useState(false);
  const stage = useRef<HTMLDivElement>(null);
  const [fit, setFit] = useState<{ k: number; h: number } | null>(null);

  useEffect(() => {
    const el = stage.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const measure = () => {
      const w = el.clientWidth;
      const h = el.clientHeight;
      setFit(w >= NARROWEST && w < WIDE ? { k: w / WIDE, h } : null);
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    if (!seen) return;
    let cancelled = false;
    fetch("/demo/health", { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then(() => !cancelled && setUp("up"))
      .catch(() => !cancelled && setUp("down"));
    return () => {
      cancelled = true;
    };
  }, [seen]);

  function go(next: (typeof SCENES)[number]) {
    setScene(next);
    try {
      const w = frame.current?.contentWindow;
      if (w) w.location.hash = next.hash;
    } catch {
      /* the frame is not ready yet: the next load uses the new hash */
    }
  }

  return (
    <section className="section" id="demo" aria-labelledby="demo-title">
      <div className="wrap">
        <Reveal as="header" className="shead shead--center">
          <h2 id="demo-title" className="h2">
            <SlideUpText v={{ en: "Try the", th: "ลอง" }} />{" "}
            <HighlightedText delay={400}>
              <SlideUpText v={{ en: "real app.", th: "แอปตัวจริง" }} delay={120} />
            </HighlightedText>
          </h2>
          <p className="shead__lede">
            <T
              v={{
                en: "Not a mock-up: the app's own code, running here.",
                th: "ไม่ใช่ภาพจำลอง: โค้ดของแอปเอง รันอยู่ตรงนี้",
              }}
            />
          </p>
        </Reveal>

        <Reveal i={1}>
          <div ref={rootRef} className="demo">
            <div className="tabs tabs--center demo__tabs" role="tablist" aria-label={t({ en: "Screens of the demo", th: "หน้าจอของ demo" })}>
              {SCENES.map((s) => (
                <button key={s.id} role="tab" type="button" aria-selected={scene.id === s.id} className="tabs__tab" onClick={() => go(s)}>
                  <T v={s.label} />
                </button>
              ))}
            </div>

            <div className="demo__window">
              <div className="demo__bar" aria-hidden="true">
                <span />
                <span />
                <span />
                <em className="mono">127.0.0.1:8187/next/ · demo</em>
              </div>
              <div ref={stage} className="demo__stage">
                {up === "up" && (
                  <iframe
                    ref={frame}
                    className={"demo__frame" + (loaded ? " is-loaded" : "")}
                    src={SRC + scene.hash}
                    title={t({ en: "Strata-xeno web app, running as a demo", th: "เว็บแอป Strata-xeno ที่รันเป็น demo" })}
                    onLoad={() => setLoaded(true)}
                    style={fit ? { inset: "0 auto auto 0", width: WIDE, height: fit.h / fit.k, transform: `scale(${fit.k})`, transformOrigin: "0 0" } : undefined}
                    loading="lazy"
                  />
                )}
                {up !== "up" && (
                  <div className="demo__off">
                    <p className="demo__offtitle">
                      {up === "checking" ? <T v={{ en: "Starting the demo…", th: "กำลังเริ่ม demo…" }} /> : <T v={{ en: "The demo backend is not running.", th: "demo backend ยังไม่ได้รัน" }} />}
                    </p>
                    {up === "down" && (
                      <>
                        <p>
                          <T
                            v={{
                              en: "Start the real server in its no-GPU demo mode, then reload this page:",
                              th: "เริ่มเซิร์ฟเวอร์จริงในโหมด demo ที่ไม่ใช้ GPU แล้วโหลดหน้านี้ใหม่:",
                            }}
                          />
                        </p>
                        <code className="mono">STRATA_SRC=&lt;your Strata-xeno checkout&gt; npm run demo</code>
                      </>
                    )}
                  </div>
                )}
                {up === "up" && !loaded && <div className="demo__wait mono" role="status"><T v={{ en: "Loading the app…", th: "กำลังโหลดแอป…" }} /></div>}
              </div>
            </div>

            <div className="demo__foot">
              <p className="cap mono">
                <T
                  v={{
                    en: "A script answers instead of the model; the engine's statistics are fixtures; the hardware shown is this PC's. Nothing is sent anywhere.",
                    th: "สคริปต์ตอบแทนโมเดล สถิติของเอนจินเป็นข้อมูลตัวอย่าง ฮาร์ดแวร์ที่แสดงเป็นของเครื่องนี้ ไม่มีอะไรถูกส่งออกไปที่ไหน",
                  }}
                />
              </p>
              <p className="demo__links">
                {scene.hint && (
                  <span className="demo__hint">
                    <T v={scene.hint} />
                  </span>
                )}
                <a className="btn btn--ghost" href={SRC + scene.hash} target="_blank" rel="noopener noreferrer">
                  <T v={{ en: "Open in a new tab", th: "เปิดในแท็บใหม่" }} /> <span aria-hidden="true">↗</span>
                </a>
              </p>
            </div>
          </div>
        </Reveal>
      </div>
    </section>
  );
}
