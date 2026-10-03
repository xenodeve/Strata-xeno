"use client";

import { useEffect, useRef, useState } from "react";
import { T, useT, type L } from "@/lib/i18n";
import { usePrefersReducedMotion } from "../Avatar";
import { Reveal } from "../Reveal";
import { HighlightedText, SlideUpText } from "../text/Fx";

export type FilmItem = { id: string; src: string; poster: string; tab: L; length: L; aria: string; caption: L };

const DWELL = 700; // how long half of the film must stay in view before it starts, in ms

/**
 * The short films made with Motion from the app's real screenshots.
 *
 * Nothing is downloaded while the visitor scrolls past: the poster is all there is. When half of the film has stayed on screen
 * for a moment it is fetched by the page itself and played from memory (a blob), muted (the only kind of autoplay a browser
 * allows); it pauses when it leaves. Because the video element is never given the file's address, a download manager that
 * watches video elements and media requests has nothing to catch and nothing to pop up over it. The native controls come
 * with the film: sound, taking it over, a pause that the visitor makes is respected until they press play. With reduced
 * motion, or with the browser's data saver on, it waits for the Play button.
 */
export function Film({ films }: { films: FilmItem[] }) {
  const t = useT();
  const [i, setI] = useState(0);
  const film = films[i];
  const ref = useRef<HTMLVideoElement>(null);
  const reduced = usePrefersReducedMotion();
  const [url, setUrl] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [failed, setFailed] = useState(false);
  const wantPlay = useRef(false); // the film is on screen and should be playing
  const byPage = useRef(false); // true while a pause is the page's own (the film left the screen)
  const byVisitor = useRef(false); // the visitor paused it: leave it alone
  const fetching = useRef<AbortController | null>(null);

  // another film: forget the first one's memory
  useEffect(() => {
    return () => {
      fetching.current?.abort();
      setUrl((u) => {
        if (u) URL.revokeObjectURL(u);
        return null;
      });
      setBusy(false);
      setFailed(false);
    };
  }, [i]);

  /** Fetch the file once, into memory. Resolves to its blob address, or null. */
  async function load(): Promise<string | null> {
    if (url) return url;
    if (fetching.current && busy) return null;
    const ac = new AbortController();
    fetching.current = ac;
    setBusy(true);
    setFailed(false);
    try {
      const res = await fetch(film.src, { signal: ac.signal });
      if (!res.ok) throw new Error("not found");
      const u = URL.createObjectURL(await res.blob());
      setUrl(u);
      return u;
    } catch {
      if (!ac.signal.aborted) setFailed(true);
      return null;
    } finally {
      if (fetching.current === ac) fetching.current = null;
      setBusy(false);
    }
  }

  // once it is in memory, play it if it is still meant to be playing
  useEffect(() => {
    const v = ref.current;
    if (url && v && wantPlay.current && !byVisitor.current) void v.play().catch(() => {});
  }, [url]);

  // start when half of it has stayed on screen for a moment, pause when it goes
  useEffect(() => {
    const v = ref.current;
    byPage.current = false;
    byVisitor.current = false;
    const saving = (navigator as Navigator & { connection?: { saveData?: boolean } }).connection?.saveData === true;
    if (!v || reduced || saving || typeof IntersectionObserver === "undefined") return;
    let timer = 0;
    const io = new IntersectionObserver(
      (entries) => {
        const on = entries.some((x) => x.isIntersecting && x.intersectionRatio >= 0.5);
        window.clearTimeout(timer);
        wantPlay.current = on;
        if (on) {
          if (!byVisitor.current) timer = window.setTimeout(() => (url ? void v.play().catch(() => {}) : void load()), DWELL);
        } else {
          // left the screen: stop a download that has not finished, pause a film that is playing
          if (!url) fetching.current?.abort();
          if (!v.paused) {
            byPage.current = true;
            v.pause();
          }
        }
      },
      { threshold: [0, 0.5, 1] },
    );
    io.observe(v);
    return () => {
      window.clearTimeout(timer);
      io.disconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reduced, i, url]);

  /** The Play button: the visitor asked for it, so fetch it and play. */
  async function start() {
    wantPlay.current = true;
    byVisitor.current = false;
    const u = await load();
    if (u) void ref.current?.play().catch(() => {});
  }

  return (
    <section className="section section--tight" id="film" aria-labelledby="film-title">
      <div className="wrap">
        <Reveal as="header" className="shead shead--center">
          <h2 id="film-title" className="h2">
            <SlideUpText v={{ en: "Watch it in", th: "ดูใน" }} />{" "}
            <HighlightedText delay={400}>
              <SlideUpText key={film.id} v={film.length} delay={120} />
            </HighlightedText>
          </h2>
        </Reveal>
        {films.length > 1 && (
          <div className="tabs tabs--center film__tabs" role="group" aria-label={t({ en: "Which film", th: "ฟิล์มเรื่องไหน" })}>
            {films.map((f, k) => (
              <button key={f.id} type="button" className="tabs__tab" aria-pressed={k === i} onClick={() => setI(k)}>
                <T v={f.tab} />
              </button>
            ))}
          </div>
        )}
        <Reveal i={1}>
          <figure className="film">
            <video
              key={film.id}
              ref={ref}
              src={url ?? undefined}
              controls={!!url}
              controlsList="nodownload noremoteplayback"
              disablePictureInPicture
              disableRemotePlayback
              muted
              loop
              playsInline
              preload="none"
              poster={film.poster}
              onPause={() => {
                if (byPage.current) byPage.current = false;
                else if (url) byVisitor.current = true;
              }}
              onPlay={() => {
                byVisitor.current = false;
              }}
              aria-label={film.aria}
            />
            {!url && (
              <button type="button" className="film__play" onClick={start} disabled={busy} aria-label={t({ en: "Play the film", th: "เล่นฟิล์ม" })}>
                <span aria-hidden="true">{busy ? "…" : "▶"}</span>
                <span className="mono">{failed ? <T v={{ en: "Could not load. Try again", th: "โหลดไม่สำเร็จ ลองอีกครั้ง" }} /> : busy ? <T v={{ en: "Loading", th: "กำลังโหลด" }} /> : <T v={{ en: "Play", th: "เล่น" }} />}</span>
              </button>
            )}
            <figcaption className="cap mono">
              <T v={film.caption} />
            </figcaption>
          </figure>
        </Reveal>
      </div>
    </section>
  );
}
