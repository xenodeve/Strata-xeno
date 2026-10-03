"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { T, type L } from "@/lib/i18n";
import { usePrefersReducedMotion } from "../Avatar";
import { CHARTS, type ChartId } from "@/data/numbers";
import { Chart } from "../charts/Chart";
import { Reveal, useInView } from "../Reveal";
import { HighlightedText, SlideUpText } from "../text/Fx";

function useCountUp(to: number, run: boolean, ms = 1100, decimals = 1) {
  const reduced = usePrefersReducedMotion();
  const [v, setV] = useState(0);
  useEffect(() => {
    if (!run) return;
    if (reduced) {
      setV(to);
      return;
    }
    let raf = 0;
    const t0 = performance.now();
    const tick = (now: number) => {
      const p = Math.min(1, (now - t0) / ms);
      const e = 1 - Math.pow(1 - p, 4); // ease-out quart, the app's own feel
      setV(to * e);
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [to, run, ms, reduced]);
  return v.toFixed(decimals);
}

type Stat = {
  id: string;
  label: L;
  mine: { value: number; unit: string; tag: L };
  theirs: { value: number; tag: L };
  href: string;
};

const STATS: Stat[] = [
  {
    id: "ram",
    label: { en: "RAM, two GPUs · lower is better", th: "RAM, สอง GPU · ยิ่งต่ำยิ่งดี" },
    mine: { value: 16.5, unit: "GiB", tag: { en: "Strata-xeno", th: "Strata-xeno" } },
    theirs: { value: 35.3, tag: { en: "upstream v0.1.26", th: "upstream v0.1.26" } },
    href: "/details#ram",
  },
  {
    id: "decode",
    label: { en: "Decode speed", th: "ความเร็ว decode" },
    mine: { value: 59.0, unit: "tok/s", tag: { en: "Strata-xeno", th: "Strata-xeno" } },
    theirs: { value: 41.0, tag: { en: "upstream v0.1.37", th: "upstream v0.1.37" } },
    href: "/details#decode",
  },
];

function StatCard({ s, k }: { s: Stat; k: number }) {
  const [ref, seen] = useInView<HTMLDivElement>("0px 0px -12% 0px");
  const mine = useCountUp(s.mine.value, seen);
  const theirs = useCountUp(s.theirs.value, seen);
  return (
    <Reveal i={k} className="stat">
      <div ref={ref}>
        <p className="mono stat__label">
          <T v={s.label} />
        </p>
        <div className="stat__row">
          <div className="stat__col stat__col--mine">
            <b className="stat__n num">{mine}</b>
            <span className="stat__u">{s.mine.unit}</span>
            <small className="mono">
              <T v={s.mine.tag} />
            </small>
          </div>
          <div className="stat__col">
            <b className="stat__n num">{theirs}</b>
            <span className="stat__u">{s.mine.unit}</span>
            <small className="mono">
              <T v={s.theirs.tag} />
            </small>
          </div>
        </div>
        <Chart spec={CHARTS[s.id as ChartId]} compact />
        <Link className="stat__link" href={s.href}>
          <T v={{ en: "Full conditions", th: "เงื่อนไขทั้งหมด" }} /> <span aria-hidden="true">→</span>
        </Link>
      </div>
    </Reveal>
  );
}

export function Stats() {
  return (
    <section className="section" id="numbers" aria-labelledby="numbers-title">
      <div className="wrap">
        <Reveal as="header" className="shead shead--center">
          <h2 id="numbers-title" className="h2">
            <SlideUpText v={{ en: "Two numbers,", th: "สองตัวเลข" }} />{" "}
            <HighlightedText delay={400}>
              <SlideUpText v={{ en: "with their conditions.", th: "พร้อมเงื่อนไข" }} delay={120} />
            </HighlightedText>
          </h2>
          <p className="shead__lede">
            <T v={{ en: "Same session, same machine, like for like.", th: "เซสชันเดียวกัน เครื่องเดียวกัน เทียบแบบเดียวกัน" }} />
          </p>
        </Reveal>
        <div className="stats">
          {STATS.map((s, k) => (
            <StatCard key={s.id} s={s} k={k} />
          ))}
        </div>
      </div>
    </section>
  );
}
