"use client";

import { useEffect, useState } from "react";
import { T, useT, type L } from "@/lib/i18n";
import { usePrefersReducedMotion } from "../Avatar";
import { useInView } from "../Reveal";

export type ChartSeries = {
  name: L;
  /** mine = Strata-xeno (the accent), theirs = what it is compared with (grey), plain = a single series. */
  tone: "mine" | "theirs" | "plain";
  values: number[];
  /** What to write beside each bar when the figure is not just the number (a range, a rounded percentage). */
  labels?: string[];
};

/**
 * A comparison chart: horizontal grouped bars on a scale that starts at zero, the unit on every figure, which direction is
 * better, and the conditions underneath. It only ever draws numbers that the Numbers section gives, taken from the same data.
 */
export type ChartSpec = {
  id: string;
  title: L;
  unit: string;
  better?: "lower" | "higher";
  groups: L[];
  series: ChartSeries[];
  /** The end of the scale and the distance between grid lines. */
  max: number;
  step: number;
  digits?: number;
  /** One line per group, written at the end of its bars (the difference between the two). */
  deltas?: L[];
  /** The conditions, in a line: same session, same machine, what was measured. */
  caption: L;
};

export function Chart({ spec, compact = false }: { spec: ChartSpec; compact?: boolean }) {
  const t = useT();
  const reduced = usePrefersReducedMotion();
  const [ref, seen] = useInView<HTMLElement>("0px 0px -12% 0px");
  const [on, setOn] = useState(false);
  useEffect(() => {
    if (seen) setOn(true);
  }, [seen]);
  const grown = on || reduced;
  const ticks = Array.from({ length: Math.round(spec.max / spec.step) + 1 }, (_, i) => i * spec.step);
  const d = spec.digits ?? 1;
  const fmt = (v: number) => v.toFixed(Number.isInteger(v) && d === 0 ? 0 : d);

  return (
    <figure ref={ref} className={"chart" + (compact ? " chart--compact" : "")} aria-label={t(spec.title)}>
      <div className="chart__top">
        <figcaption className="chart__title mono">
          <T v={spec.title} />
        </figcaption>
        {spec.better && (
          <span className="chart__dir mono">
            <span aria-hidden="true">{spec.better === "lower" ? "↓" : "↑"}</span>{" "}
            <T v={spec.better === "lower" ? { en: "lower is better", th: "ยิ่งต่ำยิ่งดี" } : { en: "higher is better", th: "ยิ่งสูงยิ่งดี" }} />
          </span>
        )}
      </div>

      <div className="chart__plot" style={{ "--n": ticks.length - 1 } as React.CSSProperties} aria-hidden="true">
        {spec.groups.map((g, gi) => (
          <div className="chart__group" key={gi}>
            <div className="chart__glabel">
              <T v={g} />
            </div>
            <div className="chart__bars">
              {spec.series.map((s, si) => {
                const v = s.values[gi];
                const pct = (v / spec.max) * 100;
                const inside = pct > 68; // a long bar carries its figure inside, so it never runs off the scale
                const delay = `${gi * 160 + si * 90}ms`;
                return (
                  <div className="chart__row" key={si}>
                    <span className={"chart__bar chart__bar--" + s.tone} style={{ width: grown ? `${pct}%` : "0%", transitionDelay: delay }} />
                    <span
                      className={"chart__val num chart__val--" + s.tone + (inside ? " is-in" : "")}
                      style={{ left: inside ? undefined : `calc(${pct}% + 8px)`, right: inside ? `${100 - pct + 1.2}%` : undefined, opacity: grown ? 1 : 0, transitionDelay: `${gi * 160 + si * 90 + 450}ms` }}
                    >
                      {s.labels?.[gi] ?? fmt(v)} {spec.unit && <small>{spec.unit}</small>}
                    </span>
                  </div>
                );
              })}
            </div>
            {spec.deltas ? (
              <span className="chart__delta mono" style={{ opacity: grown ? 1 : 0, transitionDelay: `${gi * 160 + 900}ms` }}>
                <T v={spec.deltas[gi]} />
              </span>
            ) : (
              <span className="chart__none" />
            )}
          </div>
        ))}
        <span className="chart__lead" />
        <div className="chart__axis">
          {ticks.map((v) => (
            <span key={v} style={{ left: `${(v / spec.max) * 100}%` }}>
              {v}
            </span>
          ))}
        </div>
        <span className="chart__none" />
      </div>

      <ul className="chart__legend mono" aria-hidden="true">
        {spec.series.map((s, i) => (
          <li key={i}>
            <i className={"chart__sw chart__sw--" + s.tone} /> <T v={s.name} />
          </li>
        ))}
      </ul>

      <p className="chart__cap">
        <T v={spec.caption} />
      </p>

      {/* the same numbers, for a screen reader */}
      <div className="sr-only">
      <table>
        <caption>{t(spec.title)}</caption>
        <thead>
          <tr>
            <td />
            {spec.series.map((s, i) => (
              <th scope="col" key={i}>
                {t(s.name)} ({spec.unit})
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {spec.groups.map((g, gi) => (
            <tr key={gi}>
              <th scope="row">{t(g)}</th>
              {spec.series.map((s, si) => (
                <td key={si}>{s.labels?.[gi] ?? fmt(s.values[gi])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      </div>
    </figure>
  );
}
