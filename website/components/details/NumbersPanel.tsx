"use client";

import { CHARTS, COUNTS, type ChartId, type Measured } from "@/data/numbers";
import { Chart } from "../charts/Chart";
import { T } from "@/lib/i18n";

export function MeasuredCard({ m }: { m: Measured }) {
  return (
    <article className="mcard" aria-labelledby={`m-${m.id}`}>
      <h3 className="h3" id={`m-${m.id}`}>
        <T v={m.label} />
      </h3>
      <Chart spec={CHARTS[m.id as ChartId]} />
      <dl className="conds">
        {m.conditions.map((c) => (
          <div key={c.k.en}>
            <dt className="mono">
              <T v={c.k} />
            </dt>
            <dd>
              <T v={c.v} />
            </dd>
          </div>
        ))}
      </dl>
      <p className="src mono">
        <T v={{ en: "Sources", th: "แหล่งที่มา" }} />:{" "}
        {m.source.map((s, i) => (
          <span key={s.href}>
            {i > 0 && ", "}
            <a href={s.href} rel="noopener">
              {s.label} <span aria-hidden="true">↗</span>
            </a>
          </span>
        ))}
      </p>
    </article>
  );
}

export function CountsPanel() {
  return (
    <article className="topic">
      <header className="topic__head">
        <h2 className="h3">
          <T v={{ en: "What was built, counted", th: "สิ่งที่สร้าง นับจากโค้ด" }} />
        </h2>
        <p className="topic__sum">
          <T
            v={{
              en: "Counted from the code at the stamped revision. A count is a size, not a quality measure.",
              th: "นับจากโค้ดที่ revision ที่ประทับไว้ จำนวนบอกขนาด ไม่ได้บอกคุณภาพ",
            }}
          />
        </p>
      </header>
      <ul className="counts">
        {COUNTS.map((c) => (
          <li key={c.label.en}>
            <b className="num">{c.value}</b>
            <span>
              <T v={c.label} />
            </span>
            <small className="mono">{c.method}</small>
          </li>
        ))}
      </ul>
    </article>
  );
}
