"use client";

import { useMemo, useState } from "react";
import { STATUS_LABEL, type Status } from "@/data/features";
import { REG_CATEGORIES, REGISTER, type RegCategory } from "@/data/register";
import { T, useLang, useT } from "@/lib/i18n";

const STATUSES: Status[] = ["shipped", "opt-in", "off-by-default"];

export function RegisterPanel() {
  const t = useT();
  const { lang } = useLang();
  const [cat, setCat] = useState<RegCategory | "all">("all");
  const [status, setStatus] = useState<Status | "all">("all");
  const [q, setQ] = useState("");

  const rows = useMemo(() => {
    const needle = q.trim().toLowerCase();
    return REGISTER.filter((r) => {
      if (cat !== "all" && r.cat !== cat) return false;
      if (status !== "all" && r.status !== status) return false;
      if (!needle) return true;
      return (r.title.en + " " + r.title.th + " " + r.benefit.en + " " + r.benefit.th + " " + (r.tests ?? "")).toLowerCase().includes(needle);
    });
  }, [cat, status, q]);

  return (
    <article className="topic">
      <header className="topic__head">
        <h2 className="h3">
          <T v={{ en: "Everything it adds", th: "ทุกสิ่งที่เพิ่มเข้ามา" }} />
        </h2>
        <p className="topic__sum">
          <T
            v={{
              en: "The full register, found in the code rather than in the documents. Each entry: what it does, where it lives, its tests and its limits.",
              th: "ทะเบียนฉบับเต็ม ที่ค้นจากโค้ด ไม่ใช่จากเอกสาร แต่ละรายการ: ทำอะไร อยู่ที่ไหน มี test อะไร และมีข้อจำกัดอะไร",
            }}
          />
        </p>
      </header>
      <div>
        <div className="filters">
          <label className="search">
            <span className="sr-only">{t({ en: "Search the register", th: "ค้นหาในทะเบียน" })}</span>
            <input
              type="search"
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder={t({ en: "Search: hooks, NVMe, Thai, MCP…", th: "ค้นหา: hooks, NVMe, ไทย, MCP…" })}
              autoComplete="off"
            />
          </label>
          <div className="chips" role="group" aria-label={t({ en: "Category", th: "หมวด" })}>
            <button type="button" className="chip" aria-pressed={cat === "all"} onClick={() => setCat("all")}>
              <T v={{ en: "All", th: "ทั้งหมด" }} /> <span className="num">{REGISTER.length}</span>
            </button>
            {REG_CATEGORIES.map((c) => (
              <button key={c.id} type="button" className="chip" aria-pressed={cat === c.id} onClick={() => setCat(c.id)}>
                <T v={c.label} /> <span className="num">{REGISTER.filter((r) => r.cat === c.id).length}</span>
              </button>
            ))}
          </div>
          <div className="chips" role="group" aria-label={t({ en: "Status", th: "สถานะ" })}>
            <button type="button" className="chip" aria-pressed={status === "all"} onClick={() => setStatus("all")}>
              <T v={{ en: "Any status", th: "ทุกสถานะ" }} />
            </button>
            {STATUSES.map((s) => (
              <button key={s} type="button" className="chip" aria-pressed={status === s} onClick={() => setStatus(s)}>
                <T v={STATUS_LABEL[s]} />
              </button>
            ))}
          </div>
        </div>

        <p className="count mono" aria-live="polite">
          {rows.length} / {REGISTER.length}
        </p>

        {rows.length === 0 ? (
          <p className="empty">
            <T v={{ en: "Nothing matches. Clear a filter or the search.", th: "ไม่พบรายการที่ตรงกัน ลองล้างตัวกรองหรือคำค้น" }} />
          </p>
        ) : (
          <ul className="reg">
            {rows.map((r) => (
              <li key={r.id}>
                <details className="row">
                  <summary>
                    <span className="row__title">
                      <T v={r.title} />
                    </span>
                    <span className="row__meta">
                      <span className="mono row__cat">
                        <T v={REG_CATEGORIES.find((c) => c.id === r.cat)!.label} />
                      </span>
                      <span className={`badge badge--${r.status}`}>
                        <T v={STATUS_LABEL[r.status]} />
                      </span>
                    </span>
                  </summary>
                  <div className="row__body">
                    <p>{r.benefit[lang]}</p>
                    {r.note && (
                      <p className="note">
                        <T v={r.note} />
                      </p>
                    )}
                    <ul className="evidence">
                      {r.where.map((w) => (
                        <li key={w.href}>
                          <a className="mono" href={w.href} rel="noopener">
                            {w.label} <span aria-hidden="true">↗</span>
                          </a>
                        </li>
                      ))}
                    </ul>
                    {r.tests && (
                      <p className="mono row__tests">
                        <T v={{ en: "Tests", th: "Test" }} />: {r.tests}
                      </p>
                    )}
                  </div>
                </details>
              </li>
            ))}
          </ul>
        )}
      </div>
    </article>
  );
}
