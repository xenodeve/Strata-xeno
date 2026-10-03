"use client";

import Link from "next/link";
import { LINKS, REV } from "@/data/sources";
import { T } from "@/lib/i18n";

export function Footer() {
  return (
    <footer className="foot">
      <div className="wrap foot__in">
        <p>
          <T
            v={{
              en: "Strata is the work of Niko1221 and the Strata contributors (MIT). Strata-xeno is a fork of it.",
              th: "Strata เป็นผลงานของ Niko1221 และผู้ร่วมพัฒนา (MIT) Strata-xeno เป็น fork ของมัน",
            }}
          />
        </p>
        <p className="mono foot__links">
          <a href={LINKS.fork} rel="noopener">xenodeve/Strata-xeno ↗</a>
          <a href={LINKS.upstream} rel="noopener">Niko1221/Strata ↗</a>
          <Link href="/details#credits">
            <T v={{ en: "Credits & method", th: "เครดิตและวิธีวัด" }} />
          </Link>
          <span>
            {REV.fork.sha} · {REV.fork.date}
          </span>
        </p>
        <p className="mono foot__fine">
          <T
            v={{
              en: "Not affiliated with Anthropic. “Claude Code” and “Anthropic” are named only to describe API compatibility.",
              th: "ไม่เกี่ยวข้องกับ Anthropic คำว่า “Claude Code” และ “Anthropic” ใช้เพื่ออธิบายความเข้ากันได้ของ API เท่านั้น",
            }}
          />
        </p>
      </div>
    </footer>
  );
}
