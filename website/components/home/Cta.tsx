"use client";

import Link from "next/link";
import { T } from "@/lib/i18n";
import { SLIDES } from "@/data/story";
import { botAvatarPalette } from "@/vendor/bot-avatars/index.es.js";
import { GitHubButton } from "../GitHubButton";
import { Reveal } from "../Reveal";
import { HighlightedText, RandomizedText, SlideUpText } from "../text/Fx";
import { ParticleField, TechWord } from "../text/Words";

/** The nine moods' colours, for the points behind the closing name. */
const MOOD_COLORS = SLIDES.map((s) => botAvatarPalette[s.avatar]);

export function Cta() {
  return (
    <section className="section cta-band" aria-labelledby="cta-title">
      <ParticleField className="cta-band__particles" colors={MOOD_COLORS} count={60} size={200} spread={12} speed={0.06} />
      <div className="wrap">
        <Reveal className="cta-band__in">
          <TechWord text="Strata-xeno" />
          <p className="cap mono cta-band__hint">
            <T v={{ en: "Move over the name. Drag a letter.", th: "เลื่อนเมาส์ผ่านชื่อ ลองลากตัวอักษร" }} />
          </p>
          <h2 id="cta-title" className="display display--sm">
            <SlideUpText v={{ en: "Want the", th: "อยากเห็น" }} />{" "}
            <HighlightedText delay={400}>
              <SlideUpText v={{ en: "whole picture?", th: "ภาพทั้งหมด?" }} delay={120} />
            </HighlightedText>
          </h2>
          <p className="lede">
            <RandomizedText
              split="words"
              duration={1200}
              v={{
                en: "Every feature, every number and its conditions, the full register. Topic by topic, like the app's Settings.",
                th: "ทุกฟีเจอร์ ทุกตัวเลขและเงื่อนไข และทะเบียนฉบับเต็ม ทีละหัวข้อ เหมือนหน้า Settings ของแอป",
              }}
            />
          </p>
          <div className="cta">
          <Link className="btn btn--primary btn--lg" href="/details">
            <T v={{ en: "Open the details", th: "เปิดรายละเอียด" }} /> <span aria-hidden="true" className="arrow">→</span>
          </Link>
            <GitHubButton />
          </div>
        </Reveal>
      </div>
    </section>
  );
}
