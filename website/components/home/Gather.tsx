"use client";

import { T, useT } from "@/lib/i18n";
import { botAvatarPalette } from "@/vendor/bot-avatars/index.es.js";
import ParticleText from "../text/ParticleText";

/**
 * A short claim made of points: they wait scattered, and gather into the letters when the pointer comes over (or, with no pointer,
 * when the band comes into view). The words are the app's first rule: the coding tools ask before they act.
 */
export function Gather() {
  const t = useT();
  const word = t({ en: "Ask first.", th: "ถามก่อนลงมือ" });
  return (
    <section className="gather" aria-label={word}>
      <ParticleText
        text={word}
        particleSize={2}
        density={4}
        color="#ffffff"
        highlightColor={botAvatarPalette.pill}
        scatter={180}
        gatherDuration={1600}
        stagger={420}
        pointerRepel={40}
        repelRadius={120}
        idleDrift={0.7}
        trigger="hover"
        fontSize="clamp(3rem, 12vw, 8rem)"
        fontWeight={800}
        fontFamily='"Inter Variable", "Anuphan Variable", system-ui, sans-serif'
        glow
      />
      <p className="gather__hint mono" aria-hidden="true">
        <T v={{ en: "Move over it.", th: "ลองเลื่อนเมาส์ผ่าน" }} />
      </p>
    </section>
  );
}
