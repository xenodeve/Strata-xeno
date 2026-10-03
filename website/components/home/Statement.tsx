"use client";

import { T } from "@/lib/i18n";
import { Reveal } from "../Reveal";
import { SLIDES } from "@/data/story";
import { botAvatarPalette } from "@/vendor/bot-avatars/index.es.js";
import { AuroraField, WarpWord } from "../text/Words";

const by = (id: string) => botAvatarPalette[SLIDES.find((s) => s.id === id)!.avatar];
/** Three of the moods' colours, left to right: Composing (violet), Connecting (green), Searching (blue-grey). */
const STOPS: [string, string, string] = [by("composing"), by("connecting"), by("searching")];

/** One sentence, big, bending around the pointer. It is the claim the first mood makes: the prompt is read on your own PC. */
export function Statement() {
  return (
    <section className="section section--tight statement" aria-label="Nothing leaves it.">
      <AuroraField className="statement__aurora" colors={STOPS} />
      <div className="wrap">
        <Reveal>
          <WarpWord v={{ en: "Nothing leaves it.", th: "ไม่มีอะไรออกไปข้างนอก" }} />
          <p className="cap mono statement__cap">
            <T v={{ en: "Your prompt is read on your PC. Move over the words.", th: "prompt ถูกอ่านบนพีซีของคุณ ลองเลื่อนเมาส์ผ่านตัวอักษร" }} />
          </p>
        </Reveal>
      </div>
    </section>
  );
}
