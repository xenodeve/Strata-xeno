"use client";

import { SLIDES } from "@/data/story";
import { T, type L } from "@/lib/i18n";
import { botAvatarPalette } from "@/vendor/bot-avatars/index.es.js";
import { Reveal } from "../Reveal";
import { GradientWaveText, Marquee, TextMarquee } from "../text/Fx";

/** The 15 tools of the chat's coding agent, as listed under "Coding tools in the chat". */
const TOOLS = ["Read", "Write", "Edit", "Glob", "Grep", "TodoWrite", "ExitPlanMode", "NotebookEdit", "AskUserQuestion", "Bash", "BashOutput", "KillShell", "WebFetch", "WebSearch", "Task"];

/** What a person would say it does, one word each: read, search, edit, run, plan, ask. */
const VERBS: L[] = [
  { en: "read", th: "อ่าน" },
  { en: "search", th: "ค้นหา" },
  { en: "edit", th: "แก้ไข" },
  { en: "run commands", th: "รันคำสั่ง" },
  { en: "plan", th: "วางแผน" },
  { en: "ask you", th: "ถามคุณ" },
];

/**
 * A band between the moods and the demo: the line says what the chat can do, one verb at a time, and two strips carry the
 * tools (all fifteen) and the nine moods past the eye. Everything here is a fact listed elsewhere on the site.
 */
export function Tools() {
  return (
    <section className="section section--tight tools" aria-labelledby="tools-title">
      <div className="wrap">
        <Reveal>
          <h2 id="tools-title" className="tools__line">
            <TextMarquee
              prefix={<T v={{ en: "It can", th: "มันสามารถ" }} />}
              items={VERBS}
              suffix={
                <GradientWaveText>
                  <T v={{ en: "in your project.", th: "ในโปรเจกต์ของคุณ" }} />
                </GradientWaveText>
              }
            />
          </h2>
        </Reveal>
      </div>

      <div className="tools__strips">
        <Marquee duration={46} label="The 15 tools">
          {TOOLS.map((n) => (
            <span key={n} className="fxchip fxchip--tool mono">
              {n}
            </span>
          ))}
        </Marquee>
        <Marquee duration={58} reverse label="The nine moods">
          {SLIDES.map((s) => (
            <span key={s.id} className="fxchip">
              <span className="fxchip__dot" style={{ background: botAvatarPalette[s.avatar] }} aria-hidden="true" />
              {s.status}
            </span>
          ))}
        </Marquee>
      </div>
    </section>
  );
}
